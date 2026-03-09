//! Multiplicative inverse operation with Newton-Raphson refinement.
//!
//! Each `refine_step` performs one N-R iteration on both interval endpoints,
//! doubling the number of correct bits per step via quadratic convergence.
//! The N-R iterates are reused across steps (unlike the old approach which
//! recomputed a full-precision division each step).

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::Signed;
use parking_lot::RwLock;
use std::convert::TryFrom;

use crate::binary::{
    Binary, Bounds, ReciprocalRounding, UXBinary, XBinary, reciprocal_rounded_abs_extended,
};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::sane::{self, XIsize, XUsize};

/// Minimum seed precision bits for Newton-Raphson initialization.
///
/// Even when the coordinator requests low precision, we use at least this many
/// bits for the seed to ensure N-R converges quickly for typical use cases.
const MIN_SEED_PRECISION_BITS: usize = 64;

/// N-R approximation of `1/denom`: maintains `lower <= 1/denom <= upper`.
///
/// Tracks a precision budget (in mantissa bits) that doubles each N-R step.
/// After each step, both bounds are truncated to the new precision to prevent
/// quadratic mantissa blowup from exact `Binary` arithmetic.
struct ReciprocalApprox {
    lower: Binary,
    upper: Binary,
    /// Current mantissa precision budget in bits. Doubles each N-R step.
    precision: usize,
}

/// Newton-Raphson state for computing `1/x` over an interval.
///
/// Tracks two N-R sequences: one for each endpoint of the absolute inner
/// interval. As inner bounds narrow, the denominators are updated and the
/// existing iterates serve as warm-start seeds (they remain valid bounds).
pub(crate) struct NewtonState {
    /// Approximation of `1/abs_upper` — its `lower` field is the output lower bound.
    lo: ReciprocalApprox,
    /// Approximation of `1/abs_lower` — its `upper` field is the output upper bound.
    hi: ReciprocalApprox,
    /// Current positive lower endpoint of inner interval.
    abs_lower: Binary,
    /// Current positive upper endpoint of inner interval.
    abs_upper: Binary,
    /// True if the input was negative (result should be negated).
    negate_result: bool,
}

/// Inverse (reciprocal) operation with Newton-Raphson refinement.
///
/// For inv, we don't normalize to prefix form because prefix normalization
/// can expand the interval by up to ~4x, which interferes with the
/// N-R refinement. The reciprocal computation already produces bounds
/// with controlled mantissa size (proportional to the N-R iteration count).
pub struct InvOp {
    pub inner: Arc<Node>,
    /// Newton-Raphson state. `None` means not yet initialized.
    pub newton_state: RwLock<Option<NewtonState>>,
}

impl NodeOp for InvOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let state = self.newton_state.read();

        match &*state {
            None => {
                // No state yet — return infinite bounds in the appropriate direction.
                let existing = self.inner.get_bounds()?;
                let lower = existing.small();
                let upper = existing.large();
                let zero = XBinary::zero();
                if lower <= &zero && *upper >= zero {
                    Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf))
                } else if *upper < zero {
                    Ok(Bounds::new(XBinary::NegInf, XBinary::zero()))
                } else {
                    Ok(Bounds::new(XBinary::zero(), XBinary::PosInf))
                }
            }
            Some(s) => {
                // Output lower = lo.lower (lower bound on 1/abs_upper)
                // Output upper = hi.upper (upper bound on 1/abs_lower)
                let (out_lo, out_hi) = if s.negate_result {
                    (
                        XBinary::Finite(s.hi.upper.neg()),
                        XBinary::Finite(s.lo.lower.neg()),
                    )
                } else {
                    (
                        XBinary::Finite(s.lo.lower.clone()),
                        XBinary::Finite(s.hi.upper.clone()),
                    )
                };
                Ok(Bounds::new(out_lo, out_hi))
            }
        }
    }

    fn refine_step(&self, target_width_exp: XIsize) -> Result<bool, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let mut state = self.newton_state.write();

        match &mut *state {
            None => {
                let seed_precision = match target_width_exp.to_precision_bits() {
                    XUsize::Finite(bits) if bits <= crate::MAX_COMPUTATION_BITS => {
                        bits.max(MIN_SEED_PRECISION_BITS)
                    }
                    XUsize::Finite(_) | XUsize::Inf => MIN_SEED_PRECISION_BITS,
                };
                *state = try_initialize(&input_bounds, seed_precision)?;
                Ok(true)
            }
            Some(s) => {
                // Read current inner bounds and update denominators.
                update_denominators(s, &input_bounds)?;

                // One Newton step per dispatch. Newton doubles precision each
                // step (quadratic convergence), so one step always produces a
                // visible Prefix change. The coordinator handles iteration
                // count. Doing multiple steps risks astronomical mantissa
                // growth when the target is unreachable (e.g. i64::MIN for
                // irrational values).
                newton_step(&mut s.lo, &s.abs_upper);
                newton_step(&mut s.hi, &s.abs_lower);
                Ok(true)
            }
        }
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }

    /// Sensitivity of 1/x: derivative = -1/x², so max |derivative| over
    /// [a,b] is 1/min_abs² (worst case at the value closest to zero).
    /// Child budget = target × min_abs².
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        let min_abs = match self.inner.cached_bounds() {
            Some(b) => {
                let (lo, hi) = b.abs();
                std::cmp::min(lo, hi)
            }
            None => return target_width.clone(),
        };
        target_width.mul(&min_abs).mul(&min_abs)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        true
    }
}

/// Try to initialize Newton state from the current inner bounds.
/// Returns `Ok(None)` if bounds span zero or are infinite (caller should retry later).
///
/// `seed_precision` controls the number of bits used for the initial reciprocal
/// seed. A higher value means fewer N-R doublings to reach the target precision.
fn try_initialize(
    input_bounds: &Bounds,
    seed_precision: usize,
) -> Result<Option<NewtonState>, ComputableError> {
    let lower = input_bounds.small();
    let upper = input_bounds.large();
    let zero = XBinary::zero();

    if lower <= &zero && *upper >= zero {
        return Ok(None);
    }
    let (lower_finite, upper_finite) = match (lower, upper) {
        (XBinary::Finite(lo), XBinary::Finite(hi)) => (lo.clone(), hi.clone()),
        _ => return Ok(None),
    };

    let negate_result = upper_finite.mantissa().is_negative();
    let (abs_lower, abs_upper) = if negate_result {
        (upper_finite.neg(), lower_finite.neg())
    } else {
        (lower_finite, upper_finite)
    };

    let lo = seed_reciprocal(&abs_upper, seed_precision)?;
    let hi = seed_reciprocal(&abs_lower, seed_precision)?;

    Ok(Some(NewtonState {
        lo,
        hi,
        abs_lower,
        abs_upper,
        negate_result,
    }))
}

/// Compute a seed reciprocal for N-R initialization at the given precision.
fn seed_reciprocal(
    denom: &Binary,
    seed_precision: usize,
) -> Result<ReciprocalApprox, ComputableError> {
    let precision = i64::try_from(seed_precision).unwrap_or_else(|_| {
        crate::detected_computable_would_exhaust_memory!(
            "seed_precision exceeds i64 in seed_reciprocal"
        )
    });
    let xb_denom = XBinary::Finite(denom.clone());

    let lower_xb =
        reciprocal_rounded_abs_extended(&xb_denom, precision, ReciprocalRounding::Floor)?;
    let upper_xb = reciprocal_rounded_abs_extended(&xb_denom, precision, ReciprocalRounding::Ceil)?;

    let lower = match lower_xb {
        XBinary::Finite(b) => b,
        XBinary::NegInf | XBinary::PosInf => Binary::zero(),
    };
    let upper = match upper_xb {
        XBinary::Finite(b) => b,
        XBinary::NegInf | XBinary::PosInf => Binary::zero(),
    };

    Ok(ReciprocalApprox {
        lower,
        upper,
        precision: seed_precision,
    })
}

/// Update the stored denominators from current inner bounds.
///
/// When inner bounds narrow, the N-R iterates remain valid but may need
/// their error bounds refreshed:
/// - `abs_upper` decreases: `lo.lower` stays valid, `lo.upper` may need refresh.
/// - `abs_lower` increases: `hi.upper` stays valid, `hi.lower` may need refresh.
fn update_denominators(
    state: &mut NewtonState,
    input_bounds: &Bounds,
) -> Result<(), ComputableError> {
    let lower = input_bounds.small();
    let upper = &input_bounds.large();

    let (lower_finite, upper_finite) = match (lower, upper) {
        (XBinary::Finite(lo), XBinary::Finite(hi)) => (lo.clone(), hi.clone()),
        _ => return Ok(()), // still infinite, keep old denominators
    };

    let (new_abs_lower, new_abs_upper) = if state.negate_result {
        (upper_finite.neg(), lower_finite.neg())
    } else {
        (lower_finite, upper_finite)
    };

    let one = Binary::one();

    // Update abs_upper (denominator for lo sequence).
    if new_abs_upper != state.abs_upper {
        state.abs_upper = new_abs_upper.clone();

        // lo.lower ≤ 1/old_abs_upper ≤ 1/new_abs_upper: still valid ✓
        // lo.upper: was ≥ 1/old_abs_upper, but 1/new_abs_upper may be larger.
        // Check: lo.upper * new_abs_upper ≥ 1?
        if state.lo.upper.mul(&new_abs_upper) < one {
            // lo.upper is too small. Use hi.upper as a conservative replacement.
            // hi.upper ≥ 1/abs_lower ≥ 1/abs_upper since abs_lower ≤ abs_upper.
            state.lo.upper = state.hi.upper.clone();
        }
    }

    // Update abs_lower (denominator for hi sequence).
    if new_abs_lower != state.abs_lower {
        state.abs_lower = new_abs_lower.clone();

        // hi.upper ≥ 1/old_abs_lower ≥ 1/new_abs_lower: still valid ✓
        // hi.lower: was ≤ 1/old_abs_lower, but 1/new_abs_lower may be smaller.
        // Check: hi.lower * new_abs_lower ≤ 1?
        if state.hi.lower.mul(&new_abs_lower) > one {
            // hi.lower is too large. Use lo.lower as a conservative replacement.
            // lo.lower ≤ 1/abs_upper ≤ 1/abs_lower since abs_upper ≥ abs_lower.
            state.hi.lower = state.lo.lower.clone();
        }
    }

    Ok(())
}

/// Performs one Newton-Raphson iteration on a `ReciprocalApprox`.
///
/// For `a > 0`, the N-R map `f(x) = x * (2 - a*x)` satisfies:
///   `f(x) - 1/a = -a*(x - 1/a)^2`
///
/// This means `f(x) <= 1/a` always — N-R naturally produces **lower bounds**.
/// For the upper bound: given `x <= 1/a <= upper`, the error after one step is
///   `|f(x) - 1/a| = a*(x - 1/a)^2 <= a*(upper - x)^2`
/// so `upper_new = f(x) + a*(upper - x)^2` is a valid upper bound.
///
/// After computing the new bounds, both are truncated to `2 * precision` bits
/// to prevent quadratic mantissa blowup, then `precision` is doubled.
fn newton_step(approx: &mut ReciprocalApprox, denom: &Binary) {
    let x = &approx.lower;

    // gap = upper - lower
    let gap = approx.upper.sub(x);

    // x_new = x * (2 - a * x)
    // 2 = 1 * 2^1; mantissa 1 is odd, so skip normalization.
    let two = Binary::new_normalized(BigInt::from(1_i32), 1_i64);
    let ax = denom.mul(x);
    let two_minus_ax = two.sub(&ax);
    let x_new = x.mul(&two_minus_ax);

    // err = a * gap^2
    let gap_sq = gap.mul(&gap);
    let err = denom.mul(&gap_sq);

    // upper_new = x_new + err
    let upper_new = x_new.add(&err);

    // Double the precision budget for this step.
    let precision = approx.precision;
    let new_precision = crate::sane_arithmetic!(precision; precision * 2);

    // Truncate to the new precision to keep mantissa sizes bounded.
    // Lower bound: truncate toward -∞ (floor) to stay ≤ 1/denom.
    // Upper bound: truncate toward +∞ (ceil) to stay ≥ 1/denom.
    let x_trunc = truncate_floor(&x_new, new_precision);
    let upper_trunc = truncate_ceil(&upper_new, new_precision);

    // Only update if bounds actually improve.
    if x_trunc > approx.lower {
        approx.lower = x_trunc;
    }
    if upper_trunc < approx.upper {
        approx.upper = upper_trunc;
    }

    approx.precision = new_precision;
}

/// Truncate a positive `Binary` to at most `precision_bits` mantissa bits,
/// rounding toward -∞ (floor). The result is always ≤ the input.
fn truncate_floor(x: &Binary, precision_bits: usize) -> Binary {
    let bit_length = sane::bits_as_usize(x.mantissa().magnitude().bits());
    let Some(shift) = bit_length
        .checked_sub(precision_bits)
        .filter(|&s| s > 0_usize)
    else {
        return x.clone();
    };
    let shifted = x.mantissa().magnitude() >> shift;

    // For positive values, truncation toward zero IS floor.
    // For negative values, truncation toward zero is ceil, so we need to subtract 1.
    let has_remainder = (&shifted << shift) != *x.mantissa().magnitude();
    let signed = if x.mantissa().is_negative() && has_remainder {
        -BigInt::from(shifted) - BigInt::from(1_i32)
    } else if x.mantissa().is_negative() {
        -BigInt::from(shifted)
    } else {
        BigInt::from(shifted)
    };

    let new_exp = x
        .exponent()
        .checked_add(i64::try_from(shift).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!("shift exceeds i64 in inv")
        }))
        .unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in inv")
        });
    Binary::new(signed, new_exp)
}

/// Truncate a positive `Binary` to at most `precision_bits` mantissa bits,
/// rounding toward +∞ (ceil). The result is always ≥ the input.
fn truncate_ceil(x: &Binary, precision_bits: usize) -> Binary {
    let bit_length = sane::bits_as_usize(x.mantissa().magnitude().bits());
    let Some(shift) = bit_length
        .checked_sub(precision_bits)
        .filter(|&s| s > 0_usize)
    else {
        return x.clone();
    };
    let shifted = x.mantissa().magnitude() >> shift;
    let has_remainder = (&shifted << shift) != *x.mantissa().magnitude();

    // For positive values, truncation toward zero is floor, so add 1 if remainder.
    // For negative values, truncation toward zero IS ceil.
    let signed = if !x.mantissa().is_negative() && has_remainder {
        BigInt::from(shifted) + BigInt::from(1_i32)
    } else if x.mantissa().is_negative() {
        -BigInt::from(shifted)
    } else {
        BigInt::from(shifted)
    };

    let new_exp = x
        .exponent()
        .checked_add(i64::try_from(shift).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!("shift exceeds i64 in inv")
        }))
        .unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in inv")
        });
    Binary::new(signed, new_exp)
}

#[cfg(test)]
mod tests {
    use crate::binary::{Bounds, XBinary};
    use crate::refinement::bounds_width_leq;
    use crate::sane::XUsize;
    use crate::test_utils::{bin, interval_midpoint_computable, unwrap_finite};

    #[test]
    fn inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(XBinary::NegInf, XBinary::PosInf));
    }

    #[test]
    fn inv_bounds_for_positive_interval() {
        // interval_midpoint_computable(2, 4) refines to midpoint 3, so inv()
        // should produce bounds that bracket 1/3.
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let tolerance_exp = XUsize::Finite(8);
        let bounds = inv
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(bounds.large());
        let three = bin(3, 0);
        let one = bin(1, 0);

        // Verify bounds bracket 1/3 using exact arithmetic: lower * 3 <= 1 <= upper * 3.
        assert!(
            lower.mul(&three) <= one,
            "lower bound {lower} exceeds 1/3: lower * 3 = {}",
            lower.mul(&three)
        );
        assert!(
            upper.mul(&three) >= one,
            "upper bound {upper} is below 1/3: upper * 3 = {}",
            upper.mul(&three)
        );
        assert!(
            bounds_width_leq(&bounds, &tolerance_exp),
            "bounds width exceeds tolerance"
        );
    }
}
