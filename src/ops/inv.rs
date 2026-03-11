//! Multiplicative inverse via stable-prefix batched division.
//!
//! Instead of Newton-Raphson iteration, we compute `1/x` as `2^P / prefix`
//! where `prefix` is the leading bits shared by both endpoints of the inner
//! interval. As the inner interval narrows, the prefix grows and we extend
//! the division cheaply using the stored remainder.

use std::sync::Arc;

use crate::binary::{
    Binary, ReciprocalWithRemainder, UXBinary, XBinary, extend_reciprocal,
    reciprocal_with_remainder,
};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::{self, U, XI};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use parking_lot::RwLock;

/// Minimum seed precision bits for division initialization.
const MIN_SEED_PRECISION_BITS: U = 64;

/// Division state for computing 1/x via the stable prefix of the inner interval.
struct PrefixDivision {
    recip: ReciprocalWithRemainder,
    /// The stable prefix value (common high bits of aligned endpoints).
    prefix: BigUint,
    /// Number of uncertain low bits: denom ∈ [prefix << shift, (prefix+1) << shift).
    prefix_shift: U,
    /// Exponent of the aligned interval (the common exponent after alignment).
    prefix_exponent: BigInt,
}

/// Mutable state for the prefix-based division approach.
pub(crate) struct DivisionState {
    div: PrefixDivision,
    negate_result: bool,
}

/// Inverse (reciprocal) operation with stable-prefix batched division.
pub struct InvOp {
    pub inner: Arc<Node>,
    pub division_state: RwLock<Option<DivisionState>>,
}

/// Info extracted from the stable prefix of the inner interval endpoints.
struct PrefixInfo {
    prefix: BigUint,
    prefix_shift: U,
    prefix_exponent: BigInt,
    prefix_bits: U,
}

/// Extract the stable prefix from two positive Binary values.
///
/// The stable prefix consists of the leading bits common to both endpoints
/// when aligned to the same exponent. These bits are guaranteed correct
/// regardless of where the true value falls in the interval.
fn extract_prefix(abs_lower: &Binary, abs_upper: &Binary) -> Option<PrefixInfo> {
    let (lo_m, hi_m, exponent) = Binary::align_mantissas(abs_lower, abs_upper);

    // Both should be positive after abs
    debug_assert!(lo_m >= BigInt::zero());
    debug_assert!(hi_m >= BigInt::zero());

    let lo_uint = lo_m.magnitude().clone();
    let hi_uint = hi_m.magnitude().clone();

    if lo_uint.is_zero() {
        return None;
    }

    let diff = &hi_uint - &lo_uint;
    let diff_bits = if diff.is_zero() {
        // Exact: endpoints are equal, all bits are stable
        0
    } else {
        sane::bits_as_u(diff.bits())
    };

    // prefix = lo >> diff_bits (common high bits)
    let prefix = &lo_uint >> diff_bits;
    if prefix.is_zero() {
        // No common prefix (e.g. interval crosses a power-of-2 boundary)
        return None;
    }

    let prefix_bits = sane::bits_as_u(prefix.bits());

    Some(PrefixInfo {
        prefix,
        prefix_shift: diff_bits,
        prefix_exponent: BigInt::from(exponent),
        prefix_bits,
    })
}

/// Derive upper and lower bounds on 1/x from the stored division state.
///
/// Given `2^Q / prefix = q rem r`, with `prefix_shift = s`, `prefix_exponent = e`:
/// - True denominator x ∈ [prefix * 2^(s+e), (prefix+1) * 2^(s+e))
///
/// When `prefix_shift > 0` (uncertain bits exist):
/// - Upper bound on 1/x: ceil(2^Q / prefix) * 2^(-Q-s-e)
/// - Lower bound on 1/x: floor(2^Q / (prefix+1)) * 2^(-Q-s-e)
///
/// When `prefix_shift == 0` (exact denominator, all bits stable):
/// - Both bounds come from the quotient: [q, q + (1 if r>0)] * 2^(-Q-e)
fn bounds_from_division(div: &PrefixDivision) -> (Binary, Binary) {
    let q = &div.recip.quotient;
    let r = &div.recip.remainder;
    let precision = div.recip.precision_bits;

    // Common exponent: -Q - s - e
    let result_exponent =
        -BigInt::from(precision) - BigInt::from(div.prefix_shift) - &div.prefix_exponent;

    // Upper bound: ceil(2^Q / prefix) = q + (1 if r > 0 else 0)
    let upper_mantissa = if r.is_zero() {
        q.clone()
    } else {
        q + BigUint::one()
    };

    let lower_mantissa = if div.prefix_shift == 0 {
        // Exact denominator: lower bound = floor(2^Q / prefix) = q
        q.clone()
    } else {
        // Uncertain bits: lower bound = floor(2^Q / (prefix+1))
        // = q - ceil((q - r) / (prefix + 1))
        let prefix_plus_one = &div.prefix + BigUint::one();
        if q >= r {
            let q_minus_r = q - r;
            let correction = q_minus_r.div_ceil(&prefix_plus_one);
            q - &correction
        } else {
            q.clone()
        }
    };

    let result_exp_i = result_exponent.to_i32().unwrap_or_else(|| {
        crate::detected_computable_would_exhaust_memory!(
            "result exponent exceeds I in bounds_from_division"
        )
    });
    let upper = Binary::new(BigInt::from(upper_mantissa), result_exp_i);
    let lower = Binary::new(BigInt::from(lower_mantissa), result_exp_i);

    (lower, upper)
}

impl NodeOp for InvOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let state = self.division_state.read();

        match &*state {
            None => {
                let existing = self.inner.get_prefix()?;
                let lower = existing.lower();
                let upper = existing.upper();
                let zero = XBinary::zero();
                if lower <= zero && upper >= zero {
                    Ok(Prefix::unbounded())
                } else if upper < zero {
                    Ok(Prefix::from_lower_upper(XBinary::NegInf, XBinary::zero()))
                } else {
                    Ok(Prefix::from_lower_upper(XBinary::zero(), XBinary::PosInf))
                }
            }
            Some(s) => {
                let (lo, hi) = bounds_from_division(&s.div);
                let (out_lo, out_hi) = if s.negate_result {
                    (XBinary::Finite(hi.neg()), XBinary::Finite(lo.neg()))
                } else {
                    (XBinary::Finite(lo), XBinary::Finite(hi))
                };
                Ok(Prefix::from_lower_upper(out_lo, out_hi))
            }
        }
    }

    fn refine_step(&self, target_width_exp: XI) -> Result<bool, ComputableError> {
        let input_prefix = self.inner.get_prefix()?;
        let mut state = self.division_state.write();

        // Determine sign and absolute endpoints
        let lower = input_prefix.lower();
        let upper = input_prefix.upper();
        let zero = XBinary::zero();

        if lower <= zero && upper >= zero {
            // Interval spans zero — can't compute reciprocal yet
            *state = None;
            return Ok(true);
        }

        let (lower_finite, upper_finite) = match (&lower, &upper) {
            (XBinary::Finite(lo), XBinary::Finite(hi)) => (lo.clone(), hi.clone()),
            _ => {
                *state = None;
                return Ok(true);
            }
        };

        let negate_result = upper_finite.mantissa().is_negative();
        let (abs_lower, abs_upper) = if negate_result {
            (upper_finite.neg(), lower_finite.neg())
        } else {
            (lower_finite, upper_finite)
        };

        // Extract stable prefix
        let prefix_info = match extract_prefix(&abs_lower, &abs_upper) {
            Some(info) => info,
            None => {
                // No useful prefix — return wide bounds, wait for inner to refine
                *state = None;
                return Ok(true);
            }
        };

        // The coordinator dispatches us when our bounds are wider than its budget,
        // which may require tighter bounds than target_width_exp implies (e.g. summing
        // many terms). We must always make progress when dispatched.
        //
        // Strategy: jump to min_q in one shot when the target is finite and
        // reasonable. Double the existing precision as a fallback for progress
        // (handles Inf targets and cases where budget needs tighter bounds).
        // Cap at 2×prefix_bits when uncertain bits exist.
        let current_prec = match &*state {
            Some(s) => s.div.recip.precision_bits,
            None => 0,
        };
        let doubled = if current_prec == 0 {
            MIN_SEED_PRECISION_BITS
        } else {
            sane_mul_or_max(current_prec, 2)
        };

        // Convert target to precision bits and compute min quotient bits needed.
        // For Inf targets, fall back to doubling (one_shot_feasible = false).
        let (min_q, one_shot_feasible) = match target_width_exp {
            XI::Finite {
                sign: crate::sane::Sign::Neg,
                magnitude,
            } => {
                // Fine target: precision_bits = magnitude.
                let precision_bits = magnitude;
                let q = required_quotient_bits(
                    precision_bits,
                    prefix_info.prefix_shift,
                    &prefix_info.prefix_exponent,
                );
                (q, true)
            }
            XI::Finite { .. } | XI::PosInf => {
                // Coarse target (e >= 0) or unbounded: precision_bits = 0.
                let q = required_quotient_bits(
                    0,
                    prefix_info.prefix_shift,
                    &prefix_info.prefix_exponent,
                );
                (q, true)
            }
            XI::NegInf => {
                // Exact target: fall back to doubling.
                (doubled, false)
            }
        };

        let target = if prefix_info.prefix_shift > 0 {
            let max_useful = sane_mul_or_max(prefix_info.prefix_bits, 2);
            if one_shot_feasible {
                min_q
                    .max(doubled)
                    .min(max_useful)
                    .max(MIN_SEED_PRECISION_BITS)
            } else {
                doubled.min(max_useful).max(MIN_SEED_PRECISION_BITS)
            }
        } else if one_shot_feasible {
            min_q.max(doubled).max(MIN_SEED_PRECISION_BITS)
        } else {
            doubled
        };

        match &mut *state {
            None => {
                // Fresh initialization
                let recip = reciprocal_with_remainder(&prefix_info.prefix, target);
                *state = Some(DivisionState {
                    div: PrefixDivision {
                        recip,
                        prefix: prefix_info.prefix,
                        prefix_shift: prefix_info.prefix_shift,
                        prefix_exponent: prefix_info.prefix_exponent,
                    },
                    negate_result,
                });
                Ok(true)
            }
            Some(existing) => {
                if prefix_info.prefix != existing.div.prefix
                    || prefix_info.prefix_shift != existing.div.prefix_shift
                    || prefix_info.prefix_exponent != existing.div.prefix_exponent
                {
                    // Prefix changed (inner operand refined) — new divisor, fresh division
                    let recip = reciprocal_with_remainder(&prefix_info.prefix, target);
                    *state = Some(DivisionState {
                        div: PrefixDivision {
                            recip,
                            prefix: prefix_info.prefix,
                            prefix_shift: prefix_info.prefix_shift,
                            prefix_exponent: prefix_info.prefix_exponent,
                        },
                        negate_result,
                    });
                    Ok(true)
                } else if target > existing.div.recip.precision_bits {
                    // Same prefix, need more bits — extend via remainder carry
                    let new_recip =
                        extend_reciprocal(&existing.div.recip, &existing.div.prefix, target);
                    existing.div.recip = new_recip;
                    existing.negate_result = negate_result;
                    Ok(true)
                } else {
                    // Can't improve: uncertain bits cap prevents extending further.
                    // Need the inner operand to refine (extending the prefix).
                    Ok(false)
                }
            }
        }
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }

    fn child_demand_budget(&self, target_width: &UXBinary, _child_idx: bool) -> UXBinary {
        let min_abs = match self.inner.cached_prefix() {
            Some(p) => {
                let (lo, hi) = p.abs();
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

/// Compute the minimum quotient bits Q such that the output width
/// `2^(-Q - shift - exponent)` is at most `2^(-precision_bits)`.
///
/// Requires: `Q ≥ precision_bits - shift - exponent`.
/// Clamps to `[0, U::MAX]`.
fn required_quotient_bits(precision_bits: U, shift: U, exponent: &BigInt) -> U {
    let target = BigInt::from(precision_bits) - BigInt::from(shift) - exponent;
    if target <= BigInt::zero() {
        return 0;
    }
    match target.to_u32() {
        Some(q) => q,
        None => U::MAX,
    }
}

/// Saturating multiply for U: returns `a * b` or `U::MAX` on overflow.
fn sane_mul_or_max(a: U, b: U) -> U {
    a.saturating_mul(b)
}

#[cfg(test)]
mod tests {
    use crate::binary::XBinary;
    use crate::refinement::prefix_width_leq;
    use crate::sane::XI;
    use crate::test_utils::{bin, interval_midpoint_computable, unwrap_finite};

    #[test]
    fn inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let prefix = inv.prefix().expect("prefix should succeed");
        assert_eq!(prefix.lower(), XBinary::NegInf);
        assert_eq!(prefix.upper(), XBinary::PosInf);
    }

    #[test]
    fn inv_bounds_for_positive_interval() {
        // interval_midpoint_computable(2, 4) refines to midpoint 3, so inv()
        // should produce prefix that brackets 1/3.
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let tolerance_exp = XI::from_i32(-8);
        let prefix = inv
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(&prefix.lower());
        let upper = unwrap_finite(&prefix.upper());
        let three = bin(3, 0);
        let one = bin(1, 0);

        // Verify prefix brackets 1/3 using exact arithmetic: lower * 3 <= 1 <= upper * 3.
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
            prefix_width_leq(&prefix, &tolerance_exp),
            "prefix width exceeds tolerance"
        );
    }
}
