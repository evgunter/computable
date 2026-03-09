//! N-th root operation with Newton-Raphson refinement.
//!
//! This module implements the n-th root operation (x^(1/n)) using:
//! - Newton-Raphson iteration for quadratic convergence (~9 steps for 256 bits)
//! - Directed rounding for provably correct bounds
//!
//! The algorithm maintains an interval [lower, upper] where the true root lies.
//! Each Newton step doubles the number of correct bits (quadratic convergence).
//!
//! Newton iteration for y = x^(1/n): y_{k+1} = ((n-1)*y + x/y^{n-1}) / n
//! Convexity of y^n ensures Newton from above stays above the root (valid upper bound).
//! Lower bound: x / upper^{n-1} <= root when upper >= root.
//!
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
//! numbers (but aren't completely negative) currently just return (0, inf) bounds
//! instead of returning a recoverable error that would trigger refinement of the
//! input until the bounds are fully non-negative. This should be fixed to match
//! the behavior described in the README for sqrt.
//!
//! BLOCKED: This requires node-initiated refinement -- the ability for a node's
//! `refine_step` to return a recoverable error requesting that the coordinator
//! refine a specific input before retrying. The current model doesn't support
//! this: the coordinator decides which refiners to step, and nodes cannot signal
//! "my input bounds are too wide, refine them first."

use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::{Binary, Bounds, UXBinary, XBinary};
use crate::binary_utils::bisection::midpoint;
use crate::binary_utils::power::binary_pow;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::sane::{self, XIsize, XUsize};

/// Minimum seed precision bits for Newton-Raphson initialization.
///
/// Even when the coordinator requests low precision, we use at least this many
/// bits for the seed to ensure N-R converges quickly for typical use cases.
const MIN_SEED_PRECISION_BITS: usize = 64;

/// N-th root operation with Newton-Raphson refinement.
///
/// Computes x^(1/n) where n is the root degree.
/// For n=2, this is square root; n=3 is cube root, etc.
///
/// # Constraints
/// - For even n: requires x >= 0 (otherwise returns domain error)
/// - For odd n: supports all real x (negative values have negative roots)
pub struct NthRootOp {
    /// The input node whose n-th root we're computing.
    pub inner: Arc<Node>,
    /// The root degree (n in x^(1/n)). Guaranteed to be >= 1 by the type system.
    pub degree: NonZeroU32,
    /// Newton-Raphson state. `None` until first `refine_step`.
    ///
    /// This is `None` until the first `refine_step` call, which initializes
    /// it from the input bounds. We use `Option` because initialization requires
    /// calling `inner.get_bounds()` which can fail, but node construction (via
    /// `nth_root()`) is not supposed to be fallible. By deferring initialization
    /// to the first call (which returns `Result`), we can propagate errors
    /// through the normal Result path.
    pub newton_state: RwLock<Option<NthRootNewtonState>>,
}

/// State for Newton-Raphson nth root computation.
#[derive(Clone, Debug)]
pub struct NthRootNewtonState {
    /// Current lower bound on the nth root.
    pub lower: Binary,
    /// Current upper bound on the nth root.
    pub upper: Binary,
    /// The target value (x) whose n-th root we're computing.
    pub target: Binary,
    /// Whether the result should be negated (for odd roots of negative numbers).
    pub negate_result: bool,
    /// If set, the exact root value.
    pub exact_value: Option<Binary>,
    /// Current mantissa precision budget in bits. Doubles each N-R step.
    pub precision: usize,
}

impl NodeOp for NthRootOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;

        // Fast path: read lock to check if already initialized.
        {
            let read_guard = self.newton_state.read();
            if let Some(s) = &*read_guard {
                return Ok(bounds_from_newton_state(s));
            }
        }

        // Slow path: upgrade to write lock and eagerly initialize.
        // Eager initialization is critical: returning wide initial bounds
        // (e.g., [1, target]) before Newton refinement would cause massive
        // bound explosions when many nth-root nodes are summed.
        // Double-check after acquiring write lock (another thread may have initialized).
        let mut write_guard = self.newton_state.write();
        if let Some(s) = &*write_guard {
            return Ok(bounds_from_newton_state(s));
        }
        let s = initialize_nth_root_newton_state(
            &input_bounds,
            self.degree.get(),
            MIN_SEED_PRECISION_BITS,
        )?;
        let bounds = bounds_from_newton_state(&s);
        *write_guard = Some(s);
        Ok(bounds)
    }

    fn refine_step(&self, target_width_exp: XIsize) -> Result<bool, ComputableError> {
        // Ensure state is initialized (compute_bounds is always called
        // before refine_step by the coordinator, but be defensive).
        {
            let read_guard = self.newton_state.read();
            if read_guard.is_none() {
                drop(read_guard);
                // Trigger initialization via compute_bounds.
                self.compute_bounds()?;
            }
        }

        // Convert target_width_exp to a precision cap in bits.
        // This prevents unbounded precision growth: the refiner_loop calls
        // refine_step up to 16 times per dispatch, and Newton doubles precision
        // each step. Without a cap, 16 steps would escalate from 64 to
        // 64 * 2^16 = 4M bits, making binary_pow astronomically expensive.
        let precision_cap = match target_width_exp.to_precision_bits() {
            XUsize::Finite(bits) => {
                // Allow 2x headroom beyond target so Newton has room to converge
                // past the target, but don't let it grow unboundedly.
                // Also ensure at least MIN_SEED_PRECISION_BITS.
                bits.saturating_mul(2)
                    .clamp(MIN_SEED_PRECISION_BITS, crate::MAX_COMPUTATION_BITS)
            }
            XUsize::Inf => crate::MAX_COMPUTATION_BITS,
        };

        // Perform one Newton step
        let mut write_guard = self.newton_state.write();
        let s = match write_guard.as_mut() {
            Some(s) => s,
            None => return Err(ComputableError::InfiniteBounds),
        };

        // If we already have an exact value, no need to refine
        if s.exact_value.is_some() {
            return Ok(false);
        }

        let degree = self.degree.get();
        let made_progress = newton_step_nth_root(s, degree, precision_cap);
        Ok(made_progress)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }

    /// Sensitivity of x^(1/n): derivative = (1/n) * x^((1-n)/n).
    /// Max |derivative| at smallest input x = a: (1/n) * a^((1-n)/n).
    /// Child budget = target * n * a^((n-1)/n).
    ///
    /// We approximate a^((n-1)/n) ~ a, which is conservative (budget too
    /// loose) for a >= 1 and slightly tight for a < 1. This avoids needing
    /// to compute an nth root inside the budget function.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        use crate::binary::UBinary;
        use num_bigint::BigUint;

        let n = self.degree.get();
        if n == 1 {
            return target_width.clone();
        }
        let min_abs = match self.inner.cached_bounds() {
            Some(b) => {
                let (lo, hi) = b.abs();
                std::cmp::min(lo, hi)
            }
            None => return target_width.clone(),
        };
        let n_ux = UXBinary::Finite(UBinary::new(BigUint::from(n), BigInt::zero()));
        target_width.mul(&n_ux).mul(&min_abs)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        self.degree.get() > 1
    }
}

/// Extracts bounds from an initialized Newton state.
fn bounds_from_newton_state(s: &NthRootNewtonState) -> Bounds {
    if let Some(exact) = &s.exact_value {
        let val = if s.negate_result {
            exact.neg()
        } else {
            exact.clone()
        };
        return Bounds::new(XBinary::Finite(val.clone()), XBinary::Finite(val));
    }

    let (out_lo, out_hi) = if s.negate_result {
        (
            XBinary::Finite(s.upper.neg()),
            XBinary::Finite(s.lower.neg()),
        )
    } else {
        (
            XBinary::Finite(s.lower.clone()),
            XBinary::Finite(s.upper.clone()),
        )
    };
    Bounds::new(out_lo, out_hi)
}

/// Initializes Newton-Raphson state for nth root computation.
///
/// Takes the midpoint of input bounds as the target value, then sets up
/// initial bounds and performs Newton steps at the seed precision.
fn initialize_nth_root_newton_state(
    input_bounds: &Bounds,
    degree: u32,
    seed_precision: usize,
) -> Result<NthRootNewtonState, ComputableError> {
    let lower = input_bounds.small();
    let upper = &input_bounds.large();

    // Get the target value
    let target = match (lower, upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => midpoint(l, u),
        _ => return Err(ComputableError::InfiniteBounds),
    };

    let is_even = degree.is_multiple_of(2);

    // Handle negative targets for even roots
    if is_even && target.mantissa().is_negative() {
        return Err(ComputableError::DomainError);
    }

    // For odd roots of negative values, compute root of |target| and negate
    let (actual_target, negate_result) = if !is_even && target.mantissa().is_negative() {
        (target.neg(), true)
    } else {
        (target.clone(), false)
    };

    // Handle zero
    if actual_target.mantissa().is_zero() {
        return Ok(NthRootNewtonState {
            lower: Binary::zero(),
            upper: Binary::zero(),
            target: actual_target,
            negate_result,
            exact_value: Some(Binary::zero()),
            precision: seed_precision,
        });
    }

    // Handle degree 1: root is the target itself
    if degree == 1 {
        return Ok(NthRootNewtonState {
            lower: actual_target.clone(),
            upper: actual_target.clone(),
            target: actual_target.clone(),
            negate_result,
            exact_value: Some(actual_target),
            precision: seed_precision,
        });
    }

    // Initial bounds: [1, target] for target >= 1, [target, 1] for target < 1
    let one = Binary::new(BigInt::one(), BigInt::zero());
    let (init_lower, init_upper) = if actual_target < one {
        (actual_target.clone(), one)
    } else {
        (one, actual_target.clone())
    };

    let mut state = NthRootNewtonState {
        lower: init_lower,
        upper: init_upper,
        target: actual_target,
        negate_result,
        exact_value: None,
        precision: seed_precision,
    };

    // Perform 2 initial Newton steps for a good seed.
    // Use a generous cap during initialization to allow the seed to converge
    // well; the refine_step cap will control subsequent growth.
    let init_cap = seed_precision.saturating_mul(8).min(crate::MAX_COMPUTATION_BITS);
    for _ in 0_u32..2_u32 {
        newton_step_nth_root(&mut state, degree, init_cap);
    }

    Ok(state)
}

/// Performs one Newton-Raphson step on the nth root state.
///
/// Newton iteration: y_{new} = ((n-1)*y + target/y^{n-1}) / n
///
/// From convexity of y^n:
/// - Newton from above (starting at upper) stays above: valid upper bound
/// - Lower bound: target / upper^{n-1} <= root (since upper >= root)
///
/// The `precision_cap` limits how far precision can grow, preventing
/// the refiner_loop from escalating mantissa sizes exponentially.
///
/// Returns `true` if bounds improved, `false` otherwise.
fn newton_step_nth_root(
    state: &mut NthRootNewtonState,
    degree: u32,
    precision_cap: usize,
) -> bool {
    if state.exact_value.is_some() {
        return false;
    }

    let precision = state.precision;
    let y = &state.upper;
    let target = &state.target;
    let n = degree;
    let n_minus_1 = n.saturating_sub(1);

    // Compute y^(n-1)
    let y_pow = binary_pow(y, n_minus_1);

    // If y^(n-1) is zero (shouldn't happen for positive upper), bail
    if y_pow.mantissa().is_zero() {
        return false;
    }

    // Newton step: y_new = ((n-1)*y + target/y^{n-1}) / n
    // For upper bound, round up: use ceil division
    let n_minus_1_times_y = scalar_mul(y, n_minus_1);
    let quotient_ceil = binary_div_ceil(target, &y_pow, precision);
    let numerator_ceil = n_minus_1_times_y.add(&quotient_ceil);
    let new_upper = binary_div_ceil_by_u32(&numerator_ceil, n, precision);

    // Lower bound: target / upper_new^{n-1}, rounded down
    // Use the better upper (new if improved, old otherwise).
    let upper_improved = new_upper < state.upper;
    let effective_upper = if upper_improved {
        &new_upper
    } else {
        &state.upper
    };

    let effective_upper_pow = binary_pow(effective_upper, n_minus_1);
    let new_lower = if effective_upper_pow.mantissa().is_zero() {
        state.lower.clone()
    } else {
        binary_div_floor(target, &effective_upper_pow, precision)
    };

    let lower_improved = new_lower > state.lower;

    if upper_improved || lower_improved {
        if upper_improved {
            state.upper = new_upper;
        }
        if lower_improved {
            state.lower = new_lower;
        }

        // Check if we found exact root
        if state.lower == state.upper {
            state.exact_value = Some(state.lower.clone());
        }

        // Double precision for next step (quadratic convergence),
        // but cap at the precision limit to prevent runaway growth.
        state.precision = precision
            .saturating_mul(2)
            .min(precision_cap)
            .min(crate::MAX_COMPUTATION_BITS);

        true
    } else {
        // No improvement; still double precision to try harder next time,
        // but cap at the precision limit.
        state.precision = precision
            .saturating_mul(2)
            .min(precision_cap)
            .min(crate::MAX_COMPUTATION_BITS);
        false
    }
}

/// Multiply a Binary by a u32 scalar.
fn scalar_mul(x: &Binary, scalar: u32) -> Binary {
    let mantissa = x.mantissa() * BigInt::from(scalar);
    Binary::new(mantissa, x.exponent().clone())
}

/// Divides a by b, rounding toward negative infinity (floor).
/// Result has at most `precision` bits of mantissa.
fn binary_div_floor(a: &Binary, b: &Binary, precision: usize) -> Binary {
    if a.mantissa().is_zero() {
        return Binary::zero();
    }

    let a_sign = a.mantissa().sign();
    let b_sign = b.mantissa().sign();
    let result_negative =
        (a_sign == num_bigint::Sign::Minus) != (b_sign == num_bigint::Sign::Minus);

    let a_abs = a.mantissa().magnitude().clone();
    let b_abs = b.mantissa().magnitude().clone();

    let b_bits = sane::bits_as_usize(b_abs.bits());
    let shift = crate::sane_arithmetic!(precision, b_bits; precision + b_bits);

    // Shift a left by `shift` bits, then divide by b
    let a_shifted = &a_abs << shift;
    let (quotient, remainder) = num_integer::Integer::div_rem(&a_shifted, &b_abs);

    // For floor division:
    // - positive result: use quotient as-is (truncation toward zero = floor)
    // - negative result: if remainder != 0, subtract 1
    let final_quotient = if result_negative && !remainder.is_zero() {
        quotient + BigInt::from(1_u32).magnitude().clone()
    } else {
        quotient
    };

    let mantissa = if result_negative {
        -BigInt::from(final_quotient)
    } else {
        BigInt::from(final_quotient)
    };

    // Exponent: a_exp - b_exp - shift
    let exp = a.exponent() - b.exponent() - BigInt::from(shift);

    let result = Binary::new(mantissa, exp);
    truncate_floor(&result, precision)
}

/// Divides a by b, rounding toward positive infinity (ceil).
/// Result has at most `precision` bits of mantissa.
fn binary_div_ceil(a: &Binary, b: &Binary, precision: usize) -> Binary {
    if a.mantissa().is_zero() {
        return Binary::zero();
    }

    let a_sign = a.mantissa().sign();
    let b_sign = b.mantissa().sign();
    let result_negative =
        (a_sign == num_bigint::Sign::Minus) != (b_sign == num_bigint::Sign::Minus);

    let a_abs = a.mantissa().magnitude().clone();
    let b_abs = b.mantissa().magnitude().clone();

    let b_bits = sane::bits_as_usize(b_abs.bits());
    let shift = crate::sane_arithmetic!(precision, b_bits; precision + b_bits);

    let a_shifted = &a_abs << shift;
    let (quotient, remainder) = num_integer::Integer::div_rem(&a_shifted, &b_abs);

    // For ceil division:
    // - positive result with remainder: add 1
    // - negative result: use quotient as-is (truncation toward zero = ceil for negatives)
    let final_quotient = if !result_negative && !remainder.is_zero() {
        quotient + BigInt::from(1_u32).magnitude().clone()
    } else {
        quotient
    };

    let mantissa = if result_negative {
        -BigInt::from(final_quotient)
    } else {
        BigInt::from(final_quotient)
    };

    let exp = a.exponent() - b.exponent() - BigInt::from(shift);

    let result = Binary::new(mantissa, exp);
    truncate_ceil(&result, precision)
}

/// Divides a Binary by a u32 scalar, rounding toward positive infinity (ceil).
/// Result has at most `precision` bits of mantissa.
fn binary_div_ceil_by_u32(a: &Binary, divisor: u32, precision: usize) -> Binary {
    if a.mantissa().is_zero() {
        return Binary::zero();
    }

    let divisor_big = BigInt::from(divisor);

    // Shift mantissa left by `precision` bits to preserve fractional precision
    let shifted_mantissa = a.mantissa() << precision;
    let (quotient, remainder) =
        num_integer::Integer::div_rem(&shifted_mantissa, &divisor_big);

    // For ceil: if positive and remainder != 0, add 1
    // if negative and remainder != 0, keep as-is (truncation toward zero = ceil for negatives)
    let final_quotient = if quotient.is_positive() && !remainder.is_zero() {
        quotient + BigInt::one()
    } else {
        // truncation toward zero is already ceil for negatives (or zero remainder)
        quotient
    };

    let exp = a.exponent() - BigInt::from(precision);

    let result = Binary::new(final_quotient, exp);
    truncate_ceil(&result, precision)
}

/// Truncate a Binary to at most `precision_bits` mantissa bits,
/// rounding toward -infinity (floor). The result is always <= the input.
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
    // For negative values, truncation toward zero is ceil, so subtract 1 if remainder.
    let has_remainder = (&shifted << shift) != *x.mantissa().magnitude();
    let signed = if x.mantissa().is_negative() && has_remainder {
        -BigInt::from(shifted) - BigInt::from(1_i32)
    } else if x.mantissa().is_negative() {
        -BigInt::from(shifted)
    } else {
        BigInt::from(shifted)
    };

    Binary::new(signed, x.exponent() + BigInt::from(shift))
}

/// Truncate a Binary to at most `precision_bits` mantissa bits,
/// rounding toward +infinity (ceil). The result is always >= the input.
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

    Binary::new(signed, x.exponent() + BigInt::from(shift))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computable::Computable;
    use crate::refinement::bounds_width_leq;
    use crate::sane::XUsize;
    use crate::test_utils::{bin, interval_noop_computable, unwrap_finite};

    /// Helper to create NonZeroU32 from a literal in tests.
    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("test degree must be non-zero")
    }

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        tolerance_exp: &XUsize,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let upper = unwrap_finite(&upper_xb);

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected,
            lower,
            upper
        );
        assert!(
            bounds_width_leq(bounds, tolerance_exp),
            "Bounds width should be <= tolerance",
        );
    }

    #[test]
    fn sqrt_of_4() {
        // sqrt(4) = 2
        let four = Computable::constant(bin(4, 0));
        let sqrt_four = four.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let bounds = sqrt_four
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_2() {
        // sqrt(2) ~= 1.414...
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let bounds = sqrt_two
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_8() {
        // cbrt(8) = 2
        let eight = Computable::constant(bin(8, 0));
        let cbrt_eight = eight.nth_root(nz(3));
        let epsilon = XUsize::Finite(8);
        let bounds = cbrt_eight
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_negative_8() {
        // cbrt(-8) = -2
        let neg_eight = Computable::constant(bin(-8, 0));
        let cbrt_neg_eight = neg_eight.nth_root(nz(3));
        let epsilon = XUsize::Finite(8);
        let bounds = cbrt_neg_eight
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected = bin(-2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn fourth_root_of_16() {
        // 16^(1/4) = 2
        let sixteen = Computable::constant(bin(16, 0));
        let fourth_root = sixteen.nth_root(nz(4));
        let epsilon = XUsize::Finite(8);
        let bounds = fourth_root
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_half() {
        // sqrt(0.5) ~= 0.707...
        let half = Computable::constant(bin(1, -1));
        let sqrt_half = half.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let bounds = sqrt_half
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected_f64 = 0.5_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn nth_root_in_expression() {
        // Test that nth_root works in expressions: sqrt(2) + cbrt(8) = sqrt(2) + 2
        let sqrt_2 = Computable::constant(bin(2, 0)).nth_root(nz(2));
        let cbrt_8 = Computable::constant(bin(8, 0)).nth_root(nz(3));
        let sum = sqrt_2 + cbrt_8;

        let epsilon = XUsize::Finite(8);
        let bounds = sum
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt() + 2.0_f64;
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_zero() {
        // sqrt(0) = 0
        let zero = Computable::constant(bin(0, 0));
        let sqrt_zero = zero.nth_root(nz(2));
        let bounds = sqrt_zero.bounds().expect("bounds should succeed");

        let expected = bin(0, 0);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sqrt_of_interval_overlapping_zero() {
        // Test even root of a Computable with bounds overlapping zero: [-1, 4]
        // Newton targets the midpoint of the input (1.5), so output bounds
        // should contain sqrt(1.5) ~ 1.2247.
        let interval = interval_noop_computable(-1, 4);
        let sqrt_interval = interval.nth_root(nz(2));
        let bounds = sqrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Bounds should contain sqrt(1.5) ~ 1.2247
        let expected_f64 = 1.5_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);
        assert!(
            lower <= expected && expected <= upper,
            "Expected sqrt(1.5) ~ {} to be in bounds [{}, {}]",
            expected,
            lower,
            upper
        );
    }

    #[test]
    fn cbrt_of_interval_overlapping_zero() {
        // Test odd root of a Computable with bounds overlapping zero: [-8, 27]
        // Newton targets the midpoint of the input (9.5), so output bounds
        // contain cbrt(9.5) ~ 2.11, not the full range of possible roots.
        let interval = interval_noop_computable(-8, 27);
        let cbrt_interval = interval.nth_root(nz(3));
        let bounds = cbrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Newton-based bounds should contain cbrt(midpoint) ~ cbrt(9.5) ~ 2.11
        assert!(lower <= bin(2, 0), "lower {} should be <= 2", lower);
        assert!(upper >= bin(2, 0), "upper {} should be >= 2", upper);
    }
}
