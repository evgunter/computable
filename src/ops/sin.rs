//! Sine operation using Taylor series with provably correct error bounds.
//!
//! This module implements the sine function using:
//! - Range reduction to [-pi/2, pi/2] for efficient Taylor series convergence
//! - Critical point detection for tight bounds on intervals containing extrema
//! - Directed rounding for provably correct interval arithmetic

use std::sync::Arc;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use parking_lot::RwLock;

use crate::binary::{Binary, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::binary::Bounds;

/// Sine operation with Taylor series refinement.
pub struct SinOp {
    pub inner: Arc<Node>,
    pub num_terms: RwLock<BigInt>,
}

impl NodeOp for SinOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let num_terms = self.num_terms.read().clone();
        sin_bounds(&input_bounds, &num_terms)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();
        *num_terms += BigInt::one();
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

// TODO: make this a computable number so that the results remain provably correct (right now they're logically incorrect because of the approximation used for pi!)
/// Returns a high-precision representation of pi as a Binary number.
/// Uses ~64 bits of precision (~19 decimal digits).
fn pi_binary() -> Binary {
    // pi * 2^61 = 7244019458077122842.70...
    // 7244019458077122843 is odd (ends in 3)
    let mantissa = BigInt::parse_bytes(b"7244019458077122843", 10)
        .unwrap_or_else(|| BigInt::from(3));
    Binary::new(mantissa, BigInt::from(-61))
}

/// Returns 2*pi as a Binary number (for range reduction).
fn two_pi_binary() -> Binary {
    let pi = pi_binary();
    Binary::new(pi.mantissa().clone(), pi.exponent() + BigInt::one())
}

/// Reduces x to the range [-pi, pi] by subtracting multiples of 2*pi.
fn reduce_to_pi_range(x: &Binary) -> Binary {
    let two_pi = two_pi_binary();
    let pi = pi_binary();

    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };

    if abs_x <= pi {
        return x.clone();
    }

    let k = compute_reduction_factor(x, &two_pi);
    let k_times_two_pi = multiply_by_integer(&two_pi, &k);
    x.sub(&k_times_two_pi)
}

/// Reduces x to the range [-pi/2, pi/2] and returns (reduced_x, sign_flip).
/// sign_flip indicates whether the final sin value needs to be negated.
fn reduce_to_half_pi_range(x: &Binary) -> (Binary, bool) {
    let pi = pi_binary();
    let half_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::one());
    let neg_half_pi = half_pi.neg();

    let reduced = reduce_to_pi_range(x);

    if reduced > half_pi {
        // x in (pi/2, pi]: use sin(x) = sin(pi - x)
        (pi.sub(&reduced), false)
    } else if reduced < neg_half_pi {
        // x in [-pi, -pi/2): use sin(x) = -sin(pi + x)
        (pi.add(&reduced), true)
    } else {
        (reduced, false)
    }
}

/// Computes k = round(x / period).
fn compute_reduction_factor(x: &Binary, period: &Binary) -> BigInt {
    let precision_bits = 64i64;
    let mx = x.mantissa();
    let ex = x.exponent();
    let mp = period.mantissa();
    let ep = period.exponent();

    let shifted_mx = mx << precision_bits as usize;
    let quotient = &shifted_mx / mp;
    let result_exp = ex - ep - BigInt::from(precision_bits);

    if result_exp >= BigInt::zero() {
        let shift = result_exp.to_usize().unwrap_or(0);
        &quotient << shift
    } else {
        let shift = (-&result_exp).to_usize().unwrap_or(0);
        if shift == 0 {
            quotient.clone()
        } else {
            let half = BigInt::one() << (shift - 1);
            let rounded = if quotient.is_negative() {
                &quotient - &half
            } else {
                &quotient + &half
            };
            rounded >> shift
        }
    }
}

/// Multiplies a Binary by a BigInt integer.
fn multiply_by_integer(b: &Binary, k: &BigInt) -> Binary {
    Binary::new(b.mantissa() * k, b.exponent().clone())
}

/// Truncates a Binary to at most `precision_bits` of mantissa.
fn truncate_precision(x: &Binary, precision_bits: usize) -> Binary {
    let mantissa = x.mantissa();
    let exponent = x.exponent();
    let bit_length = mantissa.magnitude().bits() as usize;

    if bit_length <= precision_bits {
        return x.clone();
    }

    let shift = bit_length - precision_bits;
    let truncated_mantissa = mantissa >> shift;
    let new_exponent = exponent + BigInt::from(shift);
    Binary::new(truncated_mantissa, new_exponent)
}

/// Checks if an interval [a, b] contains critical points of sin (where sin = +/-1).
/// Returns (contains_max, contains_min).
fn interval_contains_critical_points(lower: &Binary, upper: &Binary) -> (bool, bool) {
    let pi = pi_binary();
    let two_pi = two_pi_binary();
    let half_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::one());
    let neg_half_pi = half_pi.neg();

    let width = upper.sub(lower);
    if width >= two_pi {
        return (true, true);
    }

    let reduced_lower = reduce_to_pi_range(lower);
    let reduced_upper = reduced_lower.add(&width);

    let contains_max = interval_contains_point(&reduced_lower, &reduced_upper, &half_pi, &two_pi);
    let contains_min =
        interval_contains_point(&reduced_lower, &reduced_upper, &neg_half_pi, &two_pi);

    (contains_max, contains_min)
}

/// Checks if an interval [a, b] contains a point p (or p + k*period for any integer k).
fn interval_contains_point(
    lower: &Binary,
    upper: &Binary,
    point: &Binary,
    period: &Binary,
) -> bool {
    let mut p = point.clone();

    while p < lower.sub(period) {
        p = p.add(period);
    }
    while p > upper.add(period) {
        p = p.sub(period);
    }

    if &p >= lower && &p <= upper {
        return true;
    }
    let p_plus = p.add(period);
    if &p_plus >= lower && &p_plus <= upper {
        return true;
    }
    let p_minus = p.sub(period);
    &p_minus >= lower && &p_minus <= upper
}

/// The Taylor series is: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
/// After n terms, the error is bounded by |x|^(2n+1)/(2n+1)!
///
/// This implementation uses:
/// - Range reduction to [-pi/2, pi/2] for efficient Taylor series convergence
/// - Critical point detection for tight bounds on intervals containing extrema
/// - Directed rounding for provably correct interval arithmetic
fn sin_bounds(input_bounds: &Bounds, num_terms: &BigInt) -> Result<Bounds, ComputableError> {
    let neg_one = Binary::new(BigInt::from(-1), BigInt::zero());
    let pos_one = Binary::new(BigInt::from(1), BigInt::zero());

    // Extract finite bounds, or return [-1, 1] for any infinite bounds
    let lower = input_bounds.small();
    let upper = input_bounds.large();
    let (lower_bin, upper_bin) = match (lower, &upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => (l, u),
        _ => {
            return Ok(Bounds::new(
                XBinary::Finite(neg_one),
                XBinary::Finite(pos_one),
            ));
        }
    };

    // Check for critical points (where sin reaches +/-1)
    let (contains_max, contains_min) = interval_contains_critical_points(lower_bin, upper_bin);

    // Convert num_terms to usize for computation (capped at reasonable limit)
    let n = num_terms.to_usize().unwrap_or(1).max(1);

    // Apply range reduction to both endpoints for efficient Taylor series
    let (reduced_lower, lower_sign_flip) = reduce_to_half_pi_range(lower_bin);
    let (reduced_upper, upper_sign_flip) = reduce_to_half_pi_range(upper_bin);

    // TODO: This truncation loses precision and isn't accounted for in error bounds.
    // For provable correctness, either remove truncation or add the truncation error
    // (at most 2^(-64) * |reduced_value|) to the final error bound (while increasing the
    // precision bits so that the answer still converges instead of remaining stuck at 64 bits of precision).
    // Truncate to 64 bits to keep mantissas manageable
    let truncated_lower = truncate_precision(&reduced_lower, 64);
    let truncated_upper = truncate_precision(&reduced_upper, 64);

    // Compute Taylor series bounds on reduced values
    let sin_lower_raw = taylor_sin_bounds(&truncated_lower, n);
    let sin_upper_raw = taylor_sin_bounds(&truncated_upper, n);

    // Apply sign flips if needed
    let (sin_lower_lo, sin_lower_hi) = if lower_sign_flip {
        (sin_lower_raw.1.neg(), sin_lower_raw.0.neg())
    } else {
        sin_lower_raw
    };
    let (sin_upper_lo, sin_upper_hi) = if upper_sign_flip {
        (sin_upper_raw.1.neg(), sin_upper_raw.0.neg())
    } else {
        sin_upper_raw
    };

    // Combine endpoint bounds
    let mut result_lower = if sin_lower_lo <= sin_upper_lo {
        sin_lower_lo
    } else {
        sin_upper_lo
    };
    let mut result_upper = if sin_lower_hi >= sin_upper_hi {
        sin_lower_hi
    } else {
        sin_upper_hi
    };

    // If interval contains critical points, extend bounds accordingly
    if contains_max {
        result_upper = pos_one.clone();
    }
    if contains_min {
        result_lower = neg_one.clone();
    }

    // Final clamp to [-1, 1]
    if result_lower < neg_one {
        result_lower = neg_one.clone();
    }
    if result_upper > pos_one {
        result_upper = pos_one;
    }

    // The sin algorithm is designed to produce correctly ordered bounds
    Bounds::new_checked(XBinary::Finite(result_lower), XBinary::Finite(result_upper))
        .map_err(|_| crate::internal_error!(
            "sin bounds computation produced invalid order: this indicates a bug"
        ))
}

/// Rounding direction for directed rounding in interval arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundingDirection {
    /// Round toward negative infinity (floor)
    Down,
    /// Round toward positive infinity (ceiling)
    Up,
}

/// Computes Taylor series bounds for sin(x) with n terms.
/// Returns (lower_bound, upper_bound) accounting for truncation error.
///
/// Taylor series: sin(x) = sum_{k=0}^{n-1} (-1)^k * x^(2k+1) / (2k+1)!
/// Error after n terms: |R_n| <= |x|^(2n+1) / (2n+1)!
///
/// Uses directed rounding to compute provably correct bounds:
/// - Lower bound: all intermediate operations round DOWN (toward -inf)
/// - Upper bound: all intermediate operations round UP (toward +inf)
fn taylor_sin_bounds(x: &Binary, n: usize) -> (Binary, Binary) {
    if n == 0 {
        // No terms: just use error bound (always round UP for conservative bounds)
        let error = taylor_error_bound(x, 0);
        return (error.neg(), error);
    }

    // Compute lower and upper partial sums with directed rounding
    let sum_lower = taylor_sin_partial_sum(x, n, RoundingDirection::Down);
    let sum_upper = taylor_sin_partial_sum(x, n, RoundingDirection::Up);

    // Compute error bound (always round UP for conservative bounds)
    let error = taylor_error_bound(x, n);

    // Return bounds: lower_sum - error, upper_sum + error
    (sum_lower.sub(&error), sum_upper.add(&error))
}

/// Computes Taylor series partial sum for sin(x) with directed rounding.
///
/// For RoundingDirection::Down: rounds all division operations toward -infinity
/// For RoundingDirection::Up: rounds all division operations toward +infinity
fn taylor_sin_partial_sum(x: &Binary, n: usize, rounding: RoundingDirection) -> Binary {
    let mut sum = Binary::zero();
    let mut power = x.clone(); // x^1
    let mut factorial = BigInt::one(); // 1!

    for k in 0..n {
        // Term k: (-1)^k * x^(2k+1) / (2k+1)!
        let term_num = if k % 2 == 0 {
            power.clone()
        } else {
            power.neg()
        };

        // Divide by factorial with directed rounding
        let term = divide_by_factorial_directed(&term_num, &factorial, rounding);
        sum = sum.add(&term);

        // Prepare for next term: multiply power by x^2
        if k + 1 < n {
            power = power.mul(x).mul(x);
            // factorial *= (2k+2) * (2k+3)
            let next_k = k + 1;
            factorial *= BigInt::from(2 * next_k) * BigInt::from(2 * next_k + 1);
        }
    }

    sum
}

/// Computes |x|^(2n+1) / (2n+1)! as an upper bound on Taylor series truncation error.
/// Always rounds UP to be conservative.
fn taylor_error_bound(x: &Binary, n: usize) -> Binary {
    // Compute |x|^(2n+1)
    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };

    let exp = 2 * n + 1;
    let mut power = Binary::new(BigInt::one(), BigInt::zero()); // 1
    for _ in 0..exp {
        power = power.mul(&abs_x);
    }

    // Compute (2n+1)!
    let mut factorial = BigInt::one();
    for i in 1..=exp {
        factorial *= BigInt::from(i);
    }

    // error = power / factorial (round UP for conservative error bound)
    divide_by_factorial_directed(&power, &factorial, RoundingDirection::Up)
}

/// Divides a Binary by a BigInt factorial with directed rounding.
///
/// Rounding semantics:
/// - `RoundingDirection::Up`: rounds toward +infinity (ceiling)
/// - `RoundingDirection::Down`: rounds toward -infinity (floor)
///
/// This is essential for interval arithmetic: when computing a lower bound,
/// round DOWN; when computing an upper bound, round UP.
fn divide_by_factorial_directed(
    value: &Binary,
    factorial: &BigInt,
    rounding: RoundingDirection,
) -> Binary {
    if factorial.is_zero() {
        return value.clone();
    }

    let mantissa = value.mantissa();
    let exponent = value.exponent();

    // We need to compute mantissa / factorial with the result as a Binary.
    // To get a good approximation, we shift the mantissa up by some bits before dividing.
    // The number of bits we shift determines our precision.
    let precision_bits = 64_u64; // Extra precision for intermediate computation

    // shifted_mantissa = |mantissa| * 2^precision_bits
    let abs_mantissa = mantissa.magnitude().clone();
    let shifted_mantissa = &abs_mantissa << precision_bits as usize;

    // Compute |mantissa| / factorial
    let (quot, rem) = shifted_mantissa.div_rem(factorial.magnitude());

    // Determine how to round based on direction and sign
    // For directed rounding toward +/- infinity:
    // - Round UP (+inf): positive values round away from zero, negative round toward zero
    // - Round DOWN (-inf): positive values round toward zero, negative round away from zero
    let is_negative = mantissa.is_negative();
    let has_remainder = !rem.is_zero();

    let result_magnitude = if has_remainder {
        match (rounding, is_negative) {
            // Rounding UP (toward +infinity):
            // - Positive: round away from zero (add 1)
            // - Negative: round toward zero (truncate)
            (RoundingDirection::Up, false) => quot + BigInt::one().magnitude(),
            (RoundingDirection::Up, true) => quot,
            // Rounding DOWN (toward -infinity):
            // - Positive: round toward zero (truncate)
            // - Negative: round away from zero (add 1)
            (RoundingDirection::Down, false) => quot,
            (RoundingDirection::Down, true) => quot + BigInt::one().magnitude(),
        }
    } else {
        // Exact division, no rounding needed
        quot
    };

    // Adjust sign
    let signed_mantissa = if is_negative {
        -BigInt::from(result_magnitude)
    } else {
        BigInt::from(result_magnitude)
    };

    // New exponent = original_exponent - precision_bits
    let new_exponent = exponent - BigInt::from(precision_bits);

    Binary::new(signed_mantissa, new_exponent)
}

// Test helpers - exposed for integration tests
#[cfg(test)]
pub fn taylor_sin_bounds_test(x: &Binary, n: usize) -> (Binary, Binary) {
    taylor_sin_bounds(x, n)
}

#[cfg(test)]
pub fn taylor_sin_partial_sum_test(x: &Binary, n: usize, down: bool) -> Binary {
    let rounding = if down {
        RoundingDirection::Down
    } else {
        RoundingDirection::Up
    };
    taylor_sin_partial_sum(x, n, rounding)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::UBinary;
    use crate::computable::Computable;
    use crate::test_utils::{bin, ubin, xbin, unwrap_finite, unwrap_finite_uxbinary};
    use num_traits::One;

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    fn interval_midpoint_computable(lower: i64, upper: i64) -> Computable {
        fn midpoint_between(lower: &XBinary, upper: &XBinary) -> Binary {
            let unwrap = |input: &XBinary| -> Binary {
                match input {
                    XBinary::Finite(value) => value.clone(),
                    _ => panic!("expected finite"),
                }
            };
            let mid_sum = unwrap(lower).add(&unwrap(upper));
            let exponent = mid_sum.exponent() - BigInt::one();
            Binary::new(mid_sum.mantissa().clone(), exponent)
        }

        fn interval_refine(state: Bounds) -> Bounds {
            let midpoint = midpoint_between(state.small(), &state.large());
            Bounds::new(
                XBinary::Finite(midpoint.clone()),
                XBinary::Finite(midpoint),
            )
        }

        let interval_state = Bounds::new(xbin(lower, 0), xbin(upper, 0));
        Computable::new(
            interval_state,
            |inner_state| Ok(inner_state.clone()),
            interval_refine,
        )
    }

    #[test]
    fn sin_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_zero
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0) = 0
        let expected = bin(0, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sin_of_pi_over_2() {
        // pi/2 ~= 1.5707963...
        // We approximate it as 3217/2048 ~= 1.5708...
        let pi_over_2 = Computable::constant(bin(3217, -11));
        let sin_pi_2 = pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi/2) = 1
        let expected_f64 = (std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi/2) should be very close to 1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_of_pi() {
        // pi ~= 3.14159...
        // We approximate it as 6434/2048 ~= 3.1416...
        let pi_approx = Computable::constant(bin(6434, -11));
        let sin_pi = pi_approx.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi) ~= 0 (should be close to 0)
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi) should be very close to 0
        let small_bound = bin(1, -4);
        let neg_small_bound = bin(-1, -4);
        assert!(lower >= neg_small_bound);
        assert!(upper <= small_bound);
    }

    #[test]
    fn sin_of_negative_pi_over_2() {
        // -pi/2 ~= -1.5707963...
        let neg_pi_over_2 = Computable::constant(bin(-3217, -11));
        let sin_neg_pi_2 = neg_pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_neg_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(-pi/2) = -1
        let expected_f64 = (-std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(-pi/2) should be very close to -1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_bounds_always_in_minus_one_to_one() {
        // Test with a large value that exercises argument reduction
        let large_value = Computable::constant(bin(100, 0));
        let sin_large = large_value.sin();
        let bounds = sin_large.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        assert!(lower >= neg_one);
        assert!(upper <= one);
    }

    #[test]
    fn sin_of_small_value() {
        // For small x, sin(x) ~= x
        let small = Computable::constant(bin(1, -4)); // 1/16 = 0.0625
        let sin_small = small.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_small
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0.0625) ~= 0.0624593...
        let expected = XBinary::from_f64(0.0625_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn sin_interval_spanning_maximum() {
        // An interval that spans pi/2 (where sin has maximum)
        let computable = interval_midpoint_computable(1, 2); // [1, 2] includes pi/2 ~= 1.57
        let sin_interval = computable.sin();
        let bounds = sin_interval.bounds().expect("bounds should succeed");

        let upper = unwrap_finite(&bounds.large());

        // The upper bound should be close to 1 since the interval contains pi/2
        assert!(upper >= bin(1, -1)); // Upper bound should be at least 0.5
    }

    #[test]
    fn sin_with_infinite_input_bounds() {
        let unbounded = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let sin_unbounded = unbounded.sin();
        let bounds = sin_unbounded.bounds().expect("bounds should succeed");

        // sin of unbounded input should be [-1, 1]
        assert_eq!(bounds.small(), &xbin(-1, 0));
        assert_eq!(&bounds.large(), &xbin(1, 0));
    }

    #[test]
    fn sin_expression_with_arithmetic() {
        // Test sin(x) + cos-like expression: sin(x)^2 + sin(x + pi/2)^2 should be close to 1
        // Here we just test that sin works in expressions
        let x = Computable::constant(bin(1, 0)); // x = 1
        let sin_x = x.clone().sin();
        let two = Computable::constant(bin(2, 0));
        let expr = sin_x.clone() * two; // 2 * sin(1)

        let epsilon = ubin(1, -8);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // 2 * sin(1) ~= 2 * 0.8414... ~= 1.6829...
        let expected = XBinary::from_f64(2.0 * 1.0_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn directed_rounding_produces_valid_bounds() {
        // Test that directed rounding produces well-ordered bounds that contain the true value.

        let test_cases = [
            bin(1, -2),   // 0.25
            bin(1, 0),    // 1.0
            bin(3, 0),    // 3.0
            bin(-1, 0),   // -1.0
            bin(5, -1),   // 2.5
            bin(-3, -1),  // -1.5
        ];

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        for x in &test_cases {
            // Compute Taylor bounds with directed rounding
            let (lower, upper) = taylor_sin_bounds_test(x, 10);

            // Verify bounds are ordered correctly
            assert!(
                lower <= upper,
                "Lower bound {} should be <= upper bound {} for x = {}",
                lower, upper, x
            );

            // Verify bounds are within sin's range [-1, 1]
            assert!(
                lower >= neg_one,
                "Lower bound {} should be >= -1 for x = {}",
                lower, x
            );
            assert!(
                upper <= one,
                "Upper bound {} should be <= 1 for x = {}",
                upper, x
            );
        }
    }

    #[test]
    fn directed_rounding_bounds_converge() {
        // Verify that bounds get tighter as we add more terms
        let x = bin(1, 0); // 1.0

        let (lower5, upper5) = taylor_sin_bounds_test(&x, 5);
        let (lower10, upper10) = taylor_sin_bounds_test(&x, 10);

        let width5 = upper5.sub(&lower5);
        let width10 = upper10.sub(&lower10);

        // More terms should give tighter bounds
        assert!(
            width10 < width5,
            "Bounds with 10 terms (width {}) should be tighter than 5 terms (width {})",
            width10, width5
        );
    }

    #[test]
    fn directed_rounding_symmetry() {
        // Test that sin(-x) bounds are the negation of sin(x) bounds
        // This verifies that the directed rounding handles negative inputs correctly

        let x = bin(1, -2); // 0.25
        let neg_x = bin(-1, -2); // -0.25

        let (lower_x, upper_x) = taylor_sin_bounds_test(&x, 10);
        let (lower_neg_x, upper_neg_x) = taylor_sin_bounds_test(&neg_x, 10);

        // sin(-x) = -sin(x), so bounds should be negated and swapped
        // lower(-x) should equal -upper(x)
        // upper(-x) should equal -lower(x)

        // Allow small differences due to rounding
        let neg_upper_x = upper_x.neg();
        let neg_lower_x = lower_x.neg();

        // The bounds should be approximately symmetric
        // We just verify they're in the right ballpark
        assert!(
            lower_neg_x <= neg_upper_x.add(&bin(1, -50)),
            "lower(sin(-x)) should be approximately -upper(sin(x))"
        );
        assert!(
            neg_lower_x <= upper_neg_x.add(&bin(1, -50)),
            "-lower(sin(x)) should be approximately upper(sin(-x))"
        );
    }

    #[test]
    fn directed_rounding_lower_bound_is_lower() {
        // Verify that rounding down produces smaller values than rounding up
        let x = bin(1, 0); // 1.0
        let n = 5;

        let sum_down = taylor_sin_partial_sum_test(&x, n, true);
        let sum_up = taylor_sin_partial_sum_test(&x, n, false);

        // The down-rounded sum should be <= up-rounded sum
        assert!(
            sum_down <= sum_up,
            "Rounding down {} should produce <= rounding up {}",
            sum_down, sum_up
        );
    }
}
