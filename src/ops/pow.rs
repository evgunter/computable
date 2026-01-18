//! Integer power operation for computables.
//!
//! This module implements x^n for positive integer exponents n.
//! It computes bounds more efficiently than repeated multiplication by
//! exploiting the monotonicity properties of power functions.

use std::sync::Arc;

use num_traits::Signed;

use crate::binary::{Binary, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::binary::Bounds;

/// Integer power operation.
///
/// Computes x^n where n is a positive integer exponent (n >= 1).
/// The case n=0 is handled at the `Computable::pow` level by returning constant 1.
///
/// # Bounds Computation
/// - For odd n: x^n is monotonically increasing, so bounds are [lower^n, upper^n]
/// - For even n: x^n has a minimum at 0
///   - If interval is non-negative: [lower^n, upper^n]
///   - If interval is non-positive: [upper^n, lower^n]
///   - If interval spans zero: [0, max(|lower|^n, |upper|^n)]
pub struct PowOp {
    /// The input node to raise to a power.
    pub inner: Arc<Node>,
    /// The exponent (n in x^n). Must be >= 1 (n=0 is handled at the Computable level).
    pub exponent: u32,
}

impl NodeOp for PowOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let lower = input_bounds.small();
        let upper = &input_bounds.large();

        // Handle the trivial case of exponent = 1
        if self.exponent == 1 {
            return Ok(input_bounds);
        }

        let is_even = self.exponent.is_multiple_of(2);
        
        let (result_lower, result_upper) = if is_even {
            compute_even_power_bounds(lower, upper, self.exponent)
        } else {
            compute_odd_power_bounds(lower, upper, self.exponent)
        };

        // TODO: InvalidBoundsOrder should be mathematically impossible here since we
        // carefully compute lower/upper based on monotonicity properties. We should
        // try to use the type system to constrain this so the error case is unrepresentable.
        Bounds::new_checked(result_lower, result_upper)
            .map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        // This is a passive combinator - it doesn't refine, just propagates bounds
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
    }
}

/// Computes bounds for x^n where n is odd.
///
/// Since x^n is monotonically increasing for odd n, the output bounds
/// are simply [lower^n, upper^n].
fn compute_odd_power_bounds(lower: &XBinary, upper: &XBinary, n: u32) -> (XBinary, XBinary) {
    let result_lower = xbinary_pow(lower, n);
    let result_upper = xbinary_pow(upper, n);
    (result_lower, result_upper)
}

/// Computes bounds for x^n where n is even.
///
/// For even n, x^n has a minimum at 0:
/// - If [lower, upper] is entirely non-negative: bounds are [lower^n, upper^n]
/// - If [lower, upper] is entirely non-positive: bounds are [upper^n, lower^n]
/// - If [lower, upper] spans zero: bounds are [0, max(|lower|^n, |upper|^n)]
fn compute_even_power_bounds(lower: &XBinary, upper: &XBinary, n: u32) -> (XBinary, XBinary) {
    let lower_non_negative = !is_negative(lower);
    let upper_non_positive = !is_positive(upper);

    if lower_non_negative {
        // [lower, upper] is entirely non-negative
        let result_lower = xbinary_pow(lower, n);
        let result_upper = xbinary_pow(upper, n);
        (result_lower, result_upper)
    } else if upper_non_positive {
        // [lower, upper] is entirely non-positive
        // For even powers, more negative gives larger positive result
        let result_lower = xbinary_pow(upper, n);
        let result_upper = xbinary_pow(lower, n);
        (result_lower, result_upper)
    } else {
        // Interval spans zero
        let lower_pow = xbinary_pow(lower, n);
        let upper_pow = xbinary_pow(upper, n);
        let result_upper = xbinary_max(&lower_pow, &upper_pow);
        (XBinary::zero(), result_upper)
    }
}

/// Returns true if the XBinary value is negative (less than zero).
fn is_negative(x: &XBinary) -> bool {
    match x {
        XBinary::NegInf => true,
        XBinary::PosInf => false,
        XBinary::Finite(b) => b.mantissa().is_negative(),
    }
}

/// Returns true if the XBinary value is positive (greater than zero).
fn is_positive(x: &XBinary) -> bool {
    match x {
        XBinary::NegInf => false,
        XBinary::PosInf => true,
        XBinary::Finite(b) => b.mantissa().is_positive(),
    }
}

/// Returns the maximum of two XBinary values.
fn xbinary_max(a: &XBinary, b: &XBinary) -> XBinary {
    if a >= b { a.clone() } else { b.clone() }
}

/// Computes x^n for an XBinary value.
fn xbinary_pow(x: &XBinary, n: u32) -> XBinary {
    match x {
        XBinary::NegInf => {
            // (-∞)^n = +∞ for even n, -∞ for odd n
            if n.is_multiple_of(2) {
                XBinary::PosInf
            } else {
                XBinary::NegInf
            }
        }
        XBinary::PosInf => XBinary::PosInf,
        XBinary::Finite(b) => XBinary::Finite(binary_pow(b, n)),
    }
}

/// Computes x^n for a Binary value using repeated multiplication.
fn binary_pow(x: &Binary, n: u32) -> Binary {
    if n == 0 {
        return Binary::new(num_bigint::BigInt::from(1), num_bigint::BigInt::from(0));
    }
    
    let mut result = x.clone();
    for _ in 1..n {
        result = result.mul(x);
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::UBinary;
    use crate::computable::Computable;
    use crate::test_utils::{bin, ubin, xbin, unwrap_finite};

    fn assert_bounds_contain_expected(bounds: &Bounds, expected: &Binary, _epsilon: &UBinary) {
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected, lower, upper
        );
    }

    fn interval_computable(lower: i64, upper: i64) -> Computable {
        let interval_state = Bounds::new(xbin(lower, 0), xbin(upper, 0));
        Computable::new(
            interval_state,
            |state| Ok(state.clone()),
            |state| state, // No refinement for this test
        )
    }

    #[test]
    fn pow_constant_squared() {
        // 3^2 = 9
        let three = Computable::constant(bin(3, 0));
        let squared = three.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        let expected = bin(9, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_constant_cubed() {
        // 2^3 = 8
        let two = Computable::constant(bin(2, 0));
        let cubed = two.pow(3);
        let bounds = cubed.bounds().expect("bounds should succeed");

        let expected = bin(8, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_negative_even() {
        // (-3)^2 = 9
        let neg_three = Computable::constant(bin(-3, 0));
        let squared = neg_three.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        let expected = bin(9, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_negative_odd() {
        // (-2)^3 = -8
        let neg_two = Computable::constant(bin(-2, 0));
        let cubed = neg_two.pow(3);
        let bounds = cubed.bounds().expect("bounds should succeed");

        let expected = bin(-8, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_interval_positive_even() {
        // [2, 4]^2 = [4, 16]
        let interval = interval_computable(2, 4);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(4, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(16, 0));
    }

    #[test]
    fn pow_interval_negative_even() {
        // [-4, -2]^2 = [4, 16] (note: more negative gives larger result)
        let interval = interval_computable(-4, -2);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(4, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(16, 0));
    }

    #[test]
    fn pow_interval_spanning_zero_even() {
        // [-2, 3]^2 = [0, 9] (minimum at 0, max at the larger magnitude)
        let interval = interval_computable(-2, 3);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(0, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(9, 0));
    }

    #[test]
    fn pow_interval_odd() {
        // [2, 4]^3 = [8, 64]
        let interval = interval_computable(2, 4);
        let cubed = interval.pow(3);
        let bounds = cubed.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(8, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(64, 0));
    }

    #[test]
    fn pow_interval_negative_odd() {
        // [-4, -2]^3 = [-64, -8]
        let interval = interval_computable(-4, -2);
        let cubed = interval.pow(3);
        let bounds = cubed.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(-64, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(-8, 0));
    }

    #[test]
    fn pow_exponent_one() {
        // x^1 = x
        let three = Computable::constant(bin(3, 0));
        let result = three.pow(1);
        let bounds = result.bounds().expect("bounds should succeed");

        let expected = bin(3, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_exponent_zero() {
        // x^0 = 1 for all x
        let three = Computable::constant(bin(3, 0));
        let result = three.pow(0);
        let bounds = result.bounds().expect("bounds should succeed");

        let expected = bin(1, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_zero_to_zero() {
        // 0^0 = 1 by convention
        let zero = Computable::constant(bin(0, 0));
        let result = zero.pow(0);
        let bounds = result.bounds().expect("bounds should succeed");

        let expected = bin(1, 0);
        assert_eq!(unwrap_finite(bounds.small()), expected);
        assert_eq!(unwrap_finite(&bounds.large()), expected);
    }

    #[test]
    fn pow_in_expression() {
        // Test that pow works in expressions: 2^2 + 3^2 = 4 + 9 = 13
        let two_sq = Computable::constant(bin(2, 0)).pow(2);
        let three_sq = Computable::constant(bin(3, 0)).pow(2);
        let sum = two_sq + three_sq;

        let epsilon = ubin(1, -8);
        let bounds = sum
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(13, 0);
        assert_bounds_contain_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn pow_with_sqrt() {
        // sqrt(2)^2 should be approximately 2
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(2);
        let squared = sqrt_two.pow(2);

        let epsilon = ubin(1, -8);
        let bounds = squared
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_contain_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn pow_of_zero() {
        // 0^2 = 0
        let zero = Computable::constant(bin(0, 0));
        let squared = zero.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert!(bounds.small().is_zero());
        assert!(bounds.large().is_zero());
    }

    #[test]
    fn binary_pow_function() {
        let x = bin(3, 0);
        assert_eq!(binary_pow(&x, 0), bin(1, 0));
        assert_eq!(binary_pow(&x, 1), bin(3, 0));
        assert_eq!(binary_pow(&x, 2), bin(9, 0));
        assert_eq!(binary_pow(&x, 3), bin(27, 0));
    }
}
