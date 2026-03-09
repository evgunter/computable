//! Integer power operation for computables.
//!
//! This module implements x^n for positive integer exponents n.
//! It computes bounds more efficiently than repeated multiplication by
//! exploiting the monotonicity properties of power functions.

use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::Zero;

use crate::binary::{Bounds, UBinary, UXBinary, XBinary};
use crate::binary_utils::power::{is_negative, is_positive, xbinary_max, xbinary_pow};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::sane::XIsize;

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
    /// The exponent (n in x^n). Uses `NonZeroU32` to ensure n >= 1
    /// (n=0 is handled at the Computable level).
    pub exponent: NonZeroU32,
}

impl NodeOp for PowOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let lower = input_bounds.small();
        let upper = &input_bounds.large();

        // Handle the trivial case of exponent = 1
        if self.exponent.get() == 1 {
            return Ok(input_bounds);
        }

        let is_even = self.exponent.get().is_multiple_of(2);

        let bounds = if is_even {
            compute_even_power_bounds(lower, upper, self.exponent)
        } else {
            compute_odd_power_bounds(lower, upper, self.exponent)
        };

        Ok(bounds)
    }

    fn refine_step(&self, _target_width_exp: XIsize) -> Result<bool, ComputableError> {
        // This is a passive combinator - it doesn't refine, just propagates bounds
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// Max derivative of x^n over [lower, upper] is n · max_abs^(n-1).
    /// Child budget = target / (n · max_abs^(n-1)).
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        let n = self.exponent.get();
        if n == 1 {
            return target_width.clone();
        }
        let max_abs = match self.inner.cached_bounds() {
            Some(b) => {
                let (lo, hi) = b.abs();
                std::cmp::max(lo, hi)
            }
            None => return target_width.clone(),
        };
        // Compute max_abs^(n-1) via exponentiation by squaring.
        let power = uxbinary_pow(&max_abs, n - 1);
        let n_ux = UXBinary::Finite(UBinary::new(BigUint::from(n), 0_i64));
        let denominator = n_ux.mul(&power);
        target_width.div_floor(&denominator)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        self.exponent.get() > 1
    }
}

/// Computes base^exp for UXBinary via exponentiation by squaring.
pub(crate) fn uxbinary_pow(base: &UXBinary, exp: u32) -> UXBinary {
    if exp == 0 {
        return UXBinary::Finite(UBinary::new(BigUint::from(1u32), 0_i64));
    }
    match base {
        UXBinary::Inf => UXBinary::Inf,
        UXBinary::Finite(b) => {
            if b.mantissa().is_zero() {
                return UXBinary::zero();
            }
            let mut result = UBinary::new(BigUint::from(1u32), 0_i64);
            let mut base_val = b.clone();
            let mut e = exp;
            while e > 0 {
                if e & 1 == 1 {
                    result = result.mul(&base_val);
                }
                if e > 1 {
                    base_val = base_val.mul(&base_val);
                }
                e >>= 1;
            }
            UXBinary::Finite(result)
        }
    }
}

/// Computes bounds for x^n where n is odd.
///
/// Since x^n is monotonically increasing for odd n, the output bounds
/// are simply [lower^n, upper^n].
fn compute_odd_power_bounds(lower: &XBinary, upper: &XBinary, n: NonZeroU32) -> Bounds {
    let result_lower = xbinary_pow(lower, n);
    let result_upper = xbinary_pow(upper, n);
    Bounds::new(result_lower, result_upper)
}

/// Computes bounds for x^n where n is even.
///
/// For even n, x^n has a minimum at 0:
/// - If [lower, upper] is entirely non-negative: bounds are [lower^n, upper^n]
/// - If [lower, upper] is entirely non-positive: bounds are [upper^n, lower^n]
/// - If [lower, upper] spans zero: bounds are [0, max(|lower|^n, |upper|^n)]
fn compute_even_power_bounds(lower: &XBinary, upper: &XBinary, n: NonZeroU32) -> Bounds {
    let lower_non_negative = !is_negative(lower);
    let upper_non_positive = !is_positive(upper);

    if lower_non_negative {
        // [lower, upper] is entirely non-negative
        let result_lower = xbinary_pow(lower, n);
        let result_upper = xbinary_pow(upper, n);
        Bounds::new(result_lower, result_upper)
    } else if upper_non_positive {
        // [lower, upper] is entirely non-positive
        // For even powers, more negative gives larger positive result
        let result_lower = xbinary_pow(upper, n);
        let result_upper = xbinary_pow(lower, n);
        Bounds::new(result_lower, result_upper)
    } else {
        // Interval spans zero
        let lower_pow = xbinary_pow(lower, n);
        let upper_pow = xbinary_pow(upper, n);
        let result_upper = xbinary_max(&lower_pow, &upper_pow);
        Bounds::new(XBinary::zero(), result_upper)
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::{Binary, Bounds};
    use crate::computable::Computable;
    use crate::sane::XUsize;
    use crate::test_utils::{bin, interval_noop_computable, unwrap_finite};

    fn assert_bounds_contain_expected(bounds: &Bounds, expected: &Binary, _tolerance_exp: &XUsize) {
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected,
            lower,
            upper
        );
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
        let interval = interval_noop_computable(2, 4);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(4, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(16, 0));
    }

    #[test]
    fn pow_interval_negative_even() {
        // [-4, -2]^2 = [4, 16] (note: more negative gives larger result)
        let interval = interval_noop_computable(-4, -2);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(4, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(16, 0));
    }

    #[test]
    fn pow_interval_spanning_zero_even() {
        // [-2, 3]^2 = [0, 9] (minimum at 0, max at the larger magnitude)
        let interval = interval_noop_computable(-2, 3);
        let squared = interval.pow(2);
        let bounds = squared.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(0, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(9, 0));
    }

    #[test]
    fn pow_interval_odd() {
        // [2, 4]^3 = [8, 64]
        let interval = interval_noop_computable(2, 4);
        let cubed = interval.pow(3);
        let bounds = cubed.bounds().expect("bounds should succeed");

        assert_eq!(unwrap_finite(bounds.small()), bin(8, 0));
        assert_eq!(unwrap_finite(&bounds.large()), bin(64, 0));
    }

    #[test]
    fn pow_interval_negative_odd() {
        // [-4, -2]^3 = [-64, -8]
        let interval = interval_noop_computable(-4, -2);
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

        let tolerance_exp = XUsize::Finite(8);
        let bounds = sum
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let expected = bin(13, 0);
        assert_bounds_contain_expected(&bounds, &expected, &tolerance_exp);
    }

    #[test]
    fn pow_with_sqrt() {
        // sqrt(2)^2 should be approximately 2
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(std::num::NonZeroU32::new(2).expect("2 is non-zero"));
        let squared = sqrt_two.pow(2);

        let tolerance_exp = XUsize::Finite(8);
        let bounds = squared
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_contain_expected(&bounds, &expected, &tolerance_exp);
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
}
