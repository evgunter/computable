//! Integer power operation for computables.
//!
//! This module implements x^n for positive integer exponents n.
//! It computes bounds more efficiently than repeated multiplication by
//! exploiting the monotonicity properties of power functions.

use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::Zero;

use crate::binary::{UBinary, UXBinary, XBinary};
use crate::binary_utils::power::{is_negative, is_positive, xbinary_max, xbinary_pow};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::XI;

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
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let input_prefix = self.inner.get_prefix()?;
        let lower = input_prefix.lower();
        let upper = input_prefix.upper();

        // Handle the trivial case of exponent = 1
        if self.exponent.get() == 1 {
            return Ok(input_prefix);
        }

        let is_even = self.exponent.get().is_multiple_of(2);

        let (result_lower, result_upper) = if is_even {
            compute_even_power_bounds(&lower, &upper, self.exponent)
        } else {
            compute_odd_power_bounds(&lower, &upper, self.exponent)
        };

        Ok(Prefix::from_lower_upper(result_lower, result_upper))
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
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
    fn child_demand_budget(&self, target_width: &UXBinary, _child_idx: bool) -> UXBinary {
        let n = self.exponent.get();
        if n == 1 {
            return target_width.clone();
        }
        let max_abs = match self.inner.cached_prefix() {
            Some(p) => {
                let (lo, hi) = p.abs();
                std::cmp::max(lo, hi)
            }
            None => return target_width.clone(),
        };
        // Compute max_abs^(n-1) via exponentiation by squaring.
        let power = uxbinary_pow(&max_abs, n - 1);
        let n_ux = UXBinary::Finite(UBinary::new(BigUint::from(n), 0));
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
        return UXBinary::Finite(UBinary::new(BigUint::from(1u32), 0));
    }
    match base {
        UXBinary::Inf => UXBinary::Inf,
        UXBinary::Finite(b) => {
            if b.mantissa().is_zero() {
                return UXBinary::zero();
            }
            let mut result = UBinary::new(BigUint::from(1u32), 0);
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
fn compute_odd_power_bounds(lower: &XBinary, upper: &XBinary, n: NonZeroU32) -> (XBinary, XBinary) {
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
fn compute_even_power_bounds(
    lower: &XBinary,
    upper: &XBinary,
    n: NonZeroU32,
) -> (XBinary, XBinary) {
    let lower_non_negative = !is_negative(lower);
    let upper_non_positive = !is_positive(upper);

    if lower_non_negative {
        let result_lower = xbinary_pow(lower, n);
        let result_upper = xbinary_pow(upper, n);
        (result_lower, result_upper)
    } else if upper_non_positive {
        let result_lower = xbinary_pow(upper, n);
        let result_upper = xbinary_pow(lower, n);
        (result_lower, result_upper)
    } else {
        let lower_pow = xbinary_pow(lower, n);
        let upper_pow = xbinary_pow(upper, n);
        let result_upper = xbinary_max(&lower_pow, &upper_pow);
        (XBinary::zero(), result_upper)
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::Binary;
    use crate::computable::Computable;
    use crate::prefix::Prefix;
    use crate::sane::XI;
    use crate::test_utils::{bin, interval_noop_computable, unwrap_finite};

    fn assert_prefix_contains_expected(prefix: &Prefix, expected: &Binary, _tolerance_exp: &XI) {
        let lower = unwrap_finite(&prefix.lower());
        let upper = unwrap_finite(&prefix.upper());

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in prefix [{}, {}]",
            expected,
            lower,
            upper
        );
    }

    #[test]
    fn pow_constant_squared() {
        let three = Computable::constant(bin(3, 0));
        let squared = three.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        let expected = bin(9, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_constant_cubed() {
        let two = Computable::constant(bin(2, 0));
        let cubed = two.pow(3);
        let prefix = cubed.prefix().expect("prefix should succeed");

        let expected = bin(8, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_negative_even() {
        let neg_three = Computable::constant(bin(-3, 0));
        let squared = neg_three.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        let expected = bin(9, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_negative_odd() {
        let neg_two = Computable::constant(bin(-2, 0));
        let cubed = neg_two.pow(3);
        let prefix = cubed.prefix().expect("prefix should succeed");

        let expected = bin(-8, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_interval_positive_even() {
        let interval = interval_noop_computable(2, 4);
        let squared = interval.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        // Prefix may widen bounds (power-of-2 width rounding), so check containment
        assert!(unwrap_finite(&prefix.lower()) <= bin(4, 0));
        assert!(unwrap_finite(&prefix.upper()) >= bin(16, 0));
    }

    #[test]
    fn pow_interval_negative_even() {
        let interval = interval_noop_computable(-4, -2);
        let squared = interval.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        assert!(unwrap_finite(&prefix.lower()) <= bin(4, 0));
        assert!(unwrap_finite(&prefix.upper()) >= bin(16, 0));
    }

    #[test]
    fn pow_interval_spanning_zero_even() {
        let interval = interval_noop_computable(-2, 3);
        let squared = interval.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        assert!(unwrap_finite(&prefix.lower()) <= bin(0, 0));
        assert!(unwrap_finite(&prefix.upper()) >= bin(9, 0));
    }

    #[test]
    fn pow_interval_odd() {
        let interval = interval_noop_computable(2, 4);
        let cubed = interval.pow(3);
        let prefix = cubed.prefix().expect("prefix should succeed");

        assert!(unwrap_finite(&prefix.lower()) <= bin(8, 0));
        assert!(unwrap_finite(&prefix.upper()) >= bin(64, 0));
    }

    #[test]
    fn pow_interval_negative_odd() {
        let interval = interval_noop_computable(-4, -2);
        let cubed = interval.pow(3);
        let prefix = cubed.prefix().expect("prefix should succeed");

        assert!(unwrap_finite(&prefix.lower()) <= bin(-64, 0));
        assert!(unwrap_finite(&prefix.upper()) >= bin(-8, 0));
    }

    #[test]
    fn pow_exponent_one() {
        let three = Computable::constant(bin(3, 0));
        let result = three.pow(1);
        let prefix = result.prefix().expect("prefix should succeed");

        let expected = bin(3, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_exponent_zero() {
        let three = Computable::constant(bin(3, 0));
        let result = three.pow(0);
        let prefix = result.prefix().expect("prefix should succeed");

        let expected = bin(1, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_zero_to_zero() {
        let zero = Computable::constant(bin(0, 0));
        let result = zero.pow(0);
        let prefix = result.prefix().expect("prefix should succeed");

        let expected = bin(1, 0);
        assert_eq!(unwrap_finite(&prefix.lower()), expected);
        assert_eq!(unwrap_finite(&prefix.upper()), expected);
    }

    #[test]
    fn pow_in_expression() {
        let two_sq = Computable::constant(bin(2, 0)).pow(2);
        let three_sq = Computable::constant(bin(3, 0)).pow(2);
        let sum = two_sq + three_sq;

        let tolerance_exp = XI::from_i32(-8);
        let prefix = sum
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let expected = bin(13, 0);
        assert_prefix_contains_expected(&prefix, &expected, &tolerance_exp);
    }

    #[test]
    fn pow_with_sqrt() {
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(std::num::NonZeroU32::new(2).expect("2 is non-zero"));
        let squared = sqrt_two.pow(2);

        let tolerance_exp = XI::from_i32(-8);
        let prefix = squared
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_prefix_contains_expected(&prefix, &expected, &tolerance_exp);
    }

    #[test]
    fn pow_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let squared = zero.pow(2);
        let prefix = squared.prefix().expect("prefix should succeed");

        assert!(prefix.lower().is_zero());
        assert!(prefix.upper().is_zero());
    }
}
