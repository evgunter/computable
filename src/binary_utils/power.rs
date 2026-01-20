//! Power operations for Binary and XBinary numbers.
//!
//! This module provides integer power computation functions that operate on
//! [`Binary`](crate::binary::Binary) and [`XBinary`](crate::binary::XBinary)
//! values without requiring the Node/Computable infrastructure.

use num_bigint::BigInt;
use num_traits::{One, Signed};

use crate::binary::{Binary, XBinary};

/// Computes x^n for a Binary value using repeated multiplication.
///
/// # Arguments
///
/// * `x` - The base value
/// * `n` - The exponent (non-negative integer)
///
/// # Returns
///
/// The result of x raised to the power n.
///
/// # Examples
///
/// ```
/// use computable::binary_utils::power::binary_pow;
/// use computable::Binary;
/// use num_bigint::BigInt;
///
/// let x = Binary::new(BigInt::from(3), BigInt::from(0)); // 3
/// let result = binary_pow(&x, 2);
/// assert_eq!(result, Binary::new(BigInt::from(9), BigInt::from(0))); // 9
/// ```
pub fn binary_pow(x: &Binary, n: u32) -> Binary {
    if n == 0 {
        return Binary::new(BigInt::one(), BigInt::from(0));
    }

    // Use exponentiation by squaring for O(log n) complexity.
    // Algorithm: x^n = (x^2)^(n/2) if n is even, x * (x^2)^((n-1)/2) if n is odd.
    let mut result = Binary::new(BigInt::one(), BigInt::from(0));
    let mut base = x.clone();
    let mut exp = n;

    while exp > 0 {
        if exp & 1 == 1 {
            result = result.mul(&base);
        }
        exp >>= 1;
        if exp > 0 {
            base = base.mul(&base);
        }
    }

    result
}

/// Computes x^n for an XBinary value.
///
/// Handles infinities according to mathematical conventions:
/// - (+∞)^n = +∞ for all n >= 1
/// - (-∞)^n = +∞ for even n, -∞ for odd n
/// - For finite values, delegates to [`binary_pow`]
///
/// # Arguments
///
/// * `x` - The base value (may be infinite)
/// * `n` - The exponent (non-negative integer)
///
/// # Returns
///
/// The result of x raised to the power n.
// TODO: Use NonZeroU32 for the exponent to avoid handling infinity^0 (which is indeterminate).
pub fn xbinary_pow(x: &XBinary, n: u32) -> XBinary {
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

/// Returns the maximum of two XBinary values.
///
/// Uses the standard ordering: -∞ < all finite values < +∞.
pub fn xbinary_max(a: &XBinary, b: &XBinary) -> XBinary {
    if a >= b {
        a.clone()
    } else {
        b.clone()
    }
}

/// Returns true if the XBinary value is negative (less than zero).
///
/// - `-∞` is negative
/// - `+∞` is not negative
/// - Finite values are negative if their mantissa is negative
pub fn is_negative(x: &XBinary) -> bool {
    match x {
        XBinary::NegInf => true,
        XBinary::PosInf => false,
        XBinary::Finite(b) => b.mantissa().is_negative(),
    }
}

/// Returns true if the XBinary value is positive (greater than zero).
///
/// - `-∞` is not positive
/// - `+∞` is positive
/// - Finite values are positive if their mantissa is positive
pub fn is_positive(x: &XBinary) -> bool {
    match x {
        XBinary::NegInf => false,
        XBinary::PosInf => true,
        XBinary::Finite(b) => b.mantissa().is_positive(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::test_utils::{bin, xbin};

    #[test]
    fn binary_pow_zero_exponent() {
        let x = bin(3, 0);
        assert_eq!(binary_pow(&x, 0), bin(1, 0));
    }

    #[test]
    fn binary_pow_one_exponent() {
        let x = bin(3, 0);
        assert_eq!(binary_pow(&x, 1), bin(3, 0));
    }

    #[test]
    fn binary_pow_square() {
        let x = bin(3, 0);
        assert_eq!(binary_pow(&x, 2), bin(9, 0));
    }

    #[test]
    fn binary_pow_cube() {
        let x = bin(3, 0);
        assert_eq!(binary_pow(&x, 3), bin(27, 0));
    }

    #[test]
    fn binary_pow_negative_base() {
        let x = bin(-2, 0);
        assert_eq!(binary_pow(&x, 2), bin(4, 0)); // (-2)^2 = 4
        assert_eq!(binary_pow(&x, 3), bin(-8, 0)); // (-2)^3 = -8
    }

    #[test]
    fn binary_pow_with_exponent() {
        // (2^3)^2 = 2^6 = 64
        let x = bin(1, 3); // 2^3 = 8
        let result = binary_pow(&x, 2);
        assert_eq!(result, bin(1, 6)); // 2^6 = 64
    }

    #[test]
    fn xbinary_pow_finite() {
        let x = xbin(3, 0);
        let result = xbinary_pow(&x, 2);
        assert_eq!(result, xbin(9, 0));
    }

    #[test]
    fn xbinary_pow_pos_inf() {
        let result = xbinary_pow(&XBinary::PosInf, 5);
        assert_eq!(result, XBinary::PosInf);
    }

    #[test]
    fn xbinary_pow_neg_inf_even() {
        let result = xbinary_pow(&XBinary::NegInf, 4);
        assert_eq!(result, XBinary::PosInf);
    }

    #[test]
    fn xbinary_pow_neg_inf_odd() {
        let result = xbinary_pow(&XBinary::NegInf, 3);
        assert_eq!(result, XBinary::NegInf);
    }

    #[test]
    fn xbinary_max_finite() {
        let a = xbin(3, 0);
        let b = xbin(5, 0);
        assert_eq!(xbinary_max(&a, &b), xbin(5, 0));
        assert_eq!(xbinary_max(&b, &a), xbin(5, 0));
    }

    #[test]
    fn xbinary_max_with_inf() {
        let a = xbin(3, 0);
        assert_eq!(xbinary_max(&a, &XBinary::PosInf), XBinary::PosInf);
        assert_eq!(xbinary_max(&XBinary::NegInf, &a), xbin(3, 0));
    }

    #[test]
    fn is_negative_tests() {
        assert!(is_negative(&XBinary::NegInf));
        assert!(!is_negative(&XBinary::PosInf));
        assert!(is_negative(&xbin(-5, 0)));
        assert!(!is_negative(&xbin(5, 0)));
        assert!(!is_negative(&xbin(0, 0)));
    }

    #[test]
    fn is_positive_tests() {
        assert!(!is_positive(&XBinary::NegInf));
        assert!(is_positive(&XBinary::PosInf));
        assert!(!is_positive(&xbin(-5, 0)));
        assert!(is_positive(&xbin(5, 0)));
        assert!(!is_positive(&xbin(0, 0)));
    }
}
