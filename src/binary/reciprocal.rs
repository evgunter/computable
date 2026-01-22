//! Reciprocal computation for binary numbers.
//!
//! This module provides functions for computing reciprocals of extended binary numbers
//! with controlled precision and rounding.

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::binary_impl::Binary;
use super::error::BinaryError;
use super::xbinary::XBinary;

/// Specifies the rounding direction for reciprocal computation.
#[derive(Clone, Copy, Debug)]
pub enum ReciprocalRounding {
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceil,
}

/// Computes 1/denominator where denominator is a positive integer (BigUint).
///
/// Returns `mantissa * 2^(-precision_bits)` where:
/// - For `Floor`: `mantissa = floor(2^precision_bits / denominator)`
/// - For `Ceil`: `mantissa = ceil(2^precision_bits / denominator)`
///
/// Using `BigUint` enforces positivity through the type system.
///
/// # Arguments
/// * `denominator` - A positive BigUint to take the reciprocal of
/// * `precision_bits` - Number of bits of precision for the computation (as usize for bit shifting)
/// * `rounding` - Whether to round toward floor or ceiling
pub fn reciprocal_of_biguint(
    denominator: &BigUint,
    precision_bits: usize,
    rounding: ReciprocalRounding,
) -> Binary {
    let numerator = BigUint::one() << precision_bits;
    let quotient = match rounding {
        ReciprocalRounding::Floor => numerator.div_floor(denominator),
        ReciprocalRounding::Ceil => {
            (&numerator + denominator - BigUint::one()).div_floor(denominator)
        }
    };
    let exponent = -BigInt::from(precision_bits);
    Binary::new(BigInt::from(quotient), exponent)
}

/// Computes the reciprocal of the absolute value of an extended binary number.
///
/// The result is computed with the specified precision and rounding direction.
/// For infinite values, returns zero (1/infinity = 0).
///
/// # Arguments
/// * `value` - The value to take the reciprocal of
/// * `precision_bits` - Number of bits of precision for the computation
/// * `rounding` - Whether to round toward floor or ceiling
///
/// # Errors
/// Returns `BinaryError::ReciprocalOverflow` if the precision is too large.
pub fn reciprocal_rounded_abs_extended(
    value: &XBinary,
    precision_bits: &BigInt,
    rounding: ReciprocalRounding,
) -> Result<XBinary, BinaryError> {
    match value {
        XBinary::Finite(finite_value) => {
            let abs_mantissa = finite_value.mantissa().abs();
            let shift_bits = precision_bits - finite_value.exponent();
            let quotient = if shift_bits.is_negative() {
                match rounding {
                    ReciprocalRounding::Floor => BigInt::zero(),
                    ReciprocalRounding::Ceil => BigInt::one(),
                }
            } else {
                let shift = precision_bits_to_usize(&shift_bits)?;
                let numerator = BigInt::one() << shift;
                match rounding {
                    ReciprocalRounding::Floor => numerator.div_floor(&abs_mantissa),
                    ReciprocalRounding::Ceil => numerator.div_ceil(&abs_mantissa),
                }
            };
            let exponent = reciprocal_exponent(precision_bits)?;
            Ok(XBinary::Finite(Binary::new(quotient, exponent)))
        }
        XBinary::NegInf | XBinary::PosInf => Ok(XBinary::Finite(Binary::zero())),
    }
}

/// Computes the exponent for a reciprocal result given the precision.
fn reciprocal_exponent(precision_bits: &BigInt) -> Result<BigInt, BinaryError> {
    let precision = precision_bits_to_exponent(precision_bits)?;
    Ok(-precision)
}

/// Converts precision bits to a usize shift amount.
fn precision_bits_to_usize(precision_bits: &BigInt) -> Result<usize, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    precision_bits
        .to_usize()
        .ok_or(BinaryError::ReciprocalOverflow)
}

/// Converts precision bits to an exponent value.
fn precision_bits_to_exponent(precision_bits: &BigInt) -> Result<BigInt, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    Ok(precision_bits.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::xbin;

    #[test]
    fn reciprocal_of_biguint_basic() {
        // 1/5 with precision 8
        let denom = BigUint::from(5u32);
        let result = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Floor);

        // 2^8 / 5 = 256 / 5 = 51 (floor)
        // Result is 51 * 2^-8
        // After normalization (51 is odd), mantissa = 51, exponent = -8
        assert_eq!(result.mantissa(), &BigInt::from(51));
        assert_eq!(result.exponent(), &BigInt::from(-8));
    }

    #[test]
    fn reciprocal_of_biguint_rounding_modes() {
        // 1/3 with precision 8
        let denom = BigUint::from(3u32);

        let floor = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Floor);
        let ceil = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Ceil);

        // 2^8 / 3 = 256 / 3 = 85.33...
        // Floor = 85, Ceil = 86
        // After normalization: 85 (odd), 43 * 2^1 = 86
        assert_eq!(floor.mantissa(), &BigInt::from(85));
        assert_eq!(floor.exponent(), &BigInt::from(-8));
        assert_eq!(ceil.mantissa(), &BigInt::from(43));
        assert_eq!(ceil.exponent(), &BigInt::from(-7)); // 43 * 2^-7 = 86 * 2^-8

        // Ceil should be > Floor since 1/3 is not exactly representable
        assert!(ceil > floor);
    }

    #[test]
    fn reciprocal_of_infinity_is_zero() {
        let precision = BigInt::from(10);

        let result = reciprocal_rounded_abs_extended(
            &XBinary::PosInf,
            &precision,
            ReciprocalRounding::Floor,
        )
        .expect("should succeed");
        assert!(result.is_zero());

        let result =
            reciprocal_rounded_abs_extended(&XBinary::NegInf, &precision, ReciprocalRounding::Ceil)
                .expect("should succeed");
        assert!(result.is_zero());
    }

    #[test]
    fn reciprocal_basic_computation() {
        // 1/2 with precision 4 should give approximately 0.5
        let value = xbin(1, 1); // 2
        let precision = BigInt::from(4);

        let result = reciprocal_rounded_abs_extended(&value, &precision, ReciprocalRounding::Floor)
            .expect("should succeed");

        // With precision 4, we compute 2^4 / 2 = 8, then the exponent is -4
        // So we get 8 * 2^-4 = 0.5
        if let XBinary::Finite(binary) = result {
            assert_eq!(binary.mantissa(), &BigInt::from(1)); // 8 normalized to 1 * 2^3
            assert_eq!(binary.exponent(), &BigInt::from(-1)); // -4 + 3 = -1
        } else {
            panic!("expected finite result");
        }
    }

    #[test]
    fn reciprocal_rounding_modes() {
        // 1/3 should give different results for floor vs ceil
        let value = xbin(3, 0); // 3
        let precision = BigInt::from(8);

        let floor = reciprocal_rounded_abs_extended(&value, &precision, ReciprocalRounding::Floor)
            .expect("should succeed");
        let ceil = reciprocal_rounded_abs_extended(&value, &precision, ReciprocalRounding::Ceil)
            .expect("should succeed");

        // Ceil should be >= Floor for positive values
        assert!(ceil >= floor);
    }

    #[test]
    fn reciprocal_handles_negative_mantissa() {
        // Reciprocal of absolute value of -4 should be same as reciprocal of 4
        let neg_value = xbin(-1, 2); // -4
        let pos_value = xbin(1, 2); // 4
        let precision = BigInt::from(8);

        let neg_result =
            reciprocal_rounded_abs_extended(&neg_value, &precision, ReciprocalRounding::Floor)
                .expect("should succeed");
        let pos_result =
            reciprocal_rounded_abs_extended(&pos_value, &precision, ReciprocalRounding::Floor)
                .expect("should succeed");

        assert_eq!(neg_result, pos_result);
    }

    #[test]
    fn reciprocal_negative_shift_floor_is_zero() {
        // Very large value with small precision
        let value = xbin(1, 100); // 2^100
        let precision = BigInt::from(10);

        let result = reciprocal_rounded_abs_extended(&value, &precision, ReciprocalRounding::Floor)
            .expect("should succeed");

        assert!(result.is_zero());
    }

    #[test]
    fn reciprocal_negative_shift_ceil_is_one() {
        // Very large value with small precision
        let value = xbin(1, 100); // 2^100
        let precision = BigInt::from(10);

        let result = reciprocal_rounded_abs_extended(&value, &precision, ReciprocalRounding::Ceil)
            .expect("should succeed");

        // Should be 1 * 2^-10 (the smallest positive representable value)
        if let XBinary::Finite(binary) = result {
            assert_eq!(binary.mantissa(), &BigInt::from(1));
            assert_eq!(binary.exponent(), &BigInt::from(-10));
        } else {
            panic!("expected finite result");
        }
    }
}
