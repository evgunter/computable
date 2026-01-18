//! Unsigned binary number implementation.
//!
//! This module provides `UBinary`, an unsigned variant of `Binary` for representing
//! non-negative values like interval widths.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Sub};

use num_bigint::{BigInt, BigUint};
use num_traits::Zero;

use crate::ordered_pair::Unsigned;

use super::binary_impl::Binary;
use super::error::BinaryError;
use super::shift::shift_mantissa_chunked;

/// Unsigned binary number represented as `mantissa * 2^exponent` where mantissa >= 0.
///
/// The mantissa is normalized to be odd unless the value is zero.
/// This type is used for representing non-negative quantities like interval widths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UBinary {
    mantissa: BigUint,
    exponent: BigInt,
}

impl UBinary {
    /// Creates a new UBinary number, normalizing the representation.
    pub fn new(mantissa: BigUint, exponent: BigInt) -> Self {
        Self::normalize(mantissa, exponent)
    }

    /// Returns the zero value.
    pub fn zero() -> Self {
        Self {
            mantissa: BigUint::zero(),
            exponent: BigInt::zero(),
        }
    }

    /// Returns a reference to the mantissa.
    pub fn mantissa(&self) -> &BigUint {
        &self.mantissa
    }

    /// Returns a reference to the exponent.
    pub fn exponent(&self) -> &BigInt {
        &self.exponent
    }

    /// Creates a UBinary from a Binary, returning an error if the mantissa is negative.
    pub fn try_from_binary(binary: &Binary) -> Result<Self, BinaryError> {
        use num_traits::Signed;

        if binary.mantissa().is_negative() {
            return Err(BinaryError::NegativeMantissa);
        }
        let mantissa = binary.mantissa().magnitude().clone();
        Ok(Self::new(mantissa, binary.exponent().clone()))
    }

    /// Converts this unsigned binary to a signed binary.
    pub fn to_binary(&self) -> Binary {
        Binary::new(BigInt::from(self.mantissa.clone()), self.exponent.clone())
    }

    /// Adds two UBinary numbers.
    pub fn add(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    /// Subtracts another UBinary from this one, saturating at zero.
    pub fn sub_saturating(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        if lhs >= rhs {
            Self::normalize(lhs - rhs, exponent)
        } else {
            Self::zero()
        }
    }

    /// Multiplies two UBinary numbers.
    pub fn mul(&self, other: &Self) -> Self {
        let exponent = &self.exponent + &other.exponent;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::normalize(mantissa, exponent)
    }

    /// Normalizes the representation by factoring out powers of 2 from the mantissa.
    fn normalize(mut mantissa: BigUint, mut exponent: BigInt) -> Self {
        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: BigInt::zero(),
            };
        }

        while (&mantissa % 2u32).is_zero() {
            mantissa /= 2u32;
            exponent += 1;
        }

        Self { mantissa, exponent }
    }

    /// Aligns the mantissas of two UBinary numbers to a common exponent.
    fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigUint, BigUint, BigInt) {
        let exponent = if lhs.exponent <= rhs.exponent {
            lhs.exponent.clone()
        } else {
            rhs.exponent.clone()
        };
        let lhs_shift = BigUint::try_from(&lhs.exponent - &exponent).unwrap_or_default();
        let rhs_shift = BigUint::try_from(&rhs.exponent - &exponent).unwrap_or_default();
        let lhs_mantissa = Self::shift_mantissa(&lhs.mantissa, &lhs_shift);
        let rhs_mantissa = Self::shift_mantissa(&rhs.mantissa, &rhs_shift);
        (lhs_mantissa, rhs_mantissa, exponent)
    }

    /// Shifts a mantissa left by the given amount, handling large shifts.
    fn shift_mantissa(mantissa: &BigUint, shift: &BigUint) -> BigUint {
        if shift.is_zero() {
            return mantissa.clone();
        }
        let chunk_limit = BigUint::from(usize::MAX);
        shift_mantissa_chunked::<BigUint>(mantissa, shift, &chunk_limit)
    }
}

impl Ord for UBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use the same comparison logic as Binary but with unsigned mantissas
        let self_binary = self.to_binary();
        let other_binary = other.to_binary();
        self_binary.cmp(&other_binary)
    }
}

impl PartialOrd for UBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl num_traits::Zero for UBinary {
    fn zero() -> Self {
        UBinary::zero()
    }

    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }
}

impl Add for UBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        UBinary::add(&self, &rhs)
    }
}

impl Sub for UBinary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        UBinary::sub_saturating(&self, &rhs)
    }
}

impl Mul for UBinary {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        UBinary::mul(&self, &rhs)
    }
}

impl Unsigned for UBinary {}

impl fmt::Display for UBinary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::display::format_ubinary_display(f, &self.mantissa, &self.exponent)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::{bin, ubin};

    #[test]
    fn ubinary_normalizes_even_mantissa() {
        let value = ubin(8, 0);
        assert_eq!(value.mantissa(), &BigUint::from(1u32));
        assert_eq!(value.exponent(), &BigInt::from(3));
    }

    #[test]
    fn ubinary_zero_uses_zero_exponent() {
        let value = UBinary::new(BigUint::zero(), BigInt::from(42));
        assert_eq!(value.mantissa(), &BigUint::zero());
        assert_eq!(value.exponent(), &BigInt::zero());
    }

    #[test]
    fn ubinary_ordering_works() {
        let one = ubin(1, 0);
        let two = ubin(1, 1);
        let half = ubin(1, -1);
        assert!(two > one);
        assert!(one > half);
    }

    #[test]
    fn ubinary_add_works() {
        let one = ubin(1, 0);
        let half = ubin(1, -1);
        let sum = one + half;
        let expected = ubin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn ubinary_sub_saturating_works() {
        let two = ubin(1, 1);
        let one = ubin(1, 0);
        let diff = two.sub_saturating(&one);
        let expected = ubin(1, 0);
        assert_eq!(diff, expected);

        // Test saturation at zero
        let saturated = one.sub_saturating(&two);
        assert_eq!(saturated, UBinary::zero());
    }

    #[test]
    fn ubinary_try_from_binary_works() {
        let positive = bin(5, 2);
        let result = UBinary::try_from_binary(&positive);
        assert!(result.is_ok());
        let ubinary = result.expect("should succeed");
        assert_eq!(ubinary.mantissa(), &BigUint::from(5u32));
        assert_eq!(ubinary.exponent(), &BigInt::from(2));

        let negative = bin(-5, 2);
        let result = UBinary::try_from_binary(&negative);
        assert!(result.is_err());
    }

    #[test]
    fn ubinary_to_binary_works() {
        let ubinary = ubin(7, 3);
        let binary = ubinary.to_binary();
        assert_eq!(binary.mantissa(), &BigInt::from(7));
        assert_eq!(binary.exponent(), &BigInt::from(3));
    }

    #[test]
    fn ubinary_mul_works() {
        let two = ubin(1, 1);
        let three = ubin(3, 0);
        let product = two * three;
        let expected = ubin(3, 1); // 2 * 3 = 6 = 3 * 2^1
        assert_eq!(product, expected);
    }
}
