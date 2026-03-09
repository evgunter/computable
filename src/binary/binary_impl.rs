//! Core signed binary number implementation.
//!
//! This module provides `Binary`, an exact binary representation of rational numbers
//! as `mantissa * 2^exponent` where the mantissa is normalized to be odd (unless zero).

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Shl, Shr, Sub};

use num_bigint::{BigInt, BigUint};
use num_traits::{Float, Signed, Zero};

use super::error::BinaryError;

use super::shift::shift_mantissa_chunked;

/// Exact binary number represented as `mantissa * 2^exponent`.
///
/// The mantissa is normalized to be odd unless the value is zero.
/// This normalization ensures a canonical representation for each value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binary {
    mantissa: BigInt,
    exponent: BigInt,
}

impl Binary {
    /// Creates a new Binary number, normalizing the representation.
    pub fn new(mantissa: BigInt, exponent: BigInt) -> Self {
        Self::normalize(mantissa, exponent)
    }

    /// Creates a Binary from components that are already in canonical form
    /// (mantissa is odd or zero). Skips the normalize step.
    ///
    /// # Safety invariant (unchecked)
    ///
    /// The caller must ensure: if `mantissa` is nonzero, it is odd.
    /// If `mantissa` is zero, `exponent` must be zero.
    pub(crate) fn new_normalized(mantissa: BigInt, exponent: BigInt) -> Self {
        debug_assert!(
            mantissa.is_zero() && exponent.is_zero()
                || !mantissa.is_zero()
                    && mantissa.magnitude().trailing_zeros() == Some(0),
            "new_normalized: mantissa must be odd (or both zero), got mantissa={mantissa}, exponent={exponent}"
        );
        Self { mantissa, exponent }
    }

    /// Returns the zero value.
    pub fn zero() -> Self {
        Self {
            mantissa: BigInt::zero(),
            exponent: BigInt::zero(),
        }
    }

    /// Returns a reference to the mantissa.
    pub fn mantissa(&self) -> &BigInt {
        &self.mantissa
    }

    /// Returns a reference to the exponent.
    pub fn exponent(&self) -> &BigInt {
        &self.exponent
    }

    /// Converts an f64 to a Binary.
    ///
    /// Returns an error if the input is NaN or infinite (use XBinary for infinity).
    ///
    /// # Example
    ///
    /// ```
    /// use computable::Binary;
    ///
    /// let binary = Binary::from_f64(1.5).unwrap();
    /// // 1.5 = 3 * 2^-1
    /// assert_eq!(binary.mantissa(), &num_bigint::BigInt::from(3));
    /// assert_eq!(binary.exponent(), &num_bigint::BigInt::from(-1));
    /// ```
    pub fn from_f64(value: f64) -> Result<Self, BinaryError> {
        if value.is_nan() {
            return Err(BinaryError::Nan);
        }
        if value.is_infinite() {
            return Err(BinaryError::Infinity);
        }
        if value == 0.0_f64 {
            return Ok(Self::zero());
        }
        let (mantissa, exponent, sign) = value.integer_decode();
        let signed_mantissa = BigInt::from(sign) * BigInt::from(mantissa);
        Ok(Self::new(signed_mantissa, BigInt::from(exponent)))
    }

    /// Adds two Binary numbers.
    pub fn add(&self, other: &Self) -> Self {
        if self.mantissa.is_zero() {
            return other.clone();
        }
        if other.mantissa.is_zero() {
            return self.clone();
        }
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    /// Subtracts another Binary number from this one.
    pub fn sub(&self, other: &Self) -> Self {
        if other.mantissa.is_zero() {
            return self.clone();
        }
        if self.mantissa.is_zero() {
            return other.neg();
        }
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs - rhs, exponent)
    }

    /// Negates this Binary number.
    pub fn neg(&self) -> Self {
        if self.mantissa.is_zero() {
            return self.clone();
        }
        Self {
            mantissa: -self.mantissa.clone(),
            exponent: self.exponent.clone(),
        }
    }

    /// Multiplies two Binary numbers.
    ///
    /// Since both mantissas are odd (canonical form), their product is also odd,
    /// so normalization is unnecessary.
    pub fn mul(&self, other: &Self) -> Self {
        if self.mantissa.is_zero() || other.mantissa.is_zero() {
            return Self::zero();
        }
        let exponent = &self.exponent + &other.exponent;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::new_normalized(mantissa, exponent)
    }

    /// Returns the magnitude (absolute value) of this Binary number as a UBinary.
    ///
    /// Since `self` is already normalized (odd mantissa), the magnitude is also odd,
    /// so we skip re-normalization in UBinary.
    pub fn magnitude(&self) -> super::ubinary::UBinary {
        use super::ubinary::UBinary;
        if self.mantissa.is_zero() {
            return UBinary::zero();
        }
        UBinary::new_normalized(self.mantissa.magnitude().clone(), self.exponent.clone())
    }

    /// Normalizes the representation by factoring out powers of 2 from the mantissa.
    fn normalize(mut mantissa: BigInt, mut exponent: BigInt) -> Self {
        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: BigInt::zero(),
            };
        }

        if let Some(tz_u64) = mantissa.magnitude().trailing_zeros() {
            let tz = crate::sane::bits_as_usize(tz_u64);
            mantissa >>= tz;
            exponent += BigInt::from(tz);
        }

        Self { mantissa, exponent }
    }

    /// Aligns the mantissas of two Binary numbers to a common exponent.
    /// Returns (lhs_mantissa, rhs_mantissa, common_exponent) where both mantissas
    /// are shifted to the minimum exponent of the two inputs.
    ///
    /// One of the two mantissas is always returned without shifting (the one with
    /// the smaller exponent). The other is shifted left by the exponent difference.
    pub fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigInt, BigInt, BigInt) {
        use num_traits::ToPrimitive;

        match lhs.exponent.cmp(&rhs.exponent) {
            Ordering::Equal => {
                (lhs.mantissa.clone(), rhs.mantissa.clone(), lhs.exponent.clone())
            }
            Ordering::Less => {
                let diff = &rhs.exponent - &lhs.exponent;
                let rhs_mantissa = if let Some(shift) = diff.to_usize() {
                    crate::assert_sane_computation_size!(shift);
                    &rhs.mantissa << shift
                } else {
                    let shift_bu = BigUint::try_from(diff).unwrap_or_default();
                    Self::shift_mantissa(&rhs.mantissa, &shift_bu)
                };
                (lhs.mantissa.clone(), rhs_mantissa, lhs.exponent.clone())
            }
            Ordering::Greater => {
                let diff = &lhs.exponent - &rhs.exponent;
                let lhs_mantissa = if let Some(shift) = diff.to_usize() {
                    crate::assert_sane_computation_size!(shift);
                    &lhs.mantissa << shift
                } else {
                    let shift_bu = BigUint::try_from(diff).unwrap_or_default();
                    Self::shift_mantissa(&lhs.mantissa, &shift_bu)
                };
                (lhs_mantissa, rhs.mantissa.clone(), rhs.exponent.clone())
            }
        }
    }

    /// Shifts a mantissa left by the given amount, handling large shifts.
    fn shift_mantissa(mantissa: &BigInt, shift: &BigUint) -> BigInt {
        if shift.is_zero() {
            return mantissa.clone();
        }
        shift_mantissa_chunked::<BigInt>(mantissa, shift, usize::MAX)
    }

    /// Compares two binary values with potentially different exponents.
    pub(crate) fn cmp_shifted(
        mantissa: &BigInt,
        exponent: BigInt,
        other: &BigInt,
        other_exp: BigInt,
    ) -> Ordering {
        /// Compares `large_mantissa * 2^shift_amount` against `small_mantissa`.
        fn cmp_with_shift(
            large_mantissa: &BigInt,
            small_mantissa: &BigInt,
            shift_diff: BigInt,
        ) -> Ordering {
            use num_traits::ToPrimitive;

            if let Some(shift_amount) = shift_diff.to_usize() {
                let shifted = large_mantissa << shift_amount;
                shifted.cmp(small_mantissa)
            } else if large_mantissa.is_zero() {
                BigInt::zero().cmp(small_mantissa)
            } else if large_mantissa.is_positive() {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }

        match exponent.cmp(&other_exp) {
            Ordering::Equal => mantissa.cmp(other),
            Ordering::Greater => {
                cmp_with_shift(mantissa, other, exponent - other_exp)
            }
            Ordering::Less => {
                cmp_with_shift(other, mantissa, other_exp - exponent).reverse()
            }
        }
    }
}

impl Add for Binary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Binary::add(&self, &rhs)
    }
}

impl Sub for Binary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Binary::sub(&self, &rhs)
    }
}

impl Neg for Binary {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Binary::neg(&self)
    }
}

impl Mul for Binary {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Binary::mul(&self, &rhs)
    }
}

impl Shl<u32> for Binary {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // shifting = exponent adjustment
    fn shl(self, rhs: u32) -> Self::Output {
        if self.mantissa.is_zero() {
            return self;
        }
        Self::new_normalized(self.mantissa, self.exponent + BigInt::from(rhs))
    }
}

impl Shr<u32> for Binary {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // shifting = exponent adjustment
    fn shr(self, rhs: u32) -> Self::Output {
        if self.mantissa.is_zero() {
            return self;
        }
        Self::new_normalized(self.mantissa, self.exponent - BigInt::from(rhs))
    }
}

impl num_traits::Zero for Binary {
    fn zero() -> Self {
        Binary::zero()
    }

    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }
}

impl Ord for Binary {
    fn cmp(&self, other: &Self) -> Ordering {
        Self::cmp_shifted(
            &self.mantissa,
            self.exponent.clone(),
            &other.mantissa,
            other.exponent.clone(),
        )
    }
}

impl PartialOrd for Binary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::display::format_binary_display(f, &self.mantissa, &self.exponent)
    }
}

impl crate::ordered_pair::AbsDistance<Binary, super::UBinary> for Binary {
    /// Computes the absolute distance between two Binary values, returning a UBinary.
    fn abs_distance(self, other: Self) -> super::UBinary {
        Binary::sub(&self, &other).magnitude()
    }
}

impl crate::ordered_pair::AddWidth<Binary, super::UBinary> for Binary {
    fn add_width(self, width: super::UBinary) -> Self {
        Binary::add(&self, &width.to_binary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::bin;

    #[test]
    fn binary_normalizes_even_mantissa() {
        let value = bin(8, 0);
        assert_eq!(value.mantissa(), &BigInt::from(1_i32));
        assert_eq!(value.exponent(), &BigInt::from(3_i32));
    }

    #[test]
    fn binary_zero_uses_zero_exponent() {
        let value = Binary::new(BigInt::zero(), BigInt::from(42_i32));
        assert_eq!(value.mantissa(), &BigInt::zero());
        assert_eq!(value.exponent(), &BigInt::zero());
    }

    #[test]
    fn binary_ordering_with_exponents() {
        let one = bin(1, 0);
        let half = bin(1, -1);
        assert!(one > half);
    }

    #[test]
    fn binary_ordering_handles_large_exponent_gaps() {
        use num_traits::One;

        let huge_exp = BigInt::from(usize::MAX) + BigInt::one();
        let tiny_exp = -huge_exp.clone();
        let huge_pos = Binary::new(BigInt::from(1_i32), huge_exp.clone());
        let tiny_pos = Binary::new(BigInt::from(1_i32), tiny_exp.clone());
        assert!(huge_pos > tiny_pos);

        let huge_neg = Binary::new(BigInt::from(-1_i32), huge_exp);
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_ordering_overflow_path_uses_sign() {
        use num_traits::One;

        let huge_exp = BigInt::from(usize::MAX) + BigInt::one();
        let tiny_exp = -huge_exp.clone();
        let huge_pos = Binary::new(BigInt::from(1_i32), huge_exp.clone());
        let tiny_neg = Binary::new(BigInt::from(-1_i32), tiny_exp.clone());
        assert!(huge_pos > tiny_neg);

        let huge_neg = Binary::new(BigInt::from(-1_i32), huge_exp);
        let tiny_pos = Binary::new(BigInt::from(1_i32), tiny_exp);
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_add_aligns_exponents() {
        let one = bin(1, 0);
        let half = bin(1, -1);
        let sum = one + half;
        let expected = bin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn binary_sub_handles_negative() {
        let one = bin(1, 0);
        let two = bin(1, 1);
        let diff = one - two;
        let expected = bin(-1, 0);
        assert_eq!(diff, expected);
    }

    #[test]
    fn binary_mul_adds_exponents() {
        let two = bin(1, 1);
        let half = bin(1, -1);
        let product = two.mul(half);
        let expected = bin(1, 0);
        assert_eq!(product, expected);
    }

    #[test]
    fn binary_neg_flips_sign() {
        let pos = bin(3, 2);
        let neg = -pos;
        assert_eq!(neg.mantissa(), &BigInt::from(-3_i32));
        assert_eq!(neg.exponent(), &BigInt::from(2_i32));
    }

    #[test]
    fn binary_neg_zero_is_zero() {
        let zero = Binary::zero();
        let neg_zero = -zero.clone();
        assert_eq!(neg_zero, zero);
    }

    #[test]
    fn binary_from_f64_converts_normal_values() {
        // 1.5 = 3 * 2^-1
        let result = Binary::from_f64(1.5).expect("should succeed");
        assert_eq!(result.mantissa(), &BigInt::from(3_i32));
        assert_eq!(result.exponent(), &BigInt::from(-1_i32));

        // -2.0 = -1 * 2^1
        let neg_result = Binary::from_f64(-2.0).expect("should succeed");
        assert_eq!(neg_result.mantissa(), &BigInt::from(-1_i32));
        assert_eq!(neg_result.exponent(), &BigInt::from(1_i32));
    }

    #[test]
    fn binary_from_f64_handles_zero() {
        let zero = Binary::from_f64(0.0).expect("should succeed");
        assert!(zero.mantissa().is_zero());
    }

    #[test]
    fn binary_from_f64_rejects_nan() {
        assert_eq!(Binary::from_f64(f64::NAN), Err(BinaryError::Nan));
    }

    #[test]
    fn binary_from_f64_rejects_infinity() {
        assert_eq!(Binary::from_f64(f64::INFINITY), Err(BinaryError::Infinity));
        assert_eq!(
            Binary::from_f64(f64::NEG_INFINITY),
            Err(BinaryError::Infinity)
        );
    }
}
