//! Core signed binary number implementation.
//!
//! This module provides `Binary`, an exact binary representation of rational numbers
//! as `mantissa * 2^exponent` where the mantissa is normalized to be odd (unless zero).

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Shl, Shr, Sub};

use num_bigint::BigInt;
use num_traits::{Float, Signed, Zero};

use super::error::BinaryError;

/// Exact binary number represented as `mantissa * 2^exponent`.
///
/// The mantissa is normalized to be odd unless the value is zero.
/// This normalization ensures a canonical representation for each value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binary {
    mantissa: BigInt,
    exponent: i64,
}

impl Binary {
    /// Creates a new Binary number, normalizing the representation.
    pub fn new(mantissa: BigInt, exponent: i64) -> Self {
        Self::normalize(mantissa, exponent)
    }

    /// Creates a Binary from components that are already in canonical form
    /// (mantissa is odd or zero). Skips the normalize step.
    pub(crate) fn new_normalized(mantissa: BigInt, exponent: i64) -> Self {
        debug_assert!(
            mantissa.is_zero() && exponent == 0
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
            exponent: 0_i64,
        }
    }

    /// Returns the value one (1 = 1 * 2^0).
    pub fn one() -> Self {
        Self {
            mantissa: BigInt::from(1_i32),
            exponent: 0_i64,
        }
    }

    /// Returns a reference to the mantissa.
    pub fn mantissa(&self) -> &BigInt {
        &self.mantissa
    }

    /// Returns the exponent.
    pub fn exponent(&self) -> i64 {
        self.exponent
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
    /// assert_eq!(binary.exponent(), -1_i64);
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
        Ok(Self::new(signed_mantissa, i64::from(exponent)))
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

    /// Negates this Binary number (borrowing).
    pub fn neg(&self) -> Self {
        if self.mantissa.is_zero() {
            return self.clone();
        }
        Self {
            mantissa: -self.mantissa.clone(),
            exponent: self.exponent,
        }
    }

    /// Negates this Binary number, consuming it to avoid cloning the mantissa.
    pub fn neg_owned(self) -> Self {
        if self.mantissa.is_zero() {
            return self;
        }
        Self {
            mantissa: -self.mantissa,
            exponent: self.exponent,
        }
    }

    /// Multiplies two Binary numbers.
    pub fn mul(&self, other: &Self) -> Self {
        if self.mantissa.is_zero() || other.mantissa.is_zero() {
            return Self::zero();
        }
        let exponent = self.exponent.checked_add(other.exponent).unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in Binary::mul")
        });
        let mantissa = &self.mantissa * &other.mantissa;
        Self::new_normalized(mantissa, exponent)
    }

    /// Returns the magnitude (absolute value) of this Binary number as a UBinary.
    pub fn magnitude(&self) -> super::ubinary::UBinary {
        use super::ubinary::UBinary;
        if self.mantissa.is_zero() {
            return UBinary::zero();
        }
        UBinary::new_normalized(self.mantissa.magnitude().clone(), self.exponent)
    }

    /// Normalizes the representation by factoring out powers of 2 from the mantissa.
    fn normalize(mut mantissa: BigInt, mut exponent: i64) -> Self {
        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: 0_i64,
            };
        }

        if let Some(tz_u64) = mantissa.magnitude().trailing_zeros() {
            let tz = crate::sane::bits_as_usize(tz_u64);
            mantissa >>= tz;
            exponent = exponent.checked_add(i64::try_from(tz).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "trailing zeros exceeds i64 in Binary::normalize"
                )
            }))
            .unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in Binary::normalize"
                )
            });
        }

        Self { mantissa, exponent }
    }

    /// Aligns the mantissas of two Binary numbers to a common exponent.
    /// Returns (lhs_mantissa, rhs_mantissa, common_exponent) where both mantissas
    /// are shifted to the minimum exponent of the two inputs.
    pub fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigInt, BigInt, i64) {
        let exponent = std::cmp::min(lhs.exponent, rhs.exponent);
        let lhs_shift = lhs.exponent.abs_diff(exponent);
        let rhs_shift = rhs.exponent.abs_diff(exponent);
        let lhs_mantissa = if lhs_shift == 0 {
            lhs.mantissa.clone()
        } else {
            let shift_usize = usize::try_from(lhs_shift).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "shift exceeds usize in Binary::align_mantissas"
                )
            });
            crate::assert_sane_computation_size!(shift_usize);
            &lhs.mantissa << shift_usize
        };
        let rhs_mantissa = if rhs_shift == 0 {
            rhs.mantissa.clone()
        } else {
            let shift_usize = usize::try_from(rhs_shift).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "shift exceeds usize in Binary::align_mantissas"
                )
            });
            crate::assert_sane_computation_size!(shift_usize);
            &rhs.mantissa << shift_usize
        };
        (lhs_mantissa, rhs_mantissa, exponent)
    }

    /// Compares two binary values with potentially different exponents.
    ///
    /// Uses a bit-length fast path: for values with different magnitude bit lengths,
    /// the comparison is determined in O(1) without shifting. Only falls back to
    /// mantissa alignment when bit lengths are equal.
    pub(crate) fn cmp_shifted(
        mantissa: &BigInt,
        exponent: i64,
        other: &BigInt,
        other_exp: i64,
    ) -> Ordering {
        // 1. Sign-based fast path
        let self_sign = mantissa.sign();
        let other_sign = other.sign();
        match self_sign.cmp(&other_sign) {
            Ordering::Equal => {}
            ord @ Ordering::Less | ord @ Ordering::Greater => return ord,
        }

        // Both have the same sign. Handle zero cases.
        if mantissa.is_zero() {
            // Both zero (same sign and self is zero means other is also zero)
            return Ordering::Equal;
        }

        // Both non-zero with the same sign.
        // 2. Bit-length fast path on magnitudes
        // For non-zero: |value| = |mantissa| * 2^exponent, |mantissa| is odd
        // Bit length of |value| = magnitude.bits() + exponent
        let self_bits = i128::from(mantissa.magnitude().bits()).saturating_add(i128::from(exponent));
        let other_bits =
            i128::from(other.magnitude().bits()).saturating_add(i128::from(other_exp));

        if self_bits != other_bits {
            // Different magnitude bit lengths — determined by bit length.
            // For positive values: larger bit length means larger value.
            // For negative values: larger magnitude bit length means smaller (more negative) value.
            let mag_ord = self_bits.cmp(&other_bits);
            return if mantissa.is_positive() {
                mag_ord
            } else {
                mag_ord.reverse()
            };
        }

        // 3. Same bit length — must align and compare mantissas
        fn cmp_with_shift(
            large_mantissa: &BigInt,
            small_mantissa: &BigInt,
            shift_amount: u64,
        ) -> Ordering {
            // If shift_amount exceeds MAX_COMPUTATION_BITS, the shifted value
            // would be astronomically large. Compare by sign instead.
            #[allow(clippy::as_conversions)] // usize -> u64: always widens or is no-op
            let max_shift = crate::MAX_COMPUTATION_BITS as u64;
            if shift_amount <= max_shift {
                // SAFETY: shift_amount <= MAX_COMPUTATION_BITS <= usize::MAX on 64-bit
                let shift_usize = usize::try_from(shift_amount).unwrap_or_else(|_| {
                    unreachable!("shift_amount <= MAX_COMPUTATION_BITS <= usize::MAX")
                });
                let shifted: BigInt = large_mantissa << shift_usize;
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
                cmp_with_shift(mantissa, other, exponent.abs_diff(other_exp))
            }
            Ordering::Less => {
                cmp_with_shift(other, mantissa, other_exp.abs_diff(exponent)).reverse()
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
        self.neg_owned()
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
        Self::new_normalized(
            self.mantissa,
            self.exponent.checked_add(i64::from(rhs)).unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in Binary::shl"
                )
            }),
        )
    }
}

impl Shr<u32> for Binary {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // shifting = exponent adjustment
    fn shr(self, rhs: u32) -> Self::Output {
        if self.mantissa.is_zero() {
            return self;
        }
        Self::new_normalized(
            self.mantissa,
            self.exponent.checked_sub(i64::from(rhs)).unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in Binary::shr"
                )
            }),
        )
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
            self.exponent,
            &other.mantissa,
            other.exponent,
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
        super::display::format_binary_display(f, &self.mantissa, self.exponent)
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
        assert_eq!(value.exponent(), 3_i64);
    }

    #[test]
    fn binary_zero_uses_zero_exponent() {
        let value = Binary::new(BigInt::zero(), 42_i64);
        assert_eq!(value.mantissa(), &BigInt::zero());
        assert_eq!(value.exponent(), 0_i64);
    }

    #[test]
    fn binary_ordering_with_exponents() {
        let one = bin(1, 0);
        let half = bin(1, -1);
        assert!(one > half);
    }

    #[test]
    fn binary_ordering_handles_large_exponent_gaps() {
        let huge_pos = Binary::new(BigInt::from(1_i32), i64::MAX);
        let tiny_pos = Binary::new(BigInt::from(1_i32), i64::MIN);
        assert!(huge_pos > tiny_pos);

        let huge_neg = Binary::new(BigInt::from(-1_i32), i64::MAX);
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_ordering_overflow_path_uses_sign() {
        let huge_pos = Binary::new(BigInt::from(1_i32), i64::MAX);
        let tiny_neg = Binary::new(BigInt::from(-1_i32), i64::MIN);
        assert!(huge_pos > tiny_neg);

        let huge_neg = Binary::new(BigInt::from(-1_i32), i64::MAX);
        let tiny_pos = Binary::new(BigInt::from(1_i32), i64::MIN);
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
        assert_eq!(neg.exponent(), 2_i64);
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
        assert_eq!(result.exponent(), -1_i64);

        // -2.0 = -1 * 2^1
        let neg_result = Binary::from_f64(-2.0).expect("should succeed");
        assert_eq!(neg_result.mantissa(), &BigInt::from(-1_i32));
        assert_eq!(neg_result.exponent(), 1_i64);
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
