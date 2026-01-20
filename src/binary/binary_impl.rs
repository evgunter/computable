//! Core signed binary number implementation.
//!
//! This module provides `Binary`, an exact binary representation of rational numbers
//! as `mantissa * 2^exponent` where the mantissa is normalized to be odd (unless zero).

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};

use crate::ordered_pair::Interval;

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

    /// Adds two Binary numbers.
    pub fn add(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    /// Subtracts another Binary number from this one.
    pub fn sub(&self, other: &Self) -> Self {
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
    pub fn mul(&self, other: &Self) -> Self {
        let exponent = &self.exponent + &other.exponent;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::normalize(mantissa, exponent)
    }

    /// Returns the magnitude (absolute value) of this Binary number as a UBinary.
    pub fn magnitude(&self) -> super::ubinary::UBinary {
        use super::ubinary::UBinary;
        UBinary::new(self.mantissa.magnitude().clone(), self.exponent.clone())
    }

    /// Normalizes the representation by factoring out powers of 2 from the mantissa.
    fn normalize(mut mantissa: BigInt, mut exponent: BigInt) -> Self {
        use num_integer::Integer;

        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: BigInt::zero(),
            };
        }

        while mantissa.is_even() {
            mantissa /= 2;
            exponent += 1;
        }

        Self { mantissa, exponent }
    }

    /// Aligns the mantissas of two Binary numbers to a common exponent.
    /// Returns (lhs_mantissa, rhs_mantissa, common_exponent) where both mantissas
    /// are shifted to the minimum exponent of the two inputs.
    pub fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigInt, BigInt, BigInt) {
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
    fn shift_mantissa(mantissa: &BigInt, shift: &BigUint) -> BigInt {
        if shift.is_zero() {
            return mantissa.clone();
        }
        let chunk_limit = BigUint::from(usize::MAX);
        shift_mantissa_chunked::<BigInt>(mantissa, shift, &chunk_limit)
    }

    /// Compares two binary values with potentially different exponents.
    pub(crate) fn cmp_shifted(
        mantissa: &BigInt,
        exponent: BigInt,
        other: &BigInt,
        other_exp: BigInt,
    ) -> Ordering {
        fn cmp_large_exp(
            large_mantissa: &BigInt,
            small_mantissa: &BigInt,
            pair: Interval<BigInt, BigUint>,
        ) -> Ordering {
            use num_traits::ToPrimitive;

            let shift_amount_opt = pair.width().to_usize();

            if let Some(shift_amount) = shift_amount_opt {
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
                let pair = Interval::new(exponent, other_exp);
                cmp_large_exp(mantissa, other, pair)
            }
            Ordering::Less => {
                let pair = Interval::new(other_exp, exponent);
                cmp_large_exp(other, mantissa, pair).reverse()
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
    // TODO: Investigate if the type system can prevent needing the unreachable! check below.
    // After negation (if needed), the value is guaranteed non-negative, so try_from_binary
    // should never fail. Ideally we'd have a type-level proof of this.
    #[allow(clippy::unreachable)]
    fn abs_distance(self, other: Self) -> super::UBinary {
        use num_traits::Signed;
        let diff = Binary::sub(&self, &other);
        let non_negative = if diff.mantissa().is_negative() {
            diff.neg()
        } else {
            diff
        };
        super::UBinary::try_from_binary(&non_negative).unwrap_or_else(|_| {
            unreachable!("absolute value of difference should always be non-negative")
        })
    }
}

impl crate::ordered_pair::AddWidth<Binary, super::UBinary> for Binary {
    fn add_width(self, width: super::UBinary) -> Self {
        Binary::add(&self, &width.to_binary())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::bin;

    #[test]
    fn binary_normalizes_even_mantissa() {
        let value = bin(8, 0);
        assert_eq!(value.mantissa(), &BigInt::from(1));
        assert_eq!(value.exponent(), &BigInt::from(3));
    }

    #[test]
    fn binary_zero_uses_zero_exponent() {
        let value = Binary::new(BigInt::zero(), BigInt::from(42));
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
        let huge_pos = Binary::new(BigInt::from(1), huge_exp.clone());
        let tiny_pos = Binary::new(BigInt::from(1), tiny_exp.clone());
        assert!(huge_pos > tiny_pos);

        let huge_neg = Binary::new(BigInt::from(-1), huge_exp);
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_ordering_overflow_path_uses_sign() {
        use num_traits::One;

        let huge_exp = BigInt::from(usize::MAX) + BigInt::one();
        let tiny_exp = -huge_exp.clone();
        let huge_pos = Binary::new(BigInt::from(1), huge_exp.clone());
        let tiny_neg = Binary::new(BigInt::from(-1), tiny_exp.clone());
        assert!(huge_pos > tiny_neg);

        let huge_neg = Binary::new(BigInt::from(-1), huge_exp);
        let tiny_pos = Binary::new(BigInt::from(1), tiny_exp);
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
        assert_eq!(neg.mantissa(), &BigInt::from(-3));
        assert_eq!(neg.exponent(), &BigInt::from(2));
    }

    #[test]
    fn binary_neg_zero_is_zero() {
        let zero = Binary::zero();
        let neg_zero = -zero.clone();
        assert_eq!(neg_zero, zero);
    }
}
