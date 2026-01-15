use std::cmp::Ordering;
use std::fmt;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Float, One, Signed, ToPrimitive, Zero};

use crate::ordered_pair::OrderedPair;

impl OrderedPair<BigInt> {
    pub fn delta_usize(&self) -> Option<usize> {
        let delta = self.large() - self.small();
        delta.to_usize()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    ReciprocalOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReciprocalOverflow => write!(f, "exponent overflow during reciprocal"),
        }
    }
}

impl std::error::Error for BinaryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtendedBinaryError {
    Nan,
    Binary(BinaryError),
}

impl fmt::Display for ExtendedBinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => write!(f, "cannot convert NaN to ExtendedBinary"),
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ExtendedBinaryError {}

impl From<BinaryError> for ExtendedBinaryError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

/// exact binary number represented as `mantissa * 2^exponent`.
/// `mantissa` is normalized to be odd unless the value is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binary {
    mantissa: BigInt,
    exponent: BigInt,
}

impl Binary {
    pub fn new(mantissa: BigInt, exponent: BigInt) -> Self {
        Self::normalize(mantissa, exponent)
    }

    pub fn zero() -> Self {
        Self {
            mantissa: BigInt::zero(),
            exponent: BigInt::zero(),
        }
    }

    pub fn mantissa(&self) -> &BigInt {
        &self.mantissa
    }

    pub fn exponent(&self) -> &BigInt {
        &self.exponent
    }

    pub fn add(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs - rhs, exponent)
    }

    pub fn neg(&self) -> Self {
        if self.mantissa.is_zero() {
            return self.clone();
        }
        Self {
            mantissa: -self.mantissa.clone(),
            exponent: self.exponent.clone(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let exponent = &self.exponent + &other.exponent;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::normalize(mantissa, exponent)
    }

    fn normalize(mut mantissa: BigInt, mut exponent: BigInt) -> Self {
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

    fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigInt, BigInt, BigInt) {
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

    fn shift_mantissa(mantissa: &BigInt, shift: &BigUint) -> BigInt {
        if shift.is_zero() {
            return mantissa.clone();
        }
        let mut shifted = mantissa.clone();
        let mut remaining = shift.clone();
        // BigInt only implements shifts for primitive integers, so chunk large shifts.
        let chunk_limit = BigUint::from(usize::MAX);
        while remaining > BigUint::zero() {
            let chunk_usize = if remaining > chunk_limit {
                usize::MAX
            } else {
                let digits = remaining.to_u64_digits();
                match digits.first() {
                    Some(value) => *value as usize,
                    None => 0,
                }
            };
            shifted <<= chunk_usize;
            remaining -= BigUint::from(chunk_usize);
        }
        shifted
    }

    fn cmp_shifted(
        mantissa: &BigInt,
        exponent: BigInt,
        other: &BigInt,
        other_exp: BigInt,
    ) -> Ordering {
        fn cmp_large_exp(
            large_mantissa: &BigInt,
            small_mantissa: &BigInt,
            pair: OrderedPair<BigInt>,
        ) -> Ordering {
            let shift_amount_opt = pair.delta_usize();

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
                let pair = OrderedPair::new(exponent, other_exp);
                cmp_large_exp(mantissa, other, pair)
            }
            Ordering::Less => {
                let pair = OrderedPair::new(other_exp, exponent);
                cmp_large_exp(other, mantissa, pair).reverse()
            }
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtendedBinary {
    NegInf,
    Finite(Binary),
    PosInf,
}

impl ExtendedBinary {
    pub fn zero() -> Self {
        Self::Finite(Binary::zero())
    }

    pub fn neg(&self) -> Self {
        match self {
            Self::NegInf => Self::PosInf,
            Self::PosInf => Self::NegInf,
            Self::Finite(value) => Self::Finite(value.neg()),
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Finite(value) if value.mantissa().is_zero())
    }

    pub fn from_f64(value: f64) -> Result<Self, ExtendedBinaryError> {
        if value.is_nan() {
            return Err(ExtendedBinaryError::Nan);
        }
        if value == 0.0 {
            return Ok(Self::Finite(Binary::zero()));
        }
        if value == f64::INFINITY {
            return Ok(Self::PosInf);
        }
        if value == f64::NEG_INFINITY {
            return Ok(Self::NegInf);
        }
        let (mantissa, exponent, sign) = value.integer_decode();
        let signed_mantissa = BigInt::from(sign) * BigInt::from(mantissa);
        Ok(Self::Finite(Binary::new(
            signed_mantissa,
            BigInt::from(exponent),
        )))
    }

    pub fn add_lower(&self, other: &Self) -> Self {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, _) | (_, NegInf) => NegInf,
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn add_upper(&self, other: &Self) -> Self {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        if self.is_zero() || other.is_zero() {
            return Finite(Binary::zero());
        }
        match (self, other) {
            (Finite(lhs), Finite(rhs)) => Finite(lhs.mul(rhs)),
            (Finite(lhs), PosInf) | (PosInf, Finite(lhs)) => {
                if lhs.mantissa().is_positive() {
                    PosInf
                } else {
                    NegInf
                }
            }
            (Finite(lhs), NegInf) | (NegInf, Finite(lhs)) => {
                if lhs.mantissa().is_positive() {
                    NegInf
                } else {
                    PosInf
                }
            }
            (PosInf, PosInf) | (NegInf, NegInf) => PosInf,
            (PosInf, NegInf) | (NegInf, PosInf) => NegInf,
        }
    }
}

impl Ord for ExtendedBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, NegInf) | (PosInf, PosInf) => Ordering::Equal,
            (NegInf, _) => Ordering::Less,
            (_, NegInf) => Ordering::Greater,
            (PosInf, _) => Ordering::Greater,
            (_, PosInf) => Ordering::Less,
            (Finite(lhs), Finite(rhs)) => lhs.cmp(rhs),
        }
    }
}

impl PartialOrd for ExtendedBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ReciprocalRounding {
    Floor,
    Ceil,
}

pub(crate) fn reciprocal_rounded_abs_extended(
    value: &ExtendedBinary,
    precision_bits: &BigInt,
    rounding: ReciprocalRounding,
) -> Result<ExtendedBinary, BinaryError> {
    match value {
        ExtendedBinary::Finite(finite_value) => {
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
            Ok(ExtendedBinary::Finite(Binary::new(quotient, exponent)))
        }
        ExtendedBinary::NegInf | ExtendedBinary::PosInf => {
            Ok(ExtendedBinary::Finite(Binary::zero()))
        }
    }
}

fn reciprocal_exponent(precision_bits: &BigInt) -> Result<BigInt, BinaryError> {
    let precision = precision_bits_to_exponent(precision_bits)?;
    Ok(-precision)
}

fn precision_bits_to_usize(precision_bits: &BigInt) -> Result<usize, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    precision_bits
        .to_usize()
        .ok_or(BinaryError::ReciprocalOverflow)
}

fn precision_bits_to_exponent(precision_bits: &BigInt) -> Result<BigInt, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    Ok(precision_bits.clone())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ordered_pair::OrderedPair;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

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
    fn bounds_reject_invalid_order() {
        let lower = bin(1, 0);
        let upper = bin(-1, 0);
        let result = OrderedPair::new_checked(lower, upper);
        assert!(result.is_err());
    }

    #[test]
    fn binary_ordering_handles_large_exponent_gaps() {
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
        let sum = one.add(&half);
        let expected = bin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn binary_sub_handles_negative() {
        let one = bin(1, 0);
        let two = bin(1, 1);
        let diff = one.sub(&two);
        let expected = bin(-1, 0);
        assert_eq!(diff, expected);
    }

    #[test]
    fn binary_mul_adds_exponents() {
        let two = bin(1, 1);
        let half = bin(1, -1);
        let product = two.mul(&half);
        let expected = bin(1, 0);
        assert_eq!(product, expected);
    }
}
