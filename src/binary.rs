use std::cmp::Ordering;
use std::fmt;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{Float, One, Signed, Zero};

use crate::ordered_pair::OrderedPair;

/// exponent type for `Binary`; alias to keep the representation flexible.
pub type Exponent = i64;

impl OrderedPair<Exponent> {
    pub fn delta_usize(&self) -> Option<usize> {
        self.large()
            .checked_sub(*self.small())
            .and_then(|delta| usize::try_from(delta).ok())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    ExponentOverflow,
    ShiftOverflow,
    MultiplicationOverflow,
    ReciprocalOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOverflow => write!(f, "exponent overflow during normalization"),
            Self::ShiftOverflow => write!(f, "exponent shift overflow during alignment"),
            Self::MultiplicationOverflow => write!(f, "exponent overflow during multiplication"),
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
pub enum Binary {
    PositiveBinary {
        mantissa: BigUint,
        exponent: Exponent,
    },
    ZeroBinary {
        mantissa: BigUint,
    },
    NegativeBinary {
        mantissa: BigUint,
        exponent: Exponent,
    },
}

impl Binary {
    pub fn new(mantissa: BigInt, exponent: Exponent) -> Result<Self, BinaryError> {
        Self::normalize(mantissa.sign(), mantissa.magnitude().clone(), exponent)
    }

    pub fn zero() -> Self {
        Self::ZeroBinary {
            mantissa: BigUint::zero(),
        }
    }

    pub fn mantissa(&self) -> &BigUint {
        match self {
            Self::PositiveBinary { mantissa, .. }
            | Self::ZeroBinary { mantissa }
            | Self::NegativeBinary { mantissa, .. } => mantissa,
        }
    }

    pub fn exponent(&self) -> Exponent {
        match self {
            Self::PositiveBinary { exponent, .. } | Self::NegativeBinary { exponent, .. } => {
                *exponent
            }
            Self::ZeroBinary { .. } => 0,
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, BinaryError> {
        use Binary::{NegativeBinary, PositiveBinary, ZeroBinary};

        match (self, other) {
            (ZeroBinary { .. }, _) => Ok(other.clone()),
            (_, ZeroBinary { .. }) => Ok(self.clone()),
            (
                PositiveBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                PositiveBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => {
                let (lhs, rhs, exponent) =
                    Self::align_mantissas(lhs_mantissa, *lhs_exp, rhs_mantissa, *rhs_exp)?;
                Self::normalize(Sign::Plus, lhs + rhs, exponent)
            }
            (
                NegativeBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                NegativeBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => {
                let (lhs, rhs, exponent) =
                    Self::align_mantissas(lhs_mantissa, *lhs_exp, rhs_mantissa, *rhs_exp)?;
                Self::normalize(Sign::Minus, lhs + rhs, exponent)
            }
            (
                PositiveBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                NegativeBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => Self::add_mixed(lhs_mantissa, *lhs_exp, rhs_mantissa, *rhs_exp),
            (
                NegativeBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                PositiveBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => Self::add_mixed(rhs_mantissa, *rhs_exp, lhs_mantissa, *lhs_exp),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self, BinaryError> {
        self.add(&other.neg())
    }

    pub fn neg(&self) -> Self {
        match self {
            Self::PositiveBinary { mantissa, exponent } => Self::NegativeBinary {
                mantissa: mantissa.clone(),
                exponent: *exponent,
            },
            Self::NegativeBinary { mantissa, exponent } => Self::PositiveBinary {
                mantissa: mantissa.clone(),
                exponent: *exponent,
            },
            Self::ZeroBinary { .. } => self.clone(),
        }
    }

    pub fn with_exponent(&self, exponent: Exponent) -> Result<Self, BinaryError> {
        match self {
            Self::PositiveBinary { mantissa, .. } => {
                Self::normalize(Sign::Plus, mantissa.clone(), exponent)
            }
            Self::NegativeBinary { mantissa, .. } => {
                Self::normalize(Sign::Minus, mantissa.clone(), exponent)
            }
            Self::ZeroBinary { .. } => Ok(Self::zero()),
        }
    }

    pub fn abs_mantissa(&self) -> BigUint {
        match self {
            Self::PositiveBinary { mantissa, .. }
            | Self::NegativeBinary { mantissa, .. }
            | Self::ZeroBinary { mantissa } => mantissa.clone(),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self, BinaryError> {
        use Binary::{NegativeBinary, PositiveBinary, ZeroBinary};

        let (lhs_mantissa, rhs_mantissa, exponent, sign) = match (self, other) {
            (ZeroBinary { .. }, _) | (_, ZeroBinary { .. }) => {
                return Ok(Self::zero());
            }
            (
                PositiveBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                PositiveBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => (
                lhs_mantissa,
                rhs_mantissa,
                lhs_exp
                    .checked_add(*rhs_exp)
                    .ok_or(BinaryError::MultiplicationOverflow)?,
                Sign::Plus,
            ),
            (
                NegativeBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                NegativeBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => (
                lhs_mantissa,
                rhs_mantissa,
                lhs_exp
                    .checked_add(*rhs_exp)
                    .ok_or(BinaryError::MultiplicationOverflow)?,
                Sign::Plus,
            ),
            (
                PositiveBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                NegativeBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            )
            | (
                NegativeBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                PositiveBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => (
                lhs_mantissa,
                rhs_mantissa,
                lhs_exp
                    .checked_add(*rhs_exp)
                    .ok_or(BinaryError::MultiplicationOverflow)?,
                Sign::Minus,
            ),
        };

        let mantissa = lhs_mantissa * rhs_mantissa;
        Self::normalize(sign, mantissa, exponent)
    }

    fn normalize(
        sign: Sign,
        mut mantissa: BigUint,
        mut exponent: Exponent,
    ) -> Result<Self, BinaryError> {
        if mantissa.is_zero() {
            return Ok(Self::zero());
        }

        while mantissa.is_even() {
            mantissa >>= 1;
            exponent = exponent
                .checked_add(1)
                .ok_or(BinaryError::ExponentOverflow)?;
        }

        match sign {
            Sign::Plus => Ok(Self::PositiveBinary { mantissa, exponent }),
            Sign::Minus => Ok(Self::NegativeBinary { mantissa, exponent }),
            Sign::NoSign => Ok(Self::zero()),
        }
    }

    fn align_mantissas(
        lhs_mantissa: &BigUint,
        lhs_exp: Exponent,
        rhs_mantissa: &BigUint,
        rhs_exp: Exponent,
    ) -> Result<(BigUint, BigUint, Exponent), BinaryError> {
        let exponent = lhs_exp.min(rhs_exp);
        let lhs_shift = lhs_exp
            .checked_sub(exponent)
            .ok_or(BinaryError::ShiftOverflow)?;
        let rhs_shift = rhs_exp
            .checked_sub(exponent)
            .ok_or(BinaryError::ShiftOverflow)?;
        let lhs_shifted = Self::shift_mantissa(lhs_mantissa, lhs_shift)?;
        let rhs_shifted = Self::shift_mantissa(rhs_mantissa, rhs_shift)?;
        Ok((lhs_shifted, rhs_shifted, exponent))
    }

    fn shift_mantissa(mantissa: &BigUint, shift: Exponent) -> Result<BigUint, BinaryError> {
        if shift < 0 {
            return Err(BinaryError::ShiftOverflow);
        }
        let shift_amount = usize::try_from(shift).map_err(|_| BinaryError::ShiftOverflow)?;
        Ok(mantissa << shift_amount)
    }

    fn cmp_abs(
        mantissa: &BigUint,
        exponent: Exponent,
        other: &BigUint,
        other_exp: Exponent,
    ) -> Ordering {
        fn cmp_large_exp(
            large_mantissa: &BigUint,
            small_mantissa: &BigUint,
            pair: OrderedPair<Exponent>,
        ) -> Ordering {
            let shift_amount_opt = pair.delta_usize();

            if let Some(shift_amount) = shift_amount_opt {
                let shifted = large_mantissa << shift_amount;
                shifted.cmp(small_mantissa)
            } else if large_mantissa.is_zero() {
                BigUint::zero().cmp(small_mantissa)
            } else {
                Ordering::Greater
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

    fn add_mixed(
        positive_mantissa: &BigUint,
        positive_exp: Exponent,
        negative_mantissa: &BigUint,
        negative_exp: Exponent,
    ) -> Result<Self, BinaryError> {
        let (positive, negative, exponent) = Self::align_mantissas(
            positive_mantissa,
            positive_exp,
            negative_mantissa,
            negative_exp,
        )?;
        match positive.cmp(&negative) {
            Ordering::Equal => Ok(Self::zero()),
            Ordering::Greater => Self::normalize(Sign::Plus, positive - negative, exponent),
            Ordering::Less => Self::normalize(Sign::Minus, negative - positive, exponent),
        }
    }
}

impl Ord for Binary {
    fn cmp(&self, other: &Self) -> Ordering {
        use Binary::{NegativeBinary, PositiveBinary, ZeroBinary};

        match (self, other) {
            (ZeroBinary { .. }, ZeroBinary { .. }) => Ordering::Equal,
            (ZeroBinary { .. }, PositiveBinary { .. }) => Ordering::Less,
            (ZeroBinary { .. }, NegativeBinary { .. }) => Ordering::Greater,
            (PositiveBinary { .. }, ZeroBinary { .. }) => Ordering::Greater,
            (NegativeBinary { .. }, ZeroBinary { .. }) => Ordering::Less,
            (PositiveBinary { .. }, NegativeBinary { .. }) => Ordering::Greater,
            (NegativeBinary { .. }, PositiveBinary { .. }) => Ordering::Less,
            (
                PositiveBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                PositiveBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => Self::cmp_abs(lhs_mantissa, *lhs_exp, rhs_mantissa, *rhs_exp),
            (
                NegativeBinary {
                    mantissa: lhs_mantissa,
                    exponent: lhs_exp,
                },
                NegativeBinary {
                    mantissa: rhs_mantissa,
                    exponent: rhs_exp,
                },
            ) => Self::cmp_abs(lhs_mantissa, *lhs_exp, rhs_mantissa, *rhs_exp).reverse(),
        }
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
        matches!(self, Self::Finite(Binary::ZeroBinary { .. }))
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
        let (decoded_mantissa, decoded_exponent, decoded_sign) = value.integer_decode();
        let exponent = Exponent::from(decoded_exponent);
        let mantissa = BigUint::from(decoded_mantissa);
        let sign = if decoded_sign < 0 {
            Sign::Minus
        } else {
            Sign::Plus
        };
        Ok(Self::Finite(Binary::normalize(sign, mantissa, exponent)?))
    }

    pub fn add_lower(&self, other: &Self) -> Result<Self, crate::ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, _) | (_, NegInf) => Ok(NegInf),
            (PosInf, _) | (_, PosInf) => Ok(PosInf),
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.add(rhs)?)),
        }
    }

    pub fn add_upper(&self, other: &Self) -> Result<Self, crate::ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => Ok(PosInf),
            (NegInf, _) | (_, NegInf) => Ok(NegInf),
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.add(rhs)?)),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self, crate::ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        if self.is_zero() || other.is_zero() {
            return Ok(Finite(Binary::zero()));
        }
        match (self, other) {
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.mul(rhs)?)),
            (Finite(Binary::PositiveBinary { .. }), PosInf)
            | (PosInf, Finite(Binary::PositiveBinary { .. })) => Ok(PosInf),
            (Finite(Binary::NegativeBinary { .. }), PosInf)
            | (PosInf, Finite(Binary::NegativeBinary { .. })) => Ok(NegInf),
            (Finite(Binary::PositiveBinary { .. }), NegInf)
            | (NegInf, Finite(Binary::PositiveBinary { .. })) => Ok(NegInf),
            (Finite(Binary::NegativeBinary { .. }), NegInf)
            | (NegInf, Finite(Binary::NegativeBinary { .. })) => Ok(PosInf),
            (PosInf, PosInf) | (NegInf, NegInf) => Ok(PosInf),
            (PosInf, NegInf) | (NegInf, PosInf) => Ok(NegInf),
            (Finite(Binary::ZeroBinary { .. }), PosInf)
            | (Finite(Binary::ZeroBinary { .. }), NegInf)
            | (PosInf, Finite(Binary::ZeroBinary { .. }))
            | (NegInf, Finite(Binary::ZeroBinary { .. })) => Ok(Finite(Binary::zero())),
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
            let abs_mantissa = finite_value.abs_mantissa();
            let abs_mantissa_bigint = BigInt::from_biguint(Sign::Plus, abs_mantissa);
            let shift_bits = precision_bits - BigInt::from(finite_value.exponent());
            let quotient = if shift_bits.is_negative() {
                match rounding {
                    ReciprocalRounding::Floor => BigInt::zero(),
                    ReciprocalRounding::Ceil => BigInt::one(),
                }
            } else {
                let shift = precision_bits_to_usize(&shift_bits)?;
                let numerator = BigInt::one() << shift;
                match rounding {
                    ReciprocalRounding::Floor => numerator.div_floor(&abs_mantissa_bigint),
                    ReciprocalRounding::Ceil => numerator.div_ceil(&abs_mantissa_bigint),
                }
            };
            let exponent = reciprocal_exponent(precision_bits)?;
            Ok(ExtendedBinary::Finite(Binary::new(quotient, exponent)?))
        }
        ExtendedBinary::NegInf | ExtendedBinary::PosInf => {
            Ok(ExtendedBinary::Finite(Binary::zero()))
        }
    }
}

fn reciprocal_exponent(precision_bits: &BigInt) -> Result<Exponent, BinaryError> {
    let precision = precision_bits_to_exponent(precision_bits)?;
    precision
        .checked_neg()
        .ok_or(BinaryError::ReciprocalOverflow)
}

fn precision_bits_to_usize(precision_bits: &BigInt) -> Result<usize, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    usize::try_from(precision_bits).map_err(|_| BinaryError::ReciprocalOverflow)
}

fn precision_bits_to_exponent(precision_bits: &BigInt) -> Result<Exponent, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    Exponent::try_from(precision_bits).map_err(|_| BinaryError::ReciprocalOverflow)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ordered_pair::OrderedPair;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), exponent).expect("binary should normalize")
    }

    #[test]
    fn binary_normalizes_even_mantissa() {
        let value = bin(8, 0);
        assert_eq!(value.mantissa(), &BigUint::from(1_u8));
        assert_eq!(value.exponent(), 3);
    }

    #[test]
    fn binary_zero_uses_zero_exponent() {
        let value = Binary::new(BigInt::zero(), 42).expect("binary should normalize");
        assert!(matches!(value, Binary::ZeroBinary { .. }));
        assert_eq!(value.exponent(), 0);
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
        let huge_pos =
            Binary::new(BigInt::from(1), Exponent::MAX).expect("binary should normalize");
        let tiny_pos =
            Binary::new(BigInt::from(1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_pos > tiny_pos);

        let huge_neg =
            Binary::new(BigInt::from(-1), Exponent::MAX).expect("binary should normalize");
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_ordering_overflow_path_uses_sign() {
        let huge_pos =
            Binary::new(BigInt::from(1), Exponent::MAX).expect("binary should normalize");
        let tiny_neg =
            Binary::new(BigInt::from(-1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_pos > tiny_neg);

        let huge_neg =
            Binary::new(BigInt::from(-1), Exponent::MAX).expect("binary should normalize");
        let tiny_pos =
            Binary::new(BigInt::from(1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_add_aligns_exponents() {
        let one = bin(1, 0);
        let half = bin(1, -1);
        let sum = one.add(&half).expect("binary should add");
        let expected = bin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn binary_sub_handles_negative() {
        let one = bin(1, 0);
        let two = bin(1, 1);
        let diff = one.sub(&two).expect("binary should subtract");
        let expected = bin(-1, 0);
        assert_eq!(diff, expected);
    }

    #[test]
    fn binary_mul_adds_exponents() {
        let two = bin(1, 1);
        let half = bin(1, -1);
        let product = two.mul(&half).expect("binary should multiply");
        let expected = bin(1, 0);
        assert_eq!(product, expected);
    }
}
