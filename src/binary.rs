use std::{cmp::Ordering, ops::Sub};
use std::fmt;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Float, One, Signed, ToPrimitive, Zero};

use crate::Interval;
use crate::ordered_pair::{AbsDistance, AddWidth, SubWidth};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    ReciprocalOverflow,
    NegativeMantissa,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReciprocalOverflow => write!(f, "exponent overflow during reciprocal"),
            Self::NegativeMantissa => write!(f, "cannot create unsigned binary from negative mantissa"),
        }
    }
}

impl std::error::Error for BinaryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XBinaryError {
    Nan,
    Binary(BinaryError),
}

impl fmt::Display for XBinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => write!(f, "cannot convert NaN to XBinary"),
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for XBinaryError {}

impl From<BinaryError> for XBinaryError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

impl AbsDistance<BigInt, BigUint> for BigInt {
    fn abs_distance(self, other: BigInt) -> BigUint {
        (self - other).magnitude().clone()
    }
}

impl AddWidth<BigInt, BigUint> for BigInt {
    fn add_width(self, width: BigUint) -> Self {
        self + BigInt::from(width)
    }
}

impl SubWidth<BigInt, BigUint> for BigInt {
    fn sub_width(self, width: BigUint) -> Self {
        self - BigInt::from(width)
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
        // BigInt only implements shifts for primitive integers, so chunk large shifts.
        // Note: extremely large shifts will still attempt to allocate enormous values;
        // this just keeps each individual shift within primitive bounds.
        // TODO: consider proactively erroring on huge shifts to avoid massive allocations.
        let chunk_limit = BigUint::from(usize::MAX);
        shift_mantissa_chunked::<BigInt>(mantissa, shift, &chunk_limit)
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
            pair: Interval<BigInt, BigUint>,
        ) -> Ordering {
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

impl std::ops::Add for Binary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Binary::add(&self, &rhs)
    }
}

impl std::ops::Sub for Binary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Binary::sub(&self, &rhs)
    }
}

impl std::ops::Neg for Binary {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Binary::neg(&self)
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

fn shift_mantissa_chunked<M>(mantissa: &M, shift: &BigUint, chunk_limit: &BigUint) -> M
where
    M: Clone + std::ops::ShlAssign<usize>,
{
    // chunk_limit is injected to make the chunking behavior testable with small shifts.
    let mut shifted = mantissa.clone();
    let mut remaining = shift.clone();
    let chunk_limit_usize = {
        let digits = chunk_limit.to_u64_digits();
        match digits.first() {
            Some(value) => *value as usize,
            None => 0,
        }
    };
    while remaining > BigUint::zero() {
        let chunk_usize = if &remaining > chunk_limit {
            chunk_limit_usize
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
pub enum XBinary {
    NegInf,
    Finite(Binary),
    PosInf,
}

impl XBinary {
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

    pub fn from_f64(value: f64) -> Result<Self, XBinaryError> {
        if value.is_nan() {
            return Err(XBinaryError::Nan);
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
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, _) | (_, NegInf) => NegInf,
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn add_upper(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, PosInf) | (NegInf, NegInf) => Finite(Binary::zero()),
            (PosInf, _) | (Finite(_), NegInf) => PosInf,
            (NegInf, _) | (Finite(_), PosInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.sub(rhs)),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
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

impl std::ops::Add for XBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        XBinary::add(&self, &rhs)
    }
}

impl std::ops::Sub for XBinary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        XBinary::sub(&self, &rhs)
    }
}

impl std::ops::Neg for XBinary {
    type Output = Self;

    fn neg(self) -> Self::Output {
        XBinary::neg(&self)
    }
}

impl num_traits::Zero for XBinary {
    fn zero() -> Self {
        XBinary::zero()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}


impl Ord for XBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        use XBinary::{Finite, NegInf, PosInf};
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

impl PartialOrd for XBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Unsigned binary number represented as `mantissa * 2^exponent` where mantissa >= 0.
/// `mantissa` is normalized to be odd unless the value is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UBinary {
    mantissa: BigUint,
    exponent: BigInt,
}

impl UBinary {
    pub fn new(mantissa: BigUint, exponent: BigInt) -> Self {
        Self::normalize(mantissa, exponent)
    }

    pub fn zero() -> Self {
        Self {
            mantissa: BigUint::zero(),
            exponent: BigInt::zero(),
        }
    }

    pub fn mantissa(&self) -> &BigUint {
        &self.mantissa
    }

    pub fn exponent(&self) -> &BigInt {
        &self.exponent
    }

    /// Creates a UBinary from a Binary, returning an error if the mantissa is negative.
    pub fn try_from_binary(binary: &Binary) -> Result<Self, BinaryError> {
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

    pub fn add(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    pub fn sub_saturating(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        if lhs >= rhs {
            Self::normalize(lhs - rhs, exponent)
        } else {
            Self::zero()
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let exponent = &self.exponent + &other.exponent;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::normalize(mantissa, exponent)
    }

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

impl std::ops::Add for UBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        UBinary::add(&self, &rhs)
    }
}

impl std::ops::Sub for UBinary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        UBinary::sub_saturating(&self, &rhs)
    }
}

/// Extended unsigned binary number: either a finite nonnegative value or +infinity.
/// Used for representing bounds widths which are always nonnegative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UXBinary {
    Finite(UBinary),
    PosInf,
}

impl UXBinary {
    pub fn zero() -> Self {
        Self::Finite(UBinary::zero())
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Finite(value) if value.mantissa().is_zero())
    }

    /// Creates a UXBinary from an XBinary, returning an error if the value is negative.
    pub fn try_from_xbinary(xbinary: &XBinary) -> Result<Self, BinaryError> {
        match xbinary {
            XBinary::NegInf => Err(BinaryError::NegativeMantissa),
            XBinary::PosInf => Ok(Self::PosInf),
            XBinary::Finite(binary) => {
                if binary.mantissa().is_negative() {
                    return Err(BinaryError::NegativeMantissa);
                }
                Ok(Self::Finite(UBinary::try_from_binary(binary)?))
            }
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        use UXBinary::{Finite, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    pub fn sub_saturating(&self, other: &Self) -> Self {
        use UXBinary::{Finite, PosInf};
        match (self, other) {
            (PosInf, Finite(_)) => PosInf,
            (PosInf, PosInf) => Finite(UBinary::zero()),
            (Finite(_), PosInf) => Finite(UBinary::zero()),
            (Finite(lhs), Finite(rhs)) => Finite(lhs.sub_saturating(rhs)),
        }
    }
}

impl Ord for UXBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        use UXBinary::{Finite, PosInf};
        match (self, other) {
            (PosInf, PosInf) => Ordering::Equal,
            (PosInf, _) => Ordering::Greater,
            (_, PosInf) => Ordering::Less,
            (Finite(lhs), Finite(rhs)) => lhs.cmp(rhs),
        }
    }
}

impl PartialOrd for UXBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl num_traits::Zero for UXBinary {
    fn zero() -> Self {
        UXBinary::zero()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl std::ops::Add for UXBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        UXBinary::add(&self, &rhs)
    }
}

impl std::ops::Sub for UXBinary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        UXBinary::sub_saturating(&self, &rhs)
    }
}

impl From<UXBinary> for XBinary {
    fn from(uxbinary: UXBinary) -> Self {
        match uxbinary {
            UXBinary::PosInf => XBinary::PosInf,
            UXBinary::Finite(ubinary) => XBinary::Finite(ubinary.to_binary()),
        }
    }
}

impl AbsDistance<XBinary, UXBinary> for XBinary {
    /// Computes the width between two XBinary values, returning a UXBinary.
    /// Width is always nonnegative: |other - self|.
    fn abs_distance(self, other: Self) -> UXBinary {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            // If either bound is infinite and they're different, width is infinite
            (NegInf, PosInf) | (PosInf, NegInf) => UXBinary::PosInf,
            (NegInf, NegInf) | (PosInf, PosInf) => UXBinary::zero(),
            (NegInf, Finite(_)) | (Finite(_), PosInf) => UXBinary::PosInf,
            (PosInf, Finite(_)) | (Finite(_), NegInf) => UXBinary::PosInf,
            (Finite(l), Finite(u)) => {
                // Compute |u - l|
                let diff = u.clone().sub(l.clone());
                if diff.mantissa().is_negative() {
                    // This means lower > upper, use |lower - upper| instead
                    let abs_diff = l.sub(u);
                    UXBinary::Finite(
                        UBinary::try_from_binary(&abs_diff)
                            .unwrap_or_else(|_| UBinary::zero())
                    )
                } else {
                    UXBinary::Finite(
                        UBinary::try_from_binary(&diff)
                            .unwrap_or_else(|_| UBinary::zero())
                    )
                }
            }
        }
    }
}

impl AddWidth<XBinary, UXBinary> for XBinary {
    fn add_width(self, rhs: UXBinary) -> Self {
        self + XBinary::from(rhs)
    }
}

impl SubWidth<XBinary, UXBinary> for XBinary {
    fn sub_width(self, rhs: UXBinary) -> Self {
        self - XBinary::from(rhs)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ReciprocalRounding {
    Floor,
    Ceil,
}

pub(crate) fn reciprocal_rounded_abs_extended(
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
        XBinary::NegInf | XBinary::PosInf => {
            Ok(XBinary::Finite(Binary::zero()))
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
    use crate::ordered_pair::Bounds;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn xbin(mantissa: i64, exponent: i64) -> XBinary {
        XBinary::Finite(bin(mantissa, exponent))
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
        let lower = xbin(1, 0);
        let upper = xbin(-1, 0);
        let result = Bounds::new_checked(lower, upper);
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
        let product = two.mul(&half);
        let expected = bin(1, 0);
        assert_eq!(product, expected);
    }

    #[test]
    fn shift_mantissa_chunks_large_shift() {
        let mantissa = BigInt::from(1);
        let shift = BigUint::from(128u32);
        let chunk_limit = BigUint::from(64u32);
        let chunked = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, &chunk_limit);
        let expected = &mantissa << 128usize;
        assert_eq!(chunked, expected);
    }

    // --- UBinary tests ---

    fn ubin(mantissa: u64, exponent: i64) -> UBinary {
        UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
    }

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

    // --- UXBinary tests ---

    #[test]
    fn uxbinary_zero_is_zero() {
        assert!(UXBinary::zero().is_zero());
        assert!(!UXBinary::PosInf.is_zero());
    }

    #[test]
    fn uxbinary_ordering_works() {
        let zero = UXBinary::zero();
        let one = UXBinary::Finite(ubin(1, 0));
        let inf = UXBinary::PosInf;

        assert!(zero < one);
        assert!(one < inf);
        assert!(zero < inf);
        assert_eq!(inf, UXBinary::PosInf);
    }

    #[test]
    fn uxbinary_add_works() {
        let one = UXBinary::Finite(ubin(1, 0));
        let two = UXBinary::Finite(ubin(1, 1));
        let sum = one.clone() + two.clone();
        assert_eq!(sum, UXBinary::Finite(ubin(3, 0)));

        // Adding infinity
        let inf = UXBinary::PosInf;
        assert_eq!(one.clone() + inf.clone(), UXBinary::PosInf);
        assert_eq!(inf + one, UXBinary::PosInf);
    }

    #[test]
    fn uxbinary_sub_saturating_works() {
        let two = UXBinary::Finite(ubin(1, 1));
        let one = UXBinary::Finite(ubin(1, 0));

        let diff = two.sub_saturating(&one);
        assert_eq!(diff, UXBinary::Finite(ubin(1, 0)));

        // Saturation cases
        let saturated = one.sub_saturating(&two);
        assert_eq!(saturated, UXBinary::zero());

        let inf = UXBinary::PosInf;
        assert_eq!(inf.sub_saturating(&one), UXBinary::PosInf);
        assert_eq!(inf.sub_saturating(&inf), UXBinary::zero());
        assert_eq!(one.sub_saturating(&inf), UXBinary::zero());
    }

    #[test]
    fn uxbinary_try_from_xbinary_works() {
        // Positive finite
        let pos_finite = XBinary::Finite(bin(5, 2));
        let result = UXBinary::try_from_xbinary(&pos_finite);
        assert!(result.is_ok());

        // Negative finite
        let neg_finite = XBinary::Finite(bin(-5, 2));
        let result = UXBinary::try_from_xbinary(&neg_finite);
        assert!(result.is_err());

        // Positive infinity
        let pos_inf = XBinary::PosInf;
        let result = UXBinary::try_from_xbinary(&pos_inf);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), UXBinary::PosInf);

        // Negative infinity
        let neg_inf = XBinary::NegInf;
        let result = UXBinary::try_from_xbinary(&neg_inf);
        assert!(result.is_err());
    }

    #[test]
    fn xbinary_from_uxbinary_works() {
        let ubx = UXBinary::Finite(ubin(7, 3));
        let xb = XBinary::from(ubx);
        assert_eq!(xb, XBinary::Finite(bin(7, 3)));

        assert_eq!(XBinary::from(UXBinary::PosInf), XBinary::PosInf);
    }

    // --- abs_distance tests ---

    #[test]
    fn abs_distance_finite_cases() {
        let one = XBinary::Finite(bin(1, 0));
        let three = XBinary::Finite(bin(3, 0));

        // Normal case: upper > lower
        let width = one.clone().abs_distance(three.clone());
        assert_eq!(width, UXBinary::Finite(ubin(1, 1)));

        // Equal bounds
        let width = one.clone().abs_distance(one.clone());
        assert_eq!(width, UXBinary::zero());

        // Swapped (lower > upper) - still returns absolute value
        let width = three.abs_distance(one);
        assert_eq!(width, UXBinary::Finite(ubin(1, 1)));
    }

    #[test]
    fn abs_distance_infinite_cases() {
        let one = XBinary::Finite(bin(1, 0));
        let neg_inf = XBinary::NegInf;
        let pos_inf = XBinary::PosInf;

        // One infinite bound
        assert_eq!(neg_inf.clone().abs_distance(one.clone()), UXBinary::PosInf);
        assert_eq!(one.clone().abs_distance(pos_inf.clone()), UXBinary::PosInf);

        // Both infinite (different)
        assert_eq!(neg_inf.clone().abs_distance(pos_inf.clone()), UXBinary::PosInf);

        // Both infinite (same)
        assert_eq!(neg_inf.clone().abs_distance(neg_inf), UXBinary::zero());
        assert_eq!(pos_inf.clone().abs_distance(pos_inf), UXBinary::zero());
    }
}
