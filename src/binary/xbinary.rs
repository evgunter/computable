//! Extended signed binary numbers with infinity support.
//!
//! This module provides `XBinary`, which extends `Binary` with positive and negative
//! infinity values for representing unbounded intervals and limits.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_traits::{Float, Signed, Zero};

use super::binary_impl::Binary;
use super::error::XBinaryError;

/// Extended binary number: either a finite value or positive/negative infinity.
///
/// This type is useful for representing bounds that may be unbounded in either direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XBinary {
    /// Negative infinity.
    NegInf,
    /// A finite binary value.
    Finite(Binary),
    /// Positive infinity.
    PosInf,
}

impl XBinary {
    /// Returns the zero value.
    pub fn zero() -> Self {
        Self::Finite(Binary::zero())
    }

    /// Negates this extended binary number.
    pub fn neg(&self) -> Self {
        match self {
            Self::NegInf => Self::PosInf,
            Self::PosInf => Self::NegInf,
            Self::Finite(value) => Self::Finite(value.neg()),
        }
    }

    /// Returns true if this value is zero.
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Finite(value) if value.mantissa().is_zero())
    }

    /// Returns the magnitude (absolute value) of this extended binary number as a UXBinary.
    pub fn magnitude(&self) -> super::uxbinary::UXBinary {
        use super::uxbinary::UXBinary;
        match self {
            Self::NegInf => UXBinary::Inf,
            Self::PosInf => UXBinary::Inf,
            Self::Finite(value) => UXBinary::Finite(value.magnitude()),
        }
    }

    /// Converts an f64 to an XBinary.
    ///
    /// Returns an error if the input is NaN.
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

    /// Adds two extended binary numbers, preferring the lower bound on conflict.
    ///
    /// Used when computing lower bounds of intervals.
    pub fn add_lower(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, _) | (_, NegInf) => NegInf,
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    /// Adds two extended binary numbers, preferring the upper bound on conflict.
    ///
    /// Used when computing upper bounds of intervals.
    pub fn add_upper(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    /// Standard addition of extended binary numbers.
    pub fn add(&self, other: &Self) -> Self {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    /// Fallible subtraction of extended binary numbers.
    ///
    /// Returns an error for indeterminate forms like infinity - infinity.
    pub fn try_sub(&self, other: &Self) -> Result<Self, XBinaryError> {
        use XBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, PosInf) | (NegInf, NegInf) => Err(XBinaryError::IndeterminateForm),
            (PosInf, _) | (Finite(_), NegInf) => Ok(PosInf),
            (NegInf, _) | (Finite(_), PosInf) => Ok(NegInf),
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.sub(rhs))),
        }
    }

    /// Multiplication of extended binary numbers.
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

impl Add for XBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        XBinary::add(&self, &rhs)
    }
}

impl Sub for XBinary {
    type Output = Result<Self, XBinaryError>;

    fn sub(self, rhs: Self) -> Self::Output {
        XBinary::try_sub(&self, &rhs)
    }
}

impl Neg for XBinary {
    type Output = Self;

    fn neg(self) -> Self::Output {
        XBinary::neg(&self)
    }
}

impl Mul for XBinary {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        XBinary::mul(&self, &rhs)
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

impl fmt::Display for XBinary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XBinary::NegInf => write!(f, "-∞"),
            XBinary::Finite(binary) => write!(f, "{}", binary),
            XBinary::PosInf => write!(f, "+∞"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::xbin;

    #[test]
    fn xbinary_from_f64_handles_special_values() {
        assert!(matches!(XBinary::from_f64(f64::NAN), Err(XBinaryError::Nan)));
        assert_eq!(XBinary::from_f64(f64::INFINITY).expect("should succeed"), XBinary::PosInf);
        assert_eq!(XBinary::from_f64(f64::NEG_INFINITY).expect("should succeed"), XBinary::NegInf);
        assert_eq!(XBinary::from_f64(0.0).expect("should succeed"), XBinary::zero());
    }

    #[test]
    fn xbinary_from_f64_converts_normal_values() {
        let result = XBinary::from_f64(1.5).expect("should succeed");
        if let XBinary::Finite(value) = result {
            // 1.5 = 3 * 2^-1
            assert_eq!(value.mantissa(), &BigInt::from(3));
            assert_eq!(value.exponent(), &BigInt::from(-1));
        } else {
            panic!("expected finite value");
        }
    }

    #[test]
    fn xbinary_neg_flips_infinity() {
        assert_eq!(-XBinary::PosInf, XBinary::NegInf);
        assert_eq!(-XBinary::NegInf, XBinary::PosInf);
    }

    #[test]
    fn xbinary_is_zero_checks_finite_zero() {
        assert!(XBinary::zero().is_zero());
        assert!(!XBinary::PosInf.is_zero());
        assert!(!XBinary::NegInf.is_zero());
        assert!(!xbin(1, 0).is_zero());
    }

    #[test]
    fn xbinary_add_with_infinity() {
        let one = xbin(1, 0);
        assert_eq!(one.clone() + XBinary::PosInf, XBinary::PosInf);
        assert_eq!(XBinary::NegInf + one, XBinary::NegInf);
        assert_eq!(XBinary::PosInf + XBinary::NegInf, XBinary::PosInf);
    }

    #[test]
    fn xbinary_sub_same_infinity_is_indeterminate() {
        use crate::binary::XBinaryError;
        assert_eq!(
            XBinary::PosInf.try_sub(&XBinary::PosInf),
            Err(XBinaryError::IndeterminateForm)
        );
        assert_eq!(
            XBinary::NegInf.try_sub(&XBinary::NegInf),
            Err(XBinaryError::IndeterminateForm)
        );
    }

    #[test]
    fn xbinary_sub_finite_works() {
        let one = xbin(1, 0);
        let two = xbin(1, 1);
        assert_eq!(two - one, Ok(xbin(1, 0)));
    }

    #[test]
    fn xbinary_sub_infinity_finite_works() {
        let one = xbin(1, 0);
        assert_eq!(XBinary::PosInf - one.clone(), Ok(XBinary::PosInf));
        assert_eq!(XBinary::NegInf - one.clone(), Ok(XBinary::NegInf));
        assert_eq!(one.clone() - XBinary::PosInf, Ok(XBinary::NegInf));
        assert_eq!(one - XBinary::NegInf, Ok(XBinary::PosInf));
    }

    #[test]
    fn xbinary_mul_zero_by_anything_is_zero() {
        assert!((XBinary::zero() * XBinary::PosInf).is_zero());
        assert!((XBinary::zero() * xbin(5, 2)).is_zero());
    }

    #[test]
    fn xbinary_mul_infinity_signs() {
        assert_eq!(XBinary::PosInf * XBinary::PosInf, XBinary::PosInf);
        assert_eq!(XBinary::NegInf * XBinary::NegInf, XBinary::PosInf);
        assert_eq!(XBinary::PosInf * XBinary::NegInf, XBinary::NegInf);
    }

    #[test]
    fn xbinary_ordering() {
        assert!(XBinary::NegInf < xbin(0, 0));
        assert!(xbin(0, 0) < XBinary::PosInf);
        assert!(XBinary::NegInf < XBinary::PosInf);
        assert_eq!(XBinary::PosInf, XBinary::PosInf);
    }
}
