//! Extended unsigned binary numbers with positive infinity support.
//!
//! This module provides `UXBinary`, which extends `UBinary` with positive infinity
//! for representing unbounded non-negative quantities like infinite interval widths.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Sub};

use num_traits::Zero;

use crate::ordered_pair::{AbsDistance, AddWidth, SubWidth, Unsigned};

use super::binary_impl::Binary;
use super::error::BinaryError;
use super::ubinary::UBinary;
use super::xbinary::XBinary;

/// Extended unsigned binary number: either a finite nonnegative value or +infinity.
///
/// Used for representing bounds widths which are always nonnegative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UXBinary {
    /// A finite non-negative value.
    Finite(UBinary),
    /// Positive infinity.
    PosInf,
}

impl UXBinary {
    /// Returns the zero value.
    pub fn zero() -> Self {
        Self::Finite(UBinary::zero())
    }

    /// Returns true if this value is zero.
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Finite(value) if value.mantissa().is_zero())
    }

    /// Creates a UXBinary from an XBinary, returning an error if the value is negative.
    pub fn try_from_xbinary(xbinary: &XBinary) -> Result<Self, BinaryError> {
        match xbinary {
            XBinary::NegInf => Err(BinaryError::NegativeMantissa),
            XBinary::PosInf => Ok(Self::PosInf),
            XBinary::Finite(binary) => {
                use num_traits::Signed;
                if binary.mantissa().is_negative() {
                    return Err(BinaryError::NegativeMantissa);
                }
                Ok(Self::Finite(UBinary::try_from_binary(binary)?))
            }
        }
    }

    /// Adds two extended unsigned binary numbers.
    pub fn add(&self, other: &Self) -> Self {
        use UXBinary::{Finite, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(lhs), Finite(rhs)) => Finite(lhs.add(rhs)),
        }
    }

    /// Multiplies two extended unsigned binary numbers.
    pub fn mul(&self, other: &Self) -> Self {
        use UXBinary::{Finite, PosInf};
        // 0 * anything = 0 (including 0 * infinity)
        if self.is_zero() || other.is_zero() {
            return Finite(UBinary::zero());
        }
        match (self, other) {
            (Finite(lhs), Finite(rhs)) => Finite(lhs.mul(rhs)),
            // PosInf * nonzero = PosInf
            (PosInf, _) | (_, PosInf) => PosInf,
        }
    }

    /// Subtracts another UXBinary from this one, saturating at zero.
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

impl Zero for UXBinary {
    fn zero() -> Self {
        UXBinary::zero()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl Add for UXBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        UXBinary::add(&self, &rhs)
    }
}

impl Sub for UXBinary {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        UXBinary::sub_saturating(&self, &rhs)
    }
}

impl Mul for UXBinary {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        UXBinary::mul(&self, &rhs)
    }
}

impl Unsigned for UXBinary {}

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
    ///
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
                use num_traits::Signed;
                if diff.mantissa().is_negative() {
                    // This means lower > upper, use |lower - upper| instead
                    let abs_diff = l.sub(u);
                    UXBinary::Finite(
                        UBinary::try_from_binary(&abs_diff).unwrap_or_else(|_| UBinary::zero()),
                    )
                } else {
                    UXBinary::Finite(
                        UBinary::try_from_binary(&diff).unwrap_or_else(|_| UBinary::zero()),
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
        use XBinary::{Finite, NegInf, PosInf};
        use UXBinary::{Finite as UFinite, PosInf as UPosInf};
        match (self, rhs) {
            // Subtracting finite width from finite value
            (Finite(x), UFinite(w)) => Finite(Binary::sub(&x, &w.to_binary())),
            // Subtracting finite width from infinity preserves infinity
            (PosInf, UFinite(_)) => PosInf,
            (NegInf, UFinite(_)) => NegInf,
            // Subtracting infinite width: PosInf - PosInf and Finite - PosInf are indeterminate/NegInf
            // For SubWidth semantics (shrinking interval), treat as going to negative bound
            (PosInf, UPosInf) => NegInf, // shrinking from +inf by infinite width
            (Finite(_), UPosInf) => NegInf,
            (NegInf, UPosInf) => NegInf,
        }
    }
}

impl fmt::Display for UXBinary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UXBinary::Finite(ubinary) => write!(f, "{}", ubinary),
            UXBinary::PosInf => write!(f, "+∞"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use super::super::binary_impl::Binary;
    use num_bigint::{BigInt, BigUint};

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn ubin(mantissa: u64, exponent: i64) -> UBinary {
        UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
    }

    fn xbin(mantissa: i64, exponent: i64) -> XBinary {
        XBinary::Finite(bin(mantissa, exponent))
    }

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

    #[test]
    fn abs_distance_finite_cases() {
        let one = xbin(1, 0);
        let three = xbin(3, 0);

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
        let one = xbin(1, 0);
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

    #[test]
    fn uxbinary_mul_works() {
        let two = UXBinary::Finite(ubin(1, 1));
        let three = UXBinary::Finite(ubin(3, 0));
        let product = two.clone() * three;
        assert_eq!(product, UXBinary::Finite(ubin(3, 1)));

        // Zero times anything is zero
        assert!((UXBinary::zero() * UXBinary::PosInf).is_zero());
        assert!((UXBinary::zero() * two.clone()).is_zero());

        // Infinity times nonzero is infinity
        assert_eq!(UXBinary::PosInf * two, UXBinary::PosInf);
    }
}
