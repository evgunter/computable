//! Binary number representations for exact arithmetic.
//!
//! This module provides various binary number types for exact computation:
//!
//! - [`Binary`]: Signed exact binary number `mantissa * 2^exponent`
//! - [`XBinary`]: Extended signed binary with +/-infinity
//! - [`UBinary`]: Unsigned binary number (non-negative)
//! - [`UXBinary`]: Extended unsigned binary with +infinity
//!
//! The module also provides:
//! - [`BinaryError`] and [`XBinaryError`]: Error types for binary operations
//! - [`ReciprocalRounding`]: Rounding modes for reciprocal computation
//! - [`reciprocal_rounded_abs_extended`]: Reciprocal computation function
//!
//! # Architecture
//!
//! The types are organized as two parallel families:
//!
//! ```text
//! Signed:                      Unsigned:
//! Binary (finite)              UBinary (finite)
//!    │                            │
//!    └──► XBinary (±∞)            └──► UXBinary (+∞)
//!
//! Binary ←──────────────────────► UBinary
//!       (conversions: try_from_binary, to_binary)
//! ```
//!
//! - `XBinary` is `enum { NegInf, Finite(Binary), PosInf }`
//! - `UXBinary` is `enum { Finite(UBinary), PosInf }`
//! - `Binary` and `UBinary` are independent structs with different mantissa types
//!
//! All types maintain a canonical representation where the mantissa is odd
//! (unless the value is zero).

mod binary_impl;
mod display;
mod error;
mod reciprocal;
mod shortest;
mod shift;
mod ubinary;
mod uxbinary;
mod xbinary;

// Re-export all public types
pub use binary_impl::Binary;
pub use error::{BinaryError, XBinaryError};
pub use reciprocal::{
    reciprocal_of_biguint, reciprocal_rounded_abs_extended, ReciprocalRounding,
};
pub use shortest::{margin_from_width, shortest_xbinary_in_bounds, simplify_bounds_if_needed};
pub use ubinary::UBinary;
pub use uxbinary::UXBinary;
pub use xbinary::XBinary;

// BigInt/BigUint trait implementations for ordered_pair compatibility
use num_bigint::{BigInt, BigUint};

use crate::ordered_pair::{AbsDistance, AddWidth, Interval, Unsigned};

/// Bounds on a computable number: lower and upper bounds as XBinary values.
/// The width is stored as UXBinary to guarantee non-negativity through the type system.
///
/// This type enforces the invariant from the formalism that bounds widths are
/// always nonnegative (elements of D_inf where the value is >= 0).
pub type Bounds = Interval<XBinary, UXBinary>;

/// Finite bounds on a value: lower and upper bounds as Binary values.
///
/// Unlike [`Bounds`], this type guarantees that both bounds are finite
/// (no infinities). This is useful for algorithms like bisection that
/// require finite intervals.
pub type FiniteBounds = Interval<Binary, UBinary>;

impl Unsigned for BigUint {}

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

#[cfg(test)]
mod integration_tests {
    //! Integration tests that verify cross-module functionality.

    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::{bin, xbin};

    #[test]
    fn bounds_reject_invalid_order() {
        let lower = xbin(1, 0);
        let upper = xbin(-1, 0);
        let result = Bounds::new_checked(lower, upper);
        assert!(result.is_err());
    }

    #[test]
    fn binary_to_ubinary_to_xbinary_roundtrip() {
        let original = bin(7, 3);
        let ubinary = UBinary::try_from_binary(&original).expect("should succeed");
        let back = ubinary.to_binary();
        assert_eq!(original, back);
    }

    #[test]
    fn uxbinary_xbinary_conversion() {
        use num_bigint::BigUint;

        let ub = UBinary::new(BigUint::from(5u32), BigInt::from(2));
        let uxb = UXBinary::Finite(ub);
        let xb = XBinary::from(uxb);

        if let XBinary::Finite(binary) = xb {
            assert_eq!(binary.mantissa(), &BigInt::from(5));
            assert_eq!(binary.exponent(), &BigInt::from(2));
        } else {
            panic!("expected finite value");
        }
    }

    #[test]
    fn bounds_from_lower_and_width_constructs_correctly() {
        let lower = xbin(5, 0);
        let width = UXBinary::Finite(UBinary::new(num_bigint::BigUint::from(3u32), BigInt::from(0)));

        let bounds = Bounds::from_lower_and_width(lower.clone(), width.clone());

        assert_eq!(bounds.small(), &xbin(5, 0));
        assert_eq!(bounds.width(), &width);
        assert_eq!(bounds.large(), xbin(8, 0));
    }

    #[test]
    fn bounds_from_lower_and_width_matches_new() {
        let lower = xbin(10, 0);
        let upper = xbin(25, 0);

        let via_new = Bounds::new(lower.clone(), upper.clone());
        let via_from_lower_and_width = Bounds::from_lower_and_width(
            lower.clone(),
            via_new.width().clone(),
        );

        assert_eq!(via_new, via_from_lower_and_width);
    }

    #[test]
    fn bounds_from_lower_and_width_zero_width() {
        let lower = xbin(42, 0);
        let width = UXBinary::Finite(UBinary::new(num_bigint::BigUint::from(0u32), BigInt::from(0)));

        let bounds = Bounds::from_lower_and_width(lower.clone(), width);

        assert_eq!(bounds.small(), &xbin(42, 0));
        assert_eq!(bounds.large(), xbin(42, 0));
    }

}
