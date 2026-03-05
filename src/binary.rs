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
mod shift;
mod ubinary;
mod uxbinary;
mod xbinary;

// Re-export all public types
pub use binary_impl::Binary;
pub use error::{BinaryError, XBinaryError};
pub use reciprocal::{ReciprocalRounding, reciprocal_of_biguint, reciprocal_rounded_abs_extended};
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


impl Bounds {
    /// Returns the absolute value (magnitude) of each bound as a pair.
    ///
    /// For bounds [lower, upper], returns (|lower|, |upper|).
    pub fn abs(&self) -> (UXBinary, UXBinary) {
        (self.small().magnitude(), self.large().magnitude())
    }
}

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

        let ub = UBinary::new(BigUint::from(5u32), BigInt::from(2_i32));
        let uxb = UXBinary::Finite(ub);
        let xb = XBinary::from(uxb);

        if let XBinary::Finite(binary) = xb {
            assert_eq!(binary.mantissa(), &BigInt::from(5_i32));
            assert_eq!(binary.exponent(), &BigInt::from(2_i32));
        } else {
            panic!("expected finite value");
        }
    }

    #[test]
    fn bounds_from_lower_and_width_constructs_correctly() {
        let lower = xbin(5, 0);
        let width = UXBinary::Finite(UBinary::new(
            num_bigint::BigUint::from(3u32),
            BigInt::from(0_i32),
        ));

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
        let via_from_lower_and_width =
            Bounds::from_lower_and_width(lower.clone(), via_new.width().clone());

        assert_eq!(via_new, via_from_lower_and_width);
    }

    #[test]
    fn bounds_from_lower_and_width_zero_width() {
        let lower = xbin(42, 0);
        let width = UXBinary::Finite(UBinary::new(
            num_bigint::BigUint::from(0u32),
            BigInt::from(0_i32),
        ));

        let bounds = Bounds::from_lower_and_width(lower.clone(), width);

        assert_eq!(bounds.small(), &xbin(42, 0));
        assert_eq!(bounds.large(), xbin(42, 0));
    }

    #[test]
    fn finite_interval_subtraction() {
        use crate::finite_interval::FiniteInterval;
        // Test that [a,b] - [c,d] = [a-d, b-c]
        let a = FiniteInterval::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteInterval::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = a.interval_sub(&b);
        // [1, 2] - [3, 5] = [1-5, 2-3] = [-4, -1]
        assert_eq!(result.lo(), &bin(-4, 0));
        assert_eq!(result.hi(), bin(-1, 0));
    }

    #[test]
    fn finite_interval_negation() {
        use crate::finite_interval::FiniteInterval;
        let a = FiniteInterval::new(bin(1, 0), bin(3, 0)); // [1, 3]
        let neg_a = a.interval_neg(); // [-3, -1]
        assert_eq!(neg_a.lo(), &bin(-3, 0));
        assert_eq!(neg_a.hi(), bin(-1, 0));
    }

    #[test]
    fn finite_interval_join_overlapping() {
        use crate::finite_interval::FiniteInterval;
        let a = FiniteInterval::new(bin(1, 0), bin(4, 0)); // [1, 4]
        let b = FiniteInterval::new(bin(3, 0), bin(6, 0)); // [3, 6]

        let result = a.join(&b);
        assert_eq!(result.lo(), &bin(1, 0));
        assert_eq!(result.hi(), bin(6, 0));
    }

    #[test]
    fn finite_interval_join_disjoint() {
        use crate::finite_interval::FiniteInterval;
        let a = FiniteInterval::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteInterval::new(bin(5, 0), bin(7, 0)); // [5, 7]

        let result = a.join(&b);
        assert_eq!(result.lo(), &bin(1, 0));
        assert_eq!(result.hi(), bin(7, 0));
    }
}
