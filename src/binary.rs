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

use crate::ordered_pair::{AbsDistance, AddWidth, Unsigned};


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
    use crate::test_utils::bin;

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
