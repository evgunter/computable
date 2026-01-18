//! Shared test utilities for computable operations.
//!
//! This module provides common helper functions used across test modules to reduce
//! code duplication and provide a consistent testing interface.

use num_bigint::{BigInt, BigUint};

use crate::binary::{Binary, UBinary, UXBinary, XBinary};

/// Creates a Binary from mantissa and exponent as i64 values.
///
/// # Examples
/// ```ignore
/// let two = bin(2, 0);      // 2 * 2^0 = 2
/// let half = bin(1, -1);    // 1 * 2^(-1) = 0.5
/// let eight = bin(1, 3);    // 1 * 2^3 = 8
/// ```
pub fn bin(mantissa: i64, exponent: i64) -> Binary {
    Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
}

/// Creates a UBinary (unsigned) from mantissa and exponent.
///
/// # Examples
/// ```ignore
/// let two = ubin(2, 0);     // 2 * 2^0 = 2
/// let epsilon = ubin(1, -8); // 1 * 2^(-8) ≈ 0.004
/// ```
pub fn ubin(mantissa: u64, exponent: i64) -> UBinary {
    UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
}

/// Creates an XBinary (extended binary, allowing infinity) from mantissa and exponent.
///
/// This wraps `bin` in `XBinary::Finite`.
///
/// # Examples
/// ```ignore
/// let two = xbin(2, 0);     // Finite(2 * 2^0) = 2
/// ```
pub fn xbin(mantissa: i64, exponent: i64) -> XBinary {
    XBinary::Finite(bin(mantissa, exponent))
}

/// Unwraps a finite XBinary, panicking if it's infinite.
///
/// # Panics
/// Panics with a descriptive message if the input is NegInf or PosInf.
pub fn unwrap_finite(input: &XBinary) -> Binary {
    match input {
        XBinary::Finite(value) => value.clone(),
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite extended binary, got {:?}", input)
        }
    }
}

/// Unwraps a finite UXBinary, panicking if it's infinite.
///
/// # Panics
/// Panics with a descriptive message if the input is PosInf.
pub fn unwrap_finite_uxbinary(input: &UXBinary) -> UBinary {
    match input {
        UXBinary::Finite(value) => value.clone(),
        UXBinary::PosInf => {
            panic!("expected finite unsigned extended binary, got PosInf")
        }
    }
}
