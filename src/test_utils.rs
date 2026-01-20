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
        UXBinary::Inf => {
            panic!("expected finite unsigned extended binary, got PosInf")
        }
    }
}

use crate::binary::Bounds;
use crate::computable::Computable;
use num_traits::One;

/// Computes the midpoint between two finite XBinary values.
///
/// # Panics
/// Panics if either input is infinite.
pub fn midpoint_between(lower: &XBinary, upper: &XBinary) -> Binary {
    let mid_sum = unwrap_finite(lower).add(&unwrap_finite(upper));
    let exponent = mid_sum.exponent() - BigInt::one();
    Binary::new(mid_sum.mantissa().clone(), exponent)
}

/// Refines bounds by collapsing them to their midpoint.
pub fn interval_refine(state: Bounds) -> Bounds {
    let midpoint = midpoint_between(state.small(), &state.large());
    Bounds::new(
        XBinary::Finite(midpoint.clone()),
        XBinary::Finite(midpoint),
    )
}

/// Creates a Computable that represents an interval [lower, upper] and refines to its midpoint.
///
/// This is useful for testing operations on Computables with interval arithmetic,
/// where we want to verify how operations combine and propagate bounds.
///
/// # Examples
/// ```ignore
/// let interval = interval_midpoint_computable(1, 3); // [1, 3], refines to midpoint 2
/// let bounds = interval.bounds().expect("bounds should succeed");
/// ```
pub fn interval_midpoint_computable(lower: i64, upper: i64) -> Computable {
    let interval_state = Bounds::new(xbin(lower, 0), xbin(upper, 0));
    Computable::new(
        interval_state,
        |inner_state| Ok(inner_state.clone()),
        interval_refine,
    )
}
