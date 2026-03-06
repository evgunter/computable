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

use crate::computable::Computable;
use crate::finite_interval::FiniteInterval;
use crate::prefix::Prefix;
use crate::refinement::{XUsize, prefix_width_leq};

/// Creates a Binary representing 2^(-n), for test assertions that need
/// epsilon as a Binary value for arithmetic.
pub fn epsilon_as_binary(n: usize) -> Binary {
    let n_i64 = i64::try_from(n).expect("precision fits in i64");
    Binary::new(
        BigInt::from(1_i32),
        BigInt::from(n_i64.checked_neg().expect("negation does not overflow")),
    )
}

/// Computes the midpoint between two finite XBinary values.
///
/// # Panics
/// Panics if either input is infinite.
pub fn midpoint_between(lower: &XBinary, upper: &XBinary) -> Binary {
    let lower_finite = unwrap_finite(lower);
    let upper_finite = unwrap_finite(upper);
    FiniteInterval::new(lower_finite, upper_finite).midpoint()
}

/// Refines a prefix by collapsing it to its midpoint.
pub fn interval_refine(state: Prefix) -> Result<Prefix, crate::error::ComputableError> {
    let midpoint = midpoint_between(&state.lower(), &state.upper());
    Ok(Prefix::exact(midpoint))
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
    let initial_prefix = Prefix::from_lower_upper(xbin(lower, 0), xbin(upper, 0));
    Computable::new(
        initial_prefix,
        |state| Ok(state.clone()),
        interval_refine,
    )
}

/// Creates a Computable that represents an interval [lower, upper] without refinement.
///
/// Unlike `interval_midpoint_computable`, this helper does not refine the bounds at all.
/// It simply returns the interval unchanged. Therefore, it's actually an invalid computable number;
/// it will never converge, and should get caught by refine_to's check that the state changes.
/// Nevertheless, it is useful for testing interval arithmetic
/// operations where you want to observe how bounds propagate without any refinement.
///
/// # Examples
/// ```ignore
/// let interval = interval_noop_computable(1, 3); // [1, 3], never refines
/// let bounds = interval.bounds().expect("bounds should succeed");
/// ```
pub fn interval_noop_computable(lower: i64, upper: i64) -> Computable {
    let initial_prefix = Prefix::from_lower_upper(xbin(lower, 0), xbin(upper, 0));
    Computable::new(initial_prefix, |state| Ok(state.clone()), Ok)
}

/// Lightweight view of a Prefix's lower/upper bounds for test assertions.
///
/// Provides `.small()` and `.large()` accessors that mirror the old `Bounds` API.
/// This avoids having to update all test callsites at once.
pub struct PrefixView {
    lower: XBinary,
    upper: XBinary,
}

impl PrefixView {
    pub fn small(&self) -> &XBinary {
        &self.lower
    }

    pub fn large(&self) -> XBinary {
        self.upper.clone()
    }

    pub fn width(&self) -> UXBinary {
        match (&self.lower, &self.upper) {
            (XBinary::NegInf, _) | (_, XBinary::PosInf) => UXBinary::Inf,
            (XBinary::PosInf, _) | (_, XBinary::NegInf) => UXBinary::Inf,
            (XBinary::Finite(lo), XBinary::Finite(hi)) => {
                let diff = hi.sub(lo);
                UXBinary::Finite(
                    UBinary::try_from_binary(&diff).unwrap_or_else(|_| UBinary::zero()),
                )
            }
        }
    }
}

/// Convert a Prefix to a PrefixView for test assertions.
pub fn to_bounds(prefix: &Prefix) -> PrefixView {
    PrefixView {
        lower: prefix.lower(),
        upper: prefix.upper(),
    }
}

/// Assert that the Prefix-derived bounds contain the expected interval.
/// Due to Prefix normalization (width rounded up to power of 2), the actual
/// bounds may be wider than the tight mathematical result.
pub fn assert_bounds_contain(
    prefix: &Prefix,
    expected_lower: &XBinary,
    expected_upper: &XBinary,
) {
    let lower = prefix.lower();
    let upper = prefix.upper();
    assert!(
        lower <= *expected_lower,
        "lower bound {:?} should be <= expected {:?}",
        lower,
        expected_lower
    );
    assert!(
        upper >= *expected_upper,
        "upper bound {:?} should be >= expected {:?}",
        upper,
        expected_upper
    );
}

/// Assert that a Prefix contains a specific expected value and has width within tolerance.
pub fn assert_bounds_compatible_with_expected(
    prefix: &Prefix,
    expected: &Binary,
    tolerance_exp: &XUsize,
) {
    let lower = unwrap_finite(&prefix.lower());
    let upper = unwrap_finite(&prefix.upper());

    assert!(
        lower <= *expected && *expected <= upper,
        "Expected {} to be in bounds [{}, {}]",
        expected,
        lower,
        upper
    );
    assert!(
        prefix_width_leq(prefix, tolerance_exp),
        "Bounds width should be <= tolerance",
    );
}

/// Assert that a Prefix represents an exact value (lower == upper == expected).
pub fn assert_exact(prefix: &Prefix, expected: &Binary) {
    assert_eq!(unwrap_finite(&prefix.lower()), *expected);
    assert_eq!(unwrap_finite(&prefix.upper()), *expected);
}
