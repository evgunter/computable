use std::time::Duration;

use computable::{Binary, Bounds, FiniteBounds, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::One;

/// Result of a float-based benchmark
#[derive(Debug)]
pub struct BenchmarkResult {
    pub duration: Duration,
    pub value: f64,
}

/// Result of a Computable-based benchmark
#[derive(Debug)]
pub struct ComputableResult {
    pub duration: Duration,
    pub midpoint: Binary,
    pub width: UXBinary,
}

/// Converts an f64 value to a Binary, panicking if the value is not finite.
pub fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value) {
        Ok(XBinary::Finite(b)) => b,
        _ => panic!("expected finite f64 value"),
    }
}

/// Extracts a finite Binary from an XBinary, panicking if infinite.
pub fn finite_binary(value: &XBinary) -> Binary {
    match value {
        XBinary::Finite(binary) => binary.clone(),
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite bounds")
        }
    }
}

/// Tries to extract finite bounds from Bounds, returning None if either endpoint is infinite.
pub fn try_finite_bounds(bounds: &Bounds) -> Option<FiniteBounds> {
    match (bounds.small(), bounds.large()) {
        (XBinary::Finite(lower), XBinary::Finite(upper)) => {
            Some(FiniteBounds::new(lower.clone(), upper))
        }
        _ => None,
    }
}

// TODO: remove all the implementations in other files of `midpoint` or `midpoint_between` etc and just use one implementation
// (in the main module not benchmarks) which does the + width/2 strategy rather than redundantly computing the upper bound by
// adding width to lower and then averaging that with lower

/// Computes the midpoint of bounds.
pub fn midpoint(bounds: &FiniteBounds) -> Binary {
    let lower = bounds.small();
    let width = bounds.width().to_binary();
    let half_width = Binary::new(width.mantissa().clone(), width.exponent() - BigInt::one());
    lower.add(&half_width)
}
