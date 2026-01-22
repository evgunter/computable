use std::time::Duration;

use computable::{Binary, Bounds, FiniteBounds, UXBinary, XBinary};

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

/// Computes the midpoint of bounds.
pub fn midpoint(bounds: &FiniteBounds) -> Binary {
    bounds.midpoint()
}
