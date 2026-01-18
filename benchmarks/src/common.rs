use std::time::Duration;

use computable::{Binary, Bounds, UXBinary, XBinary};
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
    match XBinary::from_f64(value).expect("expected finite f64") {
        XBinary::Finite(binary) => binary,
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite f64 input")
        }
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

/// Computes the midpoint of bounds.
pub fn midpoint(bounds: &Bounds) -> Binary {
    let lower = finite_binary(bounds.small());
    let upper = finite_binary(&bounds.large());
    let sum = lower.add(&upper);
    let half = Binary::new(BigInt::one(), BigInt::from(-1));
    sum.mul(&half)
}
