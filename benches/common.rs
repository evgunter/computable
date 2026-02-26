#![allow(dead_code)]

use computable::{Binary, Computable, XUsize};
use criterion::BenchmarkId;

/// Standard precision sweep: epsilon = 2^(-bits) for each value.
const STANDARD_BITS: &[usize] = &[1, 4, 16, 64, 256];

/// Extended sweep, enabled by `BENCH_HIGH_PRECISION=1`.
const EXTENDED_BITS: &[usize] = &[1, 4, 16, 64, 256, 1024, 2048, 4096, 8192];

/// Returns the precision sweep to use. Set `BENCH_HIGH_PRECISION=1` to include
/// higher precisions (1024+).
pub fn precision_bits() -> &'static [usize] {
    if high_precision() {
        EXTENDED_BITS
    } else {
        STANDARD_BITS
    }
}

/// Whether high-precision benchmarks are enabled (via `BENCH_HIGH_PRECISION=1`).
pub fn high_precision() -> bool {
    std::env::var("BENCH_HIGH_PRECISION").is_ok()
}

/// Create a tolerance exponent for 2^(-bits) precision.
pub fn epsilon(bits: usize) -> XUsize {
    XUsize::Finite(bits)
}

/// Create a `BenchmarkId` with a `precision-{bits}` parameter suffix.
pub fn bench_id(bits: impl std::fmt::Display) -> BenchmarkId {
    BenchmarkId::from_parameter(format!("precision-{bits}"))
}

/// Create a `BenchmarkId` with function name and `precision-{bits}` parameter.
pub fn bench_id_named(name: impl Into<String>, bits: impl std::fmt::Display) -> BenchmarkId {
    BenchmarkId::new(name, format!("precision-{bits}"))
}

/// Whether to print diagnostic info (enabled by `BENCH_VERBOSE=1`).
pub fn verbose() -> bool {
    std::env::var("BENCH_VERBOSE").is_ok()
}

/// Sums terms using a balanced reduction instead of left-associative chaining.
///
/// This keeps the computation graph shallow (O(log n) depth), avoiding deep nesting
/// that can overflow the stack or distort timing by spending most of the runtime
/// walking long expression chains.
pub fn balanced_sum(mut values: Vec<Computable>) -> Computable {
    if values.is_empty() {
        return Computable::constant(Binary::zero());
    }

    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(left + right);
            } else {
                next.push(left);
            }
        }
        values = next;
    }

    values
        .pop()
        .expect("values should contain at least one element")
}
