use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use computable::{Binary, Computable, UBinary};
use num_bigint::BigInt;
use rand::Rng;
use rand::rngs::StdRng;

use crate::UXBinary;
use crate::balanced_sum::balanced_sum;
use crate::common::{binary_from_f64, midpoint, try_finite_bounds};

pub const INTEGER_ROOTS_SAMPLE_COUNT: usize = 1_000;

#[derive(Debug)]
struct IntegerRootsResult {
    duration: Duration,
    value: f64,
}

#[derive(Debug)]
struct IntegerRootsComputableResult {
    duration: Duration,
    midpoint: Binary,
    width: UXBinary,
}

/// Computes the n-th root of a value using binary search.
/// Returns a Computable that refines by bisection.
fn nth_root_computable(value: u64, n: u32) -> Computable {
    let value_binary = Binary::new(BigInt::from(value), BigInt::from(0));
    // TODO: see if we can take the input as NonZeroU32 directly so we don't need to unwrap
    Computable::constant(value_binary).nth_root(NonZeroU32::new(n).unwrap())
}

/// Computes integer n-th root using f64 (for comparison).
fn nth_root_float(value: f64, n: u32) -> f64 {
    value.powf(1.0 / n as f64)
}

/// Benchmark: sum of integer roots using f64
fn integer_roots_float(inputs: &[(u64, u32)]) -> IntegerRootsResult {
    let start = Instant::now();
    let mut total = 0.0;
    for &(value, n) in inputs {
        total += nth_root_float(value as f64, n);
    }
    IntegerRootsResult {
        duration: start.elapsed(),
        value: total,
    }
}

/// Benchmark: sum of integer roots using Computable (binary search)
fn integer_roots_computable(inputs: &[(u64, u32)]) -> IntegerRootsComputableResult {
    let start = Instant::now();

    let terms: Vec<Computable> = inputs
        .iter()
        .map(|&(value, n)| nth_root_computable(value, n))
        .collect();

    let total = balanced_sum(terms);

    // Refine to epsilon = 1
    let epsilon = UBinary::new(num_bigint::BigUint::from(1u32), BigInt::from(0));
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");

    let finite =
        try_finite_bounds(&bounds).expect("bounds should be finite for nth_root operations");

    IntegerRootsComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&finite),
        width: bounds.width().clone(),
    }
}

pub fn run_integer_roots_benchmark(rng: &mut StdRng) {
    // Generate inputs: (value, root_degree) pairs
    // Use various values and root degrees (2=sqrt, 3=cbrt, 4=4th root, etc.)
    let integer_roots_inputs: Vec<(u64, u32)> = (0..INTEGER_ROOTS_SAMPLE_COUNT)
        .map(|i| {
            let value = rng.gen_range(2..1000) as u64;
            let n = (i % 5) as u32 + 2; // roots from 2 to 6
            (value, n)
        })
        .collect();

    let integer_roots_float_result = integer_roots_float(&integer_roots_inputs);
    let integer_roots_computable_result = integer_roots_computable(&integer_roots_inputs);

    let integer_roots_error = {
        let float_as_binary = binary_from_f64(integer_roots_float_result.value);
        let diff = float_as_binary.sub(&integer_roots_computable_result.midpoint);
        diff.magnitude()
    };

    let integer_roots_slowdown = integer_roots_computable_result.duration.as_secs_f64()
        / integer_roots_float_result.duration.as_secs_f64();

    println!("== Integer roots (binary search) benchmark ==");
    println!("samples: {INTEGER_ROOTS_SAMPLE_COUNT}");
    println!("epsilon: 1");
    println!("root degrees: 2 (sqrt), 3 (cbrt), 4, 5, 6");
    println!("float time:      {:?}", integer_roots_float_result.duration);
    println!(
        "computable time: {:?}",
        integer_roots_computable_result.duration
    );
    println!("slowdown factor: {:.2}x", integer_roots_slowdown);
    println!(
        "float value:         {}",
        binary_from_f64(integer_roots_float_result.value)
    );
    println!(
        "computable midpoint: {}",
        integer_roots_computable_result.midpoint
    );
    println!(
        "computable width: {}",
        integer_roots_computable_result.width
    );
    println!("abs(float - midpoint): {}", integer_roots_error);
}
