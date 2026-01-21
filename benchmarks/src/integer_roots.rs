use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use computable::binary_utils::bisection::{bisection_step, bisection_step_midpoint, BisectionComparison};
use computable::{Binary, Computable, FiniteBounds, UBinary};
use num_bigint::BigInt;
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::Rng;

use crate::balanced_sum::balanced_sum;
use crate::common::{binary_from_f64, midpoint};
use crate::UXBinary;

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

    IntegerRootsComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
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
    println!(
        "float time:      {:?}",
        integer_roots_float_result.duration
    );
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

/// Returns the number of bits in the mantissa of a Binary value.
fn mantissa_bits(value: &Binary) -> u64 {
    value.mantissa().magnitude().bits()
}

/// Returns the total mantissa bits used by bounds (lower + upper).
fn bounds_mantissa_bits(bounds: &FiniteBounds) -> u64 {
    mantissa_bits(bounds.small()) + mantissa_bits(&bounds.large())
}

/// Statistics from running a bisection strategy.
#[derive(Debug)]
struct BisectionStats {
    duration: Duration,
    final_width_bits: u64,
    max_mantissa_bits: u64,
    final_mantissa_bits: u64,
    total_mantissa_bits_accumulated: u64,
}

/// Runs bisection for sqrt(target) using the provided step function.
fn run_bisection<F>(
    initial_bounds: FiniteBounds,
    target: &Binary,
    iterations: usize,
    step_fn: F,
) -> BisectionStats
where
    F: Fn(FiniteBounds, &dyn Fn(&Binary) -> BisectionComparison) -> FiniteBounds,
{
    let start = Instant::now();

    let compare = |mid: &Binary| -> BisectionComparison {
        let mid_sq = mid.mul(mid);
        match mid_sq.cmp(target) {
            std::cmp::Ordering::Less => BisectionComparison::Above,
            std::cmp::Ordering::Equal => BisectionComparison::Exact,
            std::cmp::Ordering::Greater => BisectionComparison::Below,
        }
    };

    let mut bounds = initial_bounds;
    let mut max_mantissa_bits: u64 = bounds_mantissa_bits(&bounds);
    let mut total_mantissa_bits: u64 = max_mantissa_bits;

    for _ in 0..iterations {
        bounds = step_fn(bounds, &compare);
        let current_bits = bounds_mantissa_bits(&bounds);
        max_mantissa_bits = max_mantissa_bits.max(current_bits);
        total_mantissa_bits = total_mantissa_bits.saturating_add(current_bits);
        if bounds.width().mantissa().is_zero() {
            break;
        }
    }

    let duration = start.elapsed();
    let final_width_bits = bounds.width().mantissa().bits();
    let final_mantissa_bits = bounds_mantissa_bits(&bounds);

    BisectionStats {
        duration,
        final_width_bits,
        max_mantissa_bits,
        final_mantissa_bits,
        total_mantissa_bits_accumulated: total_mantissa_bits,
    }
}

/// Benchmark comparing midpoint-based bisection vs shortest-representation bisection.
///
/// This benchmark measures:
/// 1. Performance (time taken)
/// 2. Precision accumulation (mantissa bit growth)
/// 3. Final interval width
pub fn run_bisection_comparison_benchmark() {
    const ITERATIONS: usize = 100;

    // Create initial bounds [0, 4] to find sqrt(2) ~ 1.414...
    let lower = Binary::new(BigInt::from(0), BigInt::from(0));
    let upper = Binary::new(BigInt::from(4), BigInt::from(0));
    let target = Binary::new(BigInt::from(2), BigInt::from(0)); // Looking for x where x^2 = 2

    let initial_bounds = FiniteBounds::new(lower, upper);

    // Run with midpoint strategy
    let midpoint_stats = run_bisection(
        initial_bounds.clone(),
        &target,
        ITERATIONS,
        |bounds, compare| bisection_step_midpoint(bounds, compare),
    );

    // Run with shortest-representation strategy
    let shortest_stats = run_bisection(
        initial_bounds.clone(),
        &target,
        ITERATIONS,
        |bounds, compare| bisection_step(bounds, compare),
    );

    // Print results
    println!("== Bisection strategy comparison benchmark ==");
    println!("Problem: find sqrt(2) in [0, 4]");
    println!("Iterations: {}", ITERATIONS);
    println!();

    println!("--- Midpoint strategy ---");
    println!("  Time:                    {:?}", midpoint_stats.duration);
    println!(
        "  Final width bits:        {}",
        midpoint_stats.final_width_bits
    );
    println!(
        "  Max mantissa bits:       {}",
        midpoint_stats.max_mantissa_bits
    );
    println!(
        "  Final mantissa bits:     {}",
        midpoint_stats.final_mantissa_bits
    );
    println!(
        "  Total bits accumulated:  {}",
        midpoint_stats.total_mantissa_bits_accumulated
    );
    println!();

    println!("--- Shortest-representation strategy ---");
    println!("  Time:                    {:?}", shortest_stats.duration);
    println!(
        "  Final width bits:        {}",
        shortest_stats.final_width_bits
    );
    println!(
        "  Max mantissa bits:       {}",
        shortest_stats.max_mantissa_bits
    );
    println!(
        "  Final mantissa bits:     {}",
        shortest_stats.final_mantissa_bits
    );
    println!(
        "  Total bits accumulated:  {}",
        shortest_stats.total_mantissa_bits_accumulated
    );
    println!();

    // Comparison
    let speedup = midpoint_stats.duration.as_secs_f64() / shortest_stats.duration.as_secs_f64();
    let bits_reduction = midpoint_stats.max_mantissa_bits as f64
        / shortest_stats.max_mantissa_bits.max(1) as f64;

    println!("--- Comparison ---");
    if speedup > 1.0 {
        println!(
            "  Shortest-repr is {:.2}x faster",
            speedup
        );
    } else {
        println!(
            "  Midpoint is {:.2}x faster",
            1.0 / speedup
        );
    }
    println!(
        "  Max mantissa bits ratio (midpoint/shortest): {:.2}x",
        bits_reduction
    );
    println!(
        "  Total bits accumulated ratio: {:.2}x",
        midpoint_stats.total_mantissa_bits_accumulated as f64
            / shortest_stats.total_mantissa_bits_accumulated.max(1) as f64
    );
}
