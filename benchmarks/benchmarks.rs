use std::collections::HashSet;
use std::env;
use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UBinary, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::One;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Available benchmark names
const BENCHMARK_NAMES: &[&str] = &["complex", "summation", "integer-roots", "inv", "sin"];

const COMPLEX_SAMPLE_COUNT: usize = 5_000;
const SUMMATION_SAMPLE_COUNT: usize = 200_000;
const INTEGER_ROOTS_SAMPLE_COUNT: usize = 1_000;

// TODO: split this into multiple files

#[derive(Debug)]
struct BenchmarkResult {
    duration: Duration,
    value: f64,
}

#[derive(Debug)]
struct ComputableResult {
    duration: Duration,
    midpoint: Binary,
    width: UXBinary,
}

fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value).expect("expected finite f64") {
        XBinary::Finite(binary) => binary,
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite f64 input")
        }
    }
}

fn finite_binary(value: &XBinary) -> Binary {
    match value {
        XBinary::Finite(binary) => binary.clone(),
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite bounds")
        }
    }
}

fn midpoint(bounds: &Bounds) -> Binary {
    let lower = finite_binary(bounds.small());
    let upper = finite_binary(&bounds.large());
    let sum = lower.add(&upper);
    let half = Binary::new(BigInt::one(), BigInt::from(-1));
    sum.mul(&half)
}

fn complex_float(inputs: &[(f64, f64, f64, f64)]) -> BenchmarkResult {
    let start = Instant::now();
    let mut total = 0.0;
    for (a, b, c, d) in inputs {
        let mixed = (a + b) * (c - d);
        let squared = c * c + d * d;
        total += a * b + squared + mixed;
    }
    BenchmarkResult {
        duration: start.elapsed(),
        value: total,
    }
}

fn complex_computable(inputs: &[(f64, f64, f64, f64)]) -> ComputableResult {
    let start = Instant::now();
    let mut terms = Vec::with_capacity(inputs.len());

    for (a, b, c, d) in inputs {
        let a_c = Computable::constant(binary_from_f64(*a));
        let b_c = Computable::constant(binary_from_f64(*b));
        let c_c = Computable::constant(binary_from_f64(*c));
        let d_c = Computable::constant(binary_from_f64(*d));

        let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
        let squared = c_c.clone() * c_c + d_c.clone() * d_c;
        let term = a_c * b_c + squared + mixed;
        terms.push(term);
    }

    let total = balanced_sum(terms);

    let bounds = total.bounds().expect("bounds should succeed");

    ComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
        width: bounds.width().clone(),
    }
}

fn summation_float(base: f64, inputs: &[f64]) -> BenchmarkResult {
    let start = Instant::now();
    let mut total = base;
    for value in inputs {
        total += value;
    }
    BenchmarkResult {
        duration: start.elapsed(),
        value: total,
    }
}

fn summation_computable(base: f64, inputs: &[Computable]) -> ComputableResult {
    let start = Instant::now();
    let mut terms = Vec::with_capacity(inputs.len() + 1);
    terms.push(Computable::constant(binary_from_f64(base)));
    terms.extend(inputs.iter().cloned());
    let total = balanced_sum(terms);
    let bounds = total.bounds().expect("bounds should succeed");

    ComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
        width: bounds.width().clone(),
    }
}

/// Sums terms using a balanced reduction instead of left-associative chaining.
///
/// This keeps the computation graph shallow (O(log n) depth), avoiding deep nesting
/// that can overflow the stack or distort timing by spending most of the runtime
/// walking long expression chains.
fn balanced_sum(mut values: Vec<Computable>) -> Computable {
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

/// Computes the n-th root of a value using binary search.
/// Returns a Computable that refines by bisection.
fn nth_root_computable(value: u64, n: u32) -> Computable {
    let value_binary = Binary::new(BigInt::from(value), BigInt::from(0));
    Computable::constant(value_binary).nth_root(n)
}

/// Computes integer n-th root using f64 (for comparison).
fn nth_root_float(value: f64, n: u32) -> f64 {
    value.powf(1.0 / n as f64)
}

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
    let bounds = total.refine_to_default(epsilon).expect("refine_to should succeed");

    IntegerRootsComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
        width: bounds.width().clone(),
    }
}

fn print_usage() {
    println!("Usage: benchmarks [OPTIONS] [BENCHMARK...]");
    println!();
    println!("Run performance benchmarks for Computable arithmetic.");
    println!();
    println!("Options:");
    println!("  --help, -h       Show this help message");
    println!("  --list, -l       List available benchmarks");
    println!();
    println!("Arguments:");
    println!("  BENCHMARK        Benchmark(s) to run, by name or index (0-based)");
    println!("                   If no benchmarks specified, runs all benchmarks.");
    println!();
    println!("Examples:");
    println!("  benchmarks                      # Run all benchmarks");
    println!("  benchmarks complex              # Run only 'complex' benchmark");
    println!("  benchmarks 0 2                  # Run benchmarks 0 and 2");
    println!("  benchmarks summation complex    # Run 'summation' and 'complex'");
}

fn print_benchmark_list() {
    println!("Available benchmarks:");
    for (i, name) in BENCHMARK_NAMES.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
}

fn parse_benchmark_selection(args: &[String]) -> HashSet<usize> {
    let mut selected = HashSet::new();
    
    for arg in args {
        // Try parsing as index first
        if let Ok(index) = arg.parse::<usize>() {
            if index < BENCHMARK_NAMES.len() {
                selected.insert(index);
            } else {
                eprintln!("Warning: benchmark index {} out of range (0-{})", index, BENCHMARK_NAMES.len() - 1);
            }
        } else {
            // Try matching by name
            if let Some(index) = BENCHMARK_NAMES.iter().position(|&name| name == arg) {
                selected.insert(index);
            } else {
                eprintln!("Warning: unknown benchmark '{}'", arg);
            }
        }
    }
    
    selected
}

fn run_complex_benchmark(rng: &mut StdRng) {
    let complex_inputs: Vec<(f64, f64, f64, f64)> = (0..COMPLEX_SAMPLE_COUNT)
        .map(|_| {
            (
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            )
        })
        .collect();

    let complex_float_result = complex_float(&complex_inputs);
    let complex_computable_result = complex_computable(&complex_inputs);
    let complex_error = {
        let float_as_binary = binary_from_f64(complex_float_result.value);
        let diff = float_as_binary.sub(&complex_computable_result.midpoint);
        diff.magnitude()
    };

    let complex_slowdown = complex_computable_result.duration.as_secs_f64()
        / complex_float_result.duration.as_secs_f64();

    println!("== Complex expression benchmark ==");
    println!("samples: {COMPLEX_SAMPLE_COUNT}");
    println!("float time:      {:?}", complex_float_result.duration);
    println!("computable time: {:?}", complex_computable_result.duration);
    println!("slowdown factor: {:.2}x", complex_slowdown);
    println!("float value:         {}", binary_from_f64(complex_float_result.value));
    println!("computable midpoint: {}", complex_computable_result.midpoint);
    println!("computable width: {}", complex_computable_result.width);
    println!("abs(float - midpoint): {}", complex_error);
}

fn run_summation_benchmark(rng: &mut StdRng) {
    let summation_inputs: Vec<f64> = (0..SUMMATION_SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let summation_inputs_computable: Vec<Computable> = summation_inputs
        .iter()
        .map(|&v| Computable::constant(binary_from_f64(v)))
        .collect();

    let summation_base = 2_i64.pow(30) as f64;
    let summation_float_result = summation_float(summation_base, &summation_inputs);
    let summation_computable_result = summation_computable(summation_base, &summation_inputs_computable);
    let summation_error = {
        let float_as_binary = binary_from_f64(summation_float_result.value);
        let diff = float_as_binary.sub(&summation_computable_result.midpoint);
        diff.magnitude()
    };

    // Calculate true sum of inputs (without base) for comparison
    let baseless_sum_float: f64 = summation_inputs.iter().sum();
    let baseless_sum_computable = {
        let sum = balanced_sum(summation_inputs_computable.clone());
        let bounds = sum.bounds().expect("bounds should succeed");
        midpoint(&bounds)
    };

    let float_minus_base = summation_float_result.value - summation_base;
    let computable_minus_base = {
        let base_as_binary = binary_from_f64(summation_base);
        summation_computable_result.midpoint.sub(&base_as_binary)
    };
    let float_base_error = {
        let float_as_binary = binary_from_f64(float_minus_base);
        let diff = float_as_binary.sub(&baseless_sum_computable);
        diff.magnitude()
    };
    let computable_base_error = {
        let diff = computable_minus_base.sub(&baseless_sum_computable);
        diff.magnitude()
    };

    let summation_slowdown = summation_computable_result.duration.as_secs_f64()
        / summation_float_result.duration.as_secs_f64();

    println!("== Summation (catastrophic) benchmark ==");
    println!("samples: {SUMMATION_SAMPLE_COUNT}");
    println!("base value: {}", binary_from_f64(summation_base));
    println!("float time: {:?}", summation_float_result.duration);
    println!("computable time: {:?}", summation_computable_result.duration);
    println!("slowdown factor: {:.2}x", summation_slowdown);
    println!("float value:         {}", binary_from_f64(summation_float_result.value));
    println!("computable midpoint: {}", summation_computable_result.midpoint);
    println!("computable width: {}", summation_computable_result.width);
    println!("abs(float - midpoint): {}", summation_error);
    println!();
    println!("After removing base value:");
    println!("  sum without base (float):      {}", binary_from_f64(baseless_sum_float));
    println!("  sum without base (computable): {}", baseless_sum_computable);
    println!("  abs(float - midpoint): {}", (binary_from_f64(baseless_sum_float) - baseless_sum_computable).magnitude());
    println!("  sum with base minus base (float):      {}", binary_from_f64(float_minus_base));
    println!("  sum with base minus base (computable): {}", computable_minus_base);
    println!("  float precision loss: {}", float_base_error);
    println!("  computable precision loss: {}", computable_base_error);
}

fn run_integer_roots_benchmark(rng: &mut StdRng) {
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
    println!("computable time: {:?}", integer_roots_computable_result.duration);
    println!("slowdown factor: {:.2}x", integer_roots_slowdown);
    println!("float value:         {}", binary_from_f64(integer_roots_float_result.value));
    println!("computable midpoint: {}", integer_roots_computable_result.midpoint);
    println!("computable width: {}", integer_roots_computable_result.width);
    println!("abs(float - midpoint): {}", integer_roots_error);
}

const INV_SAMPLE_COUNT: usize = 100;
const INV_PRECISION_BITS: i64 = 256;

const SIN_SAMPLE_COUNT: usize = 100;
const SIN_PRECISION_BITS: i64 = 32;

/// Benchmark: inv operation with high precision
/// This specifically tests the efficiency of the inv refinement loop
fn run_inv_benchmark(rng: &mut StdRng) {
    // Generate random positive values (avoiding values too close to zero)
    let inv_inputs: Vec<f64> = (0..INV_SAMPLE_COUNT)
        .map(|_| rng.gen_range(0.1..100.0))
        .collect();

    // Float computation
    let float_start = Instant::now();
    let float_sum: f64 = inv_inputs.iter().map(|x| 1.0 / x).sum();
    let float_duration = float_start.elapsed();

    // Computable computation with high precision target
    let epsilon = UBinary::new(num_bigint::BigUint::from(1u32), BigInt::from(-INV_PRECISION_BITS));

    let computable_start = Instant::now();
    let inv_terms: Vec<Computable> = inv_inputs
        .iter()
        .map(|&x| Computable::constant(binary_from_f64(x)).inv())
        .collect();
    let total = balanced_sum(inv_terms);
    let bounds = total.refine_to_default(epsilon).expect("refine_to should succeed");
    let computable_duration = computable_start.elapsed();

    let computable_midpoint = midpoint(&bounds);
    let computable_width = bounds.width().clone();

    let inv_error = {
        let float_as_binary = binary_from_f64(float_sum);
        let diff = float_as_binary.sub(&computable_midpoint);
        diff.magnitude()
    };

    let inv_slowdown = computable_duration.as_secs_f64() / float_duration.as_secs_f64();

    println!("== Inverse (1/x) benchmark ==");
    println!("samples: {INV_SAMPLE_COUNT}");
    println!("target precision: {} bits", INV_PRECISION_BITS);
    println!("float time:      {:?}", float_duration);
    println!("computable time: {:?}", computable_duration);
    println!("slowdown factor: {:.2}x", inv_slowdown);
    println!("float value:         {}", binary_from_f64(float_sum));
    println!("computable midpoint: {}", computable_midpoint);
    println!("computable width: {}", computable_width);
    println!("abs(float - midpoint): {}", inv_error);
}

/// Benchmark: sin operation
/// Tests Taylor series computation with range reduction and directed rounding
fn run_sin_benchmark(rng: &mut StdRng) {
    use num_bigint::BigUint;

    // Generate test values: mix of small, medium, and large
    let sin_inputs: Vec<f64> = (0..SIN_SAMPLE_COUNT)
        .map(|i| {
            if i < SIN_SAMPLE_COUNT / 3 {
                // Small values |x| <= 1
                rng.gen_range(-1.0..1.0)
            } else if i < 2 * SIN_SAMPLE_COUNT / 3 {
                // Medium values |x| <= π
                rng.gen_range(-3.15..3.15)
            } else {
                // Large values (tests range reduction)
                rng.gen_range(-100.0..100.0)
            }
        })
        .collect();

    // Float computation
    let float_start = Instant::now();
    let float_sum: f64 = sin_inputs.iter().map(|x| x.sin()).sum();
    let float_duration = float_start.elapsed();

    // Computable computation
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-SIN_PRECISION_BITS));

    let computable_start = Instant::now();
    let sin_terms: Vec<Computable> = sin_inputs
        .iter()
        .map(|&x| Computable::constant(binary_from_f64(x)).sin())
        .collect();
    let total = balanced_sum(sin_terms);
    let bounds = total.refine_to_default(epsilon).expect("refine_to should succeed");
    let computable_duration = computable_start.elapsed();

    let computable_midpoint = midpoint(&bounds);
    let computable_width = bounds.width().clone();

    let sin_error = {
        let float_as_binary = binary_from_f64(float_sum);
        let diff = float_as_binary.sub(&computable_midpoint);
        diff.magnitude()
    };

    let sin_slowdown = computable_duration.as_secs_f64() / float_duration.as_secs_f64();
    let per_call = computable_duration / SIN_SAMPLE_COUNT as u32;

    println!("== Sine (sin) benchmark ==");
    println!("samples: {} (1/3 small, 1/3 medium, 1/3 large)", SIN_SAMPLE_COUNT);
    println!("target precision: {} bits", SIN_PRECISION_BITS);
    println!("float time:      {:?}", float_duration);
    println!("computable time: {:?}", computable_duration);
    println!("per sin() call:  {:?}", per_call);
    println!("slowdown factor: {:.2}x", sin_slowdown);
    println!("float value:         {}", binary_from_f64(float_sum));
    println!("computable midpoint: {}", computable_midpoint);
    println!("computable width: {}", computable_width);
    println!("abs(float - midpoint): {}", sin_error);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    
    // Handle help and list options
    for arg in &args {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                return;
            }
            "--list" | "-l" => {
                print_benchmark_list();
                return;
            }
            _ => {}
        }
    }
    
    // Filter out options and parse benchmark selection
    let benchmark_args: Vec<String> = args.into_iter()
        .filter(|arg| !arg.starts_with('-'))
        .collect();
    
    let selected = if benchmark_args.is_empty() {
        // Run all benchmarks if none specified
        (0..BENCHMARK_NAMES.len()).collect()
    } else {
        parse_benchmark_selection(&benchmark_args)
    };
    
    if selected.is_empty() {
        eprintln!("No valid benchmarks selected. Use --list to see available benchmarks.");
        return;
    }
    
    let mut rng = StdRng::seed_from_u64(7);
    let mut first = true;
    
    for i in 0..BENCHMARK_NAMES.len() {
        if selected.contains(&i) {
            if !first {
                println!();
            }
            first = false;
            
            match i {
                0 => run_complex_benchmark(&mut rng),
                1 => run_summation_benchmark(&mut rng),
                2 => run_integer_roots_benchmark(&mut rng),
                3 => run_inv_benchmark(&mut rng),
                4 => run_sin_benchmark(&mut rng),
                _ => unreachable!(),
            }
        }
    }
}
