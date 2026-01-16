use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::One;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPoolBuilder;

const COMPLEX_SAMPLE_COUNT: usize = 5_000;
const SUMMATION_SAMPLE_COUNT: usize = 200_000;
const INTEGER_ROOTS_SAMPLE_COUNT: usize = 1_000;

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
    // Initial bounds: [1, value] for roots of values >= 1
    // For the n-th root of x, we know 1 <= x^(1/n) <= x for x >= 1
    let upper_bound = if value > 1 { value as i64 } else { 1 };
    let interval_state = Bounds::new(
        XBinary::Finite(Binary::new(BigInt::one(), BigInt::from(0))),
        XBinary::Finite(Binary::new(BigInt::from(upper_bound), BigInt::from(0))),
    );

    let bounds = |inner_state: &Bounds| -> Result<Bounds, computable::ComputableError> {
        Ok(inner_state.clone())
    };

    let target = Binary::new(BigInt::from(value), BigInt::from(0));
    let exponent = n;

    let refine = move |inner_state: Bounds| -> Bounds {
        let lower = match inner_state.small() {
            XBinary::Finite(b) => b.clone(),
            _ => panic!("expected finite lower bound"),
        };
        let upper = match inner_state.large() {
            XBinary::Finite(b) => b.clone(),
            _ => panic!("expected finite upper bound"),
        };

        // Compute midpoint: (lower + upper) / 2
        let sum = lower.add(&upper);
        let half = Binary::new(BigInt::one(), BigInt::from(-1));
        let mid = sum.mul(&half);

        // Compute mid^n
        let mut mid_pow = mid.clone();
        for _ in 1..exponent {
            mid_pow = mid_pow.mul(&mid);
        }

        // Binary search: if mid^n <= target, search upper half; else search lower half
        if mid_pow <= target {
            Bounds::new(XBinary::Finite(mid), XBinary::Finite(upper))
        } else {
            Bounds::new(XBinary::Finite(lower), XBinary::Finite(mid))
        }
    };

    Computable::new(interval_state, bounds, refine)
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
    let epsilon = Binary::new(BigInt::one(), BigInt::from(0));
    let bounds = total.refine_to_default(epsilon).expect("refine_to should succeed");

    IntegerRootsComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
        width: bounds.width().clone(),
    }
}

fn main() {
    // Limit threads to avoid a bug with too many threads
    // Using 1 thread to work around concurrency issues
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok(); // Ignore error if already initialized
    let mut rng = StdRng::seed_from_u64(7);

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

    let summation_inputs: Vec<f64> = (0..SUMMATION_SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let summation_inputs_computable: Vec<Computable> = summation_inputs
        .iter()
        .map(|&v| Computable::constant(binary_from_f64(v)))
        .collect();

    let complex_float_result = complex_float(&complex_inputs);
    let complex_computable_result = complex_computable(&complex_inputs);
    let complex_error = {
        let float_as_binary = binary_from_f64(complex_float_result.value);
        let diff = float_as_binary.sub(&complex_computable_result.midpoint);
        diff.magnitude()
    };

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

    let complex_slowdown = complex_computable_result.duration.as_secs_f64()
        / complex_float_result.duration.as_secs_f64();
    let summation_slowdown = summation_computable_result.duration.as_secs_f64()
        / summation_float_result.duration.as_secs_f64();

    println!("== Complex expression benchmark ==");
    println!("samples: {COMPLEX_SAMPLE_COUNT}");
    println!("float time:      {:?}", complex_float_result.duration);
    println!("computable time: {:?}", complex_computable_result.duration);
    println!("slowdown factor: {:.2}x", complex_slowdown);
    println!("float value:         {}", binary_from_f64(complex_float_result.value));
    println!("computable midpoint: {}", complex_computable_result.midpoint);
    println!("computable width: {}", complex_computable_result.width);
    println!("abs(float - midpoint): {}", complex_error);
    println!();
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

    // === Integer Roots Benchmark ===
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

    println!();
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
