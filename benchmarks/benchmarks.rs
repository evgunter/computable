use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const COMPLEX_SAMPLE_COUNT: usize = 5_000;
const SUMMATION_SAMPLE_COUNT: usize = 200_000;

#[derive(Debug)]
struct BenchmarkResult {
    duration: Duration,
    value: f64,
}

#[derive(Debug)]
struct ComputableResult {
    duration: Duration,
    midpoint: f64,
    width: f64,
}

fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value).expect("expected finite f64") {
        XBinary::Finite(binary) => binary,
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite f64 input")
        }
    }
}

fn binary_to_f64(binary: &Binary) -> f64 {
    let mantissa = binary
        .mantissa()
        .to_f64()
        .expect("mantissa should fit in f64");
    let exponent = binary
        .exponent()
        .to_i32()
        .expect("exponent should fit in i32");
    mantissa * (2.0f64).powi(exponent)
}

fn uxbinary_to_f64(value: &UXBinary) -> f64 {
    match value {
        UXBinary::Finite(ubinary) => binary_to_f64(&ubinary.to_binary()),
        UXBinary::PosInf => f64::INFINITY,
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

fn midpoint(bounds: &Bounds) -> f64 {
    let lower = finite_binary(bounds.small());
    let upper = finite_binary(&bounds.large());
    let sum = lower.add(&upper);
    let half = Binary::new(BigInt::one(), BigInt::from(-1));
    let mid = sum.mul(&half);
    binary_to_f64(&mid)
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
        width: uxbinary_to_f64(bounds.width()),
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

fn summation_computable(base: f64, inputs: &[f64]) -> ComputableResult {
    let start = Instant::now();
    let mut terms = Vec::with_capacity(inputs.len() + 1);
    terms.push(Computable::constant(binary_from_f64(base)));
    for value in inputs {
        let term = Computable::constant(binary_from_f64(*value));
        terms.push(term);
    }
    let total = balanced_sum(terms);
    let bounds = total.bounds().expect("bounds should succeed");

    ComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&bounds),
        width: uxbinary_to_f64(bounds.width()),
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

fn main() {
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

    let complex_float_result = complex_float(&complex_inputs);
    let complex_computable_result = complex_computable(&complex_inputs);
    let complex_error = (complex_float_result.value - complex_computable_result.midpoint).abs();

    let summation_base = 1.0e12;
    let summation_float_result = summation_float(summation_base, &summation_inputs);
    let summation_computable_result = summation_computable(summation_base, &summation_inputs);
    let summation_error =
        (summation_float_result.value - summation_computable_result.midpoint).abs();

    // Calculate true sum of inputs (without base) for comparison
    let true_sum: f64 = summation_inputs.iter().sum();
    let float_minus_base = summation_float_result.value - summation_base;
    let computable_minus_base = summation_computable_result.midpoint - summation_base;
    let float_base_error = (float_minus_base - true_sum).abs();
    let computable_base_error = (computable_minus_base - true_sum).abs();

    let complex_slowdown = complex_computable_result.duration.as_secs_f64()
        / complex_float_result.duration.as_secs_f64();
    let summation_slowdown = summation_computable_result.duration.as_secs_f64()
        / summation_float_result.duration.as_secs_f64();

    println!("== Complex expression benchmark ==");
    println!("samples: {COMPLEX_SAMPLE_COUNT}");
    println!("float time: {:?}", complex_float_result.duration);
    println!("computable time: {:?}", complex_computable_result.duration);
    println!("slowdown factor: {:.2}x", complex_slowdown);
    println!("float value: {:.10}", complex_float_result.value);
    println!("computable midpoint: {:.10}", complex_computable_result.midpoint);
    println!("computable width: {:.10}", complex_computable_result.width);
    println!("abs(float - midpoint): {:.10}", complex_error);
    println!();
    println!("== Summation (catastrophic) benchmark ==");
    println!("samples: {SUMMATION_SAMPLE_COUNT}");
    println!("base value: {summation_base:.1}");
    println!("float time: {:?}", summation_float_result.duration);
    println!("computable time: {:?}", summation_computable_result.duration);
    println!("slowdown factor: {:.2}x", summation_slowdown);
    println!("float value: {:.10}", summation_float_result.value);
    println!("computable midpoint: {:.10}", summation_computable_result.midpoint);
    println!("computable width: {:.10}", summation_computable_result.width);
    println!("abs(float - midpoint): {:.10}", summation_error);
    println!();
    println!("After removing base value:");
    println!("  true sum (inputs only): {:.10}", true_sum);
    println!("  float result: {:.10}", float_minus_base);
    println!("  computable result: {:.10}", computable_minus_base);
    println!("  float precision loss: {:.10}", float_base_error);
    println!("  computable precision loss: {:.10}", computable_base_error);
}
