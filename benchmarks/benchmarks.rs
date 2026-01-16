use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::One;
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
}
