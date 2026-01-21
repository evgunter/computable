use std::time::Instant;

use computable::Computable;
use rand::rngs::StdRng;
use rand::Rng;

use crate::balanced_sum::balanced_sum;
use crate::common::{binary_from_f64, try_finite_bounds, midpoint, BenchmarkResult, ComputableResult};

pub const COMPLEX_SAMPLE_COUNT: usize = 5_000;

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
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite for arithmetic operations");

    ComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&finite),
        width: bounds.width().clone(),
    }
}

pub fn run_complex_benchmark(rng: &mut StdRng) {
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
    println!(
        "float value:         {}",
        binary_from_f64(complex_float_result.value)
    );
    println!(
        "computable midpoint: {}",
        complex_computable_result.midpoint
    );
    println!("computable width: {}", complex_computable_result.width);
    println!("abs(float - midpoint): {}", complex_error);
}
