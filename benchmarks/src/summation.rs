use std::time::Instant;

use computable::Computable;
use rand::Rng;
use rand::rngs::StdRng;

use crate::balanced_sum::balanced_sum;
use crate::common::{
    BenchmarkResult, ComputableResult, binary_from_f64, midpoint, try_finite_bounds,
};

pub const SUMMATION_SAMPLE_COUNT: usize = 200_000;

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
    let finite =
        try_finite_bounds(&bounds).expect("bounds should be finite for arithmetic operations");

    ComputableResult {
        duration: start.elapsed(),
        midpoint: midpoint(&finite),
        width: bounds.width().clone(),
    }
}

pub fn run_summation_benchmark(rng: &mut StdRng) {
    let summation_inputs: Vec<f64> = (0..SUMMATION_SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let summation_inputs_computable: Vec<Computable> = summation_inputs
        .iter()
        .map(|&v| Computable::constant(binary_from_f64(v)))
        .collect();

    let summation_base = 2_i64.pow(30) as f64;
    let summation_float_result = summation_float(summation_base, &summation_inputs);
    let summation_computable_result =
        summation_computable(summation_base, &summation_inputs_computable);
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
        let finite =
            try_finite_bounds(&bounds).expect("bounds should be finite for arithmetic operations");
        midpoint(&finite)
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
    println!(
        "computable time: {:?}",
        summation_computable_result.duration
    );
    println!("slowdown factor: {:.2}x", summation_slowdown);
    println!(
        "float value:         {}",
        binary_from_f64(summation_float_result.value)
    );
    println!(
        "computable midpoint: {}",
        summation_computable_result.midpoint
    );
    println!("computable width: {}", summation_computable_result.width);
    println!("abs(float - midpoint): {}", summation_error);
    println!();
    println!("After removing base value:");
    println!(
        "  sum without base (float):      {}",
        binary_from_f64(baseless_sum_float)
    );
    println!(
        "  sum without base (computable): {}",
        baseless_sum_computable
    );
    println!(
        "  abs(float - midpoint): {}",
        (binary_from_f64(baseless_sum_float) - baseless_sum_computable).magnitude()
    );
    println!(
        "  sum with base minus base (float):      {}",
        binary_from_f64(float_minus_base)
    );
    println!(
        "  sum with base minus base (computable): {}",
        computable_minus_base
    );
    println!("  float precision loss: {}", float_base_error);
    println!("  computable precision loss: {}", computable_base_error);
}
