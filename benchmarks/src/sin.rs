use std::time::Instant;

use computable::{Computable, UBinary};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::Rng;

use crate::balanced_sum::balanced_sum;
use crate::common::{binary_from_f64, midpoint};

pub const SIN_SAMPLE_COUNT: usize = 100;
pub const SIN_PRECISION_BITS: i64 = 32;

/// Benchmark: sin operation
/// Tests Taylor series computation with range reduction and directed rounding
pub fn run_sin_benchmark(rng: &mut StdRng) {
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
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");
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
    println!(
        "samples: {} (1/3 small, 1/3 medium, 1/3 large)",
        SIN_SAMPLE_COUNT
    );
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
