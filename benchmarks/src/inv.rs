use std::time::Instant;

use computable::{Computable, UBinary};
use num_bigint::BigInt;
use rand::Rng;
use rand::rngs::StdRng;

use crate::balanced_sum::balanced_sum;
use crate::common::{binary_from_f64, midpoint, try_finite_bounds};

pub const INV_SAMPLE_COUNT: usize = 100;
pub const INV_PRECISION_BITS: i64 = 256;

/// Benchmark: inv operation with high precision
/// This specifically tests the efficiency of the inv refinement loop
pub fn run_inv_benchmark(rng: &mut StdRng) {
    // Generate random positive values (avoiding values too close to zero)
    let inv_inputs: Vec<f64> = (0..INV_SAMPLE_COUNT)
        .map(|_| rng.gen_range(0.1..100.0))
        .collect();

    // Float computation
    let float_start = Instant::now();
    let float_sum: f64 = inv_inputs.iter().map(|x| 1.0 / x).sum();
    let float_duration = float_start.elapsed();

    // Computable computation with high precision target
    let epsilon = UBinary::new(
        num_bigint::BigUint::from(1u32),
        BigInt::from(-INV_PRECISION_BITS),
    );

    let computable_start = Instant::now();
    let inv_terms: Vec<Computable> = inv_inputs
        .iter()
        .map(|&x| Computable::constant(binary_from_f64(x)).inv())
        .collect();
    let total = balanced_sum(inv_terms);
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");
    let computable_duration = computable_start.elapsed();

    let finite = try_finite_bounds(&bounds).expect("bounds should be finite for inv operations");
    let computable_midpoint = midpoint(&finite);
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
