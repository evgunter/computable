//! Benchmark for pi computation.
//!
//! This benchmark tests the performance of the pi implementation at various
//! precision levels.

use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UBinary, XBinary, pi, pi_bounds_at_precision};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};

use crate::common::{binary_from_f64, finite_binary, midpoint, try_finite_bounds};

/// Precision levels to test (in bits)
const PRECISION_BITS: &[u64] = &[32, 64];

/// Number of iterations for timing stability
const TIMING_ITERATIONS: u32 = 5;

/// Benchmark pi refinement at various precision levels
fn benchmark_pi_refinement() {
    println!("== Pi Refinement Benchmark ==");
    println!();
    println!("Testing pi().refine_to_default(epsilon) at various precision levels");
    println!();

    for &precision_bits in PRECISION_BITS {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(precision_bits as i64)));

        // Warm up
        let pi_warmup = pi();
        let _ = pi_warmup.refine_to_default(epsilon.clone());

        // Timed runs
        let mut total_duration = Duration::ZERO;
        let mut final_bounds = Bounds::new(XBinary::NegInf, XBinary::PosInf);

        for _ in 0..TIMING_ITERATIONS {
            let pi_comp = pi();

            let start = Instant::now();
            let bounds = pi_comp
                .refine_to_default(epsilon.clone())
                .expect("pi refinement should succeed");
            total_duration += start.elapsed();

            final_bounds = bounds;
        }

        let avg_duration = total_duration / TIMING_ITERATIONS;
        let bounds = final_bounds;

        println!(
            "Precision: {} bits (epsilon = 2^-{})",
            precision_bits, precision_bits
        );
        println!("  Average time: {:?}", avg_duration);
        println!("  Width: {}", bounds.width());

        match try_finite_bounds(&bounds) {
            Some(finite) => {
                let mid = midpoint(&finite);
                // Compare to known pi digits (converting f64 to Binary for comparison)
                let pi_binary = binary_from_f64(std::f64::consts::PI);
                let error = mid.sub(&pi_binary).magnitude();
                println!("  Midpoint: {}", mid);
                println!("  |midpoint - pi|: {}", error);
            }
            None => {
                println!("  Bounds are infinite (cannot compute midpoint)");
            }
        }
        println!();
    }
}

/// Benchmark the pi_bounds_at_precision helper function
fn benchmark_pi_bounds_at_precision_fn() {
    println!("== pi_bounds_at_precision Benchmark ==");
    println!();
    println!("Testing direct bounds computation at various precision levels");
    println!();

    for &precision_bits in PRECISION_BITS {
        // Warm up
        let _ = pi_bounds_at_precision(precision_bits);

        // Timed runs
        let mut total_duration = Duration::ZERO;
        let mut final_bounds = None;

        for _ in 0..TIMING_ITERATIONS {
            let start = Instant::now();
            let bounds = pi_bounds_at_precision(precision_bits);
            total_duration += start.elapsed();

            final_bounds = Some(bounds);
        }

        let avg_duration = total_duration / TIMING_ITERATIONS;
        let (lower, upper) = final_bounds.unwrap();

        let width = upper.sub(&lower);
        let sum = lower.add(&upper);
        let mid = Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one());

        println!("Precision: {} bits", precision_bits);
        println!("  Average time: {:?}", avg_duration);
        println!("  Width: {}", width);
        println!("  Midpoint: {}", mid);
        println!();
    }
}

/// Benchmark pi in arithmetic expressions
fn benchmark_pi_arithmetic() {
    println!("== Pi Arithmetic Benchmark ==");
    println!();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-64));

    // 2 * pi
    println!("Computing 2pi:");
    let start = Instant::now();
    let two = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
    let two_pi = two * pi();
    let bounds = two_pi
        .refine_to_default(epsilon.clone())
        .expect("2pi computation should succeed");
    let duration = start.elapsed();
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let mid = midpoint(&finite);
    let expected = binary_from_f64(2.0 * std::f64::consts::PI);
    println!("  Time: {:?}", duration);
    println!("  Midpoint: {}", mid);
    println!("  Expected: {}", expected);
    println!();

    // pi / 2
    println!("Computing pi/2:");
    let start = Instant::now();
    let half = Computable::constant(Binary::new(BigInt::from(1), BigInt::from(-1)));
    let half_pi = half * pi();
    let bounds = half_pi
        .refine_to_default(epsilon.clone())
        .expect("pi/2 computation should succeed");
    let duration = start.elapsed();
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let mid = midpoint(&finite);
    let expected = binary_from_f64(std::f64::consts::FRAC_PI_2);
    println!("  Time: {:?}", duration);
    println!("  Midpoint: {}", mid);
    println!("  Expected: {}", expected);
    println!();

    // pi^2
    println!("Computing pi^2:");
    let start = Instant::now();
    let pi_squared = pi() * pi();
    let bounds = pi_squared
        .refine_to_default(epsilon.clone())
        .expect("pi^2 computation should succeed");
    let duration = start.elapsed();
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let mid = midpoint(&finite);
    let expected = binary_from_f64(std::f64::consts::PI * std::f64::consts::PI);
    println!("  Time: {:?}", duration);
    println!("  Midpoint: {}", mid);
    println!("  Expected: {}", expected);
    println!();

    // 1 / pi
    println!("Computing 1/pi:");
    let start = Instant::now();
    let inv_pi = pi().inv();
    let bounds = inv_pi
        .refine_to_default(epsilon.clone())
        .expect("1/pi computation should succeed");
    let duration = start.elapsed();
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let mid = midpoint(&finite);
    let expected = binary_from_f64(1.0 / std::f64::consts::PI);
    println!("  Time: {:?}", duration);
    println!("  Midpoint: {}", mid);
    println!("  Expected: {}", expected);
    println!();
}

/// Benchmark sin(pi) - tests integration between pi and sin
fn benchmark_sin_pi() {
    println!("== sin(pi) Benchmark ==");
    println!();
    println!("Testing sin at multiples of pi (should be ~= 0)");
    println!();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-32));

    let test_cases = [
        (1, "sin(pi)"),
        (2, "sin(2pi)"),
        (10, "sin(10pi)"),
        (100, "sin(100pi)"),
    ];

    for (multiplier, label) in test_cases {
        let start = Instant::now();

        let n_pi = if multiplier == 1 {
            pi()
        } else {
            let n = Computable::constant(Binary::new(BigInt::from(multiplier), BigInt::from(0)));
            n * pi()
        };

        let sin_n_pi = n_pi.sin();
        let bounds = sin_n_pi
            .refine_to_default(epsilon.clone())
            .expect("sin(n*pi) computation should succeed");
        let duration = start.elapsed();

        let lower = finite_binary(bounds.small());
        let upper = finite_binary(&bounds.large());
        let width = bounds.width();

        println!("{}:", label);
        println!("  Time: {:?}", duration);
        println!("  Bounds: [{}, {}]", lower, upper);
        println!("  Width: {}", width);
        println!(
            "  Contains 0: {}",
            lower.mantissa().is_negative() || lower.mantissa().is_zero()
        );
        println!();
    }
}

/// Benchmark high-precision pi computation
fn benchmark_high_precision() {
    println!("== High Precision Pi Benchmark ==");
    println!();

    let high_precisions: &[u64] = &[2048, 4096, 8192];

    for &precision_bits in high_precisions {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(precision_bits as i64)));

        println!(
            "Precision: {} bits ({} decimal digits approx)",
            precision_bits,
            (precision_bits as f64 * 0.301).round() as u64
        );

        let start = Instant::now();
        let pi_comp = pi();
        let bounds = pi_comp
            .refine_to_default(epsilon)
            .expect("high precision pi should succeed");
        let duration = start.elapsed();

        let width = bounds.width();

        println!("  Time: {:?}", duration);
        println!("  Width: {}", width);
        println!();
    }
}

pub fn run_pi_benchmark() {
    println!("== Pi Computation Benchmark ==");
    println!();

    benchmark_pi_refinement();
    benchmark_pi_bounds_at_precision_fn();
    benchmark_pi_arithmetic();
    benchmark_sin_pi();
    benchmark_high_precision();
}
