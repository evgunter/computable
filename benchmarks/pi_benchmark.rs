//! Benchmark for pi computation.
//!
//! This benchmark tests the performance of the pi implementation at various
//! precision levels. It requires the `pi()` and `pi_bounds_at_precision()`
//! functions to be implemented and exported from the computable crate.
//!
//! Run with: cargo run --release --bin pi_benchmark --features benchmarks

use std::time::{Duration, Instant};

use computable::{pi, pi_bounds_at_precision, Binary, Bounds, Computable, UBinary, XBinary};
use num_bigint::{BigInt, BigUint};
use num_traits::One;

/// Precision levels to test (in bits)
const PRECISION_BITS: &[u64] = &[32, 64, 128, 256, 512, 1024];

/// Number of iterations for timing stability
const TIMING_ITERATIONS: u32 = 5;

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

// TODO: remove this and exclusively convert f64 to Binary for comparisons
fn binary_to_f64_approx(b: &Binary) -> f64 {
    // Rough approximation for display purposes only
    let mantissa_f64 = b.mantissa().to_string().parse::<f64>().unwrap_or(0.0);
    let exp_i32 = b.exponent().to_string().parse::<i32>().unwrap_or(0);
    mantissa_f64 * 2.0_f64.powi(exp_i32)
}

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
        let mut final_bounds = None;

        for _ in 0..TIMING_ITERATIONS {
            let pi_comp = pi();

            let start = Instant::now();
            let bounds = pi_comp
                .refine_to_default(epsilon.clone())
                .expect("pi refinement should succeed");
            total_duration += start.elapsed();

            final_bounds = Some(bounds);
        }

        let avg_duration = total_duration / TIMING_ITERATIONS;
        // TODO: can we avoid using unwrap?
        let bounds = final_bounds.unwrap();

        let lower = finite_binary(bounds.small());
        let upper = finite_binary(&bounds.large());
        let width = bounds.width();
        let mid = midpoint(&bounds);

        // Compare to known pi digits
        let pi_f64 = std::f64::consts::PI;
        let mid_f64 = binary_to_f64_approx(&mid);
        let error_f64 = (mid_f64 - pi_f64).abs();

        println!("Precision: {} bits (epsilon = 2^-{})", precision_bits, precision_bits);
        println!("  Average time: {:?}", avg_duration);
        println!("  Width: {}", width);
        println!("  Midpoint ≈ {:.15}", mid_f64);
        println!("  |midpoint - π| ≈ {:.2e}", error_f64);
        println!();
    }
}

/// Benchmark the pi_bounds_at_precision helper function
fn benchmark_pi_bounds_at_precision() {
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

        let mid_f64 = binary_to_f64_approx(&mid);
        let width_f64 = binary_to_f64_approx(&width);

        println!("Precision: {} bits", precision_bits);
        println!("  Average time: {:?}", avg_duration);
        println!("  Width ≈ {:.2e}", width_f64);
        println!("  Midpoint ≈ {:.15}", mid_f64);
        println!();
    }
}

/// Benchmark pi in arithmetic expressions
fn benchmark_pi_arithmetic() {
    println!("== Pi Arithmetic Benchmark ==");
    println!();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-64));

    // 2 * pi
    println!("Computing 2π:");
    let start = Instant::now();
    let two = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
    let two_pi = two * pi();
    let bounds = two_pi
        .refine_to_default(epsilon.clone())
        .expect("2π computation should succeed");
    let duration = start.elapsed();
    let mid = midpoint(&bounds);
    println!("  Time: {:?}", duration);
    println!("  Midpoint ≈ {:.15}", binary_to_f64_approx(&mid));
    println!("  Expected: {:.15}", 2.0 * std::f64::consts::PI);
    println!();

    // pi / 2
    println!("Computing π/2:");
    let start = Instant::now();
    let half = Computable::constant(Binary::new(BigInt::from(1), BigInt::from(-1)));
    let half_pi = half * pi();
    let bounds = half_pi
        .refine_to_default(epsilon.clone())
        .expect("π/2 computation should succeed");
    let duration = start.elapsed();
    let mid = midpoint(&bounds);
    println!("  Time: {:?}", duration);
    println!("  Midpoint ≈ {:.15}", binary_to_f64_approx(&mid));
    println!("  Expected: {:.15}", std::f64::consts::FRAC_PI_2);
    println!();

    // pi^2
    println!("Computing π²:");
    let start = Instant::now();
    let pi_squared = pi() * pi();
    let bounds = pi_squared
        .refine_to_default(epsilon.clone())
        .expect("π² computation should succeed");
    let duration = start.elapsed();
    let mid = midpoint(&bounds);
    println!("  Time: {:?}", duration);
    println!("  Midpoint ≈ {:.15}", binary_to_f64_approx(&mid));
    println!("  Expected: {:.15}", std::f64::consts::PI * std::f64::consts::PI);
    println!();

    // 1 / pi
    println!("Computing 1/π:");
    let start = Instant::now();
    let inv_pi = pi().inv();
    let bounds = inv_pi
        .refine_to_default(epsilon.clone())
        .expect("1/π computation should succeed");
    let duration = start.elapsed();
    let mid = midpoint(&bounds);
    println!("  Time: {:?}", duration);
    println!("  Midpoint ≈ {:.15}", binary_to_f64_approx(&mid));
    println!("  Expected: {:.15}", 1.0 / std::f64::consts::PI);
    println!();
}

/// Benchmark sin(pi) - tests integration between pi and sin
fn benchmark_sin_pi() {
    println!("== sin(π) Benchmark ==");
    println!();
    println!("Testing sin at multiples of π (should be ≈ 0)");
    println!();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-32));

    let test_cases = [
        (1, "sin(π)"),
        (2, "sin(2π)"),
        (10, "sin(10π)"),
        (100, "sin(100π)"),
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
            .expect("sin(nπ) computation should succeed");
        let duration = start.elapsed();

        let lower = finite_binary(bounds.small());
        let upper = finite_binary(&bounds.large());
        let width = bounds.width();

        println!("{}:", label);
        println!("  Time: {:?}", duration);
        println!("  Bounds: [{:.6e}, {:.6e}]",
            binary_to_f64_approx(&lower),
            binary_to_f64_approx(&upper));
        println!("  Width: {}", width);
        println!("  Contains 0: {}",
            lower.mantissa().is_negative() || lower.mantissa().is_zero());
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

        println!("Precision: {} bits ({} decimal digits approx)",
            precision_bits,
            (precision_bits as f64 * 0.301).round() as u64);

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

fn main() {
    println!("========================================");
    println!("        Pi Computation Benchmark        ");
    println!("========================================");
    println!();

    benchmark_pi_refinement();
    benchmark_pi_bounds_at_precision();
    benchmark_pi_arithmetic();
    benchmark_sin_pi();
    benchmark_high_precision();

    println!("========================================");
    println!("           Benchmark Complete           ");
    println!("========================================");
}
