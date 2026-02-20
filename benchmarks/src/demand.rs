//! Benchmark for asymmetric convergence rates.
//!
//! Demonstrates expressions where one refiner converges much faster than another,
//! causing the fast refiner to be stepped uselessly. This is the case where
//! demand propagation (computing per-refiner precision budgets and skipping
//! refiners that are already precise enough) would help.
//!
//! Key example: `sqrt(2) + pi()`. Pi converges exponentially (term doubling),
//! while sqrt converges linearly (bisection, 1 bit per step). After ~2 rounds,
//! pi's width is ~2^-130 — far below any reasonable epsilon — but the coordinator
//! keeps stepping PiOp alongside NthRootOp. Each PiOp step doubles its term count,
//! making compute_bounds increasingly expensive. The wasted PiOp computation
//! dominates total time for moderate precision targets.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use computable::{Binary, Computable, UBinary, pi};
use num_bigint::{BigInt, BigUint};

use crate::common::try_finite_bounds;

/// Precision targets (in bits). Kept small because PiOp's exponential term
/// growth makes high precision prohibitively slow in the pathological cases.
const PRECISION_BITS: &[u64] = &[4, 6];

/// Number of timed iterations for averaging.
const TIMING_ITERATIONS: u32 = 5;

fn sqrt_2() -> Computable {
    Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)))
        .nth_root(NonZeroU32::new(2).expect("2 is non-zero"))
}

fn sqrt_3() -> Computable {
    Computable::constant(Binary::new(BigInt::from(3), BigInt::from(0)))
        .nth_root(NonZeroU32::new(3).expect("3 is non-zero"))
}

fn constant(value: i64) -> Computable {
    Computable::constant(Binary::new(BigInt::from(value), BigInt::from(0)))
}

/// Runs a benchmark for a single expression at a given precision, with warmup
/// and averaging. Returns the average duration.
fn bench_expression(
    label: &str,
    make_expr: &dyn Fn() -> Computable,
    precision_bits: u64,
) -> Duration {
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(precision_bits as i64)));

    // Warmup
    let warmup_expr = make_expr();
    let warmup_result = warmup_expr.refine_to_default(epsilon.clone());
    let warmup_ok = warmup_result.is_ok();

    // Timed runs
    let mut total = Duration::ZERO;
    for _ in 0..TIMING_ITERATIONS {
        let expr = make_expr();
        let start = Instant::now();
        let result = expr.refine_to_default(epsilon.clone());
        total += start.elapsed();

        if let Ok(bounds) = &result {
            // Sanity check: bounds should be finite
            assert!(
                try_finite_bounds(bounds).is_some(),
                "{}: expected finite bounds",
                label
            );
        }
    }

    let avg = total / TIMING_ITERATIONS;
    println!(
        "  epsilon=2^-{:>2}: avg {:>12?}  (warmup {})",
        precision_bits,
        avg,
        if warmup_ok { "ok" } else { "FAILED" }
    );
    avg
}

pub fn run_demand_benchmark() {
    println!("== Asymmetric Convergence Benchmark ==");
    println!();
    println!("Expressions where one refiner converges much faster than another.");
    println!("With demand propagation, the fast refiner would stop being stepped");
    println!("once its precision exceeds its budget, saving wasted computation.");
    println!();

    // --- Pathological cases: asymmetric convergence ---

    println!("--- sqrt(2) + pi() ---");
    println!("  PiOp converges exponentially (term doubling); NthRootOp linearly (bisection).");
    println!("  PiOp is stepped uselessly after ~2 rounds, with growing compute cost.");
    for &bits in PRECISION_BITS {
        bench_expression("sqrt(2)+pi()", &|| sqrt_2() + pi(), bits);
    }
    println!();

    println!("--- sqrt(2) * pi() ---");
    println!("  Same asymmetry in multiplication.");
    for &bits in PRECISION_BITS {
        bench_expression("sqrt(2)*pi()", &|| sqrt_2() * pi(), bits);
    }
    println!();

    // --- Controls: single-refiner baselines ---

    println!("--- sqrt(2) + constant(3) ---");
    println!("  Control: only NthRootOp refines (constant's BaseOp converges immediately).");
    for &bits in PRECISION_BITS {
        bench_expression("sqrt(2)+const(3)", &|| sqrt_2() + constant(3), bits);
    }
    println!();

    println!("--- pi() + constant(1) ---");
    println!("  Control: only PiOp refines.");
    for &bits in PRECISION_BITS {
        bench_expression("pi()+const(1)", &|| pi() + constant(1), bits);
    }
    println!();

    // --- Control: symmetric convergence ---

    println!("--- sqrt(2) + cbrt(3) ---");
    println!("  Control: both use bisection (similar convergence rates).");
    for &bits in PRECISION_BITS {
        bench_expression("sqrt(2)+cbrt(3)", &|| sqrt_2() + sqrt_3(), bits);
    }
    println!();

    // --- Analysis ---
    println!("If sqrt(2)+pi() >> sqrt(2)+const(3) + pi()+const(1), the excess is");
    println!("wasted PiOp computation that demand propagation would eliminate.");
}
