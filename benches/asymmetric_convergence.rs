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

mod common;

use std::hint::black_box;
use std::num::NonZeroU32;

use gungraun::{library_benchmark, library_benchmark_group, main};
use num_bigint::BigInt;

use common::epsilon;
use computable::{Binary, Bounds, Computable, pi};

fn sqrt_2() -> Computable {
    Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)))
        .nth_root(NonZeroU32::new(2_u32).unwrap())
}

fn cbrt_3() -> Computable {
    Computable::constant(Binary::new(BigInt::from(3_i64), BigInt::from(0_i64)))
        .nth_root(NonZeroU32::new(3_u32).unwrap())
}

fn constant(value: i64) -> Computable {
    Computable::constant(Binary::new(BigInt::from(value), BigInt::from(0_i64)))
}

// --- mixed (asymmetric) ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sqrt2_plus_pi(bits: usize) -> Bounds {
    black_box(
        (sqrt_2() + pi())
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sqrt2_times_pi(bits: usize) -> Bounds {
    black_box(
        (sqrt_2() * pi())
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

library_benchmark_group!(
    name = mixed,
    benchmarks = [bench_sqrt2_plus_pi, bench_sqrt2_times_pi]
);

// --- controls ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sqrt2_plus_const3(bits: usize) -> Bounds {
    black_box(
        (sqrt_2() + constant(3_i64))
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_pi_plus_const1(bits: usize) -> Bounds {
    black_box(
        (pi() + constant(1_i64))
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sqrt2_plus_cbrt3(bits: usize) -> Bounds {
    black_box(
        (sqrt_2() + cbrt_3())
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

library_benchmark_group!(
    name = controls,
    benchmarks = [
        bench_sqrt2_plus_const3,
        bench_pi_plus_const1,
        bench_sqrt2_plus_cbrt3
    ]
);

main!(library_benchmark_groups = mixed, controls);
