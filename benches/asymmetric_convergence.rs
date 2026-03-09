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

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;
use std::num::NonZeroU32;

use num_bigint::BigInt;

use common::{bench_group, bench_main, epsilon};
#[cfg(not(feature = "criterion-bench"))]
use computable::Bounds;
use computable::{Binary, Computable, pi};

fn sqrt_2() -> Computable {
    Computable::constant(Binary::new(BigInt::from(2_i64), 0_i64))
        .nth_root(NonZeroU32::new(2_u32).unwrap())
}

fn cbrt_3() -> Computable {
    Computable::constant(Binary::new(BigInt::from(3_i64), 0_i64))
        .nth_root(NonZeroU32::new(3_u32).unwrap())
}

fn constant(value: i64) -> Computable {
    Computable::constant(Binary::new(BigInt::from(value), 0_i64))
}

// --- mixed (asymmetric) ---

bench_group! {
    name: mixed,
    fn bench_sqrt2_plus_pi(bits) -> Bounds {
        black_box(
            (sqrt_2() + pi())
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
    fn bench_sqrt2_times_pi(bits) -> Bounds {
        black_box(
            (sqrt_2() * pi())
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
}

// --- controls ---

bench_group! {
    name: controls,
    fn bench_sqrt2_plus_const3(bits) -> Bounds {
        black_box(
            (sqrt_2() + constant(3_i64))
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
    fn bench_pi_plus_const1(bits) -> Bounds {
        black_box(
            (pi() + constant(1_i64))
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
    fn bench_sqrt2_plus_cbrt3(bits) -> Bounds {
        black_box(
            (sqrt_2() + cbrt_3())
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
}

bench_main!(mixed, controls);
