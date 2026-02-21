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

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_bigint::{BigInt, BigUint};

use computable::{Binary, Computable, UBinary, pi};

/// Precision targets (in bits). Kept small because PiOp's exponential term
/// growth makes high precision prohibitively slow in the pathological cases.
const PRECISION_BITS: &[u64] = &[4, 6];

fn sqrt_2() -> Computable {
    Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)))
        .nth_root(NonZeroU32::new(2).unwrap())
}

fn cbrt_3() -> Computable {
    Computable::constant(Binary::new(BigInt::from(3), BigInt::from(0)))
        .nth_root(NonZeroU32::new(3).unwrap())
}

fn constant(value: i64) -> Computable {
    Computable::constant(Binary::new(BigInt::from(value), BigInt::from(0)))
}

fn bench_asymmetric(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_convergence/mixed");
    group.sample_size(10);

    for &bits in PRECISION_BITS {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));

        group.bench_with_input(
            BenchmarkId::new("sqrt2+pi", bits),
            &epsilon,
            |b, epsilon| {
                b.iter(|| {
                    black_box(
                        (sqrt_2() + pi())
                            .refine_to_default(epsilon.clone())
                            .expect("should succeed"),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqrt2*pi", bits),
            &epsilon,
            |b, epsilon| {
                b.iter(|| {
                    black_box(
                        (sqrt_2() * pi())
                            .refine_to_default(epsilon.clone())
                            .expect("should succeed"),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_controls(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_convergence/controls");
    group.sample_size(10);

    for &bits in PRECISION_BITS {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));

        // Single-refiner baselines
        group.bench_with_input(
            BenchmarkId::new("sqrt2+const3", bits),
            &epsilon,
            |b, epsilon| {
                b.iter(|| {
                    black_box(
                        (sqrt_2() + constant(3))
                            .refine_to_default(epsilon.clone())
                            .expect("should succeed"),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pi+const1", bits),
            &epsilon,
            |b, epsilon| {
                b.iter(|| {
                    black_box(
                        (pi() + constant(1))
                            .refine_to_default(epsilon.clone())
                            .expect("should succeed"),
                    )
                })
            },
        );

        // Symmetric convergence control
        group.bench_with_input(
            BenchmarkId::new("sqrt2+cbrt3", bits),
            &epsilon,
            |b, epsilon| {
                b.iter(|| {
                    black_box(
                        (sqrt_2() + cbrt_3())
                            .refine_to_default(epsilon.clone())
                            .expect("should succeed"),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_asymmetric, bench_controls);
criterion_main!(benches);
