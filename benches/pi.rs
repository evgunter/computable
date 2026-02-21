mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_bigint::{BigInt, BigUint};

use common::verbose;
use computable::{Binary, Computable, UBinary, pi, pi_bounds_at_precision};

const PRECISION_BITS: &[u64] = &[32, 64, 128, 256, 512, 1024];
const HIGH_PRECISION_BITS: &[u64] = &[2048, 4096, 8192];

fn bench_pi_refinement(c: &mut Criterion) {
    if verbose() {
        for &bits in PRECISION_BITS {
            let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));
            let bounds = pi()
                .refine_to_default(epsilon)
                .expect("pi refinement should succeed");
            eprintln!("[pi/refinement/{bits}] width: {}", bounds.width());
        }
    }

    let mut group = c.benchmark_group("pi/refinement");
    group.sample_size(10);

    for &bits in PRECISION_BITS {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));
        group.bench_with_input(BenchmarkId::from_parameter(bits), &epsilon, |b, epsilon| {
            b.iter(|| {
                black_box(
                    pi().refine_to_default(epsilon.clone())
                        .expect("pi refinement should succeed"),
                )
            })
        });
    }

    group.finish();
}

fn bench_pi_bounds_at_precision(c: &mut Criterion) {
    if verbose() {
        for &bits in PRECISION_BITS {
            let (lower, upper) = pi_bounds_at_precision(bits);
            eprintln!(
                "[pi/bounds_at_precision/{bits}] width: {}",
                upper.sub(&lower)
            );
        }
    }

    let mut group = c.benchmark_group("pi/bounds_at_precision");
    group.sample_size(10);

    for &bits in PRECISION_BITS {
        group.bench_with_input(BenchmarkId::from_parameter(bits), &bits, |b, &bits| {
            b.iter(|| black_box(pi_bounds_at_precision(bits)))
        });
    }

    group.finish();
}

fn bench_pi_arithmetic(c: &mut Criterion) {
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-64));

    let mut group = c.benchmark_group("pi/arithmetic");
    group.sample_size(10);

    group.bench_function("2pi", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            let two = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
            black_box(
                (two * pi())
                    .refine_to_default(epsilon.clone())
                    .expect("2pi should succeed"),
            )
        })
    });

    group.bench_function("pi/2", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            let half = Computable::constant(Binary::new(BigInt::from(1), BigInt::from(-1)));
            black_box(
                (half * pi())
                    .refine_to_default(epsilon.clone())
                    .expect("pi/2 should succeed"),
            )
        })
    });

    group.bench_function("pi^2", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            black_box(
                (pi() * pi())
                    .refine_to_default(epsilon.clone())
                    .expect("pi^2 should succeed"),
            )
        })
    });

    group.bench_function("1/pi", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            black_box(
                pi().inv()
                    .refine_to_default(epsilon.clone())
                    .expect("1/pi should succeed"),
            )
        })
    });

    group.finish();
}

fn bench_sin_pi(c: &mut Criterion) {
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-32));

    let mut group = c.benchmark_group("pi/sin_multiples");
    group.sample_size(10);

    for &multiplier in &[1u64, 2, 10, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(multiplier),
            &multiplier,
            |b, &multiplier| {
                let epsilon = epsilon.clone();
                b.iter(|| {
                    let n_pi = if multiplier == 1 {
                        pi()
                    } else {
                        Computable::constant(Binary::new(BigInt::from(multiplier), BigInt::from(0)))
                            * pi()
                    };
                    black_box(
                        n_pi.sin()
                            .refine_to_default(epsilon.clone())
                            .expect("sin(n*pi) should succeed"),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_pi_high_precision(c: &mut Criterion) {
    if verbose() {
        for &bits in HIGH_PRECISION_BITS {
            let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));
            let bounds = pi()
                .refine_to_default(epsilon)
                .expect("high precision pi should succeed");
            eprintln!("[pi/high_precision/{bits}] width: {}", bounds.width());
        }
    }

    let mut group = c.benchmark_group("pi/high_precision");
    group.sample_size(10);

    for &bits in HIGH_PRECISION_BITS {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));
        group.bench_with_input(BenchmarkId::from_parameter(bits), &epsilon, |b, epsilon| {
            b.iter(|| {
                black_box(
                    pi().refine_to_default(epsilon.clone())
                        .expect("high precision pi should succeed"),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pi_refinement,
    bench_pi_bounds_at_precision,
    bench_pi_arithmetic,
    bench_sin_pi,
    bench_pi_high_precision,
);
criterion_main!(benches);
