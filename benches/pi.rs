mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;

use common::{bench_id, bench_id_named, epsilon, precision_bits, verbose};
use computable::{Binary, Computable, pi, pi_bounds_at_precision};

fn bench_pi_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi/refinement");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        if verbose() {
            let bounds = pi()
                .refine_to_default(eps.clone())
                .expect("pi refinement should succeed");
            eprintln!("[pi/refinement/{bits}] width: {}", bounds.width());
        }

        group.bench_with_input(bench_id(bits), &eps, |b, eps| {
            b.iter(|| {
                black_box(
                    pi().refine_to_default(eps.clone())
                        .expect("pi refinement should succeed"),
                )
            })
        });
    }

    group.finish();
}

/// Note: unlike other benchmarks where the parameter is an epsilon (2^-bits),
/// here it is passed directly to `pi_bounds_at_precision` as the number of bits
/// of precision to compute via Machin's formula (single-shot, no iterative
/// refinement). The values are the same sweep but the interpretation differs.
fn bench_pi_bounds_at_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi/bounds_at_precision");
    group.sample_size(10);

    for &bits in precision_bits() {
        let bits_usize = bits as usize;

        if verbose() {
            let (lower, upper) = pi_bounds_at_precision(bits_usize);
            eprintln!(
                "[pi/bounds_at_precision/{bits}] width: {}",
                upper.sub(&lower)
            );
        }

        group.bench_with_input(bench_id(bits), &bits_usize, |b, &bits_usize| {
            b.iter(|| black_box(pi_bounds_at_precision(bits_usize)))
        });
    }

    group.finish();
}

fn bench_pi_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi/arithmetic");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        group.bench_with_input(bench_id_named("2pi", bits), &eps, |b, eps| {
            b.iter(|| {
                let two = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
                black_box(
                    (two * pi())
                        .refine_to_default(eps.clone())
                        .expect("2pi should succeed"),
                )
            })
        });

        group.bench_with_input(bench_id_named("pi_half", bits), &eps, |b, eps| {
            b.iter(|| {
                let half = Computable::constant(Binary::new(BigInt::from(1), BigInt::from(-1)));
                black_box(
                    (half * pi())
                        .refine_to_default(eps.clone())
                        .expect("pi/2 should succeed"),
                )
            })
        });

        group.bench_with_input(bench_id_named("pi_sq", bits), &eps, |b, eps| {
            b.iter(|| {
                black_box(
                    (pi() * pi())
                        .refine_to_default(eps.clone())
                        .expect("pi^2 should succeed"),
                )
            })
        });

        group.bench_with_input(bench_id_named("inv_pi", bits), &eps, |b, eps| {
            b.iter(|| {
                black_box(
                    pi().inv()
                        .refine_to_default(eps.clone())
                        .expect("1/pi should succeed"),
                )
            })
        });
    }

    group.finish();
}

fn bench_sin_pi(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi/sin_multiples");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        for &multiplier in &[1u64, 2, 10, 100] {
            group.bench_with_input(
                bench_id_named(format!("mul_{multiplier}"), bits),
                &eps,
                |b, eps| {
                    b.iter(|| {
                        let n_pi = if multiplier == 1 {
                            pi()
                        } else {
                            Computable::constant(Binary::new(
                                BigInt::from(multiplier),
                                BigInt::from(0),
                            )) * pi()
                        };
                        black_box(
                            n_pi.sin()
                                .refine_to_default(eps.clone())
                                .expect("sin(n*pi) should succeed"),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pi_refinement,
    bench_pi_bounds_at_precision,
    bench_pi_arithmetic,
    bench_sin_pi,
);
criterion_main!(benches);
