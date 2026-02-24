mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon, precision_bits, verbose};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 5_000;

fn generate_inputs() -> Vec<(f64, f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..SAMPLE_COUNT)
        .map(|_| {
            (
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            )
        })
        .collect()
}

fn build_terms(inputs: &[(f64, f64, f64, f64)]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&(a, bv, cv, d)| {
            let a_c = Computable::constant(Binary::from_f64(a).unwrap());
            let b_c = Computable::constant(Binary::from_f64(bv).unwrap());
            let c_c = Computable::constant(Binary::from_f64(cv).unwrap());
            let d_c = Computable::constant(Binary::from_f64(d).unwrap());

            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect()
}

fn bench_complex(c: &mut Criterion) {
    let inputs = generate_inputs();

    let mut group = c.benchmark_group("complex");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        if verbose() {
            let bounds = balanced_sum(build_terms(&inputs))
                .refine_to_default(eps.clone())
                .expect("refine_to should succeed");
            eprintln!("[complex/{bits}] width: {}", bounds.width());
        }

        group.bench_with_input(BenchmarkId::from_parameter(bits), &eps, |b, eps| {
            b.iter(|| {
                black_box(
                    balanced_sum(build_terms(&inputs))
                        .refine_to_default(eps.clone())
                        .expect("refine_to should succeed"),
                )
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_complex
}
criterion_main!(benches);
