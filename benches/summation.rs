mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon, precision_bits, verbose};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 200_000;

fn bench_summation(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let float_inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let computable_inputs: Vec<Computable> = float_inputs
        .iter()
        .map(|&v| Computable::constant(Binary::from_f64(v).unwrap()))
        .collect();
    let base = 2_i64.pow(30) as f64;

    let mut group = c.benchmark_group("summation");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        if verbose() {
            let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
            terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
            terms.extend(computable_inputs.iter().cloned());
            let bounds = balanced_sum(terms)
                .refine_to_default(eps.clone())
                .expect("refine_to should succeed");
            eprintln!("[summation/{bits}] width: {}", bounds.width());
        }

        group.bench_with_input(BenchmarkId::from_parameter(bits), &eps, |b, eps| {
            b.iter(|| {
                let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
                terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
                terms.extend(computable_inputs.iter().cloned());
                black_box(
                    balanced_sum(terms)
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
    targets = bench_summation
}
criterion_main!(benches);
