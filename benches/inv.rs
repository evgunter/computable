mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, bench_id, epsilon, precision_bits, verbose};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 100;

fn build_terms(inputs: &[f64]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).inv())
        .collect()
}

fn bench_inv(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|_| rng.gen_range(0.1..100.0))
        .collect();

    let mut group = c.benchmark_group("inv");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        if verbose() {
            let bounds = balanced_sum(build_terms(&inputs))
                .refine_to_default(eps)
                .expect("refine_to should succeed");
            eprintln!("[inv/{bits}] width: {}", bounds.width());
        }

        group.bench_with_input(bench_id(bits), &eps, |b, eps| {
            b.iter(|| {
                black_box(
                    balanced_sum(build_terms(&inputs))
                        .refine_to_default(*eps)
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
    targets = bench_inv
}
criterion_main!(benches);
