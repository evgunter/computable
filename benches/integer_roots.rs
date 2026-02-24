mod common;

use std::num::NonZeroU32;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon, precision_bits, verbose};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 1_000;

fn build_terms(inputs: &[(u64, NonZeroU32)]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&(value, n)| {
            let value_binary =
                Binary::new(num_bigint::BigInt::from(value), num_bigint::BigInt::from(0));
            Computable::constant(value_binary).nth_root(n)
        })
        .collect()
}

fn bench_integer_roots(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let inputs: Vec<(u64, NonZeroU32)> = (0..SAMPLE_COUNT)
        .map(|i| {
            let value = rng.gen_range(2..1000) as u64;
            let n = NonZeroU32::new((i % 5) as u32 + 2).expect("root degree 2-6 is non-zero");
            (value, n)
        })
        .collect();

    let mut group = c.benchmark_group("integer_roots");
    group.sample_size(10);

    for &bits in precision_bits() {
        let eps = epsilon(bits);

        if verbose() {
            let bounds = balanced_sum(build_terms(&inputs))
                .refine_to_default(eps.clone())
                .expect("refine_to should succeed");
            eprintln!("[integer_roots/{bits}] width: {}", bounds.width());
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
    targets = bench_integer_roots
}
criterion_main!(benches);
