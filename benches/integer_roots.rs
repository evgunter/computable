mod common;

use std::num::NonZeroU32;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, verbose};
use computable::{Binary, Computable, UBinary};

const SAMPLE_COUNT: usize = 1_000;

fn build_terms(inputs: &[(u64, NonZeroU32)]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&(value, n)| {
            let value_binary = Binary::new(BigInt::from(value), BigInt::from(0));
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

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(0));

    if verbose() {
        let bounds = balanced_sum(build_terms(&inputs))
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        eprintln!("[integer_roots] width: {}", bounds.width());
    }

    c.bench_function("integer_roots", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            black_box(
                balanced_sum(build_terms(&inputs))
                    .refine_to_default(epsilon.clone())
                    .expect("refine_to should succeed"),
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_integer_roots
}
criterion_main!(benches);
