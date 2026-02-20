mod common;

use std::num::NonZeroU32;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::balanced_sum;
use computable::{Binary, Computable, UBinary};

const SAMPLE_COUNT: usize = 1_000;

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

    let mut group = c.benchmark_group("integer_roots");
    group.sample_size(10);

    group.bench_function("float", |b| {
        b.iter(|| {
            let mut total = 0.0f64;
            for &(value, n) in &inputs {
                total += (value as f64).powf(1.0 / n.get() as f64);
            }
            black_box(total)
        })
    });

    group.bench_function("computable", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            let terms: Vec<Computable> = inputs
                .iter()
                .map(|&(value, n)| {
                    let value_binary = Binary::new(BigInt::from(value), BigInt::from(0));
                    Computable::constant(value_binary).nth_root(n)
                })
                .collect();
            let total = balanced_sum(terms);
            black_box(
                total
                    .refine_to_default(epsilon.clone())
                    .expect("refine_to should succeed"),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_integer_roots);
criterion_main!(benches);
