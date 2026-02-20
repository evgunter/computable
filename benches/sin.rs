mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, binary_from_f64};
use computable::{Computable, UBinary};

const SAMPLE_COUNT: usize = 100;
const PRECISION_BITS: i64 = 128;

fn bench_sin(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|i| {
            if i < SAMPLE_COUNT / 3 {
                rng.gen_range(-1.0..1.0)
            } else if i < 2 * SAMPLE_COUNT / 3 {
                rng.gen_range(-3.15..3.15)
            } else {
                rng.gen_range(-100.0..100.0)
            }
        })
        .collect();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-PRECISION_BITS));

    let mut group = c.benchmark_group("sin");
    group.sample_size(10);

    group.bench_function("float", |b| {
        b.iter(|| {
            let sum: f64 = inputs.iter().map(|x| x.sin()).sum();
            black_box(sum)
        })
    });

    group.bench_function("computable", |b| {
        let epsilon = epsilon.clone();
        b.iter(|| {
            let terms: Vec<Computable> = inputs
                .iter()
                .map(|&x| Computable::constant(binary_from_f64(x)).sin())
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

criterion_group!(benches, bench_sin);
criterion_main!(benches);
