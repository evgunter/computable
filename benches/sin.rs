mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, verbose};
use computable::{Binary, Computable, UBinary};

const SAMPLE_COUNT: usize = 100;
const PRECISION_BITS: i64 = 128;

fn build_terms(inputs: &[f64]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).sin())
        .collect()
}

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

    if verbose() {
        let bounds = balanced_sum(build_terms(&inputs))
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        eprintln!("[sin] width: {}", bounds.width());
    }

    c.bench_function("sin", |b| {
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
    targets = bench_sin
}
criterion_main!(benches);
