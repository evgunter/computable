mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::{BigInt, BigUint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, verbose};
use computable::{Binary, Computable, UBinary};

const SAMPLE_COUNT: usize = 100;
const PRECISION_BITS: i64 = 256;

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

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-PRECISION_BITS));

    if verbose() {
        let bounds = balanced_sum(build_terms(&inputs))
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        eprintln!("[inv] width: {}", bounds.width());
    }

    c.bench_function("inv", |b| {
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
    targets = bench_inv
}
criterion_main!(benches);
