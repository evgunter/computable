mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, binary_from_f64};
use computable::Computable;

const SAMPLE_COUNT: usize = 200_000;

fn bench_summation(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let float_inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let computable_inputs: Vec<Computable> = float_inputs
        .iter()
        .map(|&v| Computable::constant(binary_from_f64(v)))
        .collect();
    let base = 2_i64.pow(30) as f64;

    let mut group = c.benchmark_group("summation");
    group.sample_size(10);

    group.bench_function("float", |b| {
        b.iter(|| {
            let mut total = base;
            for &value in &float_inputs {
                total += value;
            }
            black_box(total)
        })
    });

    group.bench_function("computable", |b| {
        b.iter(|| {
            let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
            terms.push(Computable::constant(binary_from_f64(base)));
            terms.extend(computable_inputs.iter().cloned());
            let total = balanced_sum(terms);
            black_box(total.bounds().expect("bounds should succeed"))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_summation);
criterion_main!(benches);
