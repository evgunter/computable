mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, verbose};
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

    if verbose() {
        let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
        terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
        terms.extend(computable_inputs.iter().cloned());
        let bounds = balanced_sum(terms).bounds().expect("bounds should succeed");
        eprintln!("[summation] width: {}", bounds.width());
    }

    c.bench_function("summation", |b| {
        b.iter(|| {
            let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
            terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
            terms.extend(computable_inputs.iter().cloned());
            black_box(balanced_sum(terms).bounds().expect("bounds should succeed"))
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_summation
}
criterion_main!(benches);
