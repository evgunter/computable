mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, verbose};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 5_000;

fn generate_inputs() -> Vec<(f64, f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..SAMPLE_COUNT)
        .map(|_| {
            (
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            )
        })
        .collect()
}

fn build_terms(inputs: &[(f64, f64, f64, f64)]) -> Vec<Computable> {
    inputs
        .iter()
        .map(|&(a, bv, cv, d)| {
            let a_c = Computable::constant(Binary::from_f64(a).unwrap());
            let b_c = Computable::constant(Binary::from_f64(bv).unwrap());
            let c_c = Computable::constant(Binary::from_f64(cv).unwrap());
            let d_c = Computable::constant(Binary::from_f64(d).unwrap());

            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect()
}

fn bench_complex(c: &mut Criterion) {
    let inputs = generate_inputs();

    if verbose() {
        let bounds = balanced_sum(build_terms(&inputs))
            .bounds()
            .expect("bounds should succeed");
        eprintln!("[complex] width: {}", bounds.width());
    }

    c.bench_function("complex", |b| {
        b.iter(|| {
            black_box(
                balanced_sum(build_terms(&inputs))
                    .bounds()
                    .expect("bounds should succeed"),
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_complex
}
criterion_main!(benches);
