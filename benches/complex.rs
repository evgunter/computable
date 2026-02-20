mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, binary_from_f64};
use computable::Computable;

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

fn bench_complex(c: &mut Criterion) {
    let inputs = generate_inputs();

    let mut group = c.benchmark_group("complex");
    group.sample_size(10);

    group.bench_function("float", |b| {
        b.iter(|| {
            let mut total = 0.0f64;
            for &(a, bv, cv, d) in &inputs {
                let mixed = (a + bv) * (cv - d);
                let squared = cv * cv + d * d;
                total += a * bv + squared + mixed;
            }
            black_box(total)
        })
    });

    group.bench_function("computable", |b| {
        b.iter(|| {
            let mut terms = Vec::with_capacity(inputs.len());
            for &(a, bv, cv, d) in &inputs {
                let a_c = Computable::constant(binary_from_f64(a));
                let b_c = Computable::constant(binary_from_f64(bv));
                let c_c = Computable::constant(binary_from_f64(cv));
                let d_c = Computable::constant(binary_from_f64(d));

                let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
                let squared = c_c.clone() * c_c + d_c.clone() * d_c;
                let term = a_c * b_c + squared + mixed;
                terms.push(term);
            }
            let total = balanced_sum(terms);
            black_box(total.bounds().expect("bounds should succeed"))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_complex);
criterion_main!(benches);
