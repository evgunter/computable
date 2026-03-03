mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon};
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 200_000_usize;

fn build_summation() -> Computable {
    let mut rng = StdRng::seed_from_u64(7);
    let base = f64::from(2_i32.pow(30));
    let mut terms = Vec::with_capacity(SAMPLE_COUNT + 1_usize);
    terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
    for _ in 0..SAMPLE_COUNT {
        let v = rng.gen_range(-1.0e-6_f64..1.0e-6_f64);
        terms.push(Computable::constant(Binary::from_f64(v).unwrap()));
    }
    balanced_sum(terms)
}

fn bench_summation(c: &mut Criterion) {
    let mut group = c.benchmark_group("summation");
    group.sample_size(10);
    for bits in [1_usize, 4] {
        group.bench_function(format!("bits={bits}"), |b| {
            b.iter_with_setup(build_summation, |expr| {
                expr.refine_to_default(epsilon(bits))
                    .expect("should succeed")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_summation);
criterion_main!(benches);
