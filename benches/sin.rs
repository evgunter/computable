mod common;

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon};
use computable::{Binary, Bounds, Computable};

const SAMPLE_COUNT: usize = 100;

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sin(bits: usize) -> Bounds {
    let mut rng = StdRng::seed_from_u64(7);
    let terms: Vec<Computable> = (0..SAMPLE_COUNT)
        .map(|i| {
            let x = if i < SAMPLE_COUNT / 3 {
                rng.gen_range(-1.0_f64..1.0_f64)
            } else if i < 2 * SAMPLE_COUNT / 3 {
                rng.gen_range(-3.15_f64..3.15_f64)
            } else {
                rng.gen_range(-100.0_f64..100.0_f64)
            };
            Computable::constant(Binary::from_f64(x).unwrap()).sin()
        })
        .collect();

    black_box(
        balanced_sum(terms)
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

library_benchmark_group!(name = sin, benchmarks = [bench_sin]);

main!(library_benchmark_groups = sin);
