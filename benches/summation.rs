mod common;

use std::hint::black_box;

use gungraun::{LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon};
use computable::{Binary, Bounds, Computable};

const SAMPLE_COUNT: usize = 200_000_usize;

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_summation(bits: usize) -> Bounds {
    let mut rng = StdRng::seed_from_u64(7);
    let base = f64::from(2_i32.pow(30));
    let mut terms = Vec::with_capacity(SAMPLE_COUNT + 1_usize);
    terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
    for _ in 0..SAMPLE_COUNT {
        let v = rng.gen_range(-1.0e-6_f64..1.0e-6_f64);
        terms.push(Computable::constant(Binary::from_f64(v).unwrap()));
    }

    black_box(
        balanced_sum(terms)
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

library_benchmark_group!(name = summation, benchmarks = [bench_summation]);

// Each constant spawns a refiner thread; raise valgrind's default limit
// of 500 to accommodate 200k+ terms.
main!(
    config = LibraryBenchmarkConfig::default().valgrind_args(["--max-threads=210000"]);
    library_benchmark_groups = summation
);
