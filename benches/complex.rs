mod common;

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon};
use computable::{Binary, Bounds, Computable};

const SAMPLE_COUNT: usize = 5_000;

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_complex(bits: usize) -> Bounds {
    let mut rng = StdRng::seed_from_u64(7);
    let terms: Vec<Computable> = (0..SAMPLE_COUNT)
        .map(|_| {
            let a = rng.gen_range(-10.0_f64..10.0_f64);
            let bv = rng.gen_range(-10.0_f64..10.0_f64);
            let cv = rng.gen_range(-10.0_f64..10.0_f64);
            let d = rng.gen_range(-10.0_f64..10.0_f64);

            let a_c = Computable::constant(Binary::from_f64(a).unwrap());
            let b_c = Computable::constant(Binary::from_f64(bv).unwrap());
            let c_c = Computable::constant(Binary::from_f64(cv).unwrap());
            let d_c = Computable::constant(Binary::from_f64(d).unwrap());

            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();

    black_box(
        balanced_sum(terms)
            .refine_to_default(epsilon(bits))
            .expect("should succeed"),
    )
}

library_benchmark_group!(name = complex, benchmarks = [bench_complex]);

main!(library_benchmark_groups = complex);
