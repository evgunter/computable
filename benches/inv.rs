mod bench_macros;
mod common;

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use bench_macros::{bench_group, bench_main, epsilon};
use common::balanced_sum;
#[cfg(not(feature = "criterion-bench"))]
use computable::Prefix;
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 100;

bench_group! {
    name: inv,
    fn bench_inv(bits) -> Prefix {
        let mut rng = StdRng::seed_from_u64(7);
        let terms: Vec<Computable> = (0..SAMPLE_COUNT)
            .map(|_| {
                let x = rng.gen_range(0.1_f64..100.0_f64);
                Computable::constant(Binary::from_f64(x).unwrap()).inv()
            })
            .collect();

        black_box(
            balanced_sum(terms)
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
}

bench_main!(inv);
