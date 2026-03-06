mod common;

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, bench_group, bench_main, epsilon};
#[cfg(not(feature = "criterion-bench"))]
use computable::Bounds;
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 100;

bench_group! {
    name: sin,
    fn bench_sin(bits) -> Bounds {
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
}

bench_main!(sin; valgrind_args: ["--fair-sched=yes"]);
