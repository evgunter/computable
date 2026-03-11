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

const SAMPLE_COUNT: usize = 200_000_usize;

// No precision sweep: all inputs are exact constants and addition preserves
// exactness, so refinement is a no-op regardless of the requested tolerance.
bench_group! {
    name: summation,
    fn bench_summation() -> Prefix {
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
                .refine_to_default(epsilon(1))
                .expect("should succeed"),
        )
    }
}

bench_main!(summation; valgrind_args: ["--max-threads=210000"]);
