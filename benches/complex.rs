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
use computable::Bounds;
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 5_000;

// No precision sweep: all inputs are exact constants and arithmetic preserves
// exactness, so refinement is a no-op regardless of the requested tolerance.
bench_group! {
    name: complex,
    fn bench_complex() -> Bounds {
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
                .refine_to_default(epsilon(1))
                .expect("should succeed"),
        )
    }
}

bench_main!(complex);
