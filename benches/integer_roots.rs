mod common;

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;
use std::num::NonZeroU32;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, bench_group, bench_main, epsilon};
#[cfg(not(feature = "criterion-bench"))]
use computable::Prefix;
use computable::{Binary, Computable};

const SAMPLE_COUNT: usize = 1_000_usize;

bench_group! {
    name: integer_roots,
    fn bench_integer_roots(bits) -> Prefix {
        let mut rng = StdRng::seed_from_u64(7);
        let terms: Vec<Computable> = (0..SAMPLE_COUNT)
            .map(|i| {
                let value = u64::from(rng.gen_range(2_u32..1000_u32));
                let n = NonZeroU32::new(u32::try_from(i % 5).unwrap().checked_add(2_u32).unwrap())
                    .expect("root degree 2-6 is non-zero");
                let value_binary = Binary::new(
                    num_bigint::BigInt::from(value),
                    num_bigint::BigInt::from(0_i64),
                );
                Computable::constant(value_binary).nth_root(n)
            })
            .collect();

        black_box(
            balanced_sum(terms)
                .refine_to_default(epsilon(bits))
                .expect("should succeed"),
        )
    }
}

bench_main!(integer_roots; valgrind_args: ["--max-threads=2500"]);
