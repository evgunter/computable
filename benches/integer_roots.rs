mod common;

use std::hint::black_box;
use std::num::NonZeroU32;

use gungraun::{library_benchmark, library_benchmark_group, main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::{balanced_sum, epsilon};
use computable::{Binary, Bounds, Computable};

// Keep low: each nth_root + constant pair spawns 2 refiner threads under
// valgrind, and valgrind defaults to ~500 max threads.
const SAMPLE_COUNT: usize = 50_usize;

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_integer_roots(bits: usize) -> Bounds {
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

library_benchmark_group!(name = integer_roots, benchmarks = [bench_integer_roots]);

main!(library_benchmark_groups = integer_roots);
