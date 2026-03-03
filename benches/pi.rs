mod common;

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use num_bigint::BigInt;

use common::epsilon;
use computable::{Binary, Bounds, Computable, pi, pi_bounds_at_precision};

// --- pi refinement ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_pi_refinement(bits: usize) -> Bounds {
    black_box(
        pi().refine_to_default(epsilon(bits))
            .expect("pi refinement should succeed"),
    )
}

library_benchmark_group!(name = pi_refinement, benchmarks = [bench_pi_refinement]);

// --- pi bounds at precision ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_pi_bounds(bits: usize) -> (Binary, Binary) {
    black_box(pi_bounds_at_precision(bits))
}

library_benchmark_group!(name = pi_bounds, benchmarks = [bench_pi_bounds]);

// --- pi arithmetic ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_two_pi(bits: usize) -> Bounds {
    let two = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)));
    black_box(
        (two * pi())
            .refine_to_default(epsilon(bits))
            .expect("2pi should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_pi_half(bits: usize) -> Bounds {
    let half = Computable::constant(Binary::new(BigInt::from(1_i64), BigInt::from(-1_i64)));
    black_box(
        (half * pi())
            .refine_to_default(epsilon(bits))
            .expect("pi/2 should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_pi_squared(bits: usize) -> Bounds {
    black_box(
        (pi() * pi())
            .refine_to_default(epsilon(bits))
            .expect("pi^2 should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_inv_pi(bits: usize) -> Bounds {
    black_box(
        pi().inv()
            .refine_to_default(epsilon(bits))
            .expect("1/pi should succeed"),
    )
}

library_benchmark_group!(
    name = pi_arithmetic,
    benchmarks = [bench_two_pi, bench_pi_half, bench_pi_squared, bench_inv_pi]
);

// --- sin(n*pi) ---

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sin_1pi(bits: usize) -> Bounds {
    black_box(
        pi().sin()
            .refine_to_default(epsilon(bits))
            .expect("sin(pi) should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sin_2pi(bits: usize) -> Bounds {
    let n_pi = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64))) * pi();
    black_box(
        n_pi.sin()
            .refine_to_default(epsilon(bits))
            .expect("sin(2*pi) should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sin_10pi(bits: usize) -> Bounds {
    let n_pi = Computable::constant(Binary::new(BigInt::from(10_i64), BigInt::from(0_i64))) * pi();
    black_box(
        n_pi.sin()
            .refine_to_default(epsilon(bits))
            .expect("sin(10*pi) should succeed"),
    )
}

#[library_benchmark]
#[benches::precision(args = [1_usize, 4, 16, 64, 256])]
fn bench_sin_100pi(bits: usize) -> Bounds {
    let n_pi = Computable::constant(Binary::new(BigInt::from(100_i64), BigInt::from(0_i64))) * pi();
    black_box(
        n_pi.sin()
            .refine_to_default(epsilon(bits))
            .expect("sin(100*pi) should succeed"),
    )
}

library_benchmark_group!(
    name = sin_multiples,
    benchmarks = [
        bench_sin_1pi,
        bench_sin_2pi,
        bench_sin_10pi,
        bench_sin_100pi
    ]
);

main!(
    library_benchmark_groups = pi_refinement,
    pi_bounds,
    pi_arithmetic,
    sin_multiples
);
