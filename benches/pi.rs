mod common;

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;

use num_bigint::BigInt;

use common::{bench_group, bench_main, epsilon};
#[cfg(not(feature = "criterion-bench"))]
use computable::Prefix;
use computable::{Binary, Computable, pi, pi_bounds_at_precision};

bench_group! {
    name: pi_refinement,
    fn bench_pi_refinement(bits) -> Prefix {
        black_box(
            pi().refine_to_default(epsilon(bits))
                .expect("pi refinement should succeed"),
        )
    }
}

bench_group! {
    name: pi_bounds,
    fn bench_pi_bounds(bits) -> (Binary, Binary) {
        black_box(pi_bounds_at_precision(bits))
    }
}

bench_group! {
    name: pi_arithmetic,
    fn bench_two_pi(bits) -> Prefix {
        let two = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)));
        black_box(
            (two * pi())
                .refine_to_default(epsilon(bits))
                .expect("2pi should succeed"),
        )
    }
    fn bench_pi_half(bits) -> Prefix {
        let half = Computable::constant(Binary::new(BigInt::from(1_i64), BigInt::from(-1_i64)));
        black_box(
            (half * pi())
                .refine_to_default(epsilon(bits))
                .expect("pi/2 should succeed"),
        )
    }
    fn bench_pi_squared(bits) -> Prefix {
        black_box(
            (pi() * pi())
                .refine_to_default(epsilon(bits))
                .expect("pi^2 should succeed"),
        )
    }
    fn bench_inv_pi(bits) -> Prefix {
        black_box(
            pi().inv()
                .refine_to_default(epsilon(bits))
                .expect("1/pi should succeed"),
        )
    }
}

bench_group! {
    name: sin_multiples,
    fn bench_sin_1pi(bits) -> Prefix {
        black_box(
            pi().sin()
                .refine_to_default(epsilon(bits))
                .expect("sin(pi) should succeed"),
        )
    }
    fn bench_sin_2pi(bits) -> Prefix {
        let n_pi = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64))) * pi();
        black_box(
            n_pi.sin()
                .refine_to_default(epsilon(bits))
                .expect("sin(2*pi) should succeed"),
        )
    }
    fn bench_sin_10pi(bits) -> Prefix {
        let n_pi = Computable::constant(Binary::new(BigInt::from(10_i64), BigInt::from(0_i64))) * pi();
        black_box(
            n_pi.sin()
                .refine_to_default(epsilon(bits))
                .expect("sin(10*pi) should succeed"),
        )
    }
    fn bench_sin_100pi(bits) -> Prefix {
        let n_pi = Computable::constant(Binary::new(BigInt::from(100_i64), BigInt::from(0_i64))) * pi();
        black_box(
            n_pi.sin()
                .refine_to_default(epsilon(bits))
                .expect("sin(100*pi) should succeed"),
        )
    }
}

bench_main!(pi_refinement, pi_bounds, pi_arithmetic, sin_multiples);
