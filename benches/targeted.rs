mod common;

#[cfg(not(feature = "criterion-bench"))]
use gungraun::*;
use std::hint::black_box;
use std::num::NonZeroU32;

use common::{bench_group, epsilon};
#[cfg(not(feature = "criterion-bench"))]
use computable::Bounds;
use computable::{Binary, Computable, pi};
use num_bigint::BigInt;

// ---------------------------------------------------------------------------
// 1. Single sqrt at 64 and 256 bits — Tests NthRoot convergence speed
// ---------------------------------------------------------------------------
bench_group! {
    name: sqrt_convergence,
    fn bench_sqrt2_64() -> Bounds {
        let two = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)));
        black_box(
            two.nth_root(NonZeroU32::new(2).unwrap())
                .refine_to_default(epsilon(64))
                .expect("sqrt(2) at 64 bits should succeed"),
        )
    }
    fn bench_sqrt2_256() -> Bounds {
        let two = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)));
        black_box(
            two.nth_root(NonZeroU32::new(2).unwrap())
                .refine_to_default(epsilon(256))
                .expect("sqrt(2) at 256 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 2. 10 inv values at 64 and 256 bits — Tests Newton-Raphson
// ---------------------------------------------------------------------------
bench_group! {
    name: inv_small,
    fn bench_inv_10_64() -> Bounds {
        let terms: Vec<Computable> = (1..=10)
            .map(|i| {
                let x = 0.1_f64 * f64::from(i) + 0.5_f64;
                Computable::constant(Binary::from_f64(x).unwrap()).inv()
            })
            .collect();
        black_box(
            common::balanced_sum(terms)
                .refine_to_default(epsilon(64))
                .expect("inv 10 at 64 bits should succeed"),
        )
    }
    fn bench_inv_10_256() -> Bounds {
        let terms: Vec<Computable> = (1..=10)
            .map(|i| {
                let x = 0.1_f64 * f64::from(i) + 0.5_f64;
                Computable::constant(Binary::from_f64(x).unwrap()).inv()
            })
            .collect();
        black_box(
            common::balanced_sum(terms)
                .refine_to_default(epsilon(256))
                .expect("inv 10 at 256 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 3. 5 sin values at 64 and 2 sin values at 256 bits — Tests Taylor series
// ---------------------------------------------------------------------------
bench_group! {
    name: sin_small,
    fn bench_sin_5_64() -> Bounds {
        let values = [0.5_f64, 1.0, 1.5, 2.0, 2.5];
        let terms: Vec<Computable> = values
            .iter()
            .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).sin())
            .collect();
        black_box(
            common::balanced_sum(terms)
                .refine_to_default(epsilon(64))
                .expect("sin 5 at 64 bits should succeed"),
        )
    }
    fn bench_sin_2_256() -> Bounds {
        let values = [0.5_f64, 1.5];
        let terms: Vec<Computable> = values
            .iter()
            .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).sin())
            .collect();
        black_box(
            common::balanced_sum(terms)
                .refine_to_default(epsilon(256))
                .expect("sin 2 at 256 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 4. Pi at 64 and 256 bits — Tests Machin formula
// ---------------------------------------------------------------------------
bench_group! {
    name: pi_targeted,
    fn bench_pi_64() -> Bounds {
        black_box(
            pi().refine_to_default(epsilon(64))
                .expect("pi at 64 bits should succeed"),
        )
    }
    fn bench_pi_256() -> Bounds {
        black_box(
            pi().refine_to_default(epsilon(256))
                .expect("pi at 256 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 5. Near-cancellation: (1+epsilon) - 1, epsilon = 2^-50, at 64 bits
//    Tests demand propagation
// ---------------------------------------------------------------------------
bench_group! {
    name: cancellation,
    fn bench_near_cancel_64() -> Bounds {
        let one = Computable::constant(Binary::new(BigInt::from(1_i64), BigInt::from(0_i64)));
        // epsilon = 2^-50 = 1 * 2^(-50)
        let eps = Computable::constant(Binary::new(BigInt::from(1_i64), BigInt::from(-50_i64)));
        let expr = (one.clone() + eps) - one;
        black_box(
            expr.refine_to_default(epsilon(64))
                .expect("near-cancellation at 64 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 6. Deep chain: 10 nested additions a + (b + (c + ...))
//    Tests graph depth
// ---------------------------------------------------------------------------
bench_group! {
    name: deep_chain,
    fn bench_deep_add_chain_64() -> Bounds {
        // Build a right-associative chain of 10 additions
        let values: Vec<Computable> = (1..=10)
            .map(|i| {
                Computable::constant(Binary::from_f64(f64::from(i) * 0.1).unwrap())
            })
            .collect();
        // Fold right: a + (b + (c + ...))
        let expr = values.into_iter().rev().reduce(|acc, x| x + acc).unwrap();
        black_box(
            expr.refine_to_default(epsilon(64))
                .expect("deep chain at 64 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 7. Shared subexpression: (a+b) + (a-b) where a=sqrt(2), b=1
//    Tests shared node handling
// ---------------------------------------------------------------------------
bench_group! {
    name: shared_subexpr,
    fn bench_shared_sqrt2_64() -> Bounds {
        let a = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)))
            .nth_root(NonZeroU32::new(2).unwrap());
        let b = Computable::constant(Binary::new(BigInt::from(1_i64), BigInt::from(0_i64)));
        // (a+b) + (a-b) = 2*a, but exercises shared node handling
        let sum = a.clone() + b.clone();
        let diff = a - b;
        let expr = sum + diff;
        black_box(
            expr.refine_to_default(epsilon(64))
                .expect("shared subexpr at 64 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 8. Sequential refinement: refine to 16, then 64, then 128 bits
//    Tests warm-start
// ---------------------------------------------------------------------------
bench_group! {
    name: sequential_refine,
    fn bench_seq_refine() -> Bounds {
        let expr = pi();

        // Refine progressively: warm-start behavior
        let _r16 = black_box(
            expr.refine_to_default(epsilon(16))
                .expect("seq refine at 16 bits should succeed"),
        );
        let _r64 = black_box(
            expr.refine_to_default(epsilon(64))
                .expect("seq refine at 64 bits should succeed"),
        );
        black_box(
            expr.refine_to_default(epsilon(128))
                .expect("seq refine at 128 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 9. Inv sum: inv(2) + inv(3) + inv(5) + inv(7)
//    Tests multiple InvOp nodes in a single expression
// ---------------------------------------------------------------------------
bench_group! {
    name: inv_sum,
    fn bench_inv_sum_64() -> Bounds {
        let c = |v: i64| Computable::constant(Binary::new(BigInt::from(v), BigInt::from(0_i64)));
        let expr = c(2).inv() + c(3).inv() + c(5).inv() + c(7).inv();
        black_box(
            expr.refine_to_default(epsilon(64))
                .expect("inv sum at 64 bits should succeed"),
        )
    }
}

// ---------------------------------------------------------------------------
// 10. Mixed expression: sqrt(2) + pi() at 64 bits
//     Tests mixed NthRoot, Add, Pi in a single graph
// ---------------------------------------------------------------------------
bench_group! {
    name: mixed_expr,
    fn bench_mixed_64() -> Bounds {
        let sqrt2 = Computable::constant(Binary::new(BigInt::from(2_i64), BigInt::from(0_i64)))
            .nth_root(NonZeroU32::new(2).unwrap());
        let expr = sqrt2 + pi();
        black_box(
            expr.refine_to_default(epsilon(64))
                .expect("mixed expr at 64 bits should succeed"),
        )
    }
}

// Custom main to configure criterion for fast execution (< 30s total).
#[cfg(not(any(feature = "criterion-bench", feature = "time-bench")))]
main!(
    config = LibraryBenchmarkConfig::default()
        .valgrind_args(["--fair-sched=yes"]);
    library_benchmark_groups =
        sqrt_convergence,
        inv_small,
        sin_small,
        pi_targeted,
        cancellation,
        deep_chain,
        shared_subexpr,
        sequential_refine,
        inv_sum,
        mixed_expr
);

#[cfg(feature = "criterion-bench")]
::criterion::criterion_group! {
    name = benches;
    config = ::criterion::Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(200))
        .measurement_time(std::time::Duration::from_millis(500))
        .sample_size(10);
    targets =
        sqrt_convergence,
        inv_small,
        sin_small,
        pi_targeted,
        cancellation,
        deep_chain,
        shared_subexpr,
        sequential_refine,
        inv_sum,
        mixed_expr
}
#[cfg(feature = "criterion-bench")]
::criterion::criterion_main!(benches);

#[cfg(feature = "time-bench")]
fn main() {
    let bits = ::std::env::args().nth(1).map(|s| s.parse::<usize>().expect("bits argument must be a valid usize"));
    sqrt_convergence(bits);
    inv_small(bits);
    sin_small(bits);
    pi_targeted(bits);
    cancellation(bits);
    deep_chain(bits);
    shared_subexpr(bits);
    sequential_refine(bits);
    inv_sum(bits);
    mixed_expr(bits);
}
