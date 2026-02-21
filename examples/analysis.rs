//! Accuracy and quality analysis for Computable arithmetic.
//!
//! This compares Computable results against f64 to demonstrate:
//! - Bound tightness (width) at various precisions
//! - Accuracy of computable vs float results
//! - Protection against catastrophic cancellation
//! - Correctness of transcendental functions (sin, pi)
//!
//! Run with: cargo run --example analysis --release

// Share bench utilities (balanced_sum, etc.) without putting them in the library.
// We only use #[path] here because this is support code for benchmarks and examples,
// not first-class library code, and we want to keep it out of the public API.
#[path = "../benches/common.rs"]
mod common;

use std::num::NonZeroU32;
use std::time::Instant;

use computable::{
    Binary, Bounds, Computable, FiniteBounds, UBinary, XBinary, pi, pi_bounds_at_precision,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use common::balanced_sum;

// ---------------------------------------------------------------------------
// Helpers (example-specific; balanced_sum comes from benches/common.rs)
// ---------------------------------------------------------------------------

fn try_finite_bounds(bounds: &Bounds) -> Option<FiniteBounds> {
    match (bounds.small(), bounds.large()) {
        (XBinary::Finite(lower), XBinary::Finite(upper)) => {
            Some(FiniteBounds::new(lower.clone(), upper))
        }
        _ => None,
    }
}

fn finite_binary(value: &XBinary) -> Binary {
    match value {
        XBinary::Finite(binary) => binary.clone(),
        XBinary::NegInf | XBinary::PosInf => panic!("expected finite bounds"),
    }
}

// ---------------------------------------------------------------------------
// Analyses
// ---------------------------------------------------------------------------

fn complex_analysis(rng: &mut StdRng) {
    const SAMPLE_COUNT: usize = 5_000;

    let inputs: Vec<(f64, f64, f64, f64)> = (0..SAMPLE_COUNT)
        .map(|_| {
            (
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            )
        })
        .collect();

    let float_start = Instant::now();
    let mut float_total = 0.0;
    for (a, b, c, d) in &inputs {
        let mixed = (a + b) * (c - d);
        let squared = c * c + d * d;
        float_total += a * b + squared + mixed;
    }
    let float_duration = float_start.elapsed();

    let comp_start = Instant::now();
    let terms: Vec<Computable> = inputs
        .iter()
        .map(|&(a, bv, cv, d)| {
            let a_c = Computable::constant(Binary::from_f64(a).unwrap());
            let b_c = Computable::constant(Binary::from_f64(bv).unwrap());
            let c_c = Computable::constant(Binary::from_f64(cv).unwrap());
            let d_c = Computable::constant(Binary::from_f64(d).unwrap());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();
    let total = balanced_sum(terms);
    let bounds = total.bounds().expect("bounds should succeed");
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let comp_duration = comp_start.elapsed();

    let midpoint = finite.midpoint();
    let float_binary = Binary::from_f64(float_total).unwrap();
    let error = float_binary.sub(&midpoint).magnitude();

    println!("== Complex Expression Analysis ==");
    println!("samples: {SAMPLE_COUNT}");
    println!("float time:      {float_duration:?}");
    println!("computable time: {comp_duration:?}");
    println!(
        "slowdown factor: {:.1}x",
        comp_duration.as_secs_f64() / float_duration.as_secs_f64()
    );
    println!(
        "float value:         {}",
        Binary::from_f64(float_total).unwrap()
    );
    println!("computable midpoint: {midpoint}");
    println!("computable width:    {}", bounds.width());
    println!("abs(float - midpoint): {error}");
}

fn summation_analysis(rng: &mut StdRng) {
    const SAMPLE_COUNT: usize = 200_000;

    let inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|_| rng.gen_range(-1.0e-6..1.0e-6))
        .collect();
    let computable_inputs: Vec<Computable> = inputs
        .iter()
        .map(|&v| Computable::constant(Binary::from_f64(v).unwrap()))
        .collect();
    let base = 2_i64.pow(30) as f64;

    // Float: sum with base
    let float_start = Instant::now();
    let mut float_total = base;
    for &v in &inputs {
        float_total += v;
    }
    let float_duration = float_start.elapsed();

    // Computable: sum with base
    let comp_start = Instant::now();
    let mut terms = Vec::with_capacity(computable_inputs.len() + 1);
    terms.push(Computable::constant(Binary::from_f64(base).unwrap()));
    terms.extend(computable_inputs.iter().cloned());
    let total = balanced_sum(terms);
    let bounds = total.bounds().expect("bounds should succeed");
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let comp_duration = comp_start.elapsed();

    let midpoint = finite.midpoint();

    // True sum without base (computable, no cancellation)
    let baseless_sum_computable = {
        let sum = balanced_sum(computable_inputs.clone());
        let b = sum.bounds().expect("bounds should succeed");
        let f = try_finite_bounds(&b).expect("bounds should be finite");
        f.midpoint()
    };
    let baseless_sum_float: f64 = inputs.iter().sum();

    // Precision loss: (sum_with_base - base) vs true baseless sum
    let float_minus_base = float_total - base;
    let comp_minus_base = midpoint.sub(&Binary::from_f64(base).unwrap());

    let float_precision_loss = Binary::from_f64(float_minus_base)
        .unwrap()
        .sub(&baseless_sum_computable)
        .magnitude();
    let comp_precision_loss = comp_minus_base.sub(&baseless_sum_computable).magnitude();

    println!("== Summation (Catastrophic Cancellation) Analysis ==");
    println!("samples: {SAMPLE_COUNT}");
    println!("base value: {}", Binary::from_f64(base).unwrap());
    println!("float time:      {float_duration:?}");
    println!("computable time: {comp_duration:?}");
    println!(
        "slowdown factor: {:.1}x",
        comp_duration.as_secs_f64() / float_duration.as_secs_f64()
    );
    println!(
        "float value:         {}",
        Binary::from_f64(float_total).unwrap()
    );
    println!("computable midpoint: {midpoint}");
    println!("computable width:    {}", bounds.width());
    println!();
    println!("After removing base value:");
    println!(
        "  sum without base (float):      {}",
        Binary::from_f64(baseless_sum_float).unwrap()
    );
    println!("  sum without base (computable): {baseless_sum_computable}");
    println!(
        "  (sum+base)-base  (float):      {}",
        Binary::from_f64(float_minus_base).unwrap()
    );
    println!("  (sum+base)-base  (computable): {comp_minus_base}");
    println!("  float precision loss:      {float_precision_loss}");
    println!("  computable precision loss: {comp_precision_loss}");
}

fn integer_roots_analysis(rng: &mut StdRng) {
    const SAMPLE_COUNT: usize = 1_000;

    let inputs: Vec<(u64, NonZeroU32)> = (0..SAMPLE_COUNT)
        .map(|i| {
            let value = rng.gen_range(2..1000) as u64;
            let n = NonZeroU32::new((i % 5) as u32 + 2).unwrap();
            (value, n)
        })
        .collect();

    let float_start = Instant::now();
    let mut float_total = 0.0f64;
    for &(value, n) in &inputs {
        float_total += (value as f64).powf(1.0 / n.get() as f64);
    }
    let float_duration = float_start.elapsed();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(0));

    let comp_start = Instant::now();
    let terms: Vec<Computable> = inputs
        .iter()
        .map(|&(value, n)| {
            Computable::constant(Binary::new(BigInt::from(value), BigInt::from(0))).nth_root(n)
        })
        .collect();
    let total = balanced_sum(terms);
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let comp_duration = comp_start.elapsed();

    let midpoint = finite.midpoint();
    let error = Binary::from_f64(float_total)
        .unwrap()
        .sub(&midpoint)
        .magnitude();

    println!("== Integer Roots Analysis ==");
    println!("samples: {SAMPLE_COUNT}");
    println!("epsilon: 1");
    println!("root degrees: 2 (sqrt), 3 (cbrt), 4, 5, 6");
    println!("float time:      {float_duration:?}");
    println!("computable time: {comp_duration:?}");
    println!(
        "slowdown factor: {:.1}x",
        comp_duration.as_secs_f64() / float_duration.as_secs_f64()
    );
    println!(
        "float value:         {}",
        Binary::from_f64(float_total).unwrap()
    );
    println!("computable midpoint: {midpoint}");
    println!("computable width:    {}", bounds.width());
    println!("abs(float - midpoint): {error}");
}

fn inv_analysis(rng: &mut StdRng) {
    const SAMPLE_COUNT: usize = 100;
    const PRECISION_BITS: i64 = 256;

    let inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|_| rng.gen_range(0.1..100.0))
        .collect();

    let float_start = Instant::now();
    let float_sum: f64 = inputs.iter().map(|x| 1.0 / x).sum();
    let float_duration = float_start.elapsed();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-PRECISION_BITS));

    let comp_start = Instant::now();
    let terms: Vec<Computable> = inputs
        .iter()
        .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).inv())
        .collect();
    let total = balanced_sum(terms);
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");
    let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
    let comp_duration = comp_start.elapsed();

    let midpoint = finite.midpoint();
    let error = Binary::from_f64(float_sum)
        .unwrap()
        .sub(&midpoint)
        .magnitude();

    println!("== Inverse (1/x) Analysis ==");
    println!("samples: {SAMPLE_COUNT}");
    println!("target precision: {PRECISION_BITS} bits");
    println!("float time:      {float_duration:?}");
    println!("computable time: {comp_duration:?}");
    println!(
        "slowdown factor: {:.1}x",
        comp_duration.as_secs_f64() / float_duration.as_secs_f64()
    );
    println!(
        "float value:         {}",
        Binary::from_f64(float_sum).unwrap()
    );
    println!("computable midpoint: {midpoint}");
    println!("computable width:    {}", bounds.width());
    println!("abs(float - midpoint): {error}");
}

fn sin_analysis(rng: &mut StdRng) {
    const SAMPLE_COUNT: usize = 100;
    const PRECISION_BITS: i64 = 128;

    let inputs: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|i| {
            if i < SAMPLE_COUNT / 3 {
                rng.gen_range(-1.0..1.0)
            } else if i < 2 * SAMPLE_COUNT / 3 {
                rng.gen_range(-3.15..3.15)
            } else {
                rng.gen_range(-100.0..100.0)
            }
        })
        .collect();

    let float_start = Instant::now();
    let float_sum: f64 = inputs.iter().map(|x| x.sin()).sum();
    let float_duration = float_start.elapsed();

    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-PRECISION_BITS));

    let comp_start = Instant::now();
    let terms: Vec<Computable> = inputs
        .iter()
        .map(|&x| Computable::constant(Binary::from_f64(x).unwrap()).sin())
        .collect();
    let total = balanced_sum(terms);
    let bounds = total
        .refine_to_default(epsilon)
        .expect("refine_to should succeed");
    let comp_duration = comp_start.elapsed();

    println!("== Sine Analysis ==");
    println!(
        "samples: {} (1/3 small, 1/3 medium, 1/3 large)",
        SAMPLE_COUNT
    );
    println!("target precision: {PRECISION_BITS} bits");
    println!("float time:      {float_duration:?}");
    println!("computable time: {comp_duration:?}");
    println!(
        "slowdown factor: {:.1}x",
        comp_duration.as_secs_f64() / float_duration.as_secs_f64()
    );
    println!(
        "float value:         {}",
        Binary::from_f64(float_sum).unwrap()
    );

    match try_finite_bounds(&bounds) {
        Some(finite) => {
            let midpoint = finite.midpoint();
            let error = Binary::from_f64(float_sum)
                .unwrap()
                .sub(&midpoint)
                .magnitude();
            println!("computable midpoint: {midpoint}");
            println!("computable width:    {}", bounds.width());
            println!("abs(float - midpoint): {error}");
        }
        None => {
            println!("computable bounds: infinite (cannot compute midpoint)");
        }
    }
}

fn pi_analysis() {
    let precision_bits: &[u64] = &[32, 64, 128, 256, 512, 1024];

    println!("== Pi Refinement Analysis ==");
    println!();
    let pi_f64 = Binary::from_f64(std::f64::consts::PI).unwrap();

    for &bits in precision_bits {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));

        let start = Instant::now();
        let bounds = pi()
            .refine_to_default(epsilon)
            .expect("pi refinement should succeed");
        let duration = start.elapsed();

        print!(
            "  {bits:>4} bits: {duration:>12?}  width: {:<30}",
            bounds.width()
        );
        if let Some(finite) = try_finite_bounds(&bounds) {
            let mid = finite.midpoint();
            let error = mid.sub(&pi_f64).magnitude();
            println!("  |mid - pi_f64|: {error}");
        } else {
            println!("  (infinite bounds)");
        }
    }

    // pi_bounds_at_precision
    println!();
    println!("== pi_bounds_at_precision ==");
    println!();
    for &bits in precision_bits {
        let start = Instant::now();
        let (lower, upper) = pi_bounds_at_precision(bits);
        let duration = start.elapsed();
        let width = upper.sub(&lower);
        let sum = lower.add(&upper);
        let mid = Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one());
        println!(
            "  {bits:>4} bits: {duration:>12?}  width: {:<30}  midpoint: {mid}",
            width
        );
    }

    // Arithmetic with pi
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-64));

    println!();
    println!("== Pi Arithmetic ==");
    println!();

    for (label, expected_f64, build_expr) in [
        (
            "2pi",
            2.0 * std::f64::consts::PI,
            Box::new(|| Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0))) * pi())
                as Box<dyn Fn() -> Computable>,
        ),
        (
            "pi/2",
            std::f64::consts::FRAC_PI_2,
            Box::new(|| {
                Computable::constant(Binary::new(BigInt::from(1), BigInt::from(-1))) * pi()
            }),
        ),
        (
            "pi^2",
            std::f64::consts::PI * std::f64::consts::PI,
            Box::new(|| pi() * pi()),
        ),
        ("1/pi", 1.0 / std::f64::consts::PI, Box::new(|| pi().inv())),
    ] {
        let start = Instant::now();
        let expr = build_expr();
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("pi arithmetic should succeed");
        let duration = start.elapsed();
        let finite = try_finite_bounds(&bounds).expect("bounds should be finite");
        let mid = finite.midpoint();
        let expected = Binary::from_f64(expected_f64).unwrap();
        println!("  {label:<5}: {duration:>10?}  midpoint: {mid}");
        println!("         expected: {expected}");
    }

    // sin(n*pi) — should be ~0
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-32));

    println!();
    println!("== sin(n * pi) — should contain 0 ==");
    println!();

    for multiplier in [1u64, 2, 10, 100] {
        let start = Instant::now();
        let n_pi = if multiplier == 1 {
            pi()
        } else {
            Computable::constant(Binary::new(BigInt::from(multiplier), BigInt::from(0))) * pi()
        };
        let bounds = n_pi
            .sin()
            .refine_to_default(epsilon.clone())
            .expect("sin(n*pi) should succeed");
        let duration = start.elapsed();

        let lower = finite_binary(bounds.small());
        let upper = finite_binary(&bounds.large());
        let contains_zero = lower.mantissa().is_negative() || lower.mantissa().is_zero();
        println!(
            "  sin({multiplier:>3}*pi): {duration:>10?}  bounds: [{lower}, {upper}]  width: {}  contains 0: {contains_zero}",
            bounds.width()
        );
    }

    // High precision
    println!();
    println!("== High Precision Pi ==");
    println!();

    for &bits in &[2048u64, 4096, 8192] {
        let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-(bits as i64)));
        let start = Instant::now();
        let bounds = pi()
            .refine_to_default(epsilon)
            .expect("high precision pi should succeed");
        let duration = start.elapsed();
        println!(
            "  {bits:>5} bits (~{} digits): {duration:>10?}  width: {}",
            (bits as f64 * 0.301).round() as u64,
            bounds.width()
        );
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);

    complex_analysis(&mut rng);
    println!();
    summation_analysis(&mut rng);
    println!();
    integer_roots_analysis(&mut rng);
    println!();
    inv_analysis(&mut rng);
    println!();
    sin_analysis(&mut rng);
    println!();
    pi_analysis();
}
