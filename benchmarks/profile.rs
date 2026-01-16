use std::time::{Duration, Instant};

use computable::{Binary, Bounds, Computable, UXBinary, XBinary};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SAMPLE_COUNT: usize = 1_000;

fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value).expect("expected finite f64") {
        XBinary::Finite(binary) => binary,
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite f64 input")
        }
    }
}

fn binary_to_f64(binary: &Binary) -> f64 {
    let mantissa = binary
        .mantissa()
        .to_f64()
        .expect("mantissa should fit in f64");
    let exponent = binary
        .exponent()
        .to_i32()
        .expect("exponent should fit in i32");
    mantissa * (2.0f64).powi(exponent)
}

fn finite_binary(value: &XBinary) -> Binary {
    match value {
        XBinary::Finite(binary) => binary.clone(),
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite bounds")
        }
    }
}

fn midpoint(bounds: &Bounds) -> f64 {
    let lower = finite_binary(bounds.small());
    let upper = finite_binary(&bounds.large());
    let sum = lower.add(&upper);
    let half = Binary::new(BigInt::one(), BigInt::from(-1));
    let mid = sum.mul(&half);
    binary_to_f64(&mid)
}

/// Sums terms using a balanced reduction instead of left-associative chaining.
fn balanced_sum(mut values: Vec<Computable>) -> Computable {
    if values.is_empty() {
        return Computable::constant(Binary::zero());
    }

    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(left + right);
            } else {
                next.push(left);
            }
        }
        values = next;
    }

    values
        .pop()
        .expect("values should contain at least one element")
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);

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

    println!("=== Phase timing breakdown ===");
    println!("samples: {SAMPLE_COUNT}");
    println!();

    // Phase 1: Create constants
    let start = Instant::now();
    let constants: Vec<_> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            (
                Computable::constant(binary_from_f64(*a)),
                Computable::constant(binary_from_f64(*b)),
                Computable::constant(binary_from_f64(*c)),
                Computable::constant(binary_from_f64(*d)),
            )
        })
        .collect();
    println!("Phase 1 - Create constants: {:?}", start.elapsed());

    // Phase 2: Build expression tree
    let start = Instant::now();
    let terms: Vec<_> = constants
        .into_iter()
        .map(|(a_c, b_c, c_c, d_c)| {
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();
    println!("Phase 2 - Build expressions: {:?}", start.elapsed());

    // Phase 3: Balanced sum
    let start = Instant::now();
    let total = balanced_sum(terms);
    println!("Phase 3 - Balanced sum tree: {:?}", start.elapsed());

    // Phase 4: Get bounds (triggers computation)
    let start = Instant::now();
    let bounds = total.bounds().expect("bounds should succeed");
    let bounds_time = start.elapsed();
    println!("Phase 4 - Compute bounds: {:?}", bounds_time);
    println!();
    println!("Result midpoint: {:.10}", midpoint(&bounds));

    // Now time individual operations
    println!();
    println!("=== Micro-benchmarks ===");

    // Binary arithmetic
    let a = binary_from_f64(123.456);
    let b = binary_from_f64(789.012);
    
    let iterations = 100_000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.add(&b);
    }
    println!("Binary::add x{}: {:?}", iterations, start.elapsed());

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.mul(&b);
    }
    println!("Binary::mul x{}: {:?}", iterations, start.elapsed());

    // XBinary creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = XBinary::from_f64(123.456);
    }
    println!("XBinary::from_f64 x{}: {:?}", iterations, start.elapsed());

    // Computable creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Computable::constant(a.clone());
    }
    println!("Computable::constant x{}: {:?}", iterations, start.elapsed());

    // Computable addition (just graph building)
    let start = Instant::now();
    let base = Computable::constant(a.clone());
    let mut sum = base.clone();
    for _ in 0..1000 {
        sum = sum + base.clone();
    }
    println!("Computable add (1000 chain): {:?}", start.elapsed());

    // Computable bounds
    let start = Instant::now();
    let _ = sum.bounds();
    println!("Computable bounds (1000 chain): {:?}", start.elapsed());
}
