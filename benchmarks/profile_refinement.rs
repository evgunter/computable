use std::time::Instant;

use computable::{Binary, Computable, XBinary};
use num_bigint::BigInt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value).expect("expected finite f64") {
        XBinary::Finite(binary) => binary,
        XBinary::NegInf | XBinary::PosInf => {
            panic!("expected finite f64 input")
        }
    }
}

/// Sums terms using a balanced reduction
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

    values.pop().expect("values should contain at least one element")
}

fn main() {
    println!("=== Thread pool and refinement profiling ===");
    println!();

    // Check thread count
    println!("Available parallelism: {:?}", std::thread::available_parallelism());
    println!("RAYON_NUM_THREADS: {:?}", std::env::var("RAYON_NUM_THREADS"));
    println!();

    let mut rng = StdRng::seed_from_u64(42);

    // Test 1: Simple expression bounds (no refinement needed)
    println!("--- Test 1: Simple constants (no refinement) ---");
    for size in [100, 500, 1000, 2000, 5000] {
        let terms: Vec<_> = (0..size)
            .map(|_| {
                let v: f64 = rng.gen_range(-10.0..10.0);
                Computable::constant(binary_from_f64(v))
            })
            .collect();
        let total = balanced_sum(terms);
        
        let start = Instant::now();
        let _ = total.bounds().expect("bounds should succeed");
        println!("  {} constants - bounds: {:?}", size, start.elapsed());
    }
    println!();

    // Test 2: More complex expression tree (still no refinement)  
    println!("--- Test 2: Complex expressions (no refinement) ---");
    for size in [100, 500, 1000, 2000, 5000] {
        let terms: Vec<_> = (0..size)
            .map(|_| {
                let a = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
                let b = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
                let c = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
                let d = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
                let mixed = (a.clone() + b.clone()) * (c.clone() - d.clone());
                let squared = c.clone() * c + d.clone() * d;
                a * b + squared + mixed
            })
            .collect();
        let total = balanced_sum(terms);
        
        let start = Instant::now();
        let _ = total.bounds().expect("bounds should succeed");
        println!("  {} complex exprs - bounds: {:?}", size, start.elapsed());
    }
    println!();

    // Test 3: Expression that needs refinement (with division)
    println!("--- Test 3: Division expressions (requires refinement) ---");
    for size in [10, 50, 100, 200] {
        let terms: Vec<_> = (0..size)
            .map(|_| {
                let a = Computable::constant(binary_from_f64(rng.gen_range(1.0..10.0)));
                let b = Computable::constant(binary_from_f64(rng.gen_range(1.0..10.0)));
                a / b
            })
            .collect();
        let total = balanced_sum(terms);
        
        let start = Instant::now();
        let bounds = total.bounds().expect("bounds should succeed");
        let bounds_time = start.elapsed();
        
        // Check width to see if refinement was triggered
        let width = bounds.width();
        println!("  {} divisions - bounds: {:?}, width: {:?}", size, bounds_time, width);
    }
    println!();

    // Test 4: Graph traversal scaling
    println!("--- Test 4: Graph traversal (repeated bounds calls) ---");
    let terms: Vec<_> = (0..1000)
        .map(|_| {
            let a = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
            let b = Computable::constant(binary_from_f64(rng.gen_range(-10.0..10.0)));
            a * b
        })
        .collect();
    let total = balanced_sum(terms);
    
    // First call
    let start = Instant::now();
    let _ = total.bounds().expect("bounds should succeed");
    println!("  First bounds call: {:?}", start.elapsed());
    
    // Subsequent calls (should be cached)
    let start = Instant::now();
    for _ in 0..100 {
        let _ = total.bounds().expect("bounds should succeed");
    }
    println!("  100 subsequent bounds calls: {:?}", start.elapsed());
}
