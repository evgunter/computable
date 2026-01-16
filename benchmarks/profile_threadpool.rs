use std::time::Instant;

use computable::{Binary, Computable, XBinary};
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

fn run_benchmark(size: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    
    let inputs: Vec<_> = (0..size)
        .map(|_| {
            (
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
            )
        })
        .collect();

    let terms: Vec<_> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            let a_c = Computable::constant(a.clone());
            let b_c = Computable::constant(b.clone());
            let c_c = Computable::constant(c.clone());
            let d_c = Computable::constant(d.clone());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();
    let total = balanced_sum(terms);
    
    let start = Instant::now();
    let _ = total.bounds().expect("bounds should succeed");
    println!("  {} exprs: {:?}", size, start.elapsed());
}

fn main() {
    println!("=== Thread pool configuration profiling ===");
    println!();
    
    // Get current configuration
    println!("System info:");
    println!("  Available parallelism: {:?}", std::thread::available_parallelism());
    println!("  RAYON_NUM_THREADS: {:?}", std::env::var("RAYON_NUM_THREADS"));
    println!();
    
    // Run benchmark with different sizes
    println!("Benchmark results (current config):");
    for size in [500, 1000, 2000, 5000] {
        run_benchmark(size);
    }
    println!();
    
    // Test rayon parallel iteration
    println!("--- Parallel bounds computation test ---");
    use rayon::prelude::*;
    
    let mut rng = StdRng::seed_from_u64(42);
    let size = 1000;
    let inputs: Vec<_> = (0..size)
        .map(|_| {
            (
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
                binary_from_f64(rng.gen_range(-10.0..10.0)),
            )
        })
        .collect();

    // Build expression tree
    let terms: Vec<Computable> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            let a_c = Computable::constant(a.clone());
            let b_c = Computable::constant(b.clone());
            let c_c = Computable::constant(c.clone());
            let d_c = Computable::constant(d.clone());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();

    // Sequential bounds on each term
    let start = Instant::now();
    for term in &terms {
        let _ = term.bounds().expect("bounds should succeed");
    }
    println!("Sequential bounds on {} terms: {:?}", size, start.elapsed());

    // Parallel bounds on each term
    let terms2: Vec<Computable> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            let a_c = Computable::constant(a.clone());
            let b_c = Computable::constant(b.clone());
            let c_c = Computable::constant(c.clone());
            let d_c = Computable::constant(d.clone());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();

    let start = Instant::now();
    terms2.par_iter().for_each(|term| {
        let _ = term.bounds().expect("bounds should succeed");
    });
    println!("Parallel bounds on {} terms: {:?}", size, start.elapsed());
    println!();

    // Test balanced_sum with parallel reduction
    println!("--- Parallel sum reduction test ---");
    
    let terms3: Vec<Computable> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            let a_c = Computable::constant(a.clone());
            let b_c = Computable::constant(b.clone());
            let c_c = Computable::constant(c.clone());
            let d_c = Computable::constant(d.clone());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();

    let start = Instant::now();
    let total_seq = balanced_sum(terms3);
    let _ = total_seq.bounds().expect("bounds should succeed");
    println!("Sequential balanced_sum + bounds: {:?}", start.elapsed());

    // Parallel version using rayon reduce
    let terms4: Vec<Computable> = inputs
        .iter()
        .map(|(a, b, c, d)| {
            let a_c = Computable::constant(a.clone());
            let b_c = Computable::constant(b.clone());
            let c_c = Computable::constant(c.clone());
            let d_c = Computable::constant(d.clone());
            let mixed = (a_c.clone() + b_c.clone()) * (c_c.clone() - d_c.clone());
            let squared = c_c.clone() * c_c + d_c.clone() * d_c;
            a_c * b_c + squared + mixed
        })
        .collect();

    let start = Instant::now();
    // First compute bounds on all terms in parallel
    terms4.par_iter().for_each(|term| {
        let _ = term.bounds();
    });
    // Then do the sum
    let total_par = balanced_sum(terms4);
    let _ = total_par.bounds().expect("bounds should succeed");
    println!("Parallel term bounds + sequential sum: {:?}", start.elapsed());
}
