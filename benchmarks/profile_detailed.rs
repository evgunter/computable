use std::time::Instant;

use computable::{Binary, Computable, XBinary};
use num_bigint::BigInt;
use num_traits::One;
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
    println!("=== Detailed computation profiling ===");
    println!();

    let mut rng = StdRng::seed_from_u64(42);

    // Profile BigInt operations which are used in Binary arithmetic
    println!("--- BigInt operation costs ---");
    
    let a_bi = BigInt::from(123456789i64);
    let b_bi = BigInt::from(987654321i64);
    
    let iterations = 1_000_000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a_bi + &b_bi;
    }
    println!("BigInt add x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a_bi * &b_bi;
    }
    println!("BigInt mul x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a_bi.clone();
    }
    println!("BigInt clone x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a_bi <= &b_bi;
    }
    println!("BigInt cmp x{}: {:?}", iterations, start.elapsed());
    println!();

    // Profile Binary operations
    println!("--- Binary operation costs ---");
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
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.clone();
    }
    println!("Binary clone x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a <= b;
    }
    println!("Binary cmp x{}: {:?}", iterations, start.elapsed());
    println!();

    // Profile XBinary operations
    println!("--- XBinary operation costs ---");
    let xa = XBinary::Finite(a.clone());
    let xb = XBinary::Finite(b.clone());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = xa.add(&xb);
    }
    println!("XBinary::add x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = xa.mul(&xb);
    }
    println!("XBinary::mul x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = xa.clone();
    }
    println!("XBinary clone x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = xa <= xb;
    }
    println!("XBinary cmp x{}: {:?}", iterations, start.elapsed());
    println!();

    // Profile Bounds creation
    println!("--- Bounds operation costs ---");
    use computable::Bounds;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Bounds::new(xa.clone(), xb.clone());
    }
    println!("Bounds::new x{}: {:?}", iterations, start.elapsed());
    println!();

    // Simulate a complex expression bounds computation
    println!("--- Expression tree traversal costs ---");
    
    // Build a tree of N complex expressions
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

    // Simulate what happens in bounds computation (without Computable)
    // Each term: a * b + c * c + d * d + (a + b) * (c - d)
    let start = Instant::now();
    let mut results = Vec::with_capacity(size);
    for (a, b, c, d) in &inputs {
        // This simulates the compute_bounds path
        let ab = a.mul(b);
        let cc = c.mul(c);
        let dd = d.mul(d);
        let a_plus_b = a.add(b);
        let c_minus_d = c.sub(d);
        let mixed = a_plus_b.mul(&c_minus_d);
        
        let sum1 = ab.add(&cc);
        let sum2 = sum1.add(&dd);
        let result = sum2.add(&mixed);
        results.push(result);
    }
    println!("Pure Binary computation ({} exprs): {:?}", size, start.elapsed());
    
    // Now with full Computable machinery
    let start = Instant::now();
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
    let _ = total.bounds().expect("bounds should succeed");
    println!("Computable computation ({} exprs): {:?}", size, start.elapsed());
    println!();
    
    // Try to identify overhead
    println!("--- Overhead analysis ---");
    
    // Cost of Arc operations
    use std::sync::Arc;
    let arc_data: Arc<Binary> = Arc::new(a.clone());
    let iterations = 1_000_000;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Arc::clone(&arc_data);
    }
    println!("Arc::clone x{}: {:?}", iterations, start.elapsed());
    
    // Cost of RwLock operations
    use parking_lot::RwLock;
    let rwlock_data: RwLock<Option<Binary>> = RwLock::new(Some(a.clone()));
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rwlock_data.read();
    }
    println!("RwLock::read x{}: {:?}", iterations, start.elapsed());
    
    let start = Instant::now();
    for _ in 0..iterations {
        let guard = rwlock_data.read();
        let _ = guard.clone();
    }
    println!("RwLock::read + clone x{}: {:?}", iterations, start.elapsed());
    
    // Atomic operations
    use std::sync::atomic::{AtomicUsize, Ordering};
    let atomic = AtomicUsize::new(0);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = atomic.fetch_add(1, Ordering::Relaxed);
    }
    println!("AtomicUsize::fetch_add x{}: {:?}", iterations, start.elapsed());
}
