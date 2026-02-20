use std::collections::HashSet;
use std::env;

use computable::UXBinary;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod balanced_sum;
mod common;
mod complex;
mod demand;
mod integer_roots;
mod inv;
mod pi;
mod sin;
mod summation;

use complex::run_complex_benchmark;
use demand::run_demand_benchmark;
use integer_roots::run_integer_roots_benchmark;
use inv::run_inv_benchmark;
use pi::run_pi_benchmark;
use sin::run_sin_benchmark;
use summation::run_summation_benchmark;

/// Available benchmark names
const BENCHMARK_NAMES: &[&str] = &[
    "complex",
    "summation",
    "integer-roots",
    "inv",
    "sin",
    "pi",
    "demand",
];

fn print_usage() {
    println!("Usage: benchmarks [OPTIONS] [BENCHMARK...]");
    println!();
    println!("Run performance benchmarks for Computable arithmetic.");
    println!();
    println!("Options:");
    println!("  --help, -h       Show this help message");
    println!("  --list, -l       List available benchmarks");
    println!();
    println!("Arguments:");
    println!("  BENCHMARK        Benchmark(s) to run, by name or index (0-based)");
    println!("                   If no benchmarks specified, runs all benchmarks.");
    println!();
    println!("Examples:");
    println!("  benchmarks                      # Run all benchmarks");
    println!("  benchmarks complex              # Run only 'complex' benchmark");
    println!("  benchmarks 0 2                  # Run benchmarks 0 and 2");
    println!("  benchmarks summation complex    # Run 'summation' and 'complex'");
}

fn print_benchmark_list() {
    println!("Available benchmarks:");
    for (i, name) in BENCHMARK_NAMES.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
}

fn parse_benchmark_selection(args: &[String]) -> HashSet<usize> {
    let mut selected = HashSet::new();

    for arg in args {
        // Try parsing as index first
        if let Ok(index) = arg.parse::<usize>() {
            if index < BENCHMARK_NAMES.len() {
                selected.insert(index);
            } else {
                eprintln!(
                    "Warning: benchmark index {} out of range (0-{})",
                    index,
                    BENCHMARK_NAMES.len() - 1
                );
            }
        } else {
            // Try matching by name
            if let Some(index) = BENCHMARK_NAMES.iter().position(|&name| name == arg) {
                selected.insert(index);
            } else {
                eprintln!("Warning: unknown benchmark '{}'", arg);
            }
        }
    }

    selected
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    // Handle help and list options
    for arg in &args {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                return;
            }
            "--list" | "-l" => {
                print_benchmark_list();
                return;
            }
            _ => {}
        }
    }

    // Filter out options and parse benchmark selection
    let benchmark_args: Vec<String> = args
        .into_iter()
        .filter(|arg| !arg.starts_with('-'))
        .collect();

    let selected = if benchmark_args.is_empty() {
        // Run all benchmarks if none specified
        (0..BENCHMARK_NAMES.len()).collect()
    } else {
        parse_benchmark_selection(&benchmark_args)
    };

    if selected.is_empty() {
        eprintln!("No valid benchmarks selected. Use --list to see available benchmarks.");
        return;
    }

    let mut rng = StdRng::seed_from_u64(7);
    let mut first = true;

    for i in 0..BENCHMARK_NAMES.len() {
        if selected.contains(&i) {
            if !first {
                println!();
            }
            first = false;

            match i {
                0 => run_complex_benchmark(&mut rng),
                1 => run_summation_benchmark(&mut rng),
                2 => run_integer_roots_benchmark(&mut rng),
                3 => run_inv_benchmark(&mut rng),
                4 => run_sin_benchmark(&mut rng),
                5 => run_pi_benchmark(),
                6 => run_demand_benchmark(),
                _ => unreachable!(),
            }
        }
    }
}
