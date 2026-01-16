# Benchmarks

This directory contains performance benchmarks comparing Computable arithmetic against standard f64 floating-point operations.

## Running the Benchmarks

The benchmarks are implemented as a binary that requires the `benchmarks` feature flag:

```bash
cargo run --features benchmarks --bin benchmarks --release
```

The `--release` flag is important for accurate performance measurements.

## What's Measured

The benchmark suite includes two scenarios:

1. **Complex expression benchmark** - Evaluates 5,000 complex arithmetic expressions involving multiple operations (addition, multiplication, subtraction) to compare computational overhead and accuracy.

2. **Summation (catastrophic) benchmark** - Demonstrates catastrophic cancellation by summing 200,000 small values to a large base value (1.0e12), highlighting Computable's ability to track error bounds.

## Output

The benchmark reports:
- Execution time for float vs Computable operations
- Final computed values
- Computable error bounds (width)
- Absolute difference between float and Computable results

These metrics help evaluate both the performance cost and accuracy benefits of using Computable arithmetic.
