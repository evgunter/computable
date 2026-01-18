# Benchmarks

This directory contains performance benchmarks comparing Computable arithmetic against standard f64 floating-point operations.

The benchmarks are implemented as a separate crate (which shouldn't be published to crates.io) that depends on the main `computable` library.

## Running the Benchmarks

From the workspace root:

```bash
cargo run --package computable-benchmarks --release
```

Or from the benchmarks directory:

```bash
cd benchmarks
cargo run --release
```

The `--release` flag is important for accurate performance measurements.

### Selecting Specific Benchmarks

You can run specific benchmarks by name or index:

```bash
# List available benchmarks
cargo run --package computable-benchmarks --release -- --list

# Run specific benchmarks by name
cargo run --package computable-benchmarks --release -- complex summation

# Run specific benchmarks by index (0-based)
cargo run --package computable-benchmarks --release -- 0 1

# Show help
cargo run --package computable-benchmarks --release -- --help
```

This is useful for skipping benchmarks that may hang due to known issues (e.g., the integer-roots benchmark may hit a threadpool deadlock bug).

## What's Measured

The benchmark suite includes five scenarios:

1. **Complex expression benchmark** - Evaluates 5,000 complex arithmetic expressions involving multiple operations (addition, multiplication, subtraction) to compare computational overhead and accuracy.

2. **Summation (catastrophic) benchmark** - Demonstrates catastrophic cancellation by summing 200,000 small values to a large base value, highlighting Computable's ability to track error bounds.

3. **Integer roots (binary search) benchmark** - Computes 1,000 integer roots (square root, cube root, 4th/5th/6th roots) using binary search bisection, summing them together. This demonstrates Computable's ability to represent irrational numbers with arbitrary precision through refinement.

4. **Inverse (1/x) benchmark** - Tests the efficiency of the inv refinement loop with high precision (256 bits) on 100 random values.

5. **Sine (sin) benchmark** - Tests Taylor series computation with range reduction and directed rounding on 100 values of varying magnitude.

## Output

The benchmark reports:
- Execution time for float vs Computable operations
- Slowdown factor (how many times slower Computable is than float)
- Final computed values
- Computable error bounds (width)
- Absolute difference between float and Computable results
- For catastrophic summation: precision loss after removing the base value

These metrics help evaluate both the performance cost and accuracy benefits of using Computable arithmetic.

## Project Structure

```
benchmarks/
├── Cargo.toml          # Separate crate configuration
├── README.md           # This file
└── src/
    ├── main.rs         # Entry point with CLI handling
    ├── common.rs       # Shared result types and utilities
    ├── balanced_sum.rs # Balanced sum reduction algorithm
    ├── complex.rs      # Complex expression benchmark
    ├── summation.rs    # Catastrophic summation benchmark
    ├── integer_roots.rs# Integer roots benchmark
    ├── inv.rs          # Inverse (1/x) benchmark
    └── sin.rs          # Sine benchmark
```
