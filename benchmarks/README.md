# Benchmarks

This directory contains performance benchmarks comparing Computable arithmetic against standard f64 floating-point operations.

## Running the Benchmarks

The benchmarks are implemented as a binary that requires the `benchmarks` feature flag:

```bash
cargo run --features benchmarks --bin benchmarks --release
```

The `--release` flag is important for accurate performance measurements.

### Selecting Specific Benchmarks

You can run specific benchmarks by name or index:

```bash
# List available benchmarks
cargo run --features benchmarks --bin benchmarks --release -- --list

# Run specific benchmarks by name
cargo run --features benchmarks --bin benchmarks --release -- complex summation

# Run specific benchmarks by index (0-based)
cargo run --features benchmarks --bin benchmarks --release -- 0 1

# Show help
cargo run --features benchmarks --bin benchmarks --release -- --help
```

This is useful for skipping benchmarks that may hang due to known issues (e.g., the integer-roots benchmark may hit a threadpool deadlock bug).

## What's Measured

The benchmark suite includes three scenarios:

1. **Complex expression benchmark** - Evaluates 5,000 complex arithmetic expressions involving multiple operations (addition, multiplication, subtraction) to compare computational overhead and accuracy.

2. **Summation (catastrophic) benchmark** - Demonstrates catastrophic cancellation by summing 200,000 small values to a large base value, highlighting Computable's ability to track error bounds.

3. **Integer roots (binary search) benchmark** - Computes 1,000 integer roots (square root, cube root, 4th/5th/6th roots) using binary search bisection, summing them together. This demonstrates Computable's ability to represent irrational numbers with arbitrary precision through refinement.

## Output

The benchmark reports:
- Execution time for float vs Computable operations
- Slowdown factor (how many times slower Computable is than float)
- Final computed values
- Computable error bounds (width)
- Absolute difference between float and Computable results
- For catastrophic summation: precision loss after removing the base value

These metrics help evaluate both the performance cost and accuracy benefits of using Computable arithmetic.
