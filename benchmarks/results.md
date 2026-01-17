# inv Function Optimization Results

This document presents benchmark results comparing the performance of the `inv` (reciprocal) function before and after optimization.

## Optimization Summary

**Problem**: The original implementation incremented `precision_bits` by 1 each refinement step, requiring O(n) steps to achieve n bits of precision.

**Solution**: Double `precision_bits` each refinement step, reducing the number of steps to O(log n) while maintaining all correctness guarantees.

## Benchmark Configuration

- **Samples**: 100 random positive values in range [0.1, 100.0]
- **Target precision**: 256 bits (epsilon = 2^-256)
- **Operation**: Sum of 100 reciprocals, refined to target precision
- **Hardware**: Benchmark run in release mode with optimizations

## Results

### Before Optimization

| Run | Computable Time | Final Width |
|-----|-----------------|-------------|
| 1   | 360.6 ms        | 2^-257      |
| 2   | 353.1 ms        | 2^-257      |
| 3   | 363.4 ms        | 2^-257      |
| **Avg** | **359.0 ms** | 2^-257      |

### After Optimization

| Run | Computable Time | Final Width |
|-----|-----------------|-------------|
| 1   | 18.5 ms         | 2^-506      |
| 2   | 17.6 ms         | 2^-506      |
| 3   | 17.1 ms         | 2^-506      |
| **Avg** | **17.7 ms**  | 2^-506      |

## Performance Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Average Time | 359.0 ms | 17.7 ms | **20.3x faster** |
| Final Precision | 257 bits | 506 bits | 1.97x more precise |

## Analysis

### Why the Optimization Works

The original algorithm incremented precision by 1 bit per refinement step:
- Step 0: precision = 1
- Step 1: precision = 2
- Step 2: precision = 3
- ...
- Step 255: precision = 256

The optimized algorithm doubles precision each step:
- Step 0: precision = 4 (initial value)
- Step 1: precision = 8
- Step 2: precision = 16
- Step 3: precision = 32
- Step 4: precision = 64
- Step 5: precision = 128
- Step 6: precision = 256
- Step 7: precision = 512

This reduces the number of refinement iterations from ~256 to ~7, providing a dramatic speedup.

### Correctness Guarantees Preserved

The optimization does not change the underlying reciprocal computation algorithm. The `reciprocal_rounded_abs_extended` function still:

1. Computes the reciprocal using exact integer division
2. Applies correct rounding (floor or ceil) to maintain bounds
3. Uses the precision parameter only to determine the output resolution

The bounds remain mathematically valid at every step - only the step size changes. Since each step produces correct bounds for its precision level, doubling the step size produces the same final result in fewer iterations.

### Why Precision Exceeds Target

With precision doubling, the final precision achieved (506 bits) exceeds the target (256 bits). This is because the algorithm continues until bounds width is at or below epsilon. With doubling:
- At 256 bits precision, width might be slightly above epsilon
- One more doubling step goes to 512 bits, which is well below epsilon

This "overshoot" is acceptable and even beneficial - the user gets more precision than requested at minimal extra cost.

## Conclusion

The optimization achieves a **20x speedup** while maintaining all correctness guarantees. The change is minimal (only the `refine_step` method is modified) but has significant impact on performance for high-precision computations.
