# Performance Optimization Research Log

## Codebase Summary
Computable real numbers library using interval refinement with parallel coordination.
~4000 lines of core code. Key hotspots: refinement coordinator (61KB), prefix computation (31KB),
binary arithmetic, and individual ops (Inv/Sin/Pi/NthRoot).

## Baseline Criterion Benchmarks (commit a158cd9, branch mng/faster)

| Benchmark | 1 bit | 4 bits | 16 bits | 64 bits | 256 bits |
|-----------|-------|--------|---------|---------|----------|
| pi_refinement | 15.5us | 15.3us | 15.1us | 97.2us | 282.6us |
| pi_bounds | 6.6us | 6.6us | 7.7us | 25.3us | 115.0us |
| inv (100 terms) | 2.49ms | 2.50ms | 2.50ms | 3.35ms | 3.57ms |
| sin (100 terms) | 9.51ms | 10.2ms | 12.4ms | 74.6ms | **6.34s** |
| integer_roots (1000) | 338ms | 379ms | 585ms | 1.44s | **4.97s** |
| sqrt2+pi | 69.4us | 137us | 222us | 807us | 3.06ms |
| sqrt2*pi | 83.5us | 138us | 257us | 993us | 3.35ms |
| sqrt2+cbrt3 | 89.5us | 138us | 344us | 1.13ms | 4.24ms |
| complex (5000 terms) | 80.6ms | — | — | — | — |
| summation (200k terms) | 347ms | — | — | — | — |
| sin(1pi) | 55.6us | 55.2us | 55.8us | 636us | 13.0ms |
| sin(100pi) | 45.5us | 45.8us | 45.5us | 592us | 22.0ms |

### Key observations:
- **sin/256 is the worst**: 6.34s per iteration. Catastrophic scaling.
- **integer_roots/256 is second worst**: 4.97s. Linear bisection convergence.
- **inv scales excellently**: 2.5ms→3.6ms (quadratic convergence, ~8 N-R steps for 256 bits)
- **pi is very fast**: 283us at 256 bits

## Profiling Results

### Hotspot 1: Sin Taylor Series Recomputation (CRITICAL)
- `SinOp::compute_bounds` called on coordinator thread during `apply_update`
- Recomputes full Taylor series from scratch on EVERY coordinator update
- With 100 sin nodes in balanced sum, each refiner response triggers ~100 sin recomputations
- At 256 bits: ~85 Taylor terms, each with growing BigInt multiplications
- Scales cubically with precision

### Hotspot 2: Thread-per-Refiner OS Overhead (integer_roots)
- 1000 threads for 1000 NthRoot refiners
- 5.77s system time vs 6.85s user time (thread mgmt dominates)
- Most threads blocked on channel recv (idle)

### Hotspot 3: Coordinator apply_update Propagation
- BFS from updated node to root, calling compute_prefix() on each parent
- For balanced sum of N terms: O(N) nodes traversed per update
- Called after EVERY refiner response

### Hotspot 4: NthRoot Linear Convergence
- Bisection: 1 bit per step, needs ~256 steps for 256 bits
- TODO in code: "investigate using target_width_exp to leap"

## Priority-Ordered Optimization Plan

### P0: NthRoot Newton's Method (integer_roots: 4.97s → expected <1s)
Replace bisection with Newton's method for quadratic convergence.
256 bits → ~9 steps instead of ~256 steps.

### P1: Sin compute_bounds Caching (sin: 6.34s → expected <1s)
Cache Taylor series intermediate state so compute_bounds doesn't recompute from scratch.
Or: avoid calling compute_bounds on coordinator thread during apply_update.

### P2: Thread Pool / Reduced Threading (integer_roots, summation)
Replace thread::scope with a thread pool or batch refiners onto fewer threads.
Would eliminate the 5.77s system time overhead.

### P3: Batch/Lazy apply_update Propagation
Don't propagate after every response - batch updates.

### P4: Binary Arithmetic Optimizations
Reduce BigInt allocations in hot paths (align_mantissas, normalize, etc.)

### P5: Targeted Smaller Benchmarks
Create faster, more targeted benchmarks for iteration speed.

## Experiments

### Experiment 1: Baseline
Status: COMPLETE (see above)

### Experiment 2: NthRoot Newton's Method
Status: STARTING (multiple strategies in parallel)

### Experiment 3: Sin Taylor Caching
Status: STARTING (multiple strategies in parallel)

### Experiment 4: Threading Improvements
Status: STARTING

### Experiment 5: Fast Benchmarks
Status: STARTING (creating targeted benchmark suite)
