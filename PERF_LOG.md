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

### Experiment 2: Combined Coordinator Optimization (commit 9b432f5)
Status: COMPLETE - MERGED
Three optimizations in one commit:
1. **Worker pool threading**: Replace thread-per-refiner with capped pool (num_cpus threads)
2. **Bounds-cache preservation**: `apply_update` uses `set_prefix_and_bounds` instead of
   `set_prefix`, so parent bounds stay cached for ancestor recomputation
3. **Sin bounds caching**: Cache `SinOp::compute_bounds` results keyed on
   (input_bounds, pi_bounds, num_terms)

Results (from coordinator agent benchmarking):
- integer_roots bits=1: 1.05s → 0.38s (2.8x)
- integer_roots bits=4: 1.00s → 0.47s (2.1x)
- integer_roots bits=16: 1.49s → 0.79s (1.9x)
- integer_roots bits=64: 3.79s → 2.79s (1.4x)
- Memory: 24MB → 6MB for integer_roots (4x reduction)

### Experiment 3: NthRoot Newton's Method
Status: IN PROGRESS (agent ad128659, benchmarking in worktree)

### Experiment 4: Binary Arithmetic Optimizations (commit 114ac29)
Status: COMPLETE - MERGED
- new_normalized() skips trailing_zeros scan (odd * odd = odd)
- Zero short-circuits in add/sub/mul/magnitude/shift
- Direct to_usize() fast path in align_mantissas
- UBinary::Ord direct comparison avoiding 2 BigInt allocations per cmp
- Impact: sin 3-27x faster, inv 2.6x faster (agent's isolated measurements)

### Experiment 5: Fast Benchmarks (commit b937c07)
Status: COMPLETE - MERGED
14 targeted benchmarks in benches/targeted.rs covering diverse scenarios.
Total runtime ~30s with criterion.

### Experiment 6: Clone Optimization + Pi Caching (commit 568dfc0)
Status: COMPLETE - MERGED
- Binary::one() constant method, used throughout
- PiOp bounds caching (same pattern as SinBoundsCache)
- Eliminated redundant clones in prefix.rs and reciprocal.rs
- new_normalized for known-odd constants in pi/sin/bisection

### Experiment 7: Targeted Benchmark After All Wave 1+2 Optimizations
Status: COMPLETE

| Benchmark | Before | After (568dfc0) | Speedup |
|-----------|--------|-----------------|---------|
| pi_64 | ~7ms | 112us | ~63x |
| pi_256 | ~15ms | 549us | ~27x |
| sin_2_256 | ~11ms | 1.3ms | ~8x |
| inv_10_256 | ~41ms | 0.7ms | ~59x |
| inv_sum_64 | ~5ms | 345us | ~14x |
| seq_refine | ~4.3ms | 543us | ~8x |
| sqrt2_64 | ~31ms | 1.05ms | ~30x |
| sqrt2_256 | ~38ms | 10.3ms | ~4x |

Note: NthRoot Newton's method still pending - sqrt2_256 would improve further.

### Experiment 8: Sin Taylor Rational Accumulation (commit ea9757c)
Status: COMPLETE - MERGED
- Replace per-term BigInt reciprocal with single-fraction P/Q accumulation
- Only 2 divisions at end instead of ~170 at 256 bits
- Hybrid: per-term for n<=16, rational for n>16
- Impact: sin/64 29x, sin/256 22x (agent's isolated measurements)

### Experiment 9: Combined Targeted Benchmark After All Optimizations (commit ea9757c)

| Benchmark | Baseline | After All | Speedup |
|-----------|----------|-----------|---------|
| sqrt2_64 | ~31ms | 376us | **~82x** |
| sqrt2_256 | ~38ms | 1.53ms | **~25x** |
| inv_10_64 | ~4ms | 344us | **~12x** |
| inv_10_256 | ~41ms | 356us | **~115x** |
| sin_5_64 | ~11ms | 705us | **~16x** |
| sin_2_256 | ~11ms | 528us | **~21x** |
| pi_64 | ~7ms | 99us | **~71x** |
| pi_256 | ~15ms | 216us | **~69x** |
| seq_refine | ~4.3ms | 226us | **~19x** |
| inv_sum_64 | ~5ms | 153us | **~33x** |

### Experiment 10: NthRoot Newton's Method (commit 22ad53f)
Status: COMPLETE - MERGED
- Complete replacement of bisection with Newton-Raphson
- Quadratic convergence: ~9 steps for 256 bits instead of ~256 bisection steps
- Precision cap from target_width_exp prevents runaway mantissa growth
- Directed rounding: binary_div_ceil for upper, binary_div_floor for lower
- Impact: sqrt2_256 ~35% faster on top of all prior optimizations

### Experiment 11: Hyperfine A/B Interleaved Comparison (commit 22ad53f vs a158cd9)
Status: COMPLETE
Method: Built targeted.rs at both commits, hyperfine with --warmup 2 --min-runs 10,
criterion with --warm-up-time 1 --measurement-time 5 --sample-size 100, all serial.

| Benchmark | Baseline (a158cd9) | Optimized (22ad53f) | Speedup |
|-----------|-------------------|---------------------|---------|
| sqrt2_256 | 1.38ms | 71.7us | **19.3x** |
| sin_2_256 | 1.85ms | 447us | **4.1x** |
| pi_256 | 276.6us | 206.3us | **1.3x** |
| inv_10_256 | 342.7us | 281.5us | **1.2x** |
| inv_sum_64 | 145.4us | 98.6us | **1.5x** |
| seq_refine | 246.2us | 146.6us | **1.7x** |

Note: These are per-iteration criterion numbers measured on a quiet system.
The targeted benchmarks measure individual operations, not the full multi-term
benchmarks (integer_roots=1000 terms, sin=100 terms) which have additional
coordinator overhead.

### Experiment 12: Sin Range Reduction Optimization (commit 0757e21)
Status: COMPLETE - MERGED
Four optimizations:
1. Exponent-shift for two_pi/half_pi (avoid BigInt multiply + normalize)
2. Early return in reduce_to_pi_range if input already in [-pi, pi]
3. Pre-compute hi() values to avoid redundant BigInt additions
4. Eliminate interval_neg() allocations with direct negation
- Impact: sin benchmarks ~35% faster (agent's measurements: bits=4 37%, bits=16 36%, bits=64 32%)

### Experiment 13: Coordinator and Arithmetic Large-Graph Optimization (commit 3b1b51b)
Status: COMPLETE - MERGED
Three optimizations targeting summation (200K terms) and complex (5K terms):
1. **Lazy prefix derivation**: `get_bounds()` skips expensive `Prefix::from_lower_upper`
   derivation at every intermediate node. For 400K-node tree, eliminates ~400K unnecessary
   Prefix derivations and mutex lock/notify cycles during initial evaluation.
2. **Exact-input fast paths**: AddOp/MulOp/NegOp detect zero-width (exact) inputs and
   compute one operation instead of multiple endpoint combinations.
3. **O(1) coordinator bookkeeping**: Replace O(N) scans over all refiners with counters
   (`outstanding_count`, `dynamic_nonleaf_count`).

Results:
- Summation (200K terms): ~485ms → ~156ms (**3.1x faster**)
- Complex (5K terms): ~115ms → ~27ms (**4.3x faster**)

### Experiment 14: Clean Serial Full Criterion Benchmarks (post all wave 1-3)

These numbers were gathered on a quiet system with serial runs (no contention).
Commit state includes worker pool, bounds-cache, sin caching, binary arithmetic,
NthRoot Newton, sin rational accumulation. Does NOT include sin range reduction
(0757e21) or coordinator large-graph optimization (3b1b51b).

| Benchmark | 1 bit | 4 bits | 16 bits | 64 bits | 256 bits |
|-----------|-------|--------|---------|---------|----------|
| pi_refinement | 14.9us | 15.2us | 14.9us | 69.8us | 192us |
| inv (100 terms) | 815us | 830us | 868us | 1.29ms | 1.61ms |
| sin (100 terms) | 14.2ms | 12.4ms | 8.70ms | 19.3ms | 153ms |
| integer_roots (1000) | 116ms | 111ms | 127ms | 137ms | 167ms |
| sqrt2+pi | 20.1us | 20.1us | 58.6us | 110us | 222us |
| complex (5000 terms) | 24.2ms | — | — | — | — |
| sin(1pi) | 58.1us | 59.0us | 56.8us | 264us | 2.24ms |
| sin(100pi) | 41.9us | 42.3us | 48.9us | 283us | 1.76ms |

Comparison vs baseline (a158cd9):

| Benchmark | 1 bit | 4 bits | 16 bits | 64 bits | 256 bits |
|-----------|-------|--------|---------|---------|----------|
| pi | ~1x | ~1x | ~1x | **1.4x** | **1.5x** |
| inv | **3.1x** | **3.0x** | **2.9x** | **2.6x** | **2.2x** |
| sin | 0.7x | 0.8x | **1.4x** | **3.9x** | **41.4x** |
| integer_roots | **2.9x** | **3.4x** | **4.6x** | **10.5x** | **29.8x** |
| sqrt2+pi | **3.5x** | **6.8x** | **3.8x** | **7.3x** | **13.8x** |
| complex | **3.3x** | — | — | — | — |
| sin(1pi) | ~1x | ~1x | ~1x | **2.4x** | **5.8x** |
| sin(100pi) | ~1x | ~1x | ~1x | **2.1x** | **12.5x** |

Note: sin at 1-4 bits shows slight regression (14.2ms vs 9.51ms baseline).
This is worker pool overhead for very low-precision work where thread setup dominates.

### Experiment 15: Inline i64 Exponents
Status: COMPLETE - see Experiment 18

### Experiment 16: Batch apply_update Propagation (commits 10ac0dd → 48c676d)
Status: PARTIALLY REVERTED
- Level-by-level upward propagation: unique parents computed once per level
- root_dirty optimization: precision checked only once after entire batch
- parse_response separates response extraction from propagation

Initial results (on top of all prior optimizations):
- integer_roots/256: ~49% faster
- pi_256: -24%, shared_subexpr: -34%, mixed_expr: -39%, inv_sum: -27%

**Revert (commit 48c676d)**: Batched propagation reverted due to DAG correctness bug.
When a node is both a direct parent and a distant ancestor, level-by-level batching
could process it before all its children were updated. The root_dirty optimization
(skip precision_met() when root bounds didn't change) was kept — this accounts for
most of the speedup on integer_roots since the coordinator was spending ~59% of time
on per-response precision checks.

### Experiment 17: Comprehensive Serial Benchmarks (commit 10ac0dd, all optimizations)

Full criterion suite run serially on quiet system. Includes batch apply_update (later
partially reverted in 48c676d, but root_dirty optimization kept).

Targeted benchmarks:
| Benchmark | Median |
|-----------|--------|
| sqrt2_64 | 49.7us |
| sqrt2_256 | 70.1us |
| inv_10_64 | 201us |
| inv_10_256 | 209us |
| sin_5_64 | 450us |
| sin_2_256 | 334us |
| pi_64 | 65.0us |
| pi_256 | 158us |
| near_cancel_64 | 1.4us |
| deep_chain_64 | 6.5us |
| shared_sqrt2_64 | 72.9us |
| seq_refine | 149us |
| inv_sum_64 | 71.8us |
| mixed_64 | 88.0us |

Full benchmark suites:
| Benchmark | 1 bit | 4 bits | 16 bits | 64 bits | 256 bits |
|-----------|-------|--------|---------|---------|----------|
| pi_refinement | 14.9us | 15.2us | 14.6us | 70.0us | 174us |
| inv (100 terms) | 1.09ms | 1.03ms | 1.04ms | 1.51ms | 1.64ms |
| sin (100 terms) | 4.41ms | 4.40ms | 4.74ms | 9.89ms | 94.7ms |
| integer_roots (1000) | 113ms | 141ms | 146ms | 143ms | 168ms |
| complex (5000 terms) | 24.7ms | — | — | — | — |
| summation (200k terms) | 162ms | — | — | — | — |

Comparison vs baseline (a158cd9):
| Benchmark | Baseline | Current | Speedup |
|-----------|----------|---------|---------|
| pi/256 | 283us | 174us | **1.6x** |
| inv/256 (100 terms) | 3.57ms | 1.64ms | **2.2x** |
| sin/256 (100 terms) | 6.34s | 94.7ms | **67x** |
| integer_roots/256 (1000) | 4.97s | 168ms | **30x** |
| complex (5000 terms) | 80.6ms | 24.7ms | **3.3x** |
| summation (200k terms) | 347ms | 162ms | **2.1x** |
| sqrt2+pi/256 | 3.06ms | 223us | **13.7x** |

Bug found: inv_pi/bits=64 panics on unreachable! in PiOp (commit a158cd9 removed
the doubling fallback). Needs fix.

### Experiment 18: Inline i64 Exponents
Status: COMPLETE - MERGED
- Changed Binary/UBinary exponent field from BigInt to i64 across 16 files
- All exponent arithmetic uses checked_add/checked_sub with overflow detection
- Eliminates heap allocation for every exponent operation
- 240 tests pass, clippy clean
