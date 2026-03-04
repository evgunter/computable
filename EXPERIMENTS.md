# Performance Regression Experiments

## Problem
CI benchmarks show regressions vs main (round-based model):
- `bench_sin`: +97% to +235% (precision 0-4)
- `bench_inv`: +120% (precision 3-4)
- `bench_sqrt2_times_pi`: +19% to +49%
- `bench_sin_1pi` (precision_4): +3977% (!!)

Improved: `bench_sqrt2_plus_cbrt3` (-8% to -15%), `bench_integer_roots` (~-1.5%)

## Key Architectural Difference: Old (main) vs New (branch)

**Old (round-based):**
1. Compute flat budget: `ε / 2^⌈log₂(N)⌉` — O(1) bit shift
2. Dispatch all active refiners above budget
3. Block & collect ALL responses, checking precision after each
4. Safety valve: if all below budget but root not met, step widest
5. Repeat for next round

**New (event-loop):**
1. Compute propagated budgets: BFS through graph with UXBinary ops — O(graph_size)
2. Dispatch eligible, non-outstanding refiners above budget
3. Block for ONE response, drain immediately available
4. Loop back → recompute budgets → dispatch → collect → ...

## Root Causes Identified

### Cause 1: Budget recomputation overhead — O(R×M) per refinement
- Old: O(M) flat budget computations (1 per round, M rounds, O(1) each)
- New: O(R×M) propagated budget computations (1 per response, R refiners × M steps)
- For inv benchmark: 100 refiners × ~3 steps = ~300 BFS walks vs 3 bit-shifts
- Each BFS walk: ~200 nodes × UXBinary ops
- Explains inv (+120%) and sin recomputations at high refiner counts

### Cause 2: Sub-refiners never get budgets (always stepped)
- `compute_propagated_budgets` stopped at refiner boundaries (`is_refiner()`)
- Sub-refiners (e.g., PiOp children of SinOp) were NOT in the budget map
- Dispatch code: `propagated.get(id).is_some_and(...)` returns false → always dispatch
- In old model: ALL refiners got flat budget and could be skipped
- PiOp converges exponentially; each extra step doubles term count (quadratic cost)
- 100 PiOps stepped ~85 times instead of ~2 = catastrophic cost explosion
- **This is the dominant cause of sin (+97-235%) and sin_Npi (+3977%)**

## Experiments

### Experiment 1: Cache propagated budgets via generation counter (commit b147404)
Track `bounds_generation`/`budget_generation`, only recompute when they diverge.
**Result: No improvement.** Budgets still recomputed every iteration because every
response changes bounds (generation always diverges immediately).

### Experiment 2: Recompute only between dispatch waves
Only recompute when `budgets_stale && !any_outstanding`.
**Result: Helps inv and large-refiner-count benchmarks.** But doesn't help
2-refiner cases (sqrt2*pi) where each wave has only 1 response, so
recomputation happens every wave anyway.

### Experiment 3: Compute budgets once, never recompute (current solution)
Compute propagated budgets once before the loop, reuse throughout.
**Safe because:** bounds only tighten → for sensitivity-based budgets like
MulOp's `ε/(2·|sibling|)`, tighter sibling bounds → smaller `|sibling|_max`
→ looser child budget. So initial budgets (widest bounds) are the most
conservative and remain provably sufficient throughout.
**Result: Eliminates all budget recomputation overhead.**

### Fix for Cause 2: Propagate budgets through refiners
Removed the `if node.is_refiner() { continue; }` guard in
`compute_propagated_budgets`. Now the BFS walks through ALL nodes,
including refiners, using their `child_demand_budget` to propagate to children.
Sub-refiners (like PiOp) get proper budgets and can be skipped when precise enough.

## Local Benchmark Comparison (wall-clock, criterion, macOS)

| Benchmark | Main | With fixes | Change |
|-----------|------|-----------|--------|
| sin bits=64 | 57ms | 47ms | **-18%** |
| sin bits=256 | 2.93s | 2.71s | **-7%** |
| inv bits=64 | 3.41ms | 3.36ms | ~0% |
| sqrt2×pi bits=64 | 859µs | 861µs | ~0% |
| sqrt2×pi bits=256 | 3.21ms | 3.14ms | ~0% |
| sqrt2+cbrt3 bits=256 | — | — | improved |
