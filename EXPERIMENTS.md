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

### Cause 3: Stale-input re-dispatch
Under valgrind's serialized threads, the event loop processes responses one at
a time. If a fast refiner (SinOp) completes before a slow one (PiOp), the
coordinator re-dispatches SinOp with stale PiOp bounds. Each re-dispatch
increments `num_terms` and recomputes sin with stale wide inner bounds —
quadratic cost growth. The old round model waits for all responses, avoiding
this pathology.

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

### Experiment 3: Compute budgets once, never recompute
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

### Experiment 4: Per-response precision checking + Vec budget lookup
**Problem found in Experiment 3 CI:** sin_10pi precision_4 went from +87% to
+4249%. Root cause: event-loop batches responses (block + drain), then checks
precision. When PiOp propagation gives SinOp tight bounds but SinOp's stale
response is also in the batch, the stale response OVERWRITES the tight bounds.
Old model checked precision after EACH response, allowing early exit.

**Fix:** Check precision_met after each apply_response, matching old model.
Also convert propagated budget HashMap → Vec<Option<UXBinary>> indexed by
refiner position for O(1) budget lookup (saves ~100 instructions per refiner
per loop iteration under valgrind).

### Experiment 5: Collect all outstanding before re-dispatch
Wait for every dispatched refiner before dispatching the next wave.
**Result:** Fixed sin_10pi (+4249% → -0.13%) and sin_2pi. But regressed
integer_roots (+8.5%) and lost the event-loop's non-blocking benefit for
independent refiners.

### Experiment 6: needs_redispatch guard (width comparison)
Track each refiner's previous response width. Only re-dispatch if bounds
improved (self-improving) or another refiner set the flag.
**Result:** Fixed sin_10pi when it worked, but fragile — SinOp's width
decreased by epsilon each step (more Taylor terms slightly tighten truncation
error), which counted as "self-improved." Sensitive to valgrind scheduling
order (different binary → different thread interleaving → sometimes works,
sometimes doesn't).

#### Width discontinuity discovered
While debugging width comparison failures, found that NthRootOp had a
two-phase `compute_bounds`: before first `refine_step`, returned conservative
bounds from `compute_initial_bounds`; after, returned bisection-based bounds.
The first `refine_step` could produce WIDER bounds (e.g. width 3 → 8 for
sqrt(4)) because the bisection's normalized prefix-form interval is wider
than the conservative estimate. Fixed by eagerly initializing bisection
state in `compute_bounds` (removed `compute_initial_bounds` and helpers,
-111 lines net).

### Experiment 7: Graph-structural leaf-refiner check (current)
Replace width comparison with a one-time graph walk at setup. For each
refiner, check if its `compute_bounds` subtree contains any other refiners:
- **Leaf refiners** (NthRootOp, InvOp): no refiners in subtree. Self-improving
  — each `refine_step` changes internal state producing tighter bounds. Always
  re-dispatch.
- **Non-leaf refiners** (SinOp): has PiOp children in subtree. `compute_bounds`
  reads other refiners' bounds. Only re-dispatch when another refiner responds
  (via `needs_redispatch` flag set in `record_completion`).

This is deterministic and robust — no sensitivity to epsilon width changes
or valgrind scheduling order. Preserves event-loop benefit: leaf refiners
advance freely, non-leaf refiners wait for meaningful input changes.

**CI Result:** All sin_Npi regressions eliminated. sin_10pi p4: -0.14%.
sin_100pi p4: -0.59%. Robust across binary changes (unlike width comparison).

## Valgrind vs Wall-Clock Divergence

Valgrind serializes threads, which hides the event-loop's parallelism benefits
and amplifies its pathologies. Local criterion (wall-clock) comparison shows
very different results from CI (Ir):

| Benchmark | CI (Ir) Δ | Wall-clock Δ | Notes |
|-----------|-----------|--------------|-------|
| sqrt2×pi (p4) | -5% | **-94%** | Parallelism invisible to valgrind |
| integer_roots (p1) | +5% | **-52%** | Same |
| inv (p3) | +137% | +103% | Both regressed, Ir overstates |
| sin_10pi (p4) | -0.14% | +34% | Stale-input effect is worse in wall-clock |
| sin (p3) | -22% | **-61%** | Both improved, wall-clock more so |
| sin (p4) | -7% | **-32%** | Same |

The event-loop architecture provides large wall-clock improvements for
expressions with asymmetric convergence (sqrt2×pi, integer_roots) that the
Ir metric cannot capture.

## CI Results (leaf-refiner check era, before further optimization)

| Benchmark | Original regression | Ir Δ at this point | Status |
|-----------|-------------------|------------|--------|
| bench_sin (p3) | +105% | **-22%** | Fixed |
| bench_sin (p4) | +235% | **-7%** | Fixed |
| bench_sin_1pi (p4) | +3977% | **-26%** | Fixed |
| bench_sin_10pi (p4) | +87% | **-0.14%** | Fixed |
| bench_sin_100pi (p4) | +86% | **-0.59%** | Fixed |
| bench_sin_2pi (p3) | +59% | **-56%** | Fixed |
| bench_sin_2pi (p4) | +86% | **-0.22%** | Fixed |
| bench_sqrt2_times_pi (p3-4) | +42-49% | **-5 to -6%** | Fixed |
| bench_sqrt2_plus_cbrt3 | -8-15% | **-9 to -22%** | Improved |
| bench_sqrt2_plus_const3 (p2-4) | +0-5% | **-13 to -15%** | Improved |
| bench_sqrt2_plus_pi (p3-4) | +3-5% | **-6 to -8%** | Improved |
| bench_complex | ~0% | **-8%** | Improved |
| bench_inv (p3-4) | +120% | +137% | Ir regressed (wall-clock +64-103%) |
| bench_inv (p0-2) | +15% | +19% | Small Ir regression |
| bench_sin (p0-1) | +97-174% | +15-26% | Partially fixed |
| bench_integer_roots | -1.5% | ~0% | Neutral in Ir (wall-clock -16 to -52%) |
| Various small (pi_half etc.) | — | +5-16% | Event-loop fixed overhead |

## Experiment 8: Static budget detection + targeted needs_redispatch

Investigation of the bench_inv +137% regression revealed it was NOT purely
structural. The smoking gun: `compute_propagated_budgets` at wave boundaries
did a BFS over 299 nodes with 200 BigUint multiplications (InvOp's
`child_demand_budget` computes `target * min_abs²`). For the inv benchmark's
pure-AddOp tree, these refreshes were entirely wasted — AddOp budgets are
`target >> depth`, independent of bounds.

### Fix 1: budget_is_static per refiner
Added `budget_depends_on_bounds()` trait method to `NodeOp` (default: false).
MulOp, PowOp (n>1), SinOp, InvOp, NthRootOp (n>1) return true. At setup,
walk root-to-refiner paths: if any op along the way has bounds-dependent
budgets, the refiner's budget is non-static. Only refresh non-static budgets
at wave boundaries. If all budgets are static, skip the entire BFS.

### Fix 2: Only mark non-leaf refiners for needs_redispatch
Leaf refiners' `compute_bounds` reads only their own subtree. `apply_update`
propagation from other refiners goes upward through parents, never into a
leaf refiner's subtree. So marking leaf refiners for redispatch when another
refiner responds is wasted work. For the inv benchmark (100 independent leaf
InvOps), this eliminates the O(100) scan per response entirely.

### Local wall-clock results (criterion, macOS)

| Benchmark | Main | Before opt | After opt | vs main |
|-----------|------|-----------|-----------|---------|
| inv bits=1 | 3.45ms | 11.0ms | 2.16ms | **-37%** |
| inv bits=64 | 3.58ms | 7.27ms | 3.71ms | **+4%** |
| inv bits=256 | 3.53ms | 5.79ms | 3.96ms | **+12%** |
| sin bits=64 | 57ms | 42.2ms | 43.7ms | **-23%** |

**CI Result (PR #59):** sqrt2_times_pi improved from +73% to **-49%** vs
first-merge baseline. inv neutral vs first-merge (improvement was in
eliminating the BFS, not the Ir delta vs the already-cached baseline).

## Experiment 9: Budget refresh gating for leaf-only expressions

The wave-boundary budget refresh fired ~256 times for sqrt2*pi (MulOp makes
`any_budget_dynamic = true`) even though both refiners are leaves. Leaf
refiners always self-redispatch; budget loosening doesn't change their dispatch.

**Fix:** Only refresh at wave boundaries if there are active non-leaf refiners
with dynamic budgets. For leaf-only expressions (inv, sqrt2*pi), this
eliminates all redundant BFS walks.

## Experiment 10: Dispatch queue + per-iteration overhead reduction

Investigation of inv +137% Ir found 75-100x more dispatch scans than the old
round model (one per loop iteration vs one per round). Three fixes:

1. **Dispatch queue** (VecDeque): replace O(N) full scan with O(1) amortized
   queue drain. Only refiners with `needs_redispatch` recently set are checked.
2. **Remove top-of-loop precision_met**: redundant (already checked per-response
   in collection phase and pre-loop). Eliminates ~1500 `root.get_bounds()` calls.
3. **eligible_count counter**: replace O(N) `any_eligible` scan with O(1) counter.

**Bug found:** Dispatch queue had duplicate entries (no dedup guard). Non-leaf
refiners got enqueued multiple times by successive PiOp responses. Fixed with
`in_queue: Vec<bool>` guard. But this was fragile — the fundamental issue was
that non-leaf refiners were marked for redispatch after ANY single sub-refiner
responded, not after ALL had responded.

## Experiment 11: All-sub-refiners-responded gate

Root cause of the recurring sin_Npi valgrind regressions: SinOp (non-leaf)
was dispatched after just ONE PiOp responded, with partially-updated inputs.
Under some valgrind schedules this caused the +4249% pathology, under others
it didn't — making the behavior fragile and binary-change-sensitive.

**Fix:** Track per-sub-refiner response status. For each non-leaf refiner,
maintain which of its sub-refiners have responded since its last dispatch.
Only mark for redispatch when ALL have responded. This is deterministic —
completely independent of thread scheduling order.

Edge cases handled:
- **Dedup sub_refiner_indices** for shared subexpressions (DAG case)
- **Exhausted sub-refiners** keep their "responded" status across dispatch
  resets (they won't respond again; clearing would make the gate unreachable)
- **Stall recovery** force-enables non-leaf refiners when all sub-refiners
  are budget-skipped (the gate can't open normally in that case)

**CI Result:** sin_10pi p4 at **-0.08%**, sin_2pi p4 at **-0.08%**,
sin_100pi p4 at **-0.08%**. All sin_Npi at precision_4 within ±0.2% of
original baseline. No catastrophic regressions across any scheduling order.

## Final CI Results (all optimizations, vs original pre-merge baseline)

| Benchmark | Original regression | Final Ir Δ | Status |
|-----------|-------------------|------------|--------|
| bench_sin (p3) | +105% | **-22%** | Fixed |
| bench_sin (p4) | +235% | **-7%** | Fixed |
| bench_sin_1pi (p4) | +3977% | **-26%** | Fixed |
| bench_sin_10pi (p4) | +87% | **~0%** | Fixed |
| bench_sin_100pi (p4) | +86% | **~0%** | Fixed |
| bench_sin_2pi (p4) | +86% | **~0%** | Fixed |
| bench_sin_2pi (p3) | +59% | +11% | Partially fixed |
| bench_sqrt2_times_pi (p3-4) | +42-49% | **-9 to -10%** | Fixed |
| bench_sqrt2_plus_cbrt3 (p0-4) | -8-15% | **-6 to -22%** | Improved |
| bench_sqrt2_plus_const3 (p1-4) | +0-5% | **-7 to -13%** | Improved |
| bench_sqrt2_plus_pi (p2-4) | +3-5% | **-8 to -12%** | Improved |
| bench_complex | ~0% | **-8%** | Improved |
| bench_inv (p3-4) | +120% | +137% | Ir regressed (wall-clock +3-12%) |
| bench_inv (p0-2) | +15% | +19% | Small Ir regression |
| bench_sin (p0-1) | +97-174% | +7-14% | Partially fixed |
| bench_integer_roots | -1.5% | +0.5% | Neutral |
| Various small (pi_half etc.) | — | +5-10% | Event-loop fixed overhead |

## Remaining Ir Regressions

### bench_inv +137% (Ir) / +3-12% (wall-clock)
The per-iteration overhead of the event loop (dispatch queue check, eligibility
check, precision_met per response) accumulates across ~1500 responses for 100
InvOp refiners. Under valgrind's serialized threads this is amplified; real
wall-clock impact is +3-12% due to thread parallelism. Further optimization
would require reducing the per-response overhead in the collection phase
(e.g., avoiding the `outstanding.iter().any()` scan, or batching responses
more aggressively).

### bench_sin low precision +7-14% (Ir)
Event-loop fixed overhead dominates when total work is tiny. At higher
precision where computation dominates, sin shows -7 to -22% improvement.

### Small constant overhead (+5-10%)
Benchmarks with few refiners and low precision show small regression from
the event-loop's per-iteration overhead. This is the fixed cost of the
architecture, offset by parallelism benefits invisible to valgrind.

## Experiment 12: Per-refiner precision_bits + incremental arctan caching

The dispatch code was forwarding the raw root tolerance to all refiners as
`precision_bits`, ignoring their individual propagated budgets. When a
refiner's budget was tighter than the root tolerance (e.g., pi feeding into
a multiplication), PiOp's leap formula underestimated needed terms, the
doubling fallback fired, and `compute_prefix` recomputed the entire Taylor
series from scratch each call.

### Changes
1. **Per-refiner precision_bits** (`refinement.rs`): Derive `precision_bits`
   from each refiner's propagated budget via `budget_to_precision_bits()`,
   falling back to global tolerance for unreachable refiners.
2. **Remove doubling fallback** (`pi.rs`): With correct per-refiner budgets,
   the leap formula always produces `needed > num_terms` when dispatched.
   Replaced with `unreachable!`.
3. **Incremental arctan caching** (`pi.rs`): `ArctanCache` stores partial
   Taylor sums and power-of-k state. `compute_prefix` extends from cached
   state in O(delta) instead of recomputing from scratch. Caches are
   invalidated only when precision requirements exceed cached precision.

### Criterion wall-clock results (macOS, vs main branch)

| Benchmark | bits=1 | bits=4 | bits=16 | bits=64 | bits=256 |
|-----------|--------|--------|---------|---------|----------|
| pi_refinement | -11% | -9% | -15% | -7% | **-21%** |
| 2*pi | -13% | -6% | -3% | -10% | **-23%** |
| pi/2 | ~0% | ~0% | -4% | -7% | **-12%** |
| pi² | ~0% | -7% | ~0% | -10% | **-11%** |
| 1/pi | **-14%** | **-18%** | **-21%** | **-26%** | -7% |
| sin(pi) | ~0% | ~0% | -3% | **-17%** | **-20%** |
| sin(2pi) | ~0% | ~0% | **-15%** | -6% | ~0% |
| sin(10pi) | ~0% | ~0% | +3% | **-20%** | ~0% |
| sin(100pi) | ~0% | ~0% | +1% | +4% | ~0% |

No statistically significant regressions. Largest improvements at higher
precision where the incremental caching avoids the most redundant work.
The `1/pi` benchmark improved up to -26% because InvOp's budget propagation
now correctly informs PiOp's precision target.
