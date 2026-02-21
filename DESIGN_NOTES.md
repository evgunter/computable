# Design Notes: Demand-Driven Refinement

## Context

The previous experiment (`experiment/event-driven-refinement`) converted the
lock-step refinement model to a fully event-driven model where refiners run
continuously. This caused 2x-507x performance regressions because refiners
race ahead to extreme precision before the coordinator can stop them.

See `EXPERIMENT_RESULTS.md` on that branch for full investigation results.

## Problem statement

We need concurrent refinement that:
1. Allows refiners to run in parallel (the whole point)
2. Prevents refiners from computing at precision far beyond what's needed
3. Doesn't reintroduce the lock-step barrier (where fast refiners wait for slow ones)

## Approaches considered

### A. Pure event-driven with bounded channel (TESTED - failed)

Refiners push updates freely through a bounded channel. The bounded channel
provides backpressure but doesn't prevent over-computation because:
- Channel capacity = num_refiners allows each refiner to be many steps ahead
- Exponential doubling (InvOp) means each extra step doubles precision
- The coordinator is the bottleneck (sequential message processing)

Result: 507x slower on inv benchmark.

### B. Shared precision target (atomic)

Give each refiner access to a shared `AtomicU64` representing the coordinator's
current estimate of how much precision is still needed. Refiners check this
before each step and skip/sleep if they've already exceeded it.

Pros: Simple, low overhead, no coordinator changes needed.
Cons: A single global target doesn't capture per-node precision needs. A leaf
node might need more precision while the root already has enough. Also doesn't
work well for non-precision-based refiners (bisection).

### C. Pull-based / demand-driven

Invert the control flow: instead of refiners pushing updates, the coordinator
requests specific work from specific refiners. The coordinator maintains a
priority queue of "which refiner step would most improve root precision" and
dispatches work accordingly.

Pros: Perfect demand signaling — no wasted computation. Can prioritize the
most impactful refiners.
Cons: Requires the coordinator to understand which refiner to step next,
which may be complex. Single-threaded coordinator becomes the bottleneck
for dispatching.

### D. Cooperative round-based (parallel within rounds)

Keep the round-based structure but parallelize within each round: all refiners
compute their step concurrently (using thread::scope or rayon), then the
coordinator collects all updates and propagates. This is essentially the
lock-step model but with concurrent execution of each round's steps.

Pros: Preserves the lock-step model's demand pacing (1 step per round).
Concurrent within rounds. Simple to implement.
Cons: Still has the barrier problem — fast refiners wait for slow ones within
each round. But this may be acceptable if most refiners take similar time.

### E. Event-driven with per-refiner precision caps

Each refiner has a shared `AtomicI64` precision cap set by the coordinator.
Refiners check this cap before each step. When the coordinator processes an
update and propagates bounds, it can estimate how much more precision each
refiner needs and update caps accordingly.

Pros: Fine-grained per-refiner control. Refiners still run concurrently.
Cons: Estimating per-refiner precision needs is non-trivial. Over-estimation
wastes work, under-estimation requires cap updates and potential stalls.

### F. Hybrid: parallel rounds with early exit

Like (D) but refiners within a round can exit early if a shared "round done"
flag is set. The coordinator checks precision after each refiner completes
(not after ALL complete). If precision is met, set the flag so remaining
refiners in the round skip their work.

Pros: No barrier for the critical path. Preserves round-based pacing.
Cons: Marginal improvement over lock-step — still round-based, still has
some synchronization overhead.

## Selected approach

_To be filled in after analysis._
