# Experiment: Round-Based Parallel Refinement with Early Exit and Per-Refiner Exhaustion

## Branch: `experiment/demand-driven-refinement`

## Summary

Extended the lock-step refinement coordinator with two improvements:
1. **Early exit**: check precision after each individual update (not after all refiners complete)
2. **Per-refiner exhaustion**: track and skip exhausted refiners instead of failing

This preserves the demand pacing that prevents precision overshoot (the fatal
flaw of the event-driven experiment) while addressing the lock-step model's
limitations.

## Architecture

### Before (lock-step)

```
loop {
    check precision → return if met
    check iteration cap → error if exceeded
    send Step to ALL refiners
    wait for ALL N responses
    apply ALL updates
}
```

Limitations:
- No early exit within rounds: waits for ALL refiners before checking precision
- Any refiner error kills the entire refinement

### After (round-based with early exit + exhaustion tracking)

```
loop {
    check precision → return if met
    count active refiners → if 0, return exhaustion error
    check iteration cap → error if exceeded
    send Step to ACTIVE refiners only
    for each response:
        apply update
        check precision → return if met (EARLY EXIT)
        if Exhausted → mark inactive, track reason
        if Error → return error
}
```

### Key types added

```rust
enum ExhaustionReason {
    Converged,      // refine_step() returned Ok(false)
    StateUnchanged, // refine_step() returned Err(StateUnchanged)
}

enum RefinerMessage {
    Update(NodeUpdate),
    Exhausted { update: NodeUpdate, reason: ExhaustionReason },
    Error(ComputableError),
}
```

### Exhaustion error selection

When all refiners exhaust before precision is met:
- If ALL were `StateUnchanged` → return `StateUnchanged`
- Otherwise → return `MaxRefinementIterations`

### BaseOp convergence fix

`BaseOp::refine_step` now checks if bounds are a point after refine succeeds:
```rust
fn refine_step(&self) -> Result<bool, ComputableError> {
    self.base.refine()?;
    let bounds = BoundsAccess::get_bounds(self.base.as_ref())?;
    if bounds.small() == &bounds.large() {
        return Ok(false);  // converged
    }
    Ok(true)
}
```

This is needed because `TypedBaseNode::refine()` returns `Ok(())` (not error)
for constants and already-converged values, causing the old code to loop forever
on converged base nodes.

### Changes made

- `src/refinement.rs`: New enums, rewritten coordinator + refiner_loop, updated docs
- `src/ops/base.rs`: Convergence detection in refine_step

## Benchmark Results

Back-to-back comparison on the same machine, release mode (`cargo run -p computable-benchmarks --release`).
Baseline = merge-base commit `f6f97e7` (lock-step, before any changes).

### Main benchmarks

| Benchmark | Baseline (lock-step) | This branch | Ratio |
|-----------|---------------------|-------------|-------|
| Complex expression (5000 samples) | 78.3ms | 79.6ms | 1.02x |
| Summation (200k samples) | 319.2ms | 309.4ms | 0.97x |
| Integer roots (1000 samples) | 438.2ms | 356.8ms | **0.81x (19% faster)** |
| Inverse (100 samples, 256-bit) | 12.2ms | 9.2ms | **0.75x (25% faster)** |
| Sine (100 samples, 128-bit) | 2,196ms | 408.5ms | **0.19x (5.4x faster)** |

### Pi refinement (refine_to)

| Precision | Baseline | This branch | Ratio |
|-----------|----------|-------------|-------|
| 32 bits | 21.9µs | 23.8µs | 1.09x |
| 64 bits | 92.7µs | 92.7µs | 1.00x |
| 128 bits | 191.0µs | 196.5µs | 1.03x |
| 256 bits | 388.4µs | 413.9µs | 1.07x |
| 512 bits | 866.3µs | 904.6µs | 1.04x |
| 1024 bits | 2,140µs | 2,217µs | 1.04x |

### Pi bounds (direct computation)

| Precision | Baseline | This branch | Ratio |
|-----------|----------|-------------|-------|
| 32 bits | 20.0µs | 21.6µs | 1.08x |
| 64 bits | 37.3µs | 40.0µs | 1.07x |
| 128 bits | 72.9µs | 82.2µs | 1.13x |
| 256 bits | 159.4µs | 171.7µs | 1.08x |
| 512 bits | 372.3µs | 390.3µs | 1.05x |
| 1024 bits | 1,031µs | 977.8µs | 0.95x |

### Pi arithmetic

| Operation | Baseline | This branch | Ratio |
|-----------|----------|-------------|-------|
| 2π | 132.0µs | 122.7µs | 0.93x |
| π/2 | 127.5µs | 122.1µs | 0.96x |
| π² | 201.2µs | 147.3µs | 0.73x |
| 1/π | 336.3µs | 219.5µs | 0.65x |

### High-precision pi

| Precision | Baseline | This branch | Ratio |
|-----------|----------|-------------|-------|
| 2048 bits | 6.53ms | 6.60ms | 1.01x |
| 4096 bits | 27.9ms | 26.5ms | 0.95x |
| 8192 bits | 223.9ms | 137.2ms | **0.61x (39% faster)** |

### sin(kπ) benchmarks

| Expression | Baseline | This branch | Ratio |
|-----------|----------|-------------|-------|
| sin(π) | 4.37ms | 3.08ms | 0.70x |
| sin(2π) | 3.23ms | 2.62ms | 0.81x |
| sin(10π) | 7.71ms | 7.23ms | 0.94x |
| sin(100π) | 7.36ms | 7.15ms | 0.97x |

## Comparison with Event-Driven Experiment

The event-driven experiment (`experiment/event-driven-refinement`) showed 2x-507x
regressions because refiners raced to extreme precision before the coordinator
could stop them. InvOp refiners reached 4M-bit precision when only 512 was needed.

| Benchmark | Event-driven | This branch | Improvement |
|-----------|-------------|-------------|-------------|
| Integer roots | 2.1x slower | 0.81x (faster) | Fixed |
| Inverse (1/x) | 507x slower | 0.75x (faster) | Fixed |
| Sine | 2.4x slower | 0.19x (faster) | Fixed |
| Pi refinement (high prec) | 1.9x slower | 1.04x (same) | Fixed |

**All event-driven regressions are eliminated.** The demand-pacing model
(refiners only advance when the coordinator sends Step) prevents precision
overshoot entirely.

## Analysis

### Why early exit helps

The sine benchmark shows the most dramatic improvement (5.4x faster). This is
because the sin computation has many refiners (pi + input + sin series terms),
and in the old lock-step model, the coordinator had to wait for ALL refiners to
complete each round before checking if precision was already sufficient. With
early exit, the coordinator can return as soon as any single update pushes the
root bounds past the target.

### Why exhaustion tracking helps

Integer roots and inverse benchmarks improve because:
1. `BaseOp::refine_step` now correctly detects convergence (bounds are a point)
2. Converged refiners are skipped in future rounds, avoiding unnecessary work
3. Constants converge immediately, so in expressions like `1/x`, the constant
   base node stops after 1 round while the InvOp refiner continues

### Why pi benchmarks show slight overhead

Pi refinement has a single refiner chain (no parallelism to exploit). The early
exit check after each update adds a small overhead (~4%) compared to the simpler
lock-step loop. This is within noise for practical use.

### Precision overshoot: resolved

The fundamental problem from the event-driven experiment was that refiners
advanced without coordinator permission, causing exponential precision overshoot.
The round-based model preserves demand pacing: refiners only call `refine_step()`
when the coordinator sends `Step`, and the coordinator only sends `Step` when it
needs more precision. This makes precision overshoot impossible by construction.

## Correctness

- All 198 tests pass (including all 17 refinement tests)
- 20/20 flakiness runs clean
- `cargo clippy` clean
- `cargo fmt --check` clean
- No test assertion changes required
