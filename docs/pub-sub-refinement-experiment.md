# Pub/Sub Refinement Experiment Status

**Date:** 2026-01-21
**Status:** In Progress - No strategy ready to merge yet

## Overview

This document tracks an experiment to switch from lock-step refinement to an event-driven pub/sub model as described in the README. Three different approaches were implemented and benchmarked.

## Branches

All branches are based on `main` at commit `6f8e2b5`:

| Branch | Strategy | Status |
|--------|----------|--------|
| `try-strategies-1769039545-strategy-1` | Event-driven select! | All tests pass |
| `try-strategies-1769039545-strategy-2` | Continuous refiners | All tests pass |
| `try-strategies-1769039545-strategy-3` | Decentralized pub/sub | All tests pass |

## Strategy Descriptions

### Strategy 1: Event-driven select! with Coordinator Pacing

**File:** `src/refinement.rs`

- Removes `RefineCommand` enum entirely
- Coordinator uses `recv()` to receive updates as they arrive (not lock-step batching)
- Refiners wait for "continue" signal after each update via per-refiner channel
- Uses `node_to_handle` HashMap to route continue signals

**Key change:** Event-driven reception but still coordinator-paced.

### Strategy 2: Continuous Refiners with Backpressure

**File:** `src/refinement.rs`

- Refiners run continuously without waiting for commands
- Uses `bounded(0)` channel (synchronous) for natural backpressure
- Removes `RefineCommand` enum entirely
- Adds 100us backoff sleep when `refine_step()` returns `Ok(false)`
- **Bonus:** Fixes `StopFlag` memory ordering (Relaxed → Release/Acquire)

**Key change:** Simplest refiner loop, backpressure via bounded channel.

### Strategy 3: Decentralized Pub/Sub with recv_timeout

**File:** `src/refinement.rs`

- Uses `recv_timeout(Duration::from_millis(10))` instead of blocking `recv()`
- Allows periodic stop flag checks for graceful shutdown
- Adds `HashSet` visited tracking in `apply_update` to prevent cycles
- Keeps `RefineCommand` enum and lock-step structure

**Key change:** Timeout-based polling for shutdown, still fundamentally lock-step.

## Benchmark Results

All benchmarks run with `cargo run --package computable-benchmarks --release -- <index>`.

| Benchmark | Main | Strategy 1 | Strategy 2 | Strategy 3 |
|-----------|------|------------|------------|------------|
| Complex (0) | 110ms | **92ms** (-16%) | 126ms (+15%) | 102ms (-7%) |
| Summation (1) | 361ms | 393ms (+9%) | 366ms (+1%) | 442ms (+22%) |
| Integer roots (2) | 504ms | 674ms (+34%) | 731ms (+45%) | 1186ms (+135%) |
| Inv (3) | 21ms | 30ms (+43%) | 27ms (+29%) | 34ms (+62%) |

**Legend:** Negative % = faster than main (good), Positive % = slower (bad)

## Judge Evaluation Scores

| Strategy | Correctness | Performance | Code Quality | README Alignment | Total |
|----------|-------------|-------------|--------------|------------------|-------|
| Strategy 1 | 40/40 | 18/30 | 17/20 | 6/10 | **81/100** |
| Strategy 2 | 40/40 | 12/30 | 14/20 | 7/10 | **73/100** |
| Strategy 3 | 40/40 | 5/30 | 10/20 | 3/10 | **58/100** |

## Current Recommendation

**Do not merge any strategy yet.** Reasons:

1. **Performance regressions:** All strategies regress on most benchmarks. Strategy 1's Complex win is offset by Integer roots (+34%) and Inv (+43%) losses.

2. **Inv benchmark is critical:** The `inv` operation underlies division and many computations. 30-60% regression is unacceptable.

3. **Integer roots shows scaling issues:** Up to 135% regression suggests these approaches break down for operations requiring many refinement iterations.

4. **None truly implement pub/sub:** All still have coordinator managing refinement pace. True pub/sub would need autonomous refiners with subscriber mechanism.

## Action Items for Future Work

1. **Extract memory ordering fix:** Strategy 2's `StopFlag` fix (Release/Acquire ordering) is a legitimate correctness improvement. Should be merged independently.

2. **Investigate Complex benchmark:** Why does Strategy 1 perform better there? Could inform targeted optimization.

3. **Consider updating README:** The current lock-step implementation works well. Maybe document that instead of implementing pub/sub.

4. **Profile Integer roots:** Why do event-driven approaches regress so badly here? Understanding this could unlock better approaches.

## Previous Attempt Issues (Round 1)

The first round of implementations had critical bugs:

- **Strategy 1:** Refiners exited when `refine_step()` returned `Ok(false)`, causing premature termination
- **Strategy 2:** Progressive slowdown caused benchmark timeouts; iteration counting was wrong
- **Strategy 3:** Deadlocks due to blocking `recv()` without stop flag checks

These were fixed in Round 2 by:
- Strategy 1: Refiners continue waiting for coordinator signals even when no local progress
- Strategy 2: Removed progressive slowdown, fixed iteration count (multiply by refiner count)
- Strategy 3: Changed to `recv_timeout` for periodic stop flag checks

## How to Resume This Work

1. Check out a strategy branch:
   ```bash
   git checkout try-strategies-1769039545-strategy-1
   ```

2. Run tests:
   ```bash
   cargo test --lib
   ```

3. Run benchmarks:
   ```bash
   cargo run --package computable-benchmarks --release -- 0 1 2 3
   ```

4. Compare with main:
   ```bash
   git diff main src/refinement.rs
   ```

## Worktree Locations

If worktrees still exist:
- Strategy 1: `../strategy-1`
- Strategy 2: `../strategy-2`
- Strategy 3: `../strategy-3`

To recreate worktrees:
```bash
git worktree add ../strategy-1 try-strategies-1769039545-strategy-1
git worktree add ../strategy-2 try-strategies-1769039545-strategy-2
git worktree add ../strategy-3 try-strategies-1769039545-strategy-3
```
