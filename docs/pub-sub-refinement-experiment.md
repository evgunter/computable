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

**Note:** First runs often show caching penalties. Results below are from 3 consecutive runs each.

### Raw Data (3 runs each)

| Benchmark | Run | Main | Strategy 1 | Strategy 2 | Strategy 3 |
|-----------|-----|------|------------|------------|------------|
| **Complex (0)** | 1 | 91ms | 120ms | 119ms | 94ms |
| | 2 | 100ms | 99ms | 98ms | 89ms |
| | 3 | 95ms | 86ms | 93ms | 95ms |
| **Summation (1)** | 1 | 350ms | 492ms | 361ms | 624ms |
| | 2 | 452ms | 392ms | 405ms | 419ms |
| | 3 | 352ms | 386ms | 392ms | 377ms |
| **Integer roots (2)** | 1 | 619ms | 699ms | 780ms | 1701ms |
| | 2 | 478ms | 740ms | 815ms | 1271ms |
| | 3 | 564ms | 719ms | 869ms | 942ms |
| **Inv (3)** | 1 | 30ms | 27ms | 29ms | 38ms |
| | 2 | 19ms | 25ms | 26ms | 19ms |
| | 3 | 20ms | 37ms | 26ms | 25ms |

### Averages

| Benchmark | Main | Strategy 1 | Strategy 2 | Strategy 3 |
|-----------|------|------------|------------|------------|
| Complex (0) | **95ms** | 101ms (+6%) | 103ms (+8%) | **92ms (-3%)** |
| Summation (1) | **385ms** | 423ms (+10%) | 386ms (+0%) | 473ms (+23%) |
| Integer roots (2) | **554ms** | 719ms (+30%) | 821ms (+48%) | 1305ms (+136%) |
| Inv (3) | **23ms** | 30ms (+30%) | 27ms (+17%) | 27ms (+17%) |

**Legend:** Negative % = faster than main (good), Positive % = slower (bad)

### Key Observations

1. **High variance:** Significant run-to-run variance, especially Integer roots on Main (478-619ms)
2. **Strategy 3 wins Complex:** 92ms avg vs 95ms main (-3%)
3. **Strategy 2 ties Summation:** 386ms vs 385ms main (+0%)
4. **Integer roots regression is consistent:** All strategies 30-136% slower
5. **First-run penalty:** Run 1 is often slower across all implementations

## Judge Evaluation Scores

| Strategy | Correctness | Performance | Code Quality | README Alignment | Total |
|----------|-------------|-------------|--------------|------------------|-------|
| Strategy 1 | 40/40 | 18/30 | 17/20 | 6/10 | **81/100** |
| Strategy 2 | 40/40 | 12/30 | 14/20 | 7/10 | **73/100** |
| Strategy 3 | 40/40 | 5/30 | 10/20 | 3/10 | **58/100** |

## Current Recommendation

**Do not merge any strategy yet.** Reasons:

1. **Performance regressions on Integer roots:** All strategies regress significantly (30-136% slower). This benchmark involves many refinement iterations and exposes scaling issues.

2. **Inv benchmark regression:** All strategies are 17-30% slower on inv, which underlies division and many computations.

3. **Mixed results on other benchmarks:**
   - Strategy 3 wins Complex by 3% (within noise)
   - Strategy 2 ties Summation
   - But these gains don't offset the Integer roots losses

4. **None truly implement pub/sub:** All still have coordinator managing refinement pace. True pub/sub would need autonomous refiners with subscriber mechanism.

5. **High variance:** Run-to-run variance is significant, making small improvements hard to distinguish from noise.

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
