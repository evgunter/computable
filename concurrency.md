# Concurrency Model

## Overview

Refinement is parallel: each refiner runs in its own scoped thread, and a single coordinator thread orchestrates them via crossbeam channels. The coordinator uses an event-loop model that processes responses one at a time, allowing fast refiners to advance while slow ones are still computing.

## Architecture

```
                        ┌─────────────┐
                        │ Coordinator │
                        │ (caller's   │
                        │  thread)    │
                        └──────┬──────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
          Step/Stop        Step/Stop         Step/Stop
         (command_tx)     (command_tx)      (command_tx)
              │                │                 │
              ▼                ▼                 ▼
        ┌───────────┐    ┌───────────┐     ┌───────────┐
        │ Refiner 0 │    │ Refiner 1 │     │ Refiner N │
        │  (thread) │    │  (thread) │ ... │  (thread) │
        └─────┬─────┘    └─────┬─────┘     └─────┬─────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                    Update/Exhausted/Error
                      (shared update_tx)
                               │
                               ▼
                        ┌─────────────┐
                        │ Coordinator │
                        │ (update_rx) │
                        └─────────────┘
```

Each refiner has its own command channel (`command_tx` → `command_rx`). All refiners share a single update channel back to the coordinator (`update_tx` → `update_rx`). This is a many-to-one fan-in pattern.

## Thread Lifecycle

**Spawning.** `thread::scope` creates a scope in which all refiner threads are spawned. The scope guarantees all threads are joined before `refine_to` returns, even on early exit or error.

**Refiner loop.** Each refiner thread runs `refiner_loop`, which blocks on `commands.recv()` waiting for `Step` or `Stop`. On `Step`, it calls `node.refine_step()`, computes new bounds, and sends the result back on the shared `update_tx`. On `Stop`, channel disconnect, or exhaustion (converged / state unchanged), the thread exits.

**Shutdown.** When the coordinator finishes (precision met, error, or iteration limit), `shutdown_refiners` sets a `StopFlag` (atomic bool) and sends `Stop` to each refiner. The `StopFlag` is checked at the top of each refiner loop iteration, but a refiner won't see it until its current `refine_step()` completes — there is no preemption mid-step.

## Coordinator Event Loop

The coordinator runs a loop with four phases per iteration:

### 1. Precision Check
Calls `get_bounds()` on the root node and checks if the width meets the tolerance. If so, returns immediately.

### 2. Eligibility and Termination
A refiner is *eligible* if it is active (not exhausted) and under the per-refiner step limit. If no refiners are eligible and none have outstanding (in-flight) steps, the coordinator returns an error: `StateUnchanged` if all refiners exhausted without changing state, `MaxRefinementIterations` otherwise.

### 3. Dispatch
The coordinator computes a **demand budget** = `ε / 2^⌈log₂(N)⌉` where `ε` is the target precision and `N` is the active refiner count. Each eligible, non-outstanding refiner whose cached width exceeds the budget is sent a `Step` command.

If all eligible refiners are below the demand budget (demand-skipped) but the root precision isn't met, a **safety valve** fires: step the least-precise eligible refiners, skipping extreme outliers whose width is negligible compared to the widest.

If nothing was dispatched and nothing is outstanding, a **stall guard** returns `MaxRefinementIterations` to prevent deadlock (this can happen when all eligible refiners are both demand-skipped and safety-valve dominated).

### 4. Response Collection
Block on `recv()` for the first response. Drain any additional responses that arrived in the meantime via `try_recv()` — this batching avoids O(N^2) overhead when many refiners respond near-simultaneously. Then check precision for early exit.

The key design choice: after collecting available responses, the loop immediately returns to the dispatch phase. If some refiners from the previous batch are still outstanding (in-flight), the coordinator does not wait for them. Instead, it re-dispatches any eligible refiners that have completed, allowing fast refiners to advance while slow ones continue computing. The `outstanding` flag per refiner prevents double-dispatch: a refiner is only sent a new `Step` after its previous response has been received.

## Channels

All channels are crossbeam unbounded MPSC channels.

- **Command channels** (one per refiner): `Sender<RefineCommand>` held by coordinator, `Receiver<RefineCommand>` held by refiner thread. Carries `Step { precision_bits }` or `Stop`.
- **Update channel** (shared): all refiners clone the same `Sender<RefinerMessage>`, coordinator holds the single `Receiver<RefinerMessage>`. The coordinator drops its clone of `update_tx` after spawning all refiners, so `update_rx` disconnects when all refiner threads exit.

## Bounds Propagation

When the coordinator receives an update, `apply_update` sets the updated node's bounds and propagates upward through the graph via BFS. Each parent recomputes its bounds from its children; if the recomputed bounds differ from the cached value, the parent's bounds are updated and its own parents are queued. This ensures the root's bounds always reflect the latest refiner state.

## Stop Flag

`StopFlag` wraps an `AtomicBool` with `Relaxed` ordering. It is monotonic (false → true only). Refiners check it at the top of each loop iteration, but not during `refine_step()`. This means a slow `refine_step` will run to completion even after the flag is set — the refiner exits on the next loop check. A future improvement could thread the stop flag into expensive computations for mid-step cancellation (see `kill-slow-refiners` in TODOS.md).

## Concurrency Invariants

- **No shared mutable state between refiners.** Each refiner operates on its own `Arc<Node>` and communicates exclusively through channels. Node bounds are updated via `set_bounds` (interior mutability with `parking_lot::RwLock`), but only the coordinator calls `set_bounds` on non-refiner nodes, and each refiner only calls `set_bounds` on its own node.
- **Single writer for bounds propagation.** Only the coordinator thread calls `apply_update`, which propagates bounds up the tree. Refiner threads set bounds on their own leaf nodes only.
- **Outstanding tracking prevents double-dispatch.** The `outstanding[i]` flag ensures a refiner is never sent a second `Step` before its first response arrives. This is coordinated entirely within the single coordinator thread — no synchronization needed.
- **Graceful shutdown.** The coordinator sends `Stop` to all refiners and `thread::scope` waits for all threads to join. Channel disconnection (from either side) is handled as a normal exit path, not a panic.
