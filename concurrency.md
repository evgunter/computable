# Concurrency Model

## Overview

Refinement is parallel: each refiner runs in its own scoped thread, and a single coordinator thread orchestrates them via crossbeam channels. The coordinator uses an event-loop model that processes responses one at a time, allowing fast refiners to advance while slow ones are still computing. Demand budgets propagate top-down through every node in the computation graph (including refiners), controlling which refiners are dispatched. A dispatch queue tracks only the refiners whose inputs have changed, avoiding a full scan each iteration.

## Architecture

```
                        +---------------+
                        |  Coordinator  |
                        |  (caller's    |
                        |   thread)     |
                        +-------+-------+
                                |
           +--------------------+---------------------+
           |                    |                     |
       Step/Stop            Step/Stop             Step/Stop
      (command_tx)         (command_tx)          (command_tx)
           |                    |                     |
           v                    v                     v
     +-----------+        +-----------+         +-----------+
     | Refiner 0 |        | Refiner 1 |         | Refiner N |
     |  (thread) |        |  (thread) |   ...   |  (thread) |
     +-----+-----+        +-----+-----+         +-----+-----+
           |                    |                     |
           +--------------------+---------------------+
                                |
                     Update/Exhausted/Error
                       (shared update_tx)
                                |
                                v
                        +---------------+
                        |  Coordinator  |
                        |  (update_rx)  |
                        +---------------+
```

Each refiner has its own command channel (`command_tx` $\to$ `command_rx`). All refiners share a single update channel back to the coordinator (`update_tx` $\to$ `update_rx`). This is a many-to-one fan-in pattern.

## Thread Lifecycle

**Spawning.** `thread::scope` creates a scope in which all refiner threads are spawned. The scope guarantees all threads are joined before `refine_to` returns, even on early exit or error. Refiners whose cached prefix is already exact (width exponent $= -\infty$) are not spawned.

**Refiner loop.** Each refiner thread runs `refiner_loop`, which blocks on `commands.recv()` waiting for `Step` or `Stop`. On `Step`, it calls `node.refine_step()`, computes the new prefix via `node.compute_prefix()`, writes it with `node.set_prefix()`, and sends the result back on the shared `update_tx`. On `Stop`, channel disconnect, or exhaustion (converged / state unchanged), the thread exits.

**Shutdown.** When the coordinator finishes (precision met, error, or iteration limit), `shutdown_refiners` sets a `StopFlag` (atomic bool) and sends `Stop` to each refiner. The `StopFlag` is checked at the top of each refiner loop iteration, but a refiner will not see it until its current `refine_step()` completes --- there is no preemption mid-step.

## Coordinator Event Loop

The coordinator runs a loop with the following phases per iteration:

### 1. Eligibility and Termination

A refiner is *eligible* if it is active (not exhausted) and under the per-refiner step limit (`MAX_REFINEMENT_ITERATIONS`). An `eligible_count` counter is maintained incrementally: it is decremented when a refiner exhausts or hits the step limit, giving $O(1)$ eligibility checks. If `eligible_count == 0` and no refiner is outstanding (in-flight), the coordinator returns an error: `StateUnchanged` if all refiners exhausted without changing state, `MaxRefinementIterations` otherwise.

### 2. Budget Computation

Per-refiner demand budgets are computed by `compute_propagated_budgets`, which walks the entire computation graph top-down from the root. Starting with the root's target width $\varepsilon$, each node's `child_demand_budget` method maps the node's budget to budgets for its children. This propagation passes through all node types --- passive combinators and refiners alike. For DAG nodes (shared subexpressions under multiple parents), the tightest (minimum) budget is kept. Operations implement `child_demand_budget` as follows:

| Operation | Budget for child $i$ | `budget_depends_on_prefix` |
|-----------|---------------------|----------------------------|
| NegOp | $\varepsilon$ (pass-through) | false |
| AddOp | $\varepsilon/2$ | false |
| MulOp ($a \cdot b$) | $\varepsilon / (2 \cdot \text{max\_abs}(\text{sibling}))$ | true |
| PowOp ($x^n$) | $\varepsilon / (n \cdot \text{max\_abs}(x)^{n-1})$ | true ($n > 1$) |
| InvOp ($1/x$) | $\varepsilon \cdot \text{min\_abs}(x)^2$ | true |
| NthRootOp ($x^{1/n}$) | $\varepsilon \cdot n \cdot \text{min\_abs}(x)$ | true ($n > 1$) |
| SinOp (child 0: input) | $\varepsilon$ ($|\sin'| \le 1$) | true |
| SinOp (child 1: $\pi$) | $\varepsilon \cdot \pi_{\text{lower}} / \text{max\_abs}(\text{input})$ | true |
| BaseOp, PiOp | unreachable (no children) | false |

The propagated budgets are provably sufficient: if every refiner meets its budget, the root meets the target. This follows from the sensitivity bounds at each node (for MulOp, $w_{\text{out}} \le |a| \cdot w_b + |b| \cdot w_a$ is tight because $|a|$ uses the interval endpoint, not the center, so the cross-term $w_a \cdot w_b$ is already captured).

**Selective refresh.** Budgets are computed once at startup. At wave boundaries (no outstanding refiners), budgets are selectively refreshed only when there exist active non-leaf refiners whose path from the root passes through at least one node with `budget_depends_on_prefix() == true`. A per-refiner `budget_is_static` flag, computed at initialization by BFS from the root, records whether the entire root-to-refiner path uses only static-budget ops (AddOp, NegOp). On refresh, only refiners with `budget_is_static == false` have their budgets updated. This avoids recomputing budgets for pure AddOp/NegOp subtrees where budgets are static functions of the tolerance.

### 3. Dispatch

The coordinator drains a **dispatch queue** (`VecDeque<usize>` + per-refiner `in_queue` flag). Initially all refiners are queued. Thereafter, only refiners whose `needs_redispatch` flag was set are enqueued, avoiding a full $O(N)$ scan each iteration.

For each refiner popped from the queue, dispatch proceeds if all of:
- `active[i]` (not exhausted)
- `!outstanding[i]` (no in-flight step)
- `steps[i] < MAX_REFINEMENT_ITERATIONS` (under step limit)
- `needs_redispatch[i]` (inputs may have changed)

The refiner is then budget-checked: if its cached prefix width $\le$ its budget, it is skipped. Otherwise, a `Step { precision_bits }` command is sent, `outstanding[i]` is set, `needs_redispatch[i]` is cleared, and for non-leaf refiners the sub-responded tracking is reset.

**Leaf vs non-leaf refiners.** At initialization, a subtree walk from each refiner determines `is_leaf_refiner[i]`: true if the refiner's subtree contains no other refiners. Leaf refiners are self-improving (each step narrows their own prefix independently), so they are always re-dispatched after responding. Non-leaf refiners depend on their sub-refiners' prefixes, so they are only re-dispatched when all sub-refiners have responded (the all-sub-refiners-responded gate, described below).

**All-sub-refiners-responded gate.** For each non-leaf refiner $i$, `sub_refiner_indices[i]` lists the indices of refiners in its subtree. `sub_responded[i]` and `sub_responded_count[i]` track which sub-refiners have responded since $i$'s last dispatch. When $i$ is dispatched, the counters are reset (keeping exhausted sub-refiners pre-marked as responded). When a sub-refiner responds and propagation reaches $i$'s node, that sub-refiner is marked responded. Only when `sub_responded_count[i] >= sub_refiner_indices[i].len()` is `needs_redispatch[i]` set, ensuring that `compute_prefix` reads fully-updated inputs rather than partially-stale ones.

**Stall recovery.** If nothing was dispatched and nothing is outstanding, the coordinator checks for active, above-budget refiners that are blocked only because `needs_redispatch` is false (their sub-refiner gate never opened). If any such refiners exist, they are force-enabled and re-queued, and dispatch retries. If none exist, it is a true stall and `MaxRefinementIterations` is returned.

### 4. Response Collection

Block on `recv()` for the first response. Drain any additional responses that arrived in the meantime via `try_recv()` --- this batching avoids $O(N^2)$ overhead when many refiners respond near-simultaneously.

**Per-response processing.** For each response:

1. `apply_response` calls `apply_update`, which sets the refiner's prefix and propagates upward through the graph via BFS. Each parent recomputes its prefix; if it changed, the parent is updated and its own parents are queued. `apply_update` returns the list of changed node IDs along the propagation path.

2. `record_completion` updates bookkeeping:
   - Clears `outstanding[idx]`, increments `steps[idx]`.
   - If exhausted or at step limit, decrements `eligible_count`.
   - **Leaf refiner:** unconditionally sets `needs_redispatch[idx]` and enqueues it.
   - **Non-leaf refiner:** if already marked for redispatch (all subs responded while it was outstanding), enqueues it.
   - **Propagation-based redispatch:** iterates the changed node IDs from `apply_update`. For each changed ID that is a non-leaf refiner $j \ne idx$, checks if the responding refiner $idx$ is a sub-refiner of $j$. If so, marks it as responded in $j$'s gate. When $j$'s gate is full, sets `needs_redispatch[j]` and enqueues $j$.

3. **Per-response precision check.** After each response (not just at the top of the loop), the coordinator checks if the root prefix meets the tolerance. This allows early exit mid-batch when `apply_update` propagation tightens the root sufficiently.

After draining all available responses, the loop returns to the dispatch phase. Outstanding refiners from the previous batch may still be in flight --- the coordinator does not wait for them, allowing fast refiners to advance while slow ones continue computing.

## Channels

All channels are crossbeam unbounded MPSC channels.

- **Command channels** (one per refiner): `Sender<RefineCommand>` held by coordinator, `Receiver<RefineCommand>` held by refiner thread. Carries `Step { precision_bits }` or `Stop`.
- **Update channel** (shared): all refiners clone the same `Sender<RefinerMessage>`, coordinator holds the single `Receiver<RefinerMessage>`. The coordinator drops its clone of `update_tx` after spawning all refiners, so `update_rx` disconnects when all refiner threads exit.

## Prefix Propagation

When the coordinator receives an update, `apply_update` sets the updated node's prefix and propagates upward through the graph via BFS. Each parent recomputes its prefix from its children via `compute_prefix()`; if the recomputed prefix differs from the cached value, the parent's prefix is updated, its ID is added to the changed set, and its own parents are queued. The changed set is returned to the caller for precise redispatch marking (only non-leaf refiners whose nodes were actually recomputed during propagation are considered for the sub-responded gate).

## Stop Flag

`StopFlag` wraps an `AtomicBool` with `Relaxed` ordering. It is monotonic ($\text{false} \to \text{true}$ only). Refiners check it at the top of each loop iteration, but not during `refine_step()`. This means a slow `refine_step` will run to completion even after the flag is set --- the refiner exits on the next loop check.

## Concurrency Invariants

- **No shared mutable state between refiners.** Each refiner operates on its own `Arc<Node>` and communicates exclusively through channels. Node prefixes are updated via `set_prefix` (interior mutability with `parking_lot::RwLock`), but only the coordinator calls `set_prefix` on non-refiner nodes, and each refiner only calls `set_prefix` on its own node.
- **Single writer for prefix propagation.** Only the coordinator thread calls `apply_update`, which propagates prefixes up the graph. Refiner threads set prefixes on their own nodes only.
- **Outstanding tracking prevents double-dispatch.** The `outstanding[i]` flag ensures a refiner is never sent a second `Step` before its first response arrives. This is coordinated entirely within the single coordinator thread --- no synchronization needed.
- **Dispatch queue consistency.** The `in_queue[i]` flag prevents duplicate entries: a refiner is enqueued only when `needs_redispatch` is set and `in_queue` is false. Both flags are maintained exclusively by the coordinator thread.
- **Eligible count consistency.** `eligible_count` is decremented exactly once per refiner (on exhaustion or step-limit hit) and is never incremented after initialization, so it monotonically decreases from `num_refiners` toward zero.
- **Sub-responded gate correctness.** For non-leaf refiners, the gate tracks responses via the `apply_update` propagation path (changed node IDs), not via a blanket "any other refiner responded" heuristic. This prevents premature redispatch when unrelated refiners respond and ensures that `compute_prefix` reads fully-updated child prefixes.
- **Graceful shutdown.** The coordinator sends `Stop` to all refiners and `thread::scope` waits for all threads to join. Channel disconnection (from either side) is handled as a normal exit path, not a panic.
