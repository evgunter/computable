# concurrency model (draft)

## goals
- correctness: bounds and refinement must remain consistent and monotonically improving under concurrency, even if refinement progress is nondeterministic.
- performance: avoid redundant recomputation and minimize locking/coordination overhead.
- composability: child refinements should propagate to parents without requiring parent-specific logic inside child refiners.
- minimal recomputation: if a child updates frequently, the parent should coalesce updates and only recompute once per batch.
- avoid wasteful refinement: prevent leaf refiners from spinning indefinitely when they are no longer affecting the final precision.

## initial proposal (from discussion)
Each computable publishes updates to its parents via **per-parent capacity-1 channels** using `crossbeam_channel`.

- For each parent edge, the child holds a sender to a capacity-1 channel.
- The signal payload is `()` for now; it should be straightforward to evolve later if we need metadata.
- When the child updates its bounds, it **non-blockingly** sends a "bounds updated" signal to each parent.
  - If the send fails because the channel is full, **the child continues without retrying**, since the parent has already been told that an update is available.
- The parent listens to all children channels. When it receives at least one signal, it:
  1) **drains all inbound channels** (clears any queued signals),
  2) **reads current bounds from each child**, and
  3) recomputes its own bounds based on those child bounds.

This provides a coalescing behavior: multiple updates from the same child are collapsed into a single recomputation at the parent.

## bounds consistency: locking
Parents must not read bounds mid-update. A standard `RwLock` around each computable's bounds should be sufficient:
- writers take an exclusive lock during refinement updates,
- parents read under a shared lock when recomputing.

This ensures parents always see a consistent bounds snapshot without requiring versioned reads.

## refiners: preventing leaf spinning
We need a mechanism to prevent leaf computable numbers from refining forever while they no longer improve the final result.

Current direction:
- **Top-down activation with epsilon scaling**: parents activate children with a target epsilon (similar to `refine_to`), children refine while publishing intermediate updates, and then stop once they hit epsilon or are explicitly deactivated. Parents can reactivate if higher precision is needed.
  - For binary combination nodes (current assumption), pass down a smaller epsilon to each child (e.g., `epsilon / 16`) so faster children can keep contributing meaningful improvements while slower ones catch up, instead of splitting evenly (`epsilon / 2`).
  - Consider a **parent-to-child stop channel**: when a parent reaches its target precision, it sends a stop signal to all children. Children check this channel each refinement loop (e.g., right after publishing an update) and halt promptly when a stop is observed.

## questions to resolve
- **Activation granularity**: do parents activate children based on absolute epsilon, relative epsilon, or a mix?
- **Stopping conditions**: should we use a parent-to-child stop channel so children can exit promptly without races?
- **Fairness**: how do we prevent starvation if some children refine slowly or have stale update signals?
- **Error handling**: how should recoverable/irrecoverable errors propagate to parents and terminate refinement?
- **Cancellation**: how do we cancel in-flight refinements when a higher-level computation finishes or fails?
- **Resource limits**: do we cap per-refinement CPU time or iteration counts beyond existing `refine_to` limits?

## alternative models to consider (before implementation)
### Dedicated update aggregation thread per composition
- Each composition has a manager that subscribes to children and serializes recomputation.
- The manager is a dedicated task/thread that owns the parent recomputation loop and aggregates child updates into a single recompute path.
- **Difference from the main approach**: in the main approach, each parent recomputes in the same execution context that receives child signals (no dedicated manager thread). In this alternative, recomputation is centralized into one manager per composition, which can simplify scheduling and batching at the cost of extra threads and less parallel parent recomputation.
- Parents never recompute in parallel; allows batching but adds thread overhead.
- Downside: introduces a top-level coordinator per composition, which breaks the “equal footing” feel of each level in the composition tree.

## extensions to evaluate later
- **Parent-side batching**: introduce short batching windows to reduce recomputation, and benchmark to confirm it helps.
- **Priority-based scheduling**: explore a thread pool where refiners that improve global precision fastest get higher priority.
- **Epsilon allocation strategy**: revisit how parents distribute epsilon to children; benchmark whether `epsilon / 16` (or other heuristics) yields better convergence behavior.

## next step
Please confirm the decisions above, especially the approach to preventing leaf refiners from spinning. Once we pick a specific model, I can start implementing it.

## detailed plan (draft)
1) **Initialization**
   - Build the computation graph (parents/children) and allocate per-parent bounded channels.
   - Each child stores a list of parent senders in its node state so refinements can publish updates.
   - Wrap each computable's bounds in an `RwLock`.
   - Register each parent's receiver with `select!` to wait for child updates.
2) **Refinement kickoff**
   - Top-level `refine_to` (or equivalent) activates the root with a target epsilon.
   - The root activates children with derived epsilons (current proposal: pass `epsilon / 16` to each child for binary combinators).
3) **Refinement loop**
   - Each refiner:
     - refines its local state,
     - updates bounds under a write lock,
     - attempts a non-blocking send on each parent channel.
   - Each parent:
     - waits for at least one child signal,
     - drains all inbound signals,
     - reads child bounds under shared locks,
     - recomputes its own bounds and publishes if changed.
4) **Convergence detection**
   - The root checks whether bounds width meets epsilon.
   - If met, it signals cancellation/deactivation to children (candidate: a parent-to-child stop channel).
5) **Conclusion**
   - Children stop refining once deactivated or when their local epsilon is met; if using a stop channel, children check it each loop and exit promptly.
   - The root returns final bounds (or an error if irrecoverable conditions occur).
