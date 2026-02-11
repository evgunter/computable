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
- The signal payload is an enum (e.g., `Update` vs `Done`) so parents can distinguish intermediate updates from completion.
- When the child updates its bounds, it **non-blockingly** sends a "bounds updated" signal to each parent.
  - If the send fails because the channel is full, **the child continues without retrying**, since the parent has already been told that an update is available.
- The parent listens to all children channels. When it receives at least one signal, it:
  1) **drains all inbound channels** (clears any queued signals),
  2) **reads current bounds from each child**, and
  3) recomputes its own bounds based on those child bounds.

This provides a coalescing behavior: multiple updates from the same child are collapsed into a single recomputation at the parent.

## bounds consistency: locking
Parents must not read bounds mid-update. A standard `RwLock` around each computable's bounds should be sufficient (storing `Result<Bounds, RefinementError>` so parents can read error state on completion):
- writers take an exclusive lock during refinement updates,
- parents read under a shared lock when recomputing.

This ensures parents always see a consistent bounds snapshot without requiring versioned reads.

## refinement control (preventing leaf spinning)
We need a mechanism to prevent leaf computables from refining forever while they no longer improve the final result.

Current direction:
- **Top-down activation with epsilon scaling**: parents activate children with a target epsilon (similar to `refine_to`), children refine while publishing intermediate updates, and then stop once they hit epsilon or are explicitly deactivated. Parents can reactivate if higher precision is needed.
  - For binary combination nodes (current assumption), pass down a smaller epsilon to each child (e.g., `epsilon / EPSILON_SPLIT_FACTOR`) so faster children can keep contributing meaningful improvements while slower ones catch up, instead of splitting evenly (`epsilon / 2`). Define `EPSILON_SPLIT_FACTOR` as a named constant (currently `16`).
  - **Parent-to-child stop channel**: when a parent reaches its target precision, it sends a stop signal to all children. Children check this channel each refinement loop (e.g., right after publishing an update) and halt promptly when a stop is observed.
  - **Refinement responsibility stays with the parent**: instead of globally storing the tightest epsilon on the child, the parent that needs extra precision should keep listening until the child finishes its current refinement run and then request a narrower refinement (to avoid conflicts if another parent with the tightest epsilon stops).
  - Children send a **completion signal** to parents when they finish a refinement run. This is captured in the update enum (`Update` vs `Done`) and supports the stop-channel handshake.
  - **Error handling**: parents read error state from child bounds when they observe a completion.

## extensions to evaluate later
- **Parent-side batching**: introduce short batching windows to reduce recomputation, and benchmark to confirm it helps.
- **Priority-based scheduling**: explore a thread pool where refiners that improve global precision fastest get higher priority.
- **Epsilon allocation strategy**: revisit how parents distribute epsilon to children; benchmark whether `epsilon / 16` (or other heuristics) yields better convergence behavior.
- **Activation granularity**: revisit whether parents should activate children based on absolute epsilon, relative epsilon, or a mix.
- **Resource limits**: consider caps on per-refinement CPU time or iteration counts beyond existing `refine_to` limits.

## detailed plan (draft)
1) **Initialization**
   - Build the computation graph (parents/children) and allocate per-parent bounded channels.
   - Each child stores a list of parent senders in its node state so refinements can publish updates.
   - Wrap each computable's bounds in an `RwLock`.
   - Register each parent's receiver with `select!` to wait for child updates.
2) **Refinement kickoff**
   - Top-level `refine_to` (or equivalent) activates the root with a target epsilon.
   - The root activates children with derived epsilons (current proposal: pass `epsilon / EPSILON_SPLIT_FACTOR` to each child for binary combinators).
3) **Refinement loop**
   - Each refiner:
     - refines its local state,
     - updates bounds under a write lock,
     - attempts a non-blocking send on each parent channel with an `Update` message.
     - checks the stop channel after publishing and exits promptly if a stop is observed.
   - Each parent:
     - waits for at least one child signal,
     - drains all inbound signals,
     - reads child bounds under shared locks,
     - recomputes its own bounds and publishes if changed.
     - if it observes a `Done` from a child, it may choose to request a narrower refinement from that child if needed.
4) **Convergence detection**
   - Each node checks whether its bounds width meets its target epsilon.
   - If met, it signals cancellation/deactivation to children via the stop channel.
5) **Conclusion**
   - Children stop refining once deactivated, when their local epsilon is met, or when they hit an error, emitting a `Done` signal to parents.
   - The root returns final bounds (or an error if irrecoverable conditions occur).
