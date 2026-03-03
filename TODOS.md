# TODOs - Ranked by Ease of Completion

## Tier 1: Medium (Unblocked)

### <a id="propagated-demand-budget"></a>propagated-demand-budget: Propagate demand budgets down through the computation graph
**File:** `src/refinement.rs`
The current demand budget is a flat $\varepsilon / 2^{\lceil \log_2 N \rceil}$ applied uniformly to all refiners. This assumes all refiners contribute additively to the root's width, which holds for addition but not for multiplication or other nonlinear operations. For example, `a * b` with `a,b ~ 100` and width `0.5` each produces output width `~100`, far above $\varepsilon = 1$ even though both inputs are within budget.

**Fix:** add a top-down demand propagation pass before dispatch. Walk the graph from root to leaves, computing per-child budgets at each operator based on its sensitivity (how child width maps to output width) and the current cached bounds of siblings:

- **Add/Sub:** $w_{\text{out}} = w_a + w_b$ — give each child $\varepsilon_{\text{parent}} / 2$
- **Mul:** $w_{\text{out}} \approx |a| \cdot w_b + |b| \cdot w_a$ — child $a$ gets $\varepsilon_{\text{parent}} / (2|b|)$, child $b$ gets $\varepsilon_{\text{parent}} / (2|a|)$
- **Inv:** $w_{\text{out}} \approx w_a / a^2$ — child gets $\varepsilon_{\text{parent}} \cdot a^2$
- **NthRoot, Pow:** derive from the derivative of the operation

Implementation:

1. Add a method to `NodeOp` (or a new trait): `fn demand_budget(&self, target_width: &XUsize, child_index: usize, current_bounds: &[Bounds]) -> XUsize` — returns the required child width to achieve the target output width.
2. Before dispatch, walk the graph top-down from root (with target $\varepsilon$). At each non-refiner node, call `demand_budget` for each child, using the cached bounds of the other children for the sensitivity calculation. For DAG nodes (shared subexpressions appearing under multiple parents), take the tightest (minimum) budget.
3. Use the per-refiner propagated budgets for demand skipping instead of the global flat budget.
4. The current flat budget becomes a fallback for operators that don't implement the sensitivity method (or a special case for all-addition graphs).

The top-down walk is cheap (same graph that `apply_update` already traverses bottom-up) and the sensitivity functions are simple arithmetic on cached bounds. Recompute each iteration since sibling bounds change as refinement progresses.

### <a id="kill-slow-refiners"></a>kill-slow-refiners: Kill outstanding refiners once precision is achieved
**File:** `src/refinement.rs`
The event-loop coordinator returns as soon as precision is met, but `thread::scope` still blocks on joining all spawned refiner threads. If a slow refiner has an outstanding step when the coordinator exits, `shutdown_refiners` sets the stop flag and sends Stop — but the refiner won't see it until its current `refine_step()` completes. With targeted stopping (e.g. a per-refiner stop flag checked inside expensive computations), the coordinator could signal outstanding refiners to abort their current step early, avoiding the `thread::scope` join delay.

---

## Tier 2: Hard (Unblocked, but complex correctness issues)

### <a id="node-initiated-refinement"></a>node-initiated-refinement: Allow nodes to request refinement of their inputs
Enable a node's `refine_step` to return a recoverable error requesting that the coordinator refine a specific input before retrying. Currently the coordinator decides which refiners to step based on demand budgets; nodes cannot signal "my input bounds are too wide, refine them first." This is needed for nth_root to handle even-degree roots of inputs overlapping with negative numbers (return a recoverable error instead of (0, ∞) bounds). **Blocks:** [nth-root-negative](#nth-root-negative)

---

## Blocked (Waiting on other items)

### <a id="nth-root-negative"></a>nth-root-negative: Handle negative inputs for even-degree roots
**File:** `src/ops/nth_root.rs:15`
```rust
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
```
**Blocked by:** [node-initiated-refinement](#node-initiated-refinement)

---

## Very Low Priority

### <a id="serde-public-types"></a>serde-public-types: Add serialization for all public types
Add serde `Serialize` and `Deserialize` for all public types. For types without invariants, `#[derive(Serialize, Deserialize)]` suffices. For types with invariants (e.g., types that validate inputs in constructors), use custom `Deserialize` implementations that enforce those invariants on deserialization.

### <a id="common-traits-audit"></a>common-traits-audit: Audit and implement common traits
Review all public types and implement standard traits (`Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`, `Display`, `Default`) where semantically meaningful. Don't implement `Ord`/`PartialOrd` unless the type has a natural ordering, and don't implement `Default` if there's no natural default value. Be cautious with `Copy` — adding it is non-breaking, but removing it later is breaking.

---

## Design/Extensibility (Large scope)

### <a id="user-defined-nodes"></a>user-defined-nodes: User-defined composed nodes
**File:** `src/node.rs:126`
```rust
// TODO: ensure it is possible to create user-defined composed nodes.
```
API extensibility feature.
