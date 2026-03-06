# TODOs - Ranked by Ease of Completion

## Tier 0: Easy (Unblocked)

### <a id="op-constructors"></a>op-constructors: Give ops proper constructors so callers don't couple to internal structure
**File:** `src/computable.rs`
`Computable::sin` currently calls `pi_node()` because constructing a `PiOp` directly would require knowing its internal fields. The same pattern applies to other ops (`SinOp`, `InvOp`, `NthRootOp`) — `computable.rs` reaches into their internal `RwLock` fields. Each op should expose a constructor (e.g., `PiOp::new()`, `SinOp::new(inner, pi_node)`) that encapsulates initialization, so callers only depend on the public API.

### <a id="pi-unreachable-path"></a>pi-unreachable-path: Investigate whether PiOp's unreachable fallback can be eliminated
**File:** `src/ops/pi.rs`
With per-refiner budgets, the leap formula in `PiOp::refine_step` always produces `needed > num_terms` when the coordinator dispatches (otherwise the refiner is skipped). The doubling fallback is now `unreachable!`. Investigate whether the dispatch logic can be tightened to avoid this dead path entirely — e.g., by having the coordinator not dispatch a leaf refiner whose budget-derived precision would not advance it.

---

## Tier 1: Medium (Unblocked)

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
