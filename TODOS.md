# TODOs - Ranked by Ease of Completion

## Tier 2: Hard (Unblocked, but complex correctness issues)

### <a id="async-refinement"></a>async-refinement: Implement async/event-driven refinement model
**File:** `src/refinement.rs:15`
```rust
//! TODO: The README describes an async/event-driven refinement model where:
```
Major architectural change. **Blocks:** [nth-root-negative](#nth-root-negative), [nth-root-async](#nth-root-async), [refiners-stop](#refiners-stop)

---

## Blocked (Waiting on other items)


### <a id="nth-root-negative"></a>nth-root-negative: Handle negative inputs for even-degree roots
**File:** `src/ops/nth_root.rs:15`
```rust
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
```
**Blocked by:** [async-refinement](#async-refinement)

### <a id="nth-root-async"></a>nth-root-async: nth_root async model dependency
**File:** `src/ops/nth_root.rs:22`
```rust
//! async/event-driven model described in the README (see TODO in refinement.rs)
```
**Blocked by:** [async-refinement](#async-refinement)

### <a id="refiners-stop"></a>refiners-stop: Allow refiners to stop individually
**File:** `src/refinement.rs:130`
```rust
// TODO: allow individual refiners to stop at the max without
```
**Blocked by:** [async-refinement](#async-refinement)

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
