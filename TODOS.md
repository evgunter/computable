# TODOs - Ranked by Ease of Completion

## Tier 3: Hard (Unblocked, but complex correctness issues)

### <a id="inv-bounds-order"></a>inv-bounds-order: Type system for bounds ordering
**File:** `src/ops/inv.rs:140`
```rust
// TODO: can the type system ensure that the bounds remain ordered?
```
Use the type system to prevent invalid bounds ordering rather than runtime checks.

### <a id="pow-type-bounds"></a>pow-type-bounds: Type system for invalid bounds in pow
**File:** `src/ops/pow.rs:53`
```rust
// TODO: Investigate if the type system can constrain this so that invalid bounds
```
Type-level prevention of invalid states.


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

## Design/Extensibility (Large scope)

### <a id="user-defined-nodes"></a>user-defined-nodes: User-defined composed nodes
**File:** `src/node.rs:126`
```rust
// TODO: ensure it is possible to create user-defined composed nodes.
```
API extensibility feature.
