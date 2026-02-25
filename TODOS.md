# TODOs - Ranked by Ease of Completion

## Tier 2: Hard (Unblocked, but complex correctness issues)

### <a id="pow-type-bounds"></a>pow-type-bounds: Type system for invalid bounds in pow
**File:** `src/ops/pow.rs:53`
```rust
// TODO: Investigate if the type system can constrain this so that invalid bounds
```
Type-level prevention of invalid states.


### <a id="epsilon-in-refine-step"></a>epsilon-in-refine-step: Pass target epsilon to refiners
**File:** `src/ops/inv.rs:27`
```rust
// TODO: Ideally the seed precision would be derived from the target epsilon
```
The refiner protocol (`RefineCommand`) currently only sends `Step`/`Stop` — refiners have no knowledge of the target precision. Passing epsilon (or a precision budget) through to `refine_step` would let refiners like `InvOp` choose a seed precision matched to the target, avoiding unnecessary N-R iterations for low-precision requests and under-seeding for high-precision ones. This would affect the `RefineCommand` enum, the `NodeOp::refine_step` trait method, and all refiner implementations (inv, sin, pi, nth_root).

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

### <a id="arithmetic-lint-audit"></a>arithmetic-lint-audit: Audit and fix remaining arithmetic_side_effects warnings
Remaining clippy::arithmetic_side_effects warnings on `usize`/`i64`/`u64` arithmetic (precision calculations, iteration counters, bit lengths). Each site should be individually reviewed: use `sane_arithmetic!` with the `Sane` newtype where the value represents a computation size (this now uses checked arithmetic internally), or convert to `checked_*` arithmetic where overflow is a genuine concern. Don't mechanically `#[allow]` — think about whether each case could actually be a bug. Note: the `sane_arithmetic!` macro was refactored to use a `Sane` newtype with checked operators instead of `#[allow(clippy::arithmetic_side_effects)]`.

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
