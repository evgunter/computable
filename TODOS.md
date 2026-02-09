# TODOs - Ranked by Ease of Completion

## Tier 2: Medium Effort (Unblocked, requires some work)

### <a id="shortest-repr-generics"></a>shortest-repr-generics: Reduce duplication in shortest representation functions
**File:** `src/binary/shortest.rs:30`
```rust
// TODO: Consider refactoring shortest_binary_in_finite_bounds and shortest_xbinary_in_bounds
// to reduce code duplication.
```
Both functions follow a similar pattern (check sign, handle zero-crossing, handle positive/negative intervals). Could potentially be unified using generics over the bound types, though different handling of infinities may make this non-trivial.

### <a id="shortest-module-eval"></a>shortest-module-eval: Evaluate if shortest module is still needed
**File:** `src/binary/shortest.rs:15`
```rust
//! # TODO: Evaluate if this module is still needed
```
With the introduction of `bounds_from_normalized` in the bisection module, it may be possible to avoid needing explicit shortest-representation searches. Evaluate if this module is only needed for cases where bounds cannot be normalized initially.

### <a id="shortest-refinement"></a>shortest-refinement: Fix refinement progress tracking
**File:** `src/binary/shortest.rs:282`
```rust
// TODO: all the cases that use this seem to not be tracking refinement progress properly.
```
Cases using `simplify_bounds` don't appear to track refinement progress properly. This likely happens when requesting too much precision for bounds on a wide interval.

## Tier 3: Hard (Unblocked, but complex correctness issues)

### <a id="bounds-dedup"></a>bounds-dedup: Deduplicate FiniteBounds and Bounds
**File:** `src/binary.rs:77`
```rust
// TODO: Investigate code deduplication between FiniteBounds and Bounds. Both types
```
Both FiniteBounds and Bounds are `Interval<T, W>` with different type parameters and have similar interval arithmetic needs. Consider whether the interval_add, interval_sub, interval_neg, scale_positive, scale_bigint, midpoint, and comparison methods could be generalized to work on any `Interval<T, W>` where T and W satisfy appropriate trait bounds.

### <a id="inv-bounds-order"></a>inv-bounds-order: Type system for bounds ordering
**File:** `src/ops/inv.rs:148`
```rust
// TODO: can the type system ensure that the bounds remain ordered?
```
Use the type system to prevent invalid bounds ordering rather than runtime checks.

### <a id="pi-f64"></a>pi-f64: Replace f64 with rigorous computation in pi
**File:** `src/ops/pi.rs:84`
```rust
// TODO(correctness): Using f64 for this calculation is not rigorous for a "provably correct"
```
Need arbitrary precision instead of f64.

### <a id="shift-extreme"></a>shift-extreme: Handle extreme exponents in shift
**File:** `src/binary/shift.rs:25`
```rust
/// TODO: Find a solution for extreme exponents causing memory issues.
```
Memory safety for edge cases.

### <a id="pow-type-bounds"></a>pow-type-bounds: Type system for invalid bounds in pow
**File:** `src/ops/pow.rs:51`
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
