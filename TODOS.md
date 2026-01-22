# TODOs - Ranked by Ease of Completion

## Tier 1: Easy

### <a id="shortest-repr-generics"></a>shortest-repr-generics: Reduce duplication in shortest representation functions
**File:** `src/binary/shortest.rs:22`
```rust
// TODO: Consider refactoring shortest_binary_in_finite_bounds and shortest_xbinary_in_bounds
// to reduce code duplication.
```
Both functions follow a similar pattern (check sign, handle zero-crossing, handle positive/negative intervals). Could potentially be unified using generics over the bound types, though different handling of infinities may make this non-trivial.


## Tier 2: Medium Effort (Unblocked, requires some work)

### <a id="bisection-bigint"></a>bisection-bigint: Change exponent parameter type
**File:** `src/binary_utils/bisection.rs:102`
```rust
// TODO: this doesn't need to take exponent as a BigInt since we don't really do that anywhere else.
```
Switch exponent parameter to a more convenient type for its callers once they're integrated.

### <a id="shortest-module-eval"></a>shortest-module-eval: Evaluate if shortest module is still needed
**File:** `src/binary/shortest.rs:15`
```rust
//! # TODO: Evaluate if this module is still needed
```
With the introduction of `bounds_from_normalized` in the bisection module, it may be possible to avoid needing explicit shortest-representation searches. Evaluate if this module is only needed for cases where bounds cannot be normalized initially.

### <a id="shortest-refinement"></a>shortest-refinement: Fix refinement progress tracking
**File:** `src/binary/shortest.rs:271`
```rust
// TODO: all the cases that use this seem to not be tracking refinement progress properly.
```
Cases using `simplify_bounds` don't appear to track refinement progress properly. This likely happens when requesting too much precision for bounds on a wide interval.

### <a id="non-negative-type"></a>non-negative-type: Ensure non-negative via type system
**File:** `src/binary/shortest.rs:179`
```rust
// TODO: can we use the type system to ensure that this is non-negative?
```
Type constraint addition.

### <a id="precision-option-type"></a>precision-option-type: Use Option for initialization state
**File:** `src/ops/inv.rs:59`
```rust
// TODO: use Option<NonZeroOrPositiveBigInt> to encode "not initialized" vs "initialized"
// at the type level, avoiding is_zero() check and making initialization state explicit
```
Currently checks `if precision.is_zero()` to determine if initialization is needed. Using `Option<NonZero...>` would encode this state in the type system, eliminating the runtime check.

### <a id="inv-precision"></a>inv-precision: Improve inv() precision strategy
**File:** `src/ops/inv.rs:27`
```rust
// TODO: Improve inv() precision strategy. Currently precision_bits starts at 0 and
```
Algorithm improvement.

### <a id="pi-neg-test"></a>pi-neg-test: Move or remove redundant neg test
**File:** `src/ops/pi.rs:496`
```rust
// TODO: should this go with `neg` tests? is this actually needed or redundant?
```
Evaluate and either move or remove the interval negation test.

### <a id="sin-sus-comment"></a>sin-sus-comment: Investigate suspicious comment
**File:** `src/ops/sin.rs:312`
```rust
// TODO: this comment is sus, what's up with this
```
There's a comment that suggests a correctness issue ('close enough'). Determine why the code says that and whether there is a correctness issue.

### <a id="sin-midpoint-usage"></a>sin-midpoint-usage: Investigate midpoint usage in sin
**File:** `src/ops/sin.rs:485`
```rust
// TODO: it's suspicious that this uses midpoints rather than bounds
```
This also suggests a correctness issue (using the midpoint rather than the bounds separately suggests that an approximation is being made, instead of correctly propagating the bounds fully)

### <a id="sin-k-midpoint"></a>sin-k-midpoint: Fix midpoint usage for k computation
**File:** `src/ops/sin.rs:288`
```rust
// TODO(correctness): Using midpoints for k computation could cause incorrect range reduction.
```
Correctness issue.

---

## Tier 3: Hard (Unblocked, but complex correctness issues)

### <a id="bounds-dedup"></a>bounds-dedup: Deduplicate FiniteBounds and Bounds
**File:** `src/binary.rs:77`
```rust
// TODO: Investigate code deduplication between FiniteBounds and Bounds. Both types
```
Both FiniteBounds and Bounds are `Interval<T, W>` with different type parameters and have similar interval arithmetic needs. Consider whether the interval_add, interval_sub, interval_neg, scale_positive, scale_bigint, midpoint, and comparison methods could be generalized to work on any `Interval<T, W>` where T and W satisfy appropriate trait bounds.

### <a id="inv-bounds-order"></a>inv-bounds-order: Type system for bounds ordering
**File:** `src/ops/inv.rs:113`
```rust
// TODO: can the type system ensure that the bounds remain ordered?
```
Use the type system to prevent invalid bounds ordering rather than runtime checks.

### <a id="sin-truncation-tracking"></a>sin-truncation-tracking: Fix truncation precision tracking
**File:** `src/ops/sin.rs:565`
```rust
/// TODO(correctness): Truncating discards precision without tracking the error.
```
Need to track truncation error properly.

### <a id="sin-truncation-sus"></a>sin-truncation-sus: Suspicious precision truncation in sin
**File:** `src/ops/sin.rs:598`
```rust
// TODO: this precision truncation is very suspicious!
```
Related to [sin-truncation-tracking](#sin-truncation-tracking).

### <a id="sin-midpoint-correctness"></a>sin-midpoint-correctness: Fix midpoint usage for correctness
**File:** `src/ops/sin.rs:131`
```rust
// TODO(correctness): Using midpoint instead of two_pi_interval.lo could cause incorrect
```
Correctness issue in range reduction.

### <a id="sin-256-cap"></a>sin-256-cap: Fix capped 256-bit precision
**File:** `src/ops/sin.rs:273`
```rust
// TODO(correctness): Capping at 256 bits may not provide sufficient pi precision
```
May need adaptive precision.

### <a id="sin-arbitrary-precision"></a>sin-arbitrary-precision: Support arbitrary precision in divide_by_factorial_directed
**File:** `src/ops/sin.rs:721`
```rust
// TODO(sin-arbitrary-precision): Support arbitrary precision instead of fixed 64 bits.
```
The `divide_by_factorial_directed` function uses a fixed 64-bit precision for reciprocal computation. This caps achievable accuracy and should be made adaptive based on the requested output precision, similar to the issues in pi.rs.

### <a id="sin-refine-default"></a>sin-refine-default: Use refine_to_default instead of custom loop
**File:** `src/ops/sin.rs:301`
```rust
// TODO: should use the thing with refine_to_default rather than a custom loop with custom max iterations
```
Refactor to use existing pattern.

### <a id="sin-large-inputs"></a>sin-large-inputs: Handle extremely large inputs in sin
**File:** `src/ops/sin.rs:298`
```rust
// TODO(correctness): For extremely large inputs where |x| >> 2^10 * 2*pi
```
Edge case handling.

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

### <a id="pi-adaptive"></a>pi-adaptive: Make pi precision adaptive (128-bit limitation)
**File:** `src/ops/pi.rs:243`
```rust
// TODO(correctness): Fixed 128-bit precision caps the achievable accuracy to ~118 bits.
```
**Blocks:** [pi-128-plus](#pi-128-plus), [pi-dead-code](#pi-dead-code), [pi-benchmark-reenable](#pi-benchmark-reenable)

### <a id="pi-adaptive-2"></a>pi-adaptive-2: Make pi precision adaptive (second instance)
**File:** `src/ops/pi.rs:286`
```rust
// TODO(correctness): Fixed 128-bit precision here has the same limitation as in
```
Related to [pi-adaptive](#pi-adaptive).

### <a id="async-refinement"></a>async-refinement: Implement async/event-driven refinement model
**File:** `src/refinement.rs:15`
```rust
//! TODO: The README describes an async/event-driven refinement model where:
```
Major architectural change. **Blocks:** [nth-root-negative](#nth-root-negative), [nth-root-async](#nth-root-async), [refiners-stop](#refiners-stop)

---

## Blocked (Waiting on other items)

### <a id="pi-128-plus"></a>pi-128-plus: Re-enable 128+ bit precision levels
**File:** `benchmarks/src/pi.rs:15`
```rust
// TODO: Re-enable 128+ bit precision levels once the fixed 128-bit intermediate precision
```
**Blocked by:** [pi-adaptive](#pi-adaptive)

### <a id="pi-dead-code"></a>pi-dead-code: Remove dead_code annotation
**File:** `benchmarks/src/pi.rs:234`
```rust
#[allow(dead_code)] // TODO: Re-enable once fixed 128-bit precision in src/ops/pi.rs is made adaptive
```
**Blocked by:** [pi-adaptive](#pi-adaptive)

### <a id="pi-benchmark-reenable"></a>pi-benchmark-reenable: Re-enable benchmark
**File:** `benchmarks/src/pi.rs:273`
```rust
// TODO: Re-enable once fixed 128-bit intermediate precision in src/ops/pi.rs is made adaptive.
```
**Blocked by:** [pi-adaptive](#pi-adaptive)

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
