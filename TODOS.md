# TODOs - Ranked by Ease of Completion

## Tier 1: Quick Wins (Unblocked, straightforward)

### 1. Use Binary for pi comparison instead of f64
**File:** `src/ops/pi.rs:494`
```rust
// TODO: fix this by converting the f64 approximation of pi to Binary and then comparing
```
More rigorous comparison.

### 2. Improve pi comparison (similar)
**File:** `src/ops/pi.rs:537`
```rust
// TODO: improve this by converting the f64 approximation of pi to Binary and then comparing
```
Same pattern as #1.

### 3. Move or remove redundant neg test
**File:** `src/ops/pi.rs:575`
```rust
// TODO: should this go with `neg` tests? is this actually needed or redundant?
```
Evaluate and either move or remove.

### 4. Investigate Binary operations location
**File:** `src/binary_utils/mod.rs:10`
```rust
//! TODO: Investigate if any pure-Binary operations from `ops/` should be moved here.
```
Code organization review.

### 5. Unify inv and pi_inv_sum code
**File:** `src/ops/pi.rs:230`
```rust
// TODO: does this have any overlap with the `inv` function? can they be unified or at least share common code?
```
Potential code deduplication.

### 6. Construct Bounds directly
**File:** `src/binary/shortest.rs:192`
```rust
// TODO: here and elsewhere it might be nice to be able to construct Bounds directly
```
API ergonomics improvement.

---

## Tier 2: Medium Effort (Unblocked, requires some work)

### 7. Use shortest representation functions
**File:** `src/binary/shortest.rs:20`
```rust
// TODO: use these functions to make binary-search-based refinement not need to represent intervals that have so many bits of precision
```
**Blocks:** #33 (epsilon = 0 feature)

### 8. Use NonZeroU32 directly in benchmark
**File:** `benchmarks/src/integer_roots.rs:32`
```rust
// TODO: see if we can take the input as NonZeroU32 directly so we don't need to unwrap
```
Minor API change to avoid an unwrap.

### 9. Avoid using unwrap in pi benchmark
**File:** `benchmarks/src/pi.rs:55`
```rust
// TODO: can we avoid using unwrap?
```
Replace with proper error handling.

### 10. Ensure non-negative via type system
**File:** `src/binary/shortest.rs:179`
```rust
// TODO: can we use the type system to ensure that this is non-negative?
```
Type constraint addition.

### 11. Improve inv() precision strategy
**File:** `src/ops/inv.rs:27`
```rust
// TODO: Improve inv() precision strategy. Currently precision_bits starts at 0 and
```
Algorithm improvement.

### 12. Convert Interval to FiniteBounds
**File:** `src/ops/pi.rs:326`
```rust
// TODO: make this into FiniteBounds instead, using the same paradigm as the Bounds type
```
Refactor to reduce code duplication. **Blocks:** #36

### 13. Investigate suspicious comment
**File:** `src/ops/sin.rs:312`
```rust
// TODO: this comment is sus, what's up with this
```
There's a comment that suggests a correctness issue ('close enough'). Determine why the code says that and whether there is a correctness issue.

### 14. Investigate midpoint usage in sin
**File:** `src/ops/sin.rs:485`
```rust
// TODO: it's suspicious that this uses midpoints rather than bounds
```
This also suggests a correctness issue (using the midpoint rather than the bounds separately suggests that an approximation is being made, instead of correctly propagating the bounds fully)

### 15. Fix midpoint usage for k computation
**File:** `src/ops/sin.rs:288`
```rust
// TODO(correctness): Using midpoints for k computation could cause incorrect range reduction.
```
Correctness issue.

---

## Tier 3: Hard (Unblocked, but complex correctness issues)

### 16. Fix truncation precision tracking
**File:** `src/ops/sin.rs:565`
```rust
/// TODO(correctness): Truncating discards precision without tracking the error.
```
Need to track truncation error properly.

### 17. Suspicious precision truncation in sin
**File:** `src/ops/sin.rs:598`
```rust
// TODO: this precision truncation is very suspicious!
```
Related to #16.

### 18. Fix midpoint usage for correctness
**File:** `src/ops/sin.rs:131`
```rust
// TODO(correctness): Using midpoint instead of two_pi_interval.lo could cause incorrect
```
Correctness issue in range reduction.

### 19. Fix capped 256-bit precision
**File:** `src/ops/sin.rs:273`
```rust
// TODO(correctness): Capping at 256 bits may not provide sufficient pi precision
```
May need adaptive precision.

### 20. Use refine_to_default instead of custom loop
**File:** `src/ops/sin.rs:301`
```rust
// TODO: should use the thing with refine_to_default rather than a custom loop with custom max iterations
```
Refactor to use existing pattern.

### 21. Handle extremely large inputs in sin
**File:** `src/ops/sin.rs:298`
```rust
// TODO(correctness): For extremely large inputs where |x| >> 2^10 * 2*pi
```
Edge case handling.

### 22. Replace f64 with rigorous computation in pi
**File:** `src/ops/pi.rs:81`
```rust
// TODO(correctness): Using f64 for this calculation is not rigorous
```
Need arbitrary precision instead of f64.

### 23. Handle extreme exponents in shift
**File:** `src/binary/shift.rs:25`
```rust
/// TODO: Find a solution for extreme exponents causing memory issues.
```
Memory safety for edge cases.

### 24. Type system for invalid bounds in pow
**File:** `src/ops/pow.rs:53`
```rust
// TODO: Investigate if the type system can constrain this so that invalid bounds
```
Type-level prevention of invalid states.

### 25. Type system for UXBinary unreachable
**File:** `src/binary/uxbinary.rs:157`
```rust
// TODO: Investigate if the type system can prevent needing the unreachable! check
```
Eliminate impossible states at type level.

### 26. Type system for Binary unreachable
**File:** `src/binary/binary_impl.rs:240`
```rust
// TODO: Investigate if the type system can prevent needing the unreachable! check
```
Similar to #25.

### 27. Make pi precision adaptive (128-bit limitation)
**File:** `src/ops/pi.rs:243`
```rust
// TODO(correctness): Fixed 128-bit precision caps the achievable accuracy to ~118 bits.
```
**Blocks:** #30, #31, #32

### 28. Make pi precision adaptive (second instance)
**File:** `src/ops/pi.rs:304`
```rust
// TODO(correctness): Fixed 128-bit precision here has the same limitation
```
Related to #27.

### 29. Implement async/event-driven refinement model
**File:** `src/refinement.rs:15`
```rust
//! TODO: The README describes an async/event-driven refinement model where:
```
Major architectural change. **Blocks:** #34, #35, #37

---

## Blocked (Waiting on other items)

### 30. Re-enable 128+ bit precision levels
**File:** `benchmarks/src/pi.rs:15`
```rust
// TODO: Re-enable 128+ bit precision levels once the fixed 128-bit intermediate precision
```
**Blocked by:** #27 (adaptive pi precision)

### 31. Remove dead_code annotation
**File:** `benchmarks/src/pi.rs:234`
```rust
#[allow(dead_code)] // TODO: Re-enable once fixed 128-bit precision in src/ops/pi.rs is made adaptive
```
**Blocked by:** #27 (adaptive pi precision)

### 32. Re-enable benchmark
**File:** `benchmarks/src/pi.rs:273`
```rust
// TODO: Re-enable once fixed 128-bit intermediate precision in src/ops/pi.rs is made adaptive.
```
**Blocked by:** #27 (adaptive pi precision)

### 33. Allow epsilon = 0 with proper checks
**File:** `src/computable.rs:73`
```rust
// TODO: it may be desirable to allow epsilon = 0, but probably only after we implement automatic checking of short-prefix bounds
```
**Blocked by:** #7 (shortest representation / prefix bound checking)

### 34. Handle negative inputs for even-degree roots
**File:** `src/ops/nth_root.rs:15`
```rust
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
```
**Blocked by:** #29 (async/event-driven model)

### 35. nth_root async model dependency
**File:** `src/ops/nth_root.rs:22`
```rust
//! async/event-driven model described in the README (see TODO in refinement.rs)
```
**Blocked by:** #29 (async/event-driven model)

### 36. Remove debug_assert from Interval::new
**File:** `src/ops/pi.rs:338`
```rust
// TODO: no debug assert! this should be just like in Bounds
```
**Blocked by:** #12 (convert Interval to FiniteBounds)

### 37. Allow refiners to stop individually
**File:** `src/refinement.rs:130`
```rust
// TODO: allow individual refiners to stop at the max without
```
**Blocked by:** #29 (async/event-driven model)

---

## Design/Extensibility (Large scope)

### 38. User-defined composed nodes
**File:** `src/node.rs:126`
```rust
// TODO: ensure it is possible to create user-defined composed nodes.
```
API extensibility feature.

---

## Not a TODO (Meta-documentation)

### Error handling documentation pattern
**File:** `src/error.rs:9`
```rust
//!   type invariants. Always include a TODO comment about investigating whether the type
```
This describes how to write TODOs, not a TODO itself.
