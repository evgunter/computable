# TODOs - Ranked by Ease of Completion

## Tier 1: Easy

### <a id="epsilon-zero"></a>epsilon-zero: Allow epsilon = 0 with proper checks
**File:** `src/computable.rs:73`
```rust
// TODO: it may be desirable to allow epsilon = 0, but probably only after we implement automatic checking of short-prefix bounds
```
Now unblocked after shortest-repr implementation.

### <a id="bisection-benchmark"></a>bisection-benchmark: Compare midpoint vs shortest-representation bisection
**File:** `benchmarks/src/integer_roots.rs:1`
```rust
// TODO: Add comparison benchmark between midpoint-based bisection (bisection_step_midpoint)
// and shortest-representation bisection (bisection_step) to measure the precision
// accumulation reduction and any performance differences.
```
Benchmark to validate that the shortest-representation bisection strategy reduces precision accumulation without significant performance cost.


## Tier 2: Medium Effort (Unblocked, requires some work)

### <a id="nonzero-benchmark"></a>nonzero-benchmark: Use NonZeroU32 directly in benchmark
**File:** `benchmarks/src/integer_roots.rs:35`
```rust
// TODO: see if we can take the input as NonZeroU32 directly so we don't need to unwrap
```
Minor API change to avoid an unwrap.

### <a id="pi-unwrap"></a>pi-unwrap: Avoid using unwrap in pi benchmark
**File:** `benchmarks/src/pi.rs:55`
```rust
// TODO: can we avoid using unwrap?
```
Replace with proper error handling.

### <a id="non-negative-type"></a>non-negative-type: Ensure non-negative via type system
**File:** `src/binary/shortest.rs:179`
```rust
// TODO: can we use the type system to ensure that this is non-negative?
```
Type constraint addition.

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

### <a id="uxbinary-unreachable"></a>uxbinary-unreachable: Type system for UXBinary unreachable
**File:** `src/binary/uxbinary.rs:157`
```rust
// TODO: Investigate if the type system can prevent needing the unreachable! check
```
Eliminate impossible states at type level.

### <a id="binary-unreachable"></a>binary-unreachable: Type system for Binary unreachable
**File:** `src/binary/binary_impl.rs:271`
```rust
// TODO: Investigate if the type system can prevent needing the unreachable! check below.
```
Similar to [uxbinary-unreachable](#uxbinary-unreachable).

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
