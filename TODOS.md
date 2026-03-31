# TODOs - Ranked by Ease of Completion

## Tier 0: Easy (Unblocked)

### <a id="sane-i64"></a>sane-i64: Extend `sane_arithmetic!` (or add a new macro) to support i64 checked arithmetic
**File:** `src/sane.rs`
The `sane_arithmetic!` macro currently only works with `usize` operands (its guard pattern requires `$guard:ident`). After the BigInt→i64 exponent migration, many sites in the codebase use verbose `checked_mul(...).unwrap_or_else(|| detected_computable_would_exhaust_memory!(...))` chains for i64 exponent arithmetic. A `sane_i64_arithmetic!` macro (or extending the existing macro to accept i64) would make these sites more concise and consistent. Key sites to refactor:
- `src/ops/sin.rs` `taylor_sin_bounds_rational`: ~10 checked ops computing `common_exp`, `shift_per_step`, `first_shift`, `two_k`/`two_k_plus_1`, and the accumulation loop
- `src/binary/binary_impl.rs`: `mul`, `normalize`, `Shl`/`Shr` impls
- `src/binary/ubinary.rs`: `mul`, `div_floor`, `normalize`, `Shl`/`Shr` impls
- `src/binary/reciprocal.rs`: `reciprocal_of_biguint`, `reciprocal_rounded_abs_extended`
- `src/binary_utils/bisection.rs`: `midpoint`, `bisection_step_normalized`, `normalize_bounds`
- `src/ops/nth_root.rs`: `binary_div_floor`, `binary_div_ceil`, `truncate_floor`/`truncate_ceil`
- `src/finite_interval.rs`: `midpoint`

---

### <a id="op-constructors"></a>op-constructors: Give ops proper constructors so callers don't couple to internal structure
**File:** `src/computable.rs`
`Computable::sin` currently reaches into `SinOp`, `InvOp`, `NthRootOp` etc. to construct them with internal `RwLock` fields. Each op should expose a constructor (e.g., `SinOp::new(inner, pi_node)`) that encapsulates initialization, so callers only depend on the public API.

---

### <a id="incremental-arctan"></a>incremental-arctan: Cache arctan Taylor series intermediate state for incremental computation
**File:** `src/ops/pi.rs`
**Prior attempt:** `mng/pi-no-double` branch (commit `7de0fcd`)
An `ArctanCache` struct was added to store partial sums and k-power state so additional Taylor terms could be appended in O(delta) instead of recomputing from scratch. The cache was invalidated and rebuilt whenever `precision_bits` increased past the cached precision. In practice, `num_terms` and `precision_bits` increase together (since `precision_bits_for_num_terms` is monotonic), so the cache was invalidated on nearly every refinement step — paying management overhead on top of the same O(N) computation. This caused +80–490% regressions on pi-related benchmarks at higher precision levels.
A working approach needs to avoid cache invalidation entirely. One idea: compute all terms at a fixed high precision from the start (or use a precision that only grows in large steps), so the cache is extended incrementally without rebuilds.

---

## Tier 1: Medium (Unblocked)

### <a id="kill-slow-refiners"></a>kill-slow-refiners: Kill outstanding refiners once precision is achieved
**File:** `src/refinement.rs`
The event-loop coordinator returns as soon as precision is met, but `thread::scope` still blocks on joining all spawned refiner threads. If a slow refiner has an outstanding step when the coordinator exits, `shutdown_refiners` sets the stop flag and sends Stop — but the refiner won't see it until its current `refine_step()` completes. With targeted stopping (e.g. a per-refiner stop flag checked inside expensive computations), the coordinator could signal outstanding refiners to abort their current step early, avoiding the `thread::scope` join delay.

---

### <a id="single-cache"></a>single-cache: Investigate single-cache Node without convergence regressions
**Experiment:** `experiment/remove-bounds-cache` branch (commit `c8c9fd8`)
The dual cache (prefix_cache + bounds_cache) adds complexity and lock contention. Removing it speeds up pi/integer_roots benchmarks by 30-44% but regresses inv/256 by +94% and sin_Npi by +40-71% because Prefix rounding degrades Newton-Raphson convergence. Investigate whether a single cache storing exact Bounds (deriving Prefix lazily) could get both benefits — less overhead AND exact arithmetic for convergence. See EXPERIMENTS.md Experiment 13.

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
