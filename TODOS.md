# TODOs - Ranked by Ease of Completion

## Tier 1: Medium (Unblocked)

### <a id="non-blocking-refinement"></a>non-blocking-refinement: Refinement blocks on slow refiners instead of continuing with fast ones
**File:** `src/refinement.rs:441`
```rust
fn compute_demand_budget(tolerance_exp: &XUsize, num_active: usize) -> XUsize {
```
The round-based refinement model sends Step commands and then blocks waiting for all responses before starting the next round. If a fast refiner and a slow refiner are both stepped in the same round, the coordinator waits for the slow refiner even though it could be stepping the fast one again. In the test `demand_skipping_unnecessarily_steps_already_precise_refiner`, y (width 3/8, below target 1/2) is slow (1s/step) and x is fast — but the coordinator blocks on y's response instead of continuing to step x. The coordinator should not block on slow refiners when there are fast refiners it could be stepping.

---

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

## Low Priority

### <a id="bench-sample-counts"></a>bench-sample-counts: Restore larger benchmark sample counts
**Files:** `benches/integer_roots.rs`, `benches/summation.rs`
Benchmark sample counts were reduced to stay under valgrind's default ~500 thread limit (each refiner node spawns a thread). `integer_roots` went from 1000→50, `summation` from 200k→1000. To restore them, either pass `--max-threads=N` to valgrind via gungraun's `valgrind_args` config, or restructure the benchmarks to avoid spawning all refiners simultaneously.

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
