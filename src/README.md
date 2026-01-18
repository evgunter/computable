# Source Code Structure

This document describes the module organization and dependencies.

## Module Descriptions

| Module | Description |
|--------|-------------|
| `lib.rs` | Crate root; declares modules and re-exports public API |
| `binary/` | Arbitrary-precision binary numbers (mantissa * 2^exponent) |
| `ordered_pair.rs` | Interval types: `Bounds` and `Interval` with ordering guarantees |
| `error.rs` | Error types for computable operations (`ComputableError`) |
| `node.rs` | Computation graph infrastructure (`Node`, `NodeOp` trait, `BaseNode`) |
| `ops/` | Operation implementations for the computation graph |
| `ops/arithmetic.rs` | `AddOp`, `NegOp`, `MulOp` |
| `ops/inv.rs` | `InvOp` (multiplicative inverse with precision refinement) |
| `ops/sin.rs` | `SinOp` (sine via Taylor series with directed rounding) |
| `ops/base.rs` | `BaseOp` (wraps user-defined leaf nodes) |
| `refinement.rs` | Parallel refinement infrastructure (`RefinementGraph`, `RefinerHandle`) |
| `computable.rs` | Main `Computable` type with arithmetic operator implementations |
| `concurrency.rs` | Concurrency utilities (`StopFlag`) |

## Dependencies

Arrows show direct dependencies (transitive edges omitted):

```
        computable
           / \
          ▼   ▼
   refinement  ops/
      /  \      |
     ▼    \     |
concurrency \   |
             \  |
              ▼ ▼
              node
                |
                ▼
              error
                |
                ▼
              binary
                |
                ▼
           ordered_pair
```

## Data Flow

1. **User creates `Computable`** values via `Computable::new()` or `Computable::constant()`
2. **Arithmetic operations** (`+`, `-`, `*`, `/`, `.inv()`, `.sin()`) build a **computation graph** of `Node`s
3. **`refine_to(epsilon)`** triggers parallel refinement:
   - `RefinementGraph` snapshots the node graph
   - Spawns refiner threads for leaf nodes
   - Propagates bound updates through the graph
   - Returns when root bounds width ≤ epsilon
4. **`bounds()`** returns current interval bounds without refinement

## Impossible Cases

Some code paths handle cases that should be mathematically impossible given the invariants
of the types involved. These cases require special handling to balance correctness, debuggability,
and future flexibility.

### Convention

**Use `unreachable!()` for truly impossible cases:**

When a case is mathematically impossible given current type invariants, use `unreachable!()`
with a descriptive message. Always add a TODO comment about investigating whether the type
system could prevent the case from being representable:

```rust
// TODO: Investigate if the type system can prevent unordered bounds.
unreachable!("bounds are not ordered: lower > upper")
```

The goal is to eventually eliminate these `unreachable!()` calls by making invalid states
unrepresentable through the type system. The TODO serves as a reminder to investigate this.

**Use `debug_assert!()` for "currently unexpected" cases:**

Some cases are unexpected with the current feature set but might become valid in the future.
For example, certain infinite bound configurations might become meaningful if extended real
number support is added. For these cases, use `debug_assert!(false, ...)` with a comment
explaining the situation:

```rust
// This debug_assert is here because nothing currently produces this case, so
// hitting it likely indicates a bug. However, this could become a valid case
// if we later support [feature X]. If that feature is added, remove this assertion.
debug_assert!(false, "unexpected case - may be valid for future feature X");
// ... handle the case gracefully for release builds ...
```

### Why Not Always Return Errors?

Returning `Result` for impossible cases has drawbacks:
- It pollutes function signatures with errors that "should never happen"
- Callers might silently discard errors, hiding bugs
- It doesn't clearly communicate that the case represents a logic error

`unreachable!()` clearly documents that the case is a bug if reached, while `debug_assert!()`
allows graceful degradation in release builds for cases that might become valid later.
