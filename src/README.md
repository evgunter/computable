# Source Code Structure

This document describes the module organization and dependencies.

## Module Dependency Graph

Each module's internal dependencies (from `use crate::*`):

```
computable.rs      → binary, error, node, ops, refinement
refinement.rs      → binary, concurrency, error, node
node.rs            → binary, error
ops/arithmetic.rs  → binary, error, node
ops/base.rs        → binary, error, node
ops/inv.rs         → binary, error, node
ops/sin.rs         → binary, error, node
binary.rs          → ordered_pair
binary/binary_impl → ordered_pair
binary/ubinary     → ordered_pair
binary/uxbinary    → ordered_pair
error.rs           → binary
ordered_pair.rs    → (none)
concurrency.rs     → (none)
```

### Visual Overview

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

### Dependency Matrix

```
                 binary  concurr  error  node  ops  ordered_pair  refinement
computable.rs      ✓                ✓      ✓     ✓                    ✓
refinement.rs      ✓        ✓       ✓      ✓
node.rs            ✓                ✓
ops/arithmetic.rs  ✓                ✓      ✓
ops/base.rs        ✓                ✓      ✓
ops/inv.rs         ✓                ✓      ✓
ops/sin.rs         ✓                ✓      ✓
binary/*                                                 ✓
error.rs           ✓
ordered_pair.rs
concurrency.rs
```

## Module Descriptions

| Module | Description |
|--------|-------------|
| `lib.rs` | Crate root; declares modules and re-exports public API |
| `binary/` | Arbitrary-precision binary numbers (mantissa × 2^exponent) |
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

## Data Flow

1. **User creates `Computable`** values via `Computable::new()` or `Computable::constant()`
2. **Arithmetic operations** (`+`, `-`, `*`, `/`, `.inv()`, `.sin()`) build a **computation graph** of `Node`s
3. **`refine_to(epsilon)`** triggers parallel refinement:
   - `RefinementGraph` snapshots the node graph
   - Spawns refiner threads for leaf nodes
   - Propagates bound updates through the graph
   - Returns when root bounds width ≤ epsilon
4. **`bounds()`** returns current interval bounds without refinement
