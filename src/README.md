# Source Code Structure

This document describes the module organization and dependencies.

## Module Dependency Graph

Each module's internal dependencies (from `use crate::*`):

```
computable.rs      → binary, error, node, ops, ordered_pair, refinement
refinement.rs      → binary, concurrency, error, node, ordered_pair
node.rs            → error, ordered_pair
ops/arithmetic.rs  → error, node, ordered_pair
ops/base.rs        → error, node, ordered_pair
ops/inv.rs         → binary, error, node, ordered_pair
ops/sin.rs         → binary, error, node, ordered_pair
ordered_pair.rs    → binary
error.rs           → binary
concurrency.rs     → (none)
binary/binary_impl → ordered_pair
binary/ubinary     → ordered_pair
binary/uxbinary    → ordered_pair
```

Note: `binary ↔ ordered_pair` is a mutual dependency. This works in Rust
because module resolution happens at the crate level.

### Visual Overview

```
                    ┌─────────────────────────────────────────┐
                    │               lib.rs                    │
                    │          (public API exports)           │
                    └─────────────────────────────────────────┘
                                       │
       ┌───────────────┬───────────────┼───────────────┬──────────────┐
       ▼               ▼               ▼               ▼              ▼
  computable     ordered_pair ◄────► binary         error       concurrency
       │               │               ▲               ▲              ▲
       │               │    (mutual)   │               │              │
       │               └───────────────┤               │              │
       │                               │               │              │
       ▼                               │               │              │
  refinement ──────────────────────────┴───────────────┴──────────────┘
       │
       ▼
     node ─────────────────────────────────────────────┐
       │                                               │
       ▼                                               ▼
     ops/ ─────────────────────────────────────► error, ordered_pair
       │
       │ (inv.rs, sin.rs only)
       ▼
    binary
```

### Dependency Matrix

```
                 binary  concurr  error  node  ops  ordered_pair  refinement
computable.rs      ✓                ✓      ✓     ✓        ✓            ✓
refinement.rs      ✓        ✓       ✓      ✓              ✓
node.rs                             ✓                     ✓
ops/arithmetic.rs                   ✓      ✓              ✓
ops/base.rs                         ✓      ✓              ✓
ops/inv.rs         ✓                ✓      ✓              ✓
ops/sin.rs         ✓                ✓      ✓              ✓
ordered_pair.rs    ✓
error.rs           ✓
binary/*           ·                                      ✓ (mutual)
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
