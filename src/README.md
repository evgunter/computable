# Source Code Structure

This document describes the module organization and dependencies.

## Module Descriptions

| Module | Description |
|--------|-------------|
| `lib.rs` | Crate root; declares modules and re-exports public API |
| `binary/` | Arbitrary-precision binary numbers (mantissa × 2^exponent) |
| `ordered_pair.rs` | Generic `Interval<T, W>` type with ordering guarantees |
| `prefix.rs` | `Prefix` type: compact interval representation (inner endpoint + width exponent) |
| `finite_interval.rs` | `FiniteInterval`: lightweight `(Binary, Binary)` wrapper for interval arithmetic |
| `error.rs` | Error types for computable operations (`ComputableError`) |
| `node.rs` | Computation graph infrastructure (`Node`, `NodeOp` trait, `BaseNode`, `PrefixAccess`) |
| `ops/` | Operation implementations for the computation graph |
| `ops/arithmetic.rs` | `AddOp`, `NegOp`, `MulOp` |
| `ops/inv.rs` | `InvOp` (multiplicative inverse with Newton-Raphson refinement) |
| `ops/pow.rs` | `PowOp` (integer power) |
| `ops/sin.rs` | `SinOp` (sine via Taylor series with directed rounding) |
| `ops/pi.rs` | `PiOp` (pi via Machin's formula with interval propagation) |
| `ops/nth_root.rs` | `NthRootOp` (n-th root via normalized bisection) |
| `ops/base.rs` | `BaseOp` (wraps user-defined leaf nodes) |
| `refinement.rs` | Parallel refinement infrastructure (`RefinementGraph`, `RefinerHandle`) |
| `computable.rs` | Main `Computable` type with arithmetic operator implementations |
| `sane.rs` | Sanity-check assertions for catching infinite/huge computations early |
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
              / |
             /  ▼
            / prefix
            |   |
            ▼   ▼
           error
             |
             ▼
           binary
            / \
           ▼   ▼
  ordered_pair  finite_interval
```

## Data Flow

1. **User creates `Computable`** values via `Computable::new()` or `Computable::constant()`
2. **Arithmetic operations** (`+`, `-`, `*`, `/`, `.inv()`, `.sin()`, `.pow()`, `.nth_root()`) build a **computation graph** of `Node`s
3. **`refine_to(epsilon)`** triggers parallel refinement:
   - `RefinementGraph` discovers all refiner nodes and their dependencies
   - Spawns refiner threads for leaf nodes and self-refining ops (InvOp, NthRootOp)
   - Propagates prefix updates upward through passive combinators
   - Uses demand propagation to skip refiners that are already precise enough
   - Returns when root prefix width ≤ epsilon
4. **`prefix()`** returns the current `Prefix` (interval in compact form) without refinement
