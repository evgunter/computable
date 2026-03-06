# Source Code Structure

This document describes the module organization and dependencies.

## Module Descriptions

| Module | Description |
|--------|-------------|
| `lib.rs` | Crate root; declares modules and re-exports public API |
| `binary/` | Arbitrary-precision binary numbers (mantissa * 2^exponent) |
| `ordered_pair.rs` | Interval types: `Bounds` and `Interval` with ordering guarantees |
| `prefix.rs` | `Prefix` type: power-of-2 width intervals for the refinement system |
| `finite_interval.rs` | `FiniteInterval`: lightweight interval arithmetic for Taylor series |
| `error.rs` | Error types for computable operations (`ComputableError`) |
| `node.rs` | Computation graph infrastructure (`Node`, `NodeOp` trait, `BaseNode`) |
| `ops/` | Operation implementations for the computation graph |
| `ops/arithmetic.rs` | `AddOp`, `NegOp`, `MulOp` |
| `ops/inv.rs` | `InvOp` (multiplicative inverse via Newton-Raphson) |
| `ops/nth_root.rs` | `NthRootOp` (nth root via bisection) |
| `ops/pow.rs` | `PowOp` (integer exponentiation) |
| `ops/sin.rs` | `SinOp` (sine via Taylor series with directed rounding) |
| `ops/pi.rs` | `PiOp` (pi via Machin-like formula) |
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
             / |
            ▼  ▼
        prefix error
            |   |
            ▼   ▼
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
   - Each refiner targets a precision budget (width exponent) via demand propagation
   - Propagates `Prefix` updates through the graph (power-of-2 width for efficient dispatch)
   - Exact `Bounds` are maintained separately for arithmetic precision
   - Returns when root prefix width ≤ epsilon
4. **`bounds()`** returns current exact interval bounds without refinement
5. **`prefix()`** returns current `Prefix` (power-of-2 width interval) without refinement
