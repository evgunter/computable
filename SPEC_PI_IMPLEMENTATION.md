# Specification: Implement Pi as a Computable Number

## Overview

This library implements **exact real arithmetic** using interval refinement. Every `Computable` value maintains provable bounds that can be refined to arbitrary precision. **There is zero tolerance for approximations without tracked error bounds.**

Your task is to:
1. Implement π (pi) as a proper `Computable` number
2. Integrate it with the existing `sin` implementation so that **all error from pi is properly propagated through the final sin bounds**

## Current State

### The Problem

In `src/ops/sin.rs`, there is a hardcoded pi approximation:

```rust
// TODO: make this a computable number so that the results remain provably correct
// (right now they're logically incorrect because of the approximation used for pi!)
fn pi_binary() -> Binary {
    // pi * 2^61 = 7244019458077122842.70...
    let mantissa = BigInt::parse_bytes(b"7244019458077122843", 10)
        .unwrap_or_else(|| BigInt::from(3));
    Binary::new(mantissa, BigInt::from(-61))
}
```

This hardcoded value is used for:
1. **Range reduction**: Reducing input `x` to `[-π, π]` by subtracting multiples of `2π`
2. **Further reduction**: Reducing to `[-π/2, π/2]` using identities like `sin(x) = sin(π - x)`
3. **Critical point detection**: Checking if an interval contains `π/2 + nπ` (where sin = ±1)

The current implementation is **mathematically incorrect** because the error in the pi approximation is not tracked or propagated to the final bounds.

### Existing Infrastructure

Study these files to understand the patterns:
- `src/computable.rs` - The `Computable` type and how to create base computables
- `src/node.rs` - The `NodeOp` trait for implementing refinable operations
- `src/binary.rs` - The `Binary` type (mantissa × 2^exponent) and `Bounds` type
- `src/ops/sin.rs` - The existing sin implementation you'll be modifying
- `src/ops/inv.rs` - Example of a refinable operation with proper error handling

## Requirements

### Part 1: Implement Pi as a Computable

Create `src/ops/pi.rs` implementing π using **Machin's formula**:

```
π/4 = 4·arctan(1/5) - arctan(1/239)
```

The arctan Taylor series is:
```
arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
```

For `|x| < 1`, the error after `n` terms is bounded by `|x|^(2n+1) / (2n+1)`.

**Requirements:**
1. Return a `Computable` that can be refined to arbitrary precision
2. Use **directed rounding**: lower bounds round down, upper bounds round up
3. The bounds must **provably contain** the true value of π
4. Each refinement step must produce tighter (or equal) bounds

**Suggested API:**
```rust
/// Returns π as a Computable that can be refined to arbitrary precision.
pub fn pi() -> Computable

/// Returns bounds on π with at least `precision_bits` bits of accuracy.
/// This is a helper for use in sin.rs.
pub fn pi_bounds_at_precision(precision_bits: u64) -> (Binary, Binary)
```

### Part 2: Integrate Pi with Sin (CRITICAL)

This is where previous attempts failed. You must ensure that **any error in the pi computation is propagated to the final sin bounds**.

#### The Core Problem

When we compute `sin(x)`, we:
1. Compute `k = round(x / 2π)` to find how many periods to subtract
2. Compute `x_reduced = x - k·2π`
3. Compute `sin(x_reduced)` using Taylor series

The error in step 2 depends on the error in our value of `2π`. If our pi bounds are `[π_lo, π_hi]`, then:
- `k·2π` is actually in the interval `[k·2π_lo, k·2π_hi]` (for k > 0)
- So `x_reduced` is in an interval, not a point

**This interval must be tracked through to the final bounds.**

#### What You Must Implement

The approach is to track the interval of possible reduced values and handle boundary cases correctly.

##### Step 1: Compute pi to sufficient precision

Compute pi precision dynamically based on input magnitude:

```rust
// k = approximate number of 2π periods to subtract
let k = compute_reduction_factor(&x, &two_pi_approx);

// We need pi error small enough that we can determine which "branch" we're in.
// Specifically, we want the reduced value's uncertainty to be < π/4, so we're
// not straddling multiple branches.
//
// Reduced value uncertainty = k * 2 * π_error
// Requirement: k * 2 * π_error < π/4
// Therefore: π_error < π / (8k)
// In bits: π_precision > log2(8k/π) ≈ log2(k) + 1.7
//
// For final answer precision ε, we also need: π_error < ε / (2k)
// In bits: π_precision > log2(2k/ε)
//
// Take the max of these two requirements.
let precision_for_branch = log2(k) + 2;  // +2 for safety margin
let precision_for_answer = log2(k) + log2(1/target_epsilon) + 1;
let required_pi_bits = max(precision_for_branch, precision_for_answer);

let (pi_lo, pi_hi) = pi_bounds_at_precision(required_pi_bits);
let pi_mid = (pi_lo + pi_hi) / 2;
let pi_err = (pi_hi - pi_lo) / 2;
```

##### Step 2: Compute reduced value as an interval

```rust
// Range reduction: x_reduced = x - k * 2π
// Using midpoint of π, the reduced value has uncertainty k * 2 * π_err
let two_pi_mid = pi_mid * 2;
let reduced_mid = x - k * two_pi_mid;
let reduced_err = k * 2 * pi_err;

let reduced_lo = reduced_mid - reduced_err;
let reduced_hi = reduced_mid + reduced_err;
```

##### Step 3: Handle branch selection with interval awareness

When further reducing from `[-π, π]` to `[-π/2, π/2]`, check which branch the interval falls into:

```rust
let half_pi_lo = pi_lo / 2;
let half_pi_hi = pi_hi / 2;

if reduced_hi <= half_pi_lo && reduced_lo >= -half_pi_hi {
    // Entire interval is in [-π/2, π/2], use directly
    // No transformation needed, sign_flip = false

} else if reduced_lo >= half_pi_lo {
    // Entire interval is in [π/2, π], use sin(x) = sin(π - x)
    // Transform: new_reduced = π - reduced
    // new_reduced_lo = π_lo - reduced_hi
    // new_reduced_hi = π_hi - reduced_lo
    // sign_flip = false

} else if reduced_hi <= -half_pi_lo {
    // Entire interval is in [-π, -π/2], use sin(x) = -sin(-π - x)
    // Transform and set sign_flip = true

} else {
    // Interval STRADDLES a critical point (±π/2)!
    // The interval contains a point where sin = ±1.
    // Must set bounds accordingly:
    //   - If straddling +π/2: upper bound = 1
    //   - If straddling -π/2: lower bound = -1
}
```

**This is the critical part that previous implementations got wrong.** When the reduced interval straddles `π/2`, the sin value reaches its maximum of 1 somewhere in that interval. You MUST detect this and set the upper bound to 1 (or lower bound to -1 for `-π/2`).

##### Step 4: Compute Taylor series on the reduced interval

Pass the reduced interval `[reduced_lo, reduced_hi]` to the Taylor series computation:

```rust
let (sin_lo, sin_hi) = taylor_sin_bounds_on_interval(reduced_lo, reduced_hi, n);
```

**Important:** Since sin is NOT monotonic, you cannot just evaluate at endpoints. Within `[-π/2, π/2]`, sin IS monotonic (increasing), so endpoint evaluation works there. But if your interval is wider or in a different range, you need to be careful.

Since we've already reduced to `[-π/2, π/2]` and handled straddling cases, and sin is monotonic increasing on this interval:
```rust
// sin is monotonic increasing on [-π/2, π/2]
let sin_lo = taylor_sin_lower_bound(reduced_lo, n);  // round down
let sin_hi = taylor_sin_upper_bound(reduced_hi, n);  // round up
```

##### Step 5: Apply sign flip if needed

```rust
if sign_flip {
    let (sin_lo, sin_hi) = (-sin_hi, -sin_lo);  // negate and swap
}
```

##### Step 6: Clamp to [-1, 1]

```rust
result_lo = max(result_lo, -1);
result_hi = min(result_hi, 1);
```

#### Alternative: Full Interval Propagation (Option B)

Instead of tracking a midpoint ± error, you can propagate full intervals `[lo, hi]` through every operation. This is more complex but can give tighter bounds in some cases.

With this approach:
- `reduce_to_pi_range(x)` returns `(Binary, Binary)` instead of `Binary`
- `reduce_to_half_pi_range(x)` returns `((Binary, Binary), bool)`
- All arithmetic uses interval arithmetic: `[a,b] + [c,d] = [a+c, b+d]`, etc.

**Caution:** Previous attempts at this approach introduced bugs because:
1. Interval arithmetic for subtraction requires care: `[a,b] - [c,d] = [a-d, b-c]` (note the swap)
2. When transforming via `sin(x) = sin(π - x)`, the interval `π - [a,b]` becomes `[π_lo - b, π_hi - a]`
3. It's easy to lose track of which bound is which after multiple transformations

Only choose this approach if you're confident with interval arithmetic. The recommended approach above (tracking midpoint ± error) is easier to get right and sufficient for correctness.

#### Critical Points Detection

The existing `interval_contains_critical_points()` function checks if an interval contains `π/2 + nπ`. This must also use pi bounds:

```rust
fn interval_contains_critical_points(
    lower: &Binary,
    upper: &Binary,
    pi_lo: &Binary,
    pi_hi: &Binary,
) -> (bool, bool) {
    // Use OUTER bounds for conservative detection:
    // - To check if interval MIGHT contain π/2, use the smallest possible π/2
    // - This ensures we never miss a critical point

    let half_pi_inner = pi_lo / 2;  // smallest possible π/2
    let half_pi_outer = pi_hi / 2;  // largest possible π/2

    // Check if [lower, upper] might contain half_pi (or half_pi + n*2π)
    // Be conservative: if there's any chance it contains the critical point, say yes

    // ... implementation details ...
}
```

### Part 3: Testing

Add tests that verify:

1. **Pi bounds contain true pi:**
```rust
#[test]
fn pi_bounds_contain_pi() {
    // Use a known high-precision value of pi to verify bounds
    // π = 3.14159265358979323846...
}
```

2. **Pi refines correctly:**
```rust
#[test]
fn pi_refines_to_arbitrary_precision() {
    let pi = pi();
    let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-100)); // 2^-100
    let bounds = pi.refine_to_default(epsilon).unwrap();
    assert!(bounds.width() <= epsilon);
}
```

3. **Sin of pi is near zero with correct bounds:**
```rust
#[test]
fn sin_of_pi_bounds_contain_zero() {
    let x = /* pi as f64 converted to Binary */;
    let sin_x = Computable::constant(x).sin();
    let bounds = sin_x.refine_to_default(epsilon).unwrap();
    // Bounds must contain 0, and the width should account for pi error
}
```

4. **Sin at large multiples of pi:**
```rust
#[test]
fn sin_of_100_pi() {
    // sin(100π) = 0, but range reduction uses pi 100 times
    // Error should accumulate appropriately
}
```

## What NOT To Do

Based on previous failed attempts:

1. **DO NOT** just use a higher-precision fixed approximation and hope it's enough. Pi precision must scale with input magnitude.

2. **DO NOT** compute pi bounds but then only use the midpoint without tracking the error interval through range reduction.

3. **DO NOT** evaluate sin at interval endpoints and assume that gives valid bounds. Sin is NOT monotonic in general. You must either:
   - Reduce to `[-π/2, π/2]` first (where sin IS monotonic), OR
   - Detect and handle critical points within the interval

4. **DO NOT** ignore the case where the reduced interval straddles `π/2` or `-π/2`. This is where sin reaches ±1, and failing to detect this gives incorrect bounds.

5. **DO NOT** use floating-point operations (like `f64::log2()`) for precision calculations in the final implementation. Use integer/BigInt arithmetic. (Pseudocode in this spec uses float for clarity.)

6. **DO NOT** modify the core sin Taylor series logic. The existing `taylor_sin_bounds()` function is correct for point inputs. You need to:
   - Fix pi-related range reduction
   - Handle interval inputs to the Taylor series correctly

## Correctness Criteria

Your implementation is correct if and only if:

1. For ANY input `x` and ANY precision `ε`, the bounds returned by `sin(x).refine_to_default(ε)` **provably contain** the true mathematical value of sin(x).

2. The width of the bounds converges to the requested precision (accounting for any fundamental limitations from pi precision).

3. No approximation is made without its error being tracked and included in the final bounds.

## Performance

Performance is NOT a priority. A slower correct implementation is infinitely better than a faster incorrect one. That said:
- Don't be gratuitously wasteful
- Recomputing pi bounds on every sin call is acceptable
- For large inputs, pi precision must scale with `log2(|x|)` - this is unavoidable for correctness

## Deliverables

1. `src/ops/pi.rs` - Pi implementation
2. Modified `src/ops/sin.rs` - Integration with proper error propagation
3. Modified `src/ops/mod.rs` - Export the new pi module
4. Modified `src/lib.rs` - Re-export `pi()` function
5. Tests demonstrating correctness

## Questions to Answer Before Starting

1. How will you compute the required pi precision given input magnitude `|x|` and target precision `ε`?

2. After range reduction, you have an interval `[reduced_lo, reduced_hi]`. How will you determine which branch of the reduction to `[-π/2, π/2]` applies? What if the interval straddles a branch boundary?

3. If the reduced interval straddles `π/2` (meaning it contains a maximum of sin), what bounds should you return?

4. The existing `taylor_sin_bounds(x, n)` takes a point `x`. How will you adapt it to take an interval `[x_lo, x_hi]`? (Hint: sin is monotonic increasing on `[-π/2, π/2]`)

5. How will you modify `interval_contains_critical_points()` to use pi bounds instead of a fixed pi value?

Answer these questions in comments in your code before implementing. If you're unsure about any of them, that's a sign you need to think more carefully before writing code.
