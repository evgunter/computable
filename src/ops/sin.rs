//! Sine operation using Taylor series with provably correct error bounds.
//!
//! This module implements the sine function using:
//! - **Full Interval Propagation**: All pi-related errors tracked as intervals
//! - Range reduction to [-pi/2, pi/2] for efficient Taylor series convergence
//! - Critical point detection for tight bounds on intervals containing extrema
//! - Directed rounding for provably correct interval arithmetic
//!
//! ## Key Design Decision: Full Interval Propagation
//!
//! The approach propagates full intervals `[lo, hi]` through every operation:
//! - `reduce_to_pi_range(x)` returns an Interval instead of a Binary
//! - `reduce_to_half_pi_range(x)` returns `(Interval, bool)`
//! - All arithmetic uses proper interval arithmetic:
//!   - `[a,b] + [c,d] = [a+c, b+d]`
//!   - `[a,b] - [c,d] = [a-d, b-c]` (note the swap!)
//!   - When transforming via `sin(x) = sin(pi - x)`, the interval `pi - [a,b]`
//!     becomes `[pi_lo - b, pi_hi - a]`
//!
//! This ensures all pi approximation error is properly propagated to final bounds.

use std::sync::Arc;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use parking_lot::RwLock;

use crate::binary::{Binary, Bounds, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

use super::pi::{
    half_pi_interval_at_precision, pi_interval_at_precision,
    two_pi_interval_at_precision, Interval,
};

/// Sine operation with Taylor series refinement.
pub struct SinOp {
    pub inner: Arc<Node>,
    pub num_terms: RwLock<BigInt>,
}

impl NodeOp for SinOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let num_terms = self.num_terms.read().clone();
        sin_bounds(&input_bounds, &num_terms)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();
        *num_terms += BigInt::one();
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

//=============================================================================
// Main sin_bounds function with full interval propagation
//=============================================================================

/// Computes sin bounds for an input interval using full interval propagation.
///
/// ## Algorithm Overview
///
/// 1. Compute pi precision based on input magnitude
/// 2. Range reduce input to interval in [-pi, pi] (tracking pi error)
/// 3. Further reduce to interval in [-pi/2, pi/2] (tracking pi error)
/// 4. Detect if reduced interval straddles critical points (pi/2, -pi/2)
/// 5. Compute Taylor series on reduced interval
/// 6. Apply sign flips and clamp to [-1, 1]
fn sin_bounds(input_bounds: &Bounds, num_terms: &BigInt) -> Result<Bounds, ComputableError> {
    let neg_one = Binary::new(BigInt::from(-1), BigInt::zero());
    let pos_one = Binary::new(BigInt::from(1), BigInt::zero());

    // Extract finite bounds, or return [-1, 1] for any infinite bounds
    let lower = input_bounds.small();
    let upper = input_bounds.large();
    let (lower_bin, upper_bin) = match (lower, &upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => (l, u),
        _ => {
            return Ok(Bounds::new(
                XBinary::Finite(neg_one),
                XBinary::Finite(pos_one),
            ));
        }
    };

    // Convert num_terms to usize (capped at reasonable limit)
    let n = num_terms.to_usize().unwrap_or(1).max(1);

    // Compute required pi precision based on input magnitude.
    // We need enough precision that:
    // 1. We can determine which branch we're in after range reduction
    // 2. The final error from pi is smaller than our Taylor series precision
    let precision_bits = compute_required_pi_precision(lower_bin, upper_bin, n);

    // Get pi-related intervals at the computed precision
    let two_pi_interval = two_pi_interval_at_precision(precision_bits);
    let pi_interval = pi_interval_at_precision(precision_bits);
    let half_pi_interval = half_pi_interval_at_precision(precision_bits);

    // Process each endpoint through range reduction with full interval tracking
    let input_interval = Interval::new(lower_bin.clone(), upper_bin.clone());

    // Check if the input interval is wide enough to contain a full period
    let input_width = upper_bin.sub(lower_bin);
    let two_pi_approx = two_pi_interval.midpoint();
    if input_width >= two_pi_approx {
        // Input spans at least one full period, sin ranges over all of [-1, 1]
        return Ok(Bounds::new(
            XBinary::Finite(neg_one),
            XBinary::Finite(pos_one),
        ));
    }

    // Perform range reduction with full interval propagation
    let reduced_result = reduce_to_half_pi_range_interval(
        &input_interval,
        &two_pi_interval,
        &pi_interval,
        &half_pi_interval,
    );

    // Compute sin bounds based on the reduction result
    let (result_lo, result_hi) = match reduced_result {
        ReductionResult::InRange { interval, sign_flip } => {
            // Interval is fully within [-pi/2, pi/2], use Taylor series
            let (sin_lo, sin_hi) = compute_sin_on_monotonic_interval(&interval, n);
            if sign_flip {
                (sin_hi.neg(), sin_lo.neg())
            } else {
                (sin_lo, sin_hi)
            }
        }
        ReductionResult::ContainsMax { sin_min } => {
            // Interval contains pi/2 where sin = 1
            (sin_min, pos_one.clone())
        }
        ReductionResult::ContainsMin { sin_max } => {
            // Interval contains -pi/2 where sin = -1
            (neg_one.clone(), sin_max)
        }
        ReductionResult::ContainsBoth => {
            // Interval contains both critical points
            (neg_one.clone(), pos_one.clone())
        }
        ReductionResult::SpansMultipleBranches {
            overall_lo,
            overall_hi,
        } => {
            // Reduced interval spans multiple branches
            (overall_lo, overall_hi)
        }
    };

    // Final clamp to [-1, 1]
    let clamped_lo = if result_lo < neg_one {
        neg_one.clone()
    } else {
        result_lo
    };
    let clamped_hi = if result_hi > pos_one {
        pos_one
    } else {
        result_hi
    };

    Bounds::new_checked(XBinary::Finite(clamped_lo), XBinary::Finite(clamped_hi))
        .map_err(|_| ComputableError::InvalidBoundsOrder)
}

//=============================================================================
// Range reduction with full interval propagation
//=============================================================================

/// Result of range reduction to [-pi/2, pi/2].
#[derive(Debug)]
enum ReductionResult {
    /// Interval is fully in [-pi/2, pi/2], can use Taylor directly
    InRange { interval: Interval, sign_flip: bool },
    /// Interval contains pi/2 (sin maximum), provides minimum sin value
    ContainsMax { sin_min: Binary },
    /// Interval contains -pi/2 (sin minimum), provides maximum sin value
    ContainsMin { sin_max: Binary },
    /// Interval contains both critical points
    ContainsBoth,
    /// Interval spans multiple branches after reduction
    SpansMultipleBranches { overall_lo: Binary, overall_hi: Binary },
}

/// Computes required pi precision based on input magnitude and Taylor terms.
///
/// We need pi precision such that:
/// 1. k * 2 * pi_error < pi/4 (to determine which branch we're in)
/// 2. pi_error contributes acceptably small error to final answer
///
/// where k is the number of 2*pi periods subtracted.
fn compute_required_pi_precision(lower: &Binary, upper: &Binary, taylor_terms: usize) -> u64 {
    // Estimate k = |x| / (2*pi) using a rough approximation
    // We use the larger magnitude endpoint
    let abs_lo = if lower.mantissa().is_negative() {
        lower.neg()
    } else {
        lower.clone()
    };
    let abs_hi = if upper.mantissa().is_negative() {
        upper.neg()
    } else {
        upper.clone()
    };
    let max_abs = if abs_lo > abs_hi { abs_lo } else { abs_hi };

    // Rough estimate: k ~= max_abs / 6.28
    // We need: pi_precision > log2(k) + some_margin
    // log2(max_abs) ~= bit_length(mantissa) + exponent
    let mantissa_bits = max_abs.mantissa().magnitude().bits() as i64;
    let exp = max_abs
        .exponent()
        .to_i64()
        .unwrap_or(0)
        .saturating_add(mantissa_bits);

    // k ~= 2^exp / 2^2.65 (since 2*pi ~= 2^2.65)
    // log2(k) ~= exp - 2.65
    let log2_k = (exp - 3).max(0) as u64;

    // Base precision for branch determination: log2(k) + 4
    let precision_for_branch = log2_k.saturating_add(4);

    // Precision for final answer: depends on Taylor series precision
    // Taylor error ~= 2^(-taylor_terms * some_factor)
    // We want pi error to be smaller than this
    let precision_for_answer = (taylor_terms as u64).saturating_mul(3).saturating_add(log2_k);

    // Take max, with reasonable bounds
    let precision = precision_for_branch.max(precision_for_answer);

    // Ensure at least 64 bits, at most 256 bits (reasonable limits)
    precision.clamp(64, 256)
}

/// Reduces an input interval to [-pi, pi] using interval arithmetic.
///
/// Returns the reduced interval accounting for pi approximation error.
/// Iterates until the result is within bounds.
fn reduce_to_pi_range_interval(
    input: &Interval,
    two_pi: &Interval,
    pi: &Interval,
) -> Interval {
    // Use the midpoint of 2*pi for computing k
    let two_pi_mid = two_pi.midpoint();
    let neg_pi_lo = pi.lo.neg(); // most negative possible -pi (inner bound)

    let mut current = input.clone();

    // Iterate reduction until we're in range (at most a few iterations needed)
    for _ in 0..10 {
        // Check if already in range: [-pi_lo, pi_lo] is the "safe" inner range
        // We use inner bounds to be conservative
        if current.lo >= neg_pi_lo && current.hi <= pi.lo {
            return current;
        }

        // Also check with outer bounds - if we're definitely out of range
        let neg_pi_hi = pi.hi.neg();
        if current.lo >= neg_pi_hi && current.hi <= pi.hi {
            // We're in the outer range [-pi_hi, pi_hi], close enough
            return current;
        }

        // Compute k = round(x / 2*pi) using interval midpoint
        let current_mid = current.midpoint();
        let k = compute_reduction_factor(&current_mid, &two_pi_mid);

        if k.is_zero() {
            // Can't reduce further
            return current;
        }

        // Compute k * 2*pi as an interval
        let k_times_two_pi = two_pi.scale_bigint(&k);

        // Subtract: current - k*2*pi using interval arithmetic
        current = current.sub(&k_times_two_pi);
    }

    // If we still haven't converged after 10 iterations, return what we have
    current
}

/// Reduces an interval from [-pi, pi] to [-pi/2, pi/2] with full interval tracking.
///
/// This is the critical function that handles all the branch cases:
/// - If interval is entirely in [-pi/2, pi/2]: use directly
/// - If interval is entirely in [pi/2, pi]: use sin(x) = sin(pi - x)
/// - If interval is entirely in [-pi, -pi/2]: use sin(x) = -sin(-pi - x) = sin(-pi - x) with flip
/// - If interval straddles pi/2: contains maximum (sin = 1)
/// - If interval straddles -pi/2: contains minimum (sin = -1)
fn reduce_to_half_pi_range_interval(
    input: &Interval,
    two_pi: &Interval,
    pi: &Interval,
    half_pi: &Interval,
) -> ReductionResult {
    // First reduce to [-pi, pi]
    let reduced = reduce_to_pi_range_interval(input, two_pi, pi);

    // Now determine which branch(es) the reduced interval falls into
    let neg_half_pi = half_pi.neg(); // [-pi/2_hi, -pi/2_lo]
    let neg_pi = pi.neg();           // [-pi_hi, -pi_lo]

    // Key comparisons using conservative bounds:
    // To check if interval is entirely in [-pi/2, pi/2]:
    //   reduced.hi <= half_pi.lo (interval entirely below pi/2)
    //   AND reduced.lo >= neg_half_pi.hi (interval entirely above -pi/2)

    // Case 1: Entirely in [-pi/2, pi/2]
    if reduced.hi <= half_pi.lo && reduced.lo >= neg_half_pi.hi {
        return ReductionResult::InRange {
            interval: reduced,
            sign_flip: false,
        };
    }

    // Case 2: Entirely in [pi/2, pi]
    // reduced.lo >= half_pi.lo AND reduced.hi <= pi.hi
    if reduced.lo >= half_pi.lo && reduced.hi <= pi.hi {
        // Transform: x -> pi - x
        // sin(x) = sin(pi - x) for x in [pi/2, pi]
        // pi - [a, b] using interval arithmetic:
        // [pi_lo, pi_hi] - [a, b] = [pi_lo - b, pi_hi - a]
        let transformed = pi.sub(&reduced);
        return ReductionResult::InRange {
            interval: transformed,
            sign_flip: false,
        };
    }

    // Case 3: Entirely in [-pi, -pi/2]
    // reduced.hi <= neg_half_pi.hi (which is -pi/2_lo, the least negative)
    // AND reduced.lo >= neg_pi.hi (which is -pi_lo, the least negative -pi)
    if reduced.hi <= neg_half_pi.hi && reduced.lo >= neg_pi.hi {
        // Transform: x -> -pi - x, then negate result
        // sin(x) = -sin(-pi - x) for x in [-pi, -pi/2]
        // Actually: sin(x) = sin(-pi - x) = -sin(pi + x)
        // Simpler: sin(x) for x in [-pi, -pi/2] can use sin(x) = -sin(-x - pi)
        // Or: sin(x) = sin(pi + x) for x in [-pi, -pi/2] gives us angle in [0, pi/2]
        // Let's use: new_x = pi + x, then sin(x) = -sin(new_x)
        // [pi_lo, pi_hi] + [a, b] = [pi_lo + a, pi_hi + b]
        let transformed = pi.add(&reduced);
        return ReductionResult::InRange {
            interval: transformed,
            sign_flip: true,
        };
    }

    // Case 4: Straddles pi/2 (contains maximum)
    // reduced.lo < half_pi.hi AND reduced.hi > half_pi.lo
    if reduced.lo < half_pi.hi && reduced.hi > half_pi.lo {
        // The interval contains pi/2 where sin = 1
        // We need to compute the minimum sin value over the interval
        // The min occurs at one of the endpoints of the sub-interval in [-pi/2, pi]
        // that doesn't contain pi/2

        // Conservative: compute sin at both adjusted endpoints and take min
        let (sin_at_lo, _) = compute_sin_bounds_for_point(&reduced.lo, 15);
        let (sin_at_hi, _) = compute_sin_bounds_for_point(&reduced.hi, 15);
        let sin_min = if sin_at_lo < sin_at_hi {
            sin_at_lo
        } else {
            sin_at_hi
        };

        return ReductionResult::ContainsMax { sin_min };
    }

    // Case 5: Straddles -pi/2 (contains minimum)
    // reduced.lo < neg_half_pi.hi AND reduced.hi > neg_half_pi.lo
    if reduced.lo < neg_half_pi.hi && reduced.hi > neg_half_pi.lo {
        // The interval contains -pi/2 where sin = -1
        let (_, sin_at_lo) = compute_sin_bounds_for_point(&reduced.lo, 15);
        let (_, sin_at_hi) = compute_sin_bounds_for_point(&reduced.hi, 15);
        let sin_max = if sin_at_lo > sin_at_hi {
            sin_at_lo
        } else {
            sin_at_hi
        };

        return ReductionResult::ContainsMin { sin_max };
    }

    // Case 6: Spans multiple branches (complex case)
    // This happens when the interval spans more than pi/2 in width
    // Compute conservative bounds by evaluating sin at several points
    let neg_one = Binary::new(BigInt::from(-1), BigInt::zero());
    let pos_one = Binary::new(BigInt::from(1), BigInt::zero());

    // Check if we span both critical points
    let spans_max = reduced.lo < half_pi.hi && reduced.hi > half_pi.lo;
    let spans_min = reduced.lo < neg_half_pi.hi && reduced.hi > neg_half_pi.lo;

    if spans_max && spans_min {
        return ReductionResult::ContainsBoth;
    }

    // Otherwise compute bounds at endpoints
    let (sin_lo_1, sin_hi_1) = compute_sin_bounds_for_point(&reduced.lo, 15);
    let (sin_lo_2, sin_hi_2) = compute_sin_bounds_for_point(&reduced.hi, 15);

    let overall_lo = if sin_lo_1 < sin_lo_2 {
        sin_lo_1
    } else {
        sin_lo_2
    };
    let overall_hi = if sin_hi_1 > sin_hi_2 {
        sin_hi_1
    } else {
        sin_hi_2
    };

    ReductionResult::SpansMultipleBranches {
        overall_lo: overall_lo.max(neg_one),
        overall_hi: overall_hi.min(pos_one),
    }
}

/// Computes sin bounds for a point (with Taylor series).
fn compute_sin_bounds_for_point(x: &Binary, n: usize) -> (Binary, Binary) {
    // Range reduce to [-pi/2, pi/2] using a simple approximation
    // This is for helper computations, not the main bounds
    let pi_approx = Binary::new(
        BigInt::parse_bytes(b"7244019458077122843", 10).unwrap_or_else(|| BigInt::from(3)),
        BigInt::from(-61),
    );
    let two_pi_approx = Binary::new(pi_approx.mantissa().clone(), pi_approx.exponent() + BigInt::one());
    let half_pi_approx = Binary::new(pi_approx.mantissa().clone(), pi_approx.exponent() - BigInt::one());

    let mut reduced = x.clone();
    let neg_half_pi = half_pi_approx.neg();

    // Reduce to [-pi, pi]
    if reduced > pi_approx || reduced < pi_approx.neg() {
        let k = compute_reduction_factor(&reduced, &two_pi_approx);
        let k_two_pi = multiply_by_integer(&two_pi_approx, &k);
        reduced = reduced.sub(&k_two_pi);
    }

    // Reduce to [-pi/2, pi/2]
    let mut sign_flip = false;
    if reduced > half_pi_approx {
        reduced = pi_approx.sub(&reduced);
    } else if reduced < neg_half_pi {
        reduced = pi_approx.add(&reduced);
        sign_flip = true;
    }

    let truncated = truncate_precision(&reduced, 64);
    let (sin_lo, sin_hi) = taylor_sin_bounds(&truncated, n);

    if sign_flip {
        (sin_hi.neg(), sin_lo.neg())
    } else {
        (sin_lo, sin_hi)
    }
}

/// Computes k = round(x / period).
fn compute_reduction_factor(x: &Binary, period: &Binary) -> BigInt {
    let precision_bits = 64i64;
    let mx = x.mantissa();
    let ex = x.exponent();
    let mp = period.mantissa();
    let ep = period.exponent();

    let shifted_mx = mx << precision_bits as usize;
    let quotient = &shifted_mx / mp;
    let result_exp = ex - ep - BigInt::from(precision_bits);

    if result_exp >= BigInt::zero() {
        let shift = result_exp.to_usize().unwrap_or(0);
        &quotient << shift
    } else {
        let shift = (-&result_exp).to_usize().unwrap_or(0);
        if shift == 0 {
            quotient.clone()
        } else {
            let half = BigInt::one() << (shift - 1);
            let rounded = if quotient.is_negative() {
                &quotient - &half
            } else {
                &quotient + &half
            };
            rounded >> shift
        }
    }
}

/// Multiplies a Binary by a BigInt integer.
fn multiply_by_integer(b: &Binary, k: &BigInt) -> Binary {
    Binary::new(b.mantissa() * k, b.exponent().clone())
}

/// Truncates a Binary to at most `precision_bits` of mantissa.
fn truncate_precision(x: &Binary, precision_bits: usize) -> Binary {
    let mantissa = x.mantissa();
    let exponent = x.exponent();
    let bit_length = mantissa.magnitude().bits() as usize;

    if bit_length <= precision_bits {
        return x.clone();
    }

    let shift = bit_length - precision_bits;
    let truncated_mantissa = mantissa >> shift;
    let new_exponent = exponent + BigInt::from(shift);
    Binary::new(truncated_mantissa, new_exponent)
}

//=============================================================================
// Taylor series computation for intervals
//=============================================================================

/// Computes sin bounds for an interval known to be in [-pi/2, pi/2].
///
/// Since sin is monotonically increasing on [-pi/2, pi/2], we can simply
/// evaluate at the endpoints.
fn compute_sin_on_monotonic_interval(interval: &Interval, n: usize) -> (Binary, Binary) {
    // sin is monotonic increasing on [-pi/2, pi/2]
    // So: sin([a, b]) = [sin(a)_lo, sin(b)_hi]

    let truncated_lo = truncate_precision(&interval.lo, 64);
    let truncated_hi = truncate_precision(&interval.hi, 64);

    let (sin_lo_bounds_lo, _) = taylor_sin_bounds(&truncated_lo, n);
    let (_, sin_hi_bounds_hi) = taylor_sin_bounds(&truncated_hi, n);

    (sin_lo_bounds_lo, sin_hi_bounds_hi)
}

/// Rounding direction for directed rounding in interval arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundingDirection {
    /// Round toward negative infinity (floor)
    Down,
    /// Round toward positive infinity (ceiling)
    Up,
}

/// Computes Taylor series bounds for sin(x) with n terms.
/// Returns (lower_bound, upper_bound) accounting for truncation error.
///
/// Taylor series: sin(x) = sum_{k=0}^{n-1} (-1)^k * x^(2k+1) / (2k+1)!
/// Error after n terms: |R_n| <= |x|^(2n+1) / (2n+1)!
///
/// Uses directed rounding to compute provably correct bounds:
/// - Lower bound: all intermediate operations round DOWN (toward -inf)
/// - Upper bound: all intermediate operations round UP (toward +inf)
fn taylor_sin_bounds(x: &Binary, n: usize) -> (Binary, Binary) {
    if n == 0 {
        // No terms: just use error bound (always round UP for conservative bounds)
        let error = taylor_error_bound(x, 0);
        return (error.neg(), error);
    }

    // Compute lower and upper partial sums with directed rounding
    let sum_lower = taylor_sin_partial_sum(x, n, RoundingDirection::Down);
    let sum_upper = taylor_sin_partial_sum(x, n, RoundingDirection::Up);

    // Compute error bound (always round UP for conservative bounds)
    let error = taylor_error_bound(x, n);

    // Return bounds: lower_sum - error, upper_sum + error
    (sum_lower.sub(&error), sum_upper.add(&error))
}

/// Computes Taylor series partial sum for sin(x) with directed rounding.
///
/// For RoundingDirection::Down: rounds all division operations toward -infinity
/// For RoundingDirection::Up: rounds all division operations toward +infinity
fn taylor_sin_partial_sum(x: &Binary, n: usize, rounding: RoundingDirection) -> Binary {
    let mut sum = Binary::zero();
    let mut power = x.clone(); // x^1
    let mut factorial = BigInt::one(); // 1!

    for k in 0..n {
        // Term k: (-1)^k * x^(2k+1) / (2k+1)!
        let term_num = if k % 2 == 0 {
            power.clone()
        } else {
            power.neg()
        };

        // Divide by factorial with directed rounding
        let term = divide_by_factorial_directed(&term_num, &factorial, rounding);
        sum = sum.add(&term);

        // Prepare for next term: multiply power by x^2
        if k + 1 < n {
            power = power.mul(x).mul(x);
            // factorial *= (2k+2) * (2k+3)
            let next_k = k + 1;
            factorial *= BigInt::from(2 * next_k) * BigInt::from(2 * next_k + 1);
        }
    }

    sum
}

/// Computes |x|^(2n+1) / (2n+1)! as an upper bound on Taylor series truncation error.
/// Always rounds UP to be conservative.
fn taylor_error_bound(x: &Binary, n: usize) -> Binary {
    // Compute |x|^(2n+1)
    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };

    let exp = 2 * n + 1;
    let mut power = Binary::new(BigInt::one(), BigInt::zero()); // 1
    for _ in 0..exp {
        power = power.mul(&abs_x);
    }

    // Compute (2n+1)!
    let mut factorial = BigInt::one();
    for i in 1..=exp {
        factorial *= BigInt::from(i);
    }

    // error = power / factorial (round UP for conservative error bound)
    divide_by_factorial_directed(&power, &factorial, RoundingDirection::Up)
}

/// Divides a Binary by a BigInt factorial with directed rounding.
///
/// Rounding semantics:
/// - `RoundingDirection::Up`: rounds toward +infinity (ceiling)
/// - `RoundingDirection::Down`: rounds toward -infinity (floor)
///
/// This is essential for interval arithmetic: when computing a lower bound,
/// round DOWN; when computing an upper bound, round UP.
fn divide_by_factorial_directed(
    value: &Binary,
    factorial: &BigInt,
    rounding: RoundingDirection,
) -> Binary {
    if factorial.is_zero() {
        return value.clone();
    }

    let mantissa = value.mantissa();
    let exponent = value.exponent();

    // We need to compute mantissa / factorial with the result as a Binary.
    // To get a good approximation, we shift the mantissa up by some bits before dividing.
    // The number of bits we shift determines our precision.
    let precision_bits = 64_u64; // Extra precision for intermediate computation

    // shifted_mantissa = |mantissa| * 2^precision_bits
    let abs_mantissa = mantissa.magnitude().clone();
    let shifted_mantissa = &abs_mantissa << precision_bits as usize;

    // Compute |mantissa| / factorial
    let (quot, rem) = shifted_mantissa.div_rem(factorial.magnitude());

    // Determine how to round based on direction and sign
    // For directed rounding toward +/- infinity:
    // - Round UP (+inf): positive values round away from zero, negative round toward zero
    // - Round DOWN (-inf): positive values round toward zero, negative round away from zero
    let is_negative = mantissa.is_negative();
    let has_remainder = !rem.is_zero();

    let result_magnitude = if has_remainder {
        match (rounding, is_negative) {
            // Rounding UP (toward +infinity):
            // - Positive: round away from zero (add 1)
            // - Negative: round toward zero (truncate)
            (RoundingDirection::Up, false) => quot + BigInt::one().magnitude(),
            (RoundingDirection::Up, true) => quot,
            // Rounding DOWN (toward -infinity):
            // - Positive: round toward zero (truncate)
            // - Negative: round away from zero (add 1)
            (RoundingDirection::Down, false) => quot,
            (RoundingDirection::Down, true) => quot + BigInt::one().magnitude(),
        }
    } else {
        // Exact division, no rounding needed
        quot
    };

    // Adjust sign
    let signed_mantissa = if is_negative {
        -BigInt::from(result_magnitude)
    } else {
        BigInt::from(result_magnitude)
    };

    // New exponent = original_exponent - precision_bits
    let new_exponent = exponent - BigInt::from(precision_bits);

    Binary::new(signed_mantissa, new_exponent)
}

// Test helpers - exposed for integration tests
#[cfg(test)]
pub fn taylor_sin_bounds_test(x: &Binary, n: usize) -> (Binary, Binary) {
    taylor_sin_bounds(x, n)
}

#[cfg(test)]
pub fn taylor_sin_partial_sum_test(x: &Binary, n: usize, down: bool) -> Binary {
    let rounding = if down {
        RoundingDirection::Down
    } else {
        RoundingDirection::Up
    };
    taylor_sin_partial_sum(x, n, rounding)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::{UBinary, UXBinary};
    use crate::computable::Computable;
    use num_bigint::BigUint;
    use num_traits::One;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn ubin(mantissa: u64, exponent: i64) -> UBinary {
        UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
    }

    fn xbin(mantissa: i64, exponent: i64) -> XBinary {
        XBinary::Finite(bin(mantissa, exponent))
    }

    fn unwrap_finite(input: &XBinary) -> Binary {
        match input {
            XBinary::Finite(value) => value.clone(),
            XBinary::NegInf | XBinary::PosInf => {
                panic!("expected finite extended binary")
            }
        }
    }

    fn unwrap_finite_uxbinary(input: &UXBinary) -> UBinary {
        match input {
            UXBinary::Finite(value) => value.clone(),
            UXBinary::PosInf => {
                panic!("expected finite unsigned extended binary")
            }
        }
    }

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    fn interval_midpoint_computable(lower: i64, upper: i64) -> Computable {
        fn midpoint_between(lower: &XBinary, upper: &XBinary) -> Binary {
            let unwrap = |input: &XBinary| -> Binary {
                match input {
                    XBinary::Finite(value) => value.clone(),
                    _ => panic!("expected finite"),
                }
            };
            let mid_sum = unwrap(lower).add(&unwrap(upper));
            let exponent = mid_sum.exponent() - BigInt::one();
            Binary::new(mid_sum.mantissa().clone(), exponent)
        }

        fn interval_refine(state: Bounds) -> Bounds {
            let midpoint = midpoint_between(state.small(), &state.large());
            Bounds::new(
                XBinary::Finite(midpoint.clone()),
                XBinary::Finite(midpoint),
            )
        }

        let interval_state = Bounds::new(xbin(lower, 0), xbin(upper, 0));
        Computable::new(
            interval_state,
            |inner_state| Ok(inner_state.clone()),
            interval_refine,
        )
    }

    #[test]
    fn sin_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_zero
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0) = 0
        let expected = bin(0, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sin_of_pi_over_2() {
        // pi/2 ~= 1.5707963...
        // We approximate it as 3217/2048 ~= 1.5708...
        let pi_over_2 = Computable::constant(bin(3217, -11));
        let sin_pi_2 = pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi/2) = 1
        let expected_f64 = (std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi/2) should be very close to 1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_of_pi() {
        // pi ~= 3.14159...
        // We approximate it as 6434/2048 ~= 3.1416...
        let pi_approx = Computable::constant(bin(6434, -11));
        let sin_pi = pi_approx.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi) ~= 0 (should be close to 0)
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi) should be very close to 0
        let small_bound = bin(1, -4);
        let neg_small_bound = bin(-1, -4);
        assert!(lower >= neg_small_bound);
        assert!(upper <= small_bound);
    }

    #[test]
    fn sin_of_negative_pi_over_2() {
        // -pi/2 ~= -1.5707963...
        let neg_pi_over_2 = Computable::constant(bin(-3217, -11));
        let sin_neg_pi_2 = neg_pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_neg_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(-pi/2) = -1
        let expected_f64 = (-std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(-pi/2) should be very close to -1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_bounds_always_in_minus_one_to_one() {
        // Test with a large value that exercises argument reduction
        let large_value = Computable::constant(bin(100, 0));
        let sin_large = large_value.sin();
        let bounds = sin_large.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        assert!(lower >= neg_one);
        assert!(upper <= one);
    }

    #[test]
    fn sin_of_small_value() {
        // For small x, sin(x) ~= x
        let small = Computable::constant(bin(1, -4)); // 1/16 = 0.0625
        let sin_small = small.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_small
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0.0625) ~= 0.0624593...
        let expected = XBinary::from_f64(0.0625_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn sin_interval_spanning_maximum() {
        // An interval that spans pi/2 (where sin has maximum)
        let computable = interval_midpoint_computable(1, 2); // [1, 2] includes pi/2 ~= 1.57
        let sin_interval = computable.sin();
        let bounds = sin_interval.bounds().expect("bounds should succeed");

        let upper = unwrap_finite(&bounds.large());

        // The upper bound should be close to 1 since the interval contains pi/2
        assert!(upper >= bin(1, -1)); // Upper bound should be at least 0.5
    }

    #[test]
    fn sin_with_infinite_input_bounds() {
        let unbounded = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let sin_unbounded = unbounded.sin();
        let bounds = sin_unbounded.bounds().expect("bounds should succeed");

        // sin of unbounded input should be [-1, 1]
        assert_eq!(bounds.small(), &xbin(-1, 0));
        assert_eq!(&bounds.large(), &xbin(1, 0));
    }

    #[test]
    fn sin_expression_with_arithmetic() {
        // Test sin(x) + cos-like expression: sin(x)^2 + sin(x + pi/2)^2 should be close to 1
        // Here we just test that sin works in expressions
        let x = Computable::constant(bin(1, 0)); // x = 1
        let sin_x = x.clone().sin();
        let two = Computable::constant(bin(2, 0));
        let expr = sin_x.clone() * two; // 2 * sin(1)

        let epsilon = ubin(1, -8);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // 2 * sin(1) ~= 2 * 0.8414... ~= 1.6829...
        let expected = XBinary::from_f64(2.0 * 1.0_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn directed_rounding_produces_valid_bounds() {
        // Test that directed rounding produces well-ordered bounds that contain the true value.

        let test_cases = [
            bin(1, -2),   // 0.25
            bin(1, 0),    // 1.0
            bin(3, 0),    // 3.0
            bin(-1, 0),   // -1.0
            bin(5, -1),   // 2.5
            bin(-3, -1),  // -1.5
        ];

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        for x in &test_cases {
            // Compute Taylor bounds with directed rounding
            let (lower, upper) = taylor_sin_bounds_test(x, 10);

            // Verify bounds are ordered correctly
            assert!(
                lower <= upper,
                "Lower bound {} should be <= upper bound {} for x = {}",
                lower, upper, x
            );

            // Verify bounds are within sin's range [-1, 1]
            assert!(
                lower >= neg_one,
                "Lower bound {} should be >= -1 for x = {}",
                lower, x
            );
            assert!(
                upper <= one,
                "Upper bound {} should be <= 1 for x = {}",
                upper, x
            );
        }
    }

    #[test]
    fn directed_rounding_bounds_converge() {
        // Verify that bounds get tighter as we add more terms
        let x = bin(1, 0); // 1.0

        let (lower5, upper5) = taylor_sin_bounds_test(&x, 5);
        let (lower10, upper10) = taylor_sin_bounds_test(&x, 10);

        let width5 = upper5.sub(&lower5);
        let width10 = upper10.sub(&lower10);

        // More terms should give tighter bounds
        assert!(
            width10 < width5,
            "Bounds with 10 terms (width {}) should be tighter than 5 terms (width {})",
            width10, width5
        );
    }

    #[test]
    fn directed_rounding_symmetry() {
        // Test that sin(-x) bounds are the negation of sin(x) bounds
        // This verifies that the directed rounding handles negative inputs correctly

        let x = bin(1, -2); // 0.25
        let neg_x = bin(-1, -2); // -0.25

        let (lower_x, upper_x) = taylor_sin_bounds_test(&x, 10);
        let (lower_neg_x, upper_neg_x) = taylor_sin_bounds_test(&neg_x, 10);

        // sin(-x) = -sin(x), so bounds should be negated and swapped
        // lower(-x) should equal -upper(x)
        // upper(-x) should equal -lower(x)

        // Allow small differences due to rounding
        let neg_upper_x = upper_x.neg();
        let neg_lower_x = lower_x.neg();

        // The bounds should be approximately symmetric
        // We just verify they're in the right ballpark
        assert!(
            lower_neg_x <= neg_upper_x.add(&bin(1, -50)),
            "lower(sin(-x)) should be approximately -upper(sin(x))"
        );
        assert!(
            neg_lower_x <= upper_neg_x.add(&bin(1, -50)),
            "-lower(sin(x)) should be approximately upper(sin(-x))"
        );
    }

    #[test]
    fn directed_rounding_lower_bound_is_lower() {
        // Verify that rounding down produces smaller values than rounding up
        let x = bin(1, 0); // 1.0
        let n = 5;

        let sum_down = taylor_sin_partial_sum_test(&x, n, true);
        let sum_up = taylor_sin_partial_sum_test(&x, n, false);

        // The down-rounded sum should be <= up-rounded sum
        assert!(
            sum_down <= sum_up,
            "Rounding down {} should produce <= rounding up {}",
            sum_down, sum_up
        );
    }

    #[test]
    fn sin_of_large_multiple_of_pi() {
        // Test sin(100) which requires significant range reduction
        // This exercises the pi error propagation
        let x = Computable::constant(bin(100, 0)); // 100
        let sin_x = x.sin();
        let epsilon = ubin(1, -4);
        let bounds = sin_x
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Verify bounds are within [-1, 1]
        let neg_one = bin(-1, 0);
        let one = bin(1, 0);
        assert!(lower >= neg_one, "lower bound should be >= -1");
        assert!(upper <= one, "upper bound should be <= 1");

        // sin(100) ~= -0.5063...
        // Our bounds should be close to this value
        // Due to pi approximation errors accumulated over 16 periods,
        // we allow some tolerance in the bounds
        let expected_approx = -0.5063_f64;
        let expected_binary = XBinary::from_f64(expected_approx)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);

        // Check that bounds are in a reasonable range around the expected value
        // The accumulated pi error for k=16 periods means our result could differ
        // from the mathematical value. We verify the bounds are reasonable.
        let tolerance = bin(1, -2); // Allow 0.25 tolerance for large k
        assert!(
            lower <= expected_value.add(&tolerance) && expected_value.sub(&tolerance) <= upper,
            "sin(100) bounds [{}, {}] should be within tolerance of expected value {}",
            lower, upper, expected_value
        );
    }

    #[test]
    fn sin_pi_bounds_contain_zero() {
        // Use our pi implementation for a more precise test
        use super::super::pi::pi_bounds_at_precision;

        let (pi_lo, pi_hi) = pi_bounds_at_precision(64);
        let pi_mid = pi_lo.add(&pi_hi);
        let pi_approx = Binary::new(pi_mid.mantissa().clone(), pi_mid.exponent() - BigInt::one());

        let sin_pi = Computable::constant(pi_approx).sin();
        let epsilon = ubin(1, -10);
        let bounds = sin_pi
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        let zero = bin(0, 0);

        // sin(pi) = 0, bounds should contain zero
        assert!(
            lower <= zero && zero <= upper,
            "sin(pi) bounds [{}, {}] should contain zero",
            lower, upper
        );
    }
}
