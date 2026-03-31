//! Sine operation using Taylor series with provably correct error bounds.
//!
//! This module implements the sine function using:
//! - **Full FiniteBounds Propagation**: All pi-related errors tracked as intervals
//! - Range reduction to [-pi/2, pi/2] for efficient Taylor series convergence
//! - Critical point detection for tight bounds on intervals containing extrema
//! - Directed rounding for provably correct interval arithmetic
//!
//! ## Key Design Decision: Full FiniteBounds Propagation
//!
//! The approach propagates full intervals `[lo, hi]` through every operation:
//! - `reduce_to_pi_range(x)` returns an FiniteBounds instead of a Binary
//! - `reduce_to_half_pi_range(x)` returns `(FiniteBounds, bool)`
//! - All arithmetic uses proper interval arithmetic:
//!   - `[a,b] + [c,d] = [a+c, b+d]`
//!   - `[a,b] - [c,d] = [a-d, b-c]` (note the swap!)
//!   - When transforming via `sin(x) = sin(pi - x)`, the interval `pi - [a,b]`
//!     becomes `[pi_lo - b, pi_hi - a]`
//!
//! This ensures all pi approximation error is properly propagated to final bounds.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::UXBinary;
use crate::binary::{
    Binary, FiniteBounds, ReciprocalRounding, UBinary, XBinary, reciprocal_of_biguint,
};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::{I, U, XI};

/// Cached inputs and result for `sin_prefix`, avoiding redundant Taylor series
/// recomputation when `compute_prefix` is called multiple times with the same inputs.
pub struct SinPrefixCache {
    input_prefix: Prefix,
    pi_prefix: Prefix,
    num_terms: U,
    result: Prefix,
}

/// Sine operation with Taylor series refinement.
pub struct SinOp {
    inner: Arc<Node>,
    pi_node: Arc<Node>,
    num_terms: RwLock<U>,
    /// Cache of the last `compute_prefix` result, keyed on inputs.
    /// Eliminates redundant Taylor series recomputation during prefix propagation.
    prefix_cache: RwLock<Option<SinPrefixCache>>,
}

impl SinOp {
    pub fn new(inner: Arc<Node>, pi_node: Arc<Node>) -> Self {
        Self {
            inner,
            pi_node,
            num_terms: RwLock::new(1),
            prefix_cache: RwLock::new(None),
        }
    }
}

impl NodeOp for SinOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let input_prefix = self.inner.get_prefix()?;
        let pi_prefix = self.pi_node.get_prefix()?;
        let num_terms = *self.num_terms.read();

        // Check cache: if inputs haven't changed, return the cached result.
        {
            let cache = self.prefix_cache.read();
            if let Some(cached) = cache.as_ref()
                && cached.input_prefix == input_prefix
                && cached.pi_prefix == pi_prefix
                && cached.num_terms == num_terms
            {
                return Ok(cached.result.clone());
            }
        }

        let result = sin_prefix(&input_prefix, &pi_prefix, num_terms)?;

        // Store in cache for future calls with the same inputs.
        {
            let mut cache = self.prefix_cache.write();
            *cache = Some(SinPrefixCache {
                input_prefix,
                pi_prefix,
                num_terms,
                result: result.clone(),
            });
        }

        Ok(result)
    }

    fn refine_step(&self, target_width_exp: XI) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();

        // Leap based on coordinator's precision target.
        // XI::Finite { sign: Neg, magnitude } means width ≤ 2^(-magnitude).
        // For precision we need magnitude bits when negative, and 0 bits
        // when non-negative (coarse target).
        match target_width_exp {
            XI::Finite {
                sign: crate::sane::Sign::Neg,
                magnitude,
            } => {
                // Fine target: precision_bits = magnitude.
                let precision_bits = magnitude;
                // Each Taylor term contributes ~3.5-7 bits; precision_bits / 4 is
                // conservative yet ~25% tighter than the previous / 3 estimate.
                let needed_n = (precision_bits / 4).max(1);
                if needed_n > *num_terms {
                    *num_terms = needed_n;
                }
            }
            XI::Finite { .. } | XI::PosInf => {
                // Coarse target (e >= 0) or unbounded: no precision needed
                // from the target — skip the leap.
            }
            XI::NegInf => {
                // Exact target: no finite leap possible, fall through to
                // the incremental +1 below.
            }
        }

        // Leap to match input precision when possible (complementary: handles
        // cases where inner bounds are still wide).
        if let Ok(input_prefix) = self.inner.get_prefix()
            && let Some(width_bits) = estimate_precision_bits(&input_prefix)
        {
            let needed_n = (width_bits / 4).max(1);
            if needed_n > *num_terms {
                *num_terms = needed_n;
            }
        }

        // Always +1: keeps refiner alive, sole driver for exact inputs
        *num_terms = num_terms.saturating_add(1);
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner), Arc::clone(&self.pi_node)]
    }

    fn is_refiner(&self) -> bool {
        true
    }

    /// child 0 (input): |d(sin)/dθ| ≤ 1, so input budget = target.
    /// child 1 (pi): range reduction subtracts k ≈ |input|/(2π) copies of
    /// 2π. Each copy has width 2·pi_width, so the total pi-related error
    /// is k·2·pi_width = |input|·pi_width/π. For this to be ≤ target/2
    /// we need pi_width ≤ target·π/(2·|input|). We allocate half the
    /// error budget to pi (range reduction) and leave the other half for
    /// sin's own Taylor series and Prefix rounding. We use pi's cached
    /// lower bound as a conservative estimate of π.
    fn child_demand_budget(&self, target_width: &UXBinary, child_idx: bool) -> UXBinary {
        if !child_idx {
            // Input child: sin has derivative bounded by 1.
            return target_width.clone();
        }
        // Pi child: budget = target_width · pi_lower / (2 · max_abs(input)).
        // Half the error budget goes to pi, half to sin itself.
        let input_max_abs = match self.inner.cached_prefix() {
            Some(p) => {
                let (lo, hi) = p.abs();
                std::cmp::max(lo, hi)
            }
            None => return UXBinary::zero(),
        };
        let pi_lower = match self.pi_node.cached_prefix() {
            Some(p) => {
                let (lo, _hi) = p.abs();
                lo
            }
            None => return UXBinary::zero(),
        };
        target_width.mul(&pi_lower).div_floor(&(input_max_abs << 1))
    }

    fn budget_depends_on_bounds(&self) -> bool {
        true
    }
}

//=============================================================================
// Main sin_bounds function with full interval propagation
//=============================================================================

/// Computes sin prefix for an input interval using full interval propagation.
///
/// ## Algorithm Overview
///
/// 1. Extract pi bounds from the graph node
/// 2. Range reduce input to interval in [-pi, pi] (tracking pi error)
/// 3. Further reduce to interval in [-pi/2, pi/2] (tracking pi error)
/// 4. Detect if reduced interval straddles critical points (pi/2, -pi/2)
/// 5. Compute Taylor series on reduced interval
/// 6. Apply sign flips and clamp to [-1, 1]
fn sin_prefix(
    input_prefix: &Prefix,
    pi_prefix: &Prefix,
    num_terms: U,
) -> Result<Prefix, ComputableError> {
    let pos_one = Binary::one();
    let neg_one = pos_one.neg();

    // Extract finite bounds, or return [-1, 1] for any infinite bounds
    let lower = input_prefix.lower();
    let upper = input_prefix.upper();
    let (lower_bin, upper_bin) = match (&lower, &upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => (l.clone(), u.clone()),
        _ => {
            return Ok(Prefix::from_lower_upper(
                XBinary::Finite(neg_one),
                XBinary::Finite(pos_one),
            ));
        }
    };

    let n = num_terms.max(1);

    // When the input is zero, range reduction subtracts 0 copies of 2π,
    // so π's precision is irrelevant. Compute sin directly via Taylor series.
    let (input_abs_lo, input_abs_hi) = input_prefix.abs();
    let input_max_abs = std::cmp::max(input_abs_lo, input_abs_hi);

    if input_max_abs.is_zero() {
        let input_interval = FiniteBounds::new(lower_bin, upper_bin);
        let sin_result = compute_sin_on_monotonic_interval(&input_interval, n);
        let clamped_lo = std::cmp::max(sin_result.lo().clone(), neg_one);
        let clamped_hi = std::cmp::min(sin_result.hi().clone(), pos_one);
        return Ok(Prefix::from_lower_upper(
            XBinary::Finite(clamped_lo),
            XBinary::Finite(clamped_hi),
        ));
    }

    // Pi's precision affects the result — extract finite bounds or bail.
    let pi_lo = pi_prefix.lower();
    let pi_hi_xb = pi_prefix.upper();
    let pi_interval = match (&pi_lo, &pi_hi_xb) {
        (XBinary::Finite(lo), XBinary::Finite(hi)) => FiniteBounds::new(lo.clone(), hi.clone()),
        _ => {
            return Ok(Prefix::from_lower_upper(
                XBinary::Finite(neg_one),
                XBinary::Finite(pos_one),
            ));
        }
    };

    // Derive two_pi and half_pi from pi by shifting exponents (avoiding BigInt multiply).
    // Scaling by 2 shifts exponent +1; scaling by 1/2 shifts exponent -1.
    // Using from_lower_and_width avoids the abs_distance re-computation in new().
    let pi_width = pi_interval.width();
    let pi_lo_exp = pi_interval.lo().exponent();
    let two_pi_lo = Binary::new_normalized(
        pi_interval.lo().mantissa().clone(),
        crate::sane_i_arithmetic!(pi_lo_exp; pi_lo_exp + 1),
    );
    let pi_w_exp = pi_width.exponent();
    let two_pi_width = UBinary::new_normalized(
        pi_width.mantissa().clone(),
        crate::sane_i_arithmetic!(pi_w_exp; pi_w_exp + 1),
    );
    let two_pi_interval = FiniteBounds::from_lower_and_width(two_pi_lo, two_pi_width);
    let half_pi_lo = Binary::new_normalized(
        pi_interval.lo().mantissa().clone(),
        crate::sane_i_arithmetic!(pi_lo_exp; pi_lo_exp - 1),
    );
    let half_pi_width = UBinary::new_normalized(
        pi_width.mantissa().clone(),
        crate::sane_i_arithmetic!(pi_w_exp; pi_w_exp - 1),
    );
    let half_pi_interval = FiniteBounds::from_lower_and_width(half_pi_lo, half_pi_width);

    // Process each endpoint through range reduction with full interval tracking
    let input_interval = FiniteBounds::new(lower_bin.clone(), upper_bin.clone());

    // Check if the input interval is wide enough to contain a full period.
    // Use the lower bound of 2π for the comparison: if the input width exceeds even
    // the smallest possible 2π, it definitely spans a full period. This is the
    // conservative direction — we may return [-1, 1] slightly too eagerly (when the
    // width is between two_pi_lo and true 2π), but we never miss a case where
    // the width truly exceeds 2π. Over-approximation is always sound; under-approximation
    // would be a correctness bug.
    let input_width = upper_bin.sub(&lower_bin);
    if input_width >= *two_pi_interval.lo() {
        // Input spans at least one full period, sin ranges over all of [-1, 1]
        return Ok(Prefix::from_lower_upper(
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
        n,
    );

    // Compute sin bounds based on the reduction result
    let (result_lo, result_hi) = match reduced_result {
        ReductionResult::InRange {
            interval,
            sign_flip,
        } => {
            // FiniteBounds is fully within [-pi/2, pi/2], use Taylor series
            let sin_bounds = compute_sin_on_monotonic_interval(&interval, n);
            if sign_flip {
                (sin_bounds.hi().neg(), sin_bounds.lo().neg())
            } else {
                (sin_bounds.lo().clone(), sin_bounds.hi().clone())
            }
        }
        ReductionResult::ContainsMax { sin_min } => {
            // FiniteBounds contains pi/2 where sin = 1
            (sin_min, pos_one.clone())
        }
        ReductionResult::ContainsMin { sin_max } => {
            // FiniteBounds contains -pi/2 where sin = -1
            (neg_one.clone(), sin_max)
        }
        ReductionResult::ContainsBoth => {
            // FiniteBounds contains both critical points
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

    // Normalize to prefix form to prevent precision accumulation
    Ok(Prefix::from_lower_upper(
        XBinary::Finite(clamped_lo),
        XBinary::Finite(clamped_hi),
    ))
}

//=============================================================================
// Range reduction with full interval propagation
//=============================================================================

/// Result of range reduction to [-pi/2, pi/2].
#[derive(Debug)]
enum ReductionResult {
    /// FiniteBounds is fully in [-pi/2, pi/2], can use Taylor directly
    InRange {
        interval: FiniteBounds,
        sign_flip: bool,
    },
    /// FiniteBounds contains pi/2 (sin maximum), provides minimum sin value
    ContainsMax { sin_min: Binary },
    /// FiniteBounds contains -pi/2 (sin minimum), provides maximum sin value
    ContainsMin { sin_max: Binary },
    /// FiniteBounds contains both critical points
    ContainsBoth,
    /// FiniteBounds spans multiple branches after reduction
    SpansMultipleBranches {
        overall_lo: Binary,
        overall_hi: Binary,
    },
}

/// Reduces an input interval to approximately [-pi, pi] using interval arithmetic.
///
/// Returns the reduced interval accounting for pi approximation error.
///
/// The approach computes k = round(midpoint / 2π_mid) analytically in a single step,
/// then subtracts k * [2π_lo, 2π_hi] using full interval arithmetic. This handles
/// arbitrarily large inputs correctly because k is computed as a BigInt with no
/// magnitude limitation. After the initial reduction, at most 2 fixup iterations
/// handle rounding edge cases (where the midpoint-derived k was off by 1).
///
/// Soundness argument: the subtraction `current - k * two_pi` uses full interval
/// arithmetic, so the result correctly contains all possible true values regardless
/// of which k we pick. The choice of k only affects how close we land to [-π, π] —
/// not whether the result is a valid enclosure.
fn reduce_to_pi_range_interval(
    input: &FiniteBounds,
    two_pi: &FiniteBounds,
    pi: &FiniteBounds,
) -> FiniteBounds {
    // Pre-compute pi.hi() once — used for both the range check and its negation.
    let pi_hi = pi.hi();
    let neg_pi_hi = pi_hi.neg();

    // Quick check: if input is already in range, avoid all computation.
    let input_hi = input.hi();
    if *input.lo() >= neg_pi_hi && *input_hi <= *pi_hi {
        return input.clone();
    }

    let two_pi_mid = two_pi.midpoint();

    // Compute k analytically: k = round(midpoint(input) / midpoint(2π)).
    // This is a single BigInt computation that works for arbitrarily large inputs.
    // Using midpoints here is fine because k is just an integer we choose — the
    // soundness comes from the interval subtraction, not from k being exact.
    let input_mid = input.midpoint();
    let k = compute_reduction_factor(&input_mid, &two_pi_mid);

    let mut current = if k.is_zero() {
        input.clone()
    } else {
        let k_times_two_pi = two_pi.scale_bigint(&k);
        input.interval_sub(&k_times_two_pi)
    };

    // After the analytical reduction, we should be close to [-π, π].
    // At most 2 fixup iterations handle the case where k was off by 1
    // (which can happen when the midpoint is near a multiple of 2π).
    for _ in 0_u32..2_u32 {
        // Check if we're within the outer bounds [-pi_hi, pi_hi].
        // This is acceptable because downstream code (reduce_to_half_pi_range_interval)
        // uses conservative interval comparisons with pi/half_pi that properly account
        // for the uncertainty in the pi approximation. The reduced value doesn't need
        // to be exactly in [-π, π]; it just needs to be close enough that the interval
        // comparisons can correctly determine which trigonometric identity to apply.
        let current_hi = current.hi();
        if *current.lo() >= neg_pi_hi && *current_hi <= *pi_hi {
            return current;
        }

        // Compute a small fixup k (should be -1, 0, or 1)
        let fixup_mid = current.midpoint();
        let fixup_k = compute_reduction_factor(&fixup_mid, &two_pi_mid);

        if fixup_k.is_zero() {
            // Can't reduce further — we're as close as we can get
            break;
        }

        let fixup_shift = two_pi.scale_bigint(&fixup_k);
        current = current.interval_sub(&fixup_shift);
    }

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
    input: &FiniteBounds,
    two_pi: &FiniteBounds,
    pi: &FiniteBounds,
    half_pi: &FiniteBounds,
    n: U,
) -> ReductionResult {
    // First reduce to [-pi, pi]
    let reduced = reduce_to_pi_range_interval(input, two_pi, pi);

    // Pre-compute hi() values that are used multiple times below.
    // Each hi() call computes lower + width, so caching these avoids
    // redundant BigInt additions.
    let reduced_hi = reduced.hi();
    let half_pi_hi = half_pi.hi();
    let pi_hi = pi.hi();
    // For neg_half_pi = -[half_pi_lo, half_pi_hi] = [-half_pi_hi, -half_pi_lo]:
    //   neg_half_pi.lo() = -half_pi_hi, neg_half_pi.hi() = -half_pi_lo
    let neg_half_pi_hi = half_pi.lo().neg(); // = -half_pi_lo
    let neg_half_pi_lo = half_pi_hi.neg(); // = -half_pi_hi
    // For neg_pi = -[pi_lo, pi_hi] = [-pi_hi, -pi_lo]:
    //   neg_pi.hi() = -pi_lo
    let neg_pi_hi = pi.lo().neg(); // = -pi_lo

    // Key comparisons using conservative bounds:
    // To check if interval is entirely in [-pi/2, pi/2]:
    //   reduced.hi <= half_pi.lo (interval entirely below pi/2)
    //   AND reduced.lo >= neg_half_pi.hi (interval entirely above -pi/2)

    // Case 1: Entirely in [-pi/2, pi/2]
    if *reduced_hi <= *half_pi.lo() && *reduced.lo() >= neg_half_pi_hi {
        return ReductionResult::InRange {
            interval: reduced,
            sign_flip: false,
        };
    }

    // Case 2: Entirely in [pi/2, pi]
    // reduced.lo >= half_pi.lo AND reduced.hi <= pi.hi
    if *reduced.lo() >= *half_pi.lo() && *reduced_hi <= *pi_hi {
        // Transform: x -> pi - x
        // sin(x) = sin(pi - x) for x in [pi/2, pi]
        // pi - [a, b] using interval arithmetic:
        // [pi_lo, pi_hi] - [a, b] = [pi_lo - b, pi_hi - a]
        let transformed = pi.interval_sub(&reduced);
        return ReductionResult::InRange {
            interval: transformed,
            sign_flip: false,
        };
    }

    // Case 3: Entirely in [-pi, -pi/2]
    // reduced.hi <= neg_half_pi.hi (which is -pi/2_lo, the least negative)
    // AND reduced.lo >= neg_pi.hi (which is -pi_lo, the least negative -pi)
    if *reduced_hi <= neg_half_pi_hi && *reduced.lo() >= neg_pi_hi {
        // Transform: x -> -pi - x, then negate result
        // sin(x) = -sin(-pi - x) for x in [-pi, -pi/2]
        // Actually: sin(x) = sin(-pi - x) = -sin(pi + x)
        // Simpler: sin(x) for x in [-pi, -pi/2] can use sin(x) = -sin(-x - pi)
        // Or: sin(x) = sin(pi + x) for x in [-pi, -pi/2] gives us angle in [0, pi/2]
        // Let's use: new_x = pi + x, then sin(x) = -sin(new_x)
        // [pi_lo, pi_hi] + [a, b] = [pi_lo + a, pi_hi + b]
        let transformed = pi.interval_add(&reduced);
        return ReductionResult::InRange {
            interval: transformed,
            sign_flip: true,
        };
    }

    // Determine which critical points the interval might straddle.
    // The "both" check must come before the individual cases so that an interval
    // straddling BOTH ±pi/2 returns ContainsBoth rather than just ContainsMax.
    let spans_max = *reduced.lo() < *half_pi_hi && *reduced_hi > *half_pi.lo();
    let spans_min = *reduced.lo() < neg_half_pi_hi && *reduced_hi > neg_half_pi_lo;

    // Case 4: Straddles both pi/2 and -pi/2
    if spans_max && spans_min {
        return ReductionResult::ContainsBoth;
    }

    // Case 5: Straddles pi/2 only (contains maximum)
    if spans_max {
        // The interval contains pi/2 where sin = 1
        // Compute the minimum sin value at the endpoints
        let sin_bounds_at_lo = compute_sin_bounds_for_point_with_pi(reduced.lo(), n, pi, half_pi);
        let sin_bounds_at_hi = compute_sin_bounds_for_point_with_pi(reduced_hi, n, pi, half_pi);
        let sin_min = if sin_bounds_at_lo.lo() < sin_bounds_at_hi.lo() {
            sin_bounds_at_lo.lo().clone()
        } else {
            sin_bounds_at_hi.lo().clone()
        };

        return ReductionResult::ContainsMax { sin_min };
    }

    // Case 6: Straddles -pi/2 only (contains minimum)
    if spans_min {
        // The interval contains -pi/2 where sin = -1
        let sin_bounds_at_lo = compute_sin_bounds_for_point_with_pi(reduced.lo(), n, pi, half_pi);
        let sin_bounds_at_hi = compute_sin_bounds_for_point_with_pi(reduced_hi, n, pi, half_pi);
        // Cache hi() to avoid cloning twice per FiniteBounds.
        let hi_at_lo = sin_bounds_at_lo.hi();
        let hi_at_hi = sin_bounds_at_hi.hi();
        let sin_max = if hi_at_lo > hi_at_hi {
            hi_at_lo.clone()
        } else {
            hi_at_hi.clone()
        };

        return ReductionResult::ContainsMin { sin_max };
    }

    // Case 7: Spans multiple branches but doesn't straddle either critical point.
    // This happens when the interval crosses a branch boundary (e.g., between
    // center and upper regions) without containing pi/2 or -pi/2.
    // Compute conservative bounds at both endpoints and take their union.
    let pos_one = Binary::one();
    let neg_one = pos_one.neg();

    let sin_bounds_1 = compute_sin_bounds_for_point_with_pi(reduced.lo(), n, pi, half_pi);
    let sin_bounds_2 = compute_sin_bounds_for_point_with_pi(reduced_hi, n, pi, half_pi);
    let combined = sin_bounds_1.join(&sin_bounds_2);

    ReductionResult::SpansMultipleBranches {
        overall_lo: combined.lo().clone().max(neg_one),
        overall_hi: combined.hi().clone().min(pos_one),
    }
}

/// Computes sin bounds for a point using interval-based pi for provably correct bounds.
///
/// This function uses the same structure as the original but with interval-based pi
/// for proper error tracking. The point x is assumed to already be reduced to
/// approximately [-pi, pi].
///
/// When x falls within the uncertainty interval of a branch boundary (half_pi or -half_pi),
/// both possible branches are evaluated and the union of their bounds is returned.
fn compute_sin_bounds_for_point_with_pi(
    x: &Binary,
    n: U,
    pi: &FiniteBounds,
    half_pi: &FiniteBounds,
) -> FiniteBounds {
    // Pre-compute hi() and derived negation values to avoid redundant BigInt ops.
    // neg_half_pi = -[half_pi_lo, half_pi_hi] = [-half_pi_hi, -half_pi_lo]
    let half_pi_hi = half_pi.hi();
    let neg_half_pi_hi = half_pi.lo().neg(); // = -half_pi_lo
    let neg_half_pi_lo = half_pi_hi.neg(); // = -half_pi_hi

    // For the transformation, we use the full interval to get proper bounds
    let x_interval = FiniteBounds::point(x.clone());

    // Determine which branch(es) x could be in using interval bounds.
    // We use conservative comparisons:
    // - x is definitively above half_pi if x > half_pi.hi()
    // - x is definitively below -half_pi if x < neg_half_pi.lo() (i.e., x < -half_pi.hi())
    // - Otherwise, x might be in a boundary region where we need to consider multiple branches

    let definitely_above_half_pi = x > half_pi_hi;
    let definitely_below_neg_half_pi = *x < neg_half_pi_lo;
    let definitely_in_center = *x >= neg_half_pi_hi && x <= half_pi.lo();

    if definitely_in_center {
        // x is definitively in [-pi/2, pi/2], use directly
        compute_sin_on_monotonic_interval(&x_interval, n)
    } else if definitely_above_half_pi {
        // x is definitively in (pi/2, pi], transform: sin(x) = sin(pi - x)
        let reduced_interval = pi.interval_sub(&x_interval);
        compute_sin_on_monotonic_interval(&reduced_interval, n)
    } else if definitely_below_neg_half_pi {
        // x is definitively in [-pi, -pi/2), transform: sin(x) = -sin(pi + x)
        let reduced_interval = pi.interval_add(&x_interval);
        let sin_bounds = compute_sin_on_monotonic_interval(&reduced_interval, n);
        FiniteBounds::new(sin_bounds.hi().neg(), sin_bounds.lo().neg())
    } else if x >= half_pi.lo() {
        // x is in the boundary region around half_pi: [half_pi.lo, half_pi.hi]
        // Need to consider both the center branch and the upper branch

        // Center branch: use x directly
        let center_bounds = compute_sin_on_monotonic_interval(&x_interval, n);

        // Upper branch: transform sin(x) = sin(pi - x)
        let upper_reduced = pi.interval_sub(&x_interval);
        let upper_bounds = compute_sin_on_monotonic_interval(&upper_reduced, n);

        // Take the union of bounds from both branches
        center_bounds.join(&upper_bounds)
    } else {
        // x is in the boundary region around -half_pi: [neg_half_pi.lo, neg_half_pi.hi]
        // i.e., x is in [-half_pi.hi, -half_pi.lo]
        // Need to consider both the center branch and the lower branch

        // Center branch: use x directly
        let center_bounds = compute_sin_on_monotonic_interval(&x_interval, n);

        // Lower branch: transform sin(x) = -sin(pi + x)
        let lower_reduced = pi.interval_add(&x_interval);
        let lower_sin_bounds = compute_sin_on_monotonic_interval(&lower_reduced, n);
        let lower_bounds =
            FiniteBounds::new(lower_sin_bounds.hi().neg(), lower_sin_bounds.lo().neg());

        // Take the union of bounds from both branches
        center_bounds.join(&lower_bounds)
    }
}

/// Computes k = round(x / period).
fn compute_reduction_factor(x: &Binary, period: &Binary) -> BigInt {
    let mx = x.mantissa();
    let ex = x.exponent();
    let mp = period.mantissa();
    let ep = period.exponent();

    // We need to compute: k = (mx * 2^ex) / (mp * 2^ep) = (mx / mp) * 2^(ex - ep)
    //
    // To avoid losing precision, we shift mx up by enough bits so that
    // mx << shift_bits > mp, ensuring a non-zero quotient.
    // The shift should be at least mp.bits() - mx.bits() + some_precision.
    let mx_bits = crate::sane::bits_as_u(mx.magnitude().bits());
    let mp_bits = crate::sane::bits_as_u(mp.magnitude().bits());

    // Shift by enough bits to get a meaningful quotient, plus extra precision for rounding.
    // When mp_bits > mx_bits, we need at least the difference plus 64 extra bits.
    // When mx_bits >= mp_bits, 64 bits suffice.
    let precision_bits = if mp_bits >= mx_bits {
        crate::sane_arithmetic!(mp_bits, mx_bits; mp_bits - mx_bits + 64)
    } else {
        64
    };

    let shifted_mx = mx << crate::sane::u_as_usize(precision_bits);
    let quotient = &shifted_mx / mp;
    let precision_bits_i = I::try_from(precision_bits).unwrap_or_else(|_| {
        crate::detected_computable_would_exhaust_memory!(
            "precision_bits exceeds I in compute_reduction_factor"
        )
    });
    let result_exp = ex
        .checked_sub(ep)
        .and_then(|v| v.checked_sub(precision_bits_i))
        .unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!(
                "exponent overflow in compute_reduction_factor"
            )
        });

    if result_exp >= 0 {
        let shift = usize::try_from(result_exp).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!(
                "shift exceeds usize in compute_reduction_factor"
            )
        });
        &quotient << shift
    } else {
        let magnitude = result_exp.unsigned_abs();
        let quotient_bits = quotient.bits();
        if u64::from(magnitude) > quotient_bits {
            if quotient.is_negative() {
                return BigInt::from(-1_i32);
            } else {
                return BigInt::zero();
            }
        }
        let shift_val = magnitude;
        match std::num::NonZeroU32::new(shift_val) {
            None => quotient.clone(),
            Some(shift) => {
                let half = BigInt::one() << crate::sane::u_as_usize(crate::sane::sub_one(shift));
                let rounded = if quotient.is_negative() {
                    &quotient - &half
                } else {
                    &quotient + &half
                };
                rounded >> crate::sane::u_as_usize(shift.get())
            }
        }
    }
}

/// Truncates a Binary to at most `precision_bits` of mantissa, rounding in the
/// specified direction.
///
/// This uses directed rounding to ensure soundness in interval arithmetic:
/// - `RoundingDirection::Down` (floor): rounds toward -infinity
/// - `RoundingDirection::Up` (ceil): rounds toward +infinity
///
/// The implementation works on the mantissa magnitude to avoid any ambiguity
/// in how BigInt right-shift handles negative values.
fn truncate_precision_directed(x: &Binary, precision_bits: U, dir: RoundingDirection) -> Binary {
    let mantissa = x.mantissa();
    let exponent = x.exponent();
    let bit_length = crate::sane::bits_as_u(mantissa.magnitude().bits());

    let Some(shift_nz) = bit_length
        .checked_sub(precision_bits)
        .and_then(std::num::NonZeroU32::new)
    else {
        return x.clone();
    };
    let shift = crate::sane::u_as_usize(shift_nz.get());

    // Shift the magnitude (always truncates toward zero) and detect remainder
    let abs_shifted = mantissa.magnitude() >> shift;
    let has_remainder = (&abs_shifted << shift) != *mantissa.magnitude();
    let is_negative = mantissa.is_negative();

    // Floor (toward -inf): positive toward zero, negative away from zero
    // Ceil  (toward +inf): positive away from zero, negative toward zero
    // "Away from zero" means incrementing the magnitude when there's a remainder.
    let round_away_from_zero = has_remainder
        && matches!(
            (dir, is_negative),
            (RoundingDirection::Down, true) | (RoundingDirection::Up, false)
        );

    let abs_result = if round_away_from_zero {
        abs_shifted + 1u32
    } else {
        abs_shifted
    };

    let signed_result = if is_negative {
        -BigInt::from(abs_result)
    } else {
        BigInt::from(abs_result)
    };

    let shift_i = crate::sane::usize_as_i(shift);
    let exp = exponent;
    Binary::new(
        signed_result,
        crate::sane_i_arithmetic!(exp, shift_i; exp + shift_i),
    )
}

//=============================================================================
// Taylor series computation for intervals
//=============================================================================

/// Computes sin bounds for an interval known to be in [-pi/2, pi/2].
///
/// Since sin is monotonically increasing on [-pi/2, pi/2], we can simply
/// evaluate at the endpoints.
fn compute_sin_on_monotonic_interval(interval: &FiniteBounds, n: U) -> FiniteBounds {
    // sin is monotonic increasing on [-pi/2, pi/2]
    // So: sin([a, b]) = [sin(a)_lo, sin(b)_hi]
    //
    // Round lo DOWN and hi UP so the truncated interval contains the original,
    // preserving soundness: sin(truncated_lo) <= sin(lo) and sin(truncated_hi) >= sin(hi).
    //
    // The Taylor series with n terms yields roughly n*10 bits of accuracy
    // (conservative estimate for |x| ≤ π/2). Intermediate precision must match.
    let target_precision = crate::sane_arithmetic!(n; n * 10);
    let truncated_lo =
        truncate_precision_directed(interval.lo(), target_precision, RoundingDirection::Down);
    let truncated_hi =
        truncate_precision_directed(interval.hi(), target_precision, RoundingDirection::Up);

    let (sin_lo_bounds_lo, _) = taylor_sin_bounds(&truncated_lo, n, target_precision);
    let (_, sin_hi_bounds_hi) = taylor_sin_bounds(&truncated_hi, n, target_precision);

    FiniteBounds::new(sin_lo_bounds_lo, sin_hi_bounds_hi)
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
/// For small n (≤ 16), uses per-term division which has lower overhead.
/// For larger n, accumulates the exact rational sum as a single fraction P/Q
/// (where Q is the common factorial denominator), then performs only two BigInt
/// divisions at the end (one for floor, one for ceil), eliminating O(n)
/// expensive reciprocal computations.
fn taylor_sin_bounds(x: &Binary, n: U, target_precision: U) -> (Binary, Binary) {
    if n == 0 {
        // No terms: error bound is |x|^1 / 1! = |x|
        let abs_x = x.magnitude().to_binary();
        let error = divide_by_factorial_directed(
            &abs_x,
            &BigInt::one(),
            RoundingDirection::Up,
            target_precision,
        );
        return (error.neg(), error);
    }

    // For small n, per-term division has less overhead than rational accumulation.
    if n <= 16 {
        return taylor_sin_bounds_per_term(x, n, target_precision);
    }

    taylor_sin_bounds_rational(x, n, target_precision)
}

/// Per-term Taylor series computation. Each term is divided by its factorial
/// independently using directed rounding. Efficient for small n due to low
/// overhead, but performs O(n) expensive reciprocal computations.
fn taylor_sin_bounds_per_term(x: &Binary, n: U, target_precision: U) -> (Binary, Binary) {
    let x_squared = x.mul(x);
    let mut sum_lo = Binary::zero();
    let mut sum_hi = Binary::zero();
    let mut power = x.clone(); // x^1
    let mut factorial = BigInt::one(); // 1!

    // First term (k=0): +x / 1!
    let term_lo = divide_by_factorial_directed(
        &power,
        &factorial,
        RoundingDirection::Down,
        target_precision,
    );
    let term_hi =
        divide_by_factorial_directed(&power, &factorial, RoundingDirection::Up, target_precision);
    sum_lo = sum_lo.add(&term_lo);
    sum_hi = sum_hi.add(&term_hi);

    for k in 1..n {
        // Advance state: power *= x^2, factorial *= (2k)(2k+1)
        power = power.mul(&x_squared);
        let k_big = BigInt::from(k);
        factorial *= &k_big * 2_i64 * (&k_big * 2_i64 + 1_i64);

        // Term k: (-1)^k * x^(2k+1) / (2k+1)!
        let term_num = if k % 2 == 0 {
            power.clone()
        } else {
            power.neg()
        };
        let t_lo = divide_by_factorial_directed(
            &term_num,
            &factorial,
            RoundingDirection::Down,
            target_precision,
        );
        let t_hi = divide_by_factorial_directed(
            &term_num,
            &factorial,
            RoundingDirection::Up,
            target_precision,
        );
        sum_lo = sum_lo.add(&t_lo);
        sum_hi = sum_hi.add(&t_hi);
    }

    // Derive error bound from the loop's final power/factorial state.
    // After the loop, power = x^(2(n-1)+1) = x^(2n-1) and factorial = (2n-1)!.
    // The error term needs |x|^(2n+1) / (2n+1)!, so advance one more step.
    power = power.mul(&x_squared); // x^(2n+1)
    let n_big = BigInt::from(n);
    factorial *= &n_big * 2_i64 * (&n_big * 2_i64 + 1_i64); // (2n+1)!

    let error_power = Binary::new(
        BigInt::from(power.mantissa().magnitude().clone()),
        power.exponent(),
    );
    let error = divide_by_factorial_directed(
        &error_power,
        &factorial,
        RoundingDirection::Up,
        target_precision,
    );

    (sum_lo.sub(&error), sum_hi.add(&error))
}

/// Rational accumulation Taylor series computation. Accumulates the exact sum
/// as a single fraction P/Q, performing only two BigInt divisions at the end.
///
/// The mantissa powers x^(2k+1) have varying binary exponents e*(2k+1). To sum
/// them as integers over a common denominator, all terms are aligned to a common
/// exponent by shifting mantissas appropriately.
fn taylor_sin_bounds_rational(x: &Binary, n: U, target_precision: U) -> (Binary, Binary) {
    let m = x.mantissa();
    let e = i64::from(x.exponent()); // widen I to i64 for exponent arithmetic

    // Precompute m^2 for the recurrence power_{k} = power_{k-1} * m^2
    let m_sq = m * m;

    // Compute remaining factorial ratios: R_k = (2n-1)! / (2k+1)!
    // R_{n-1} = 1, R_k = R_{k+1} * (2k+2) * (2k+3)
    let n_usize = crate::sane::u_as_usize(n);
    let mut remaining_factors: Vec<BigInt> = Vec::with_capacity(n_usize);
    remaining_factors.resize(n_usize, BigInt::zero());
    let n_minus_1 = crate::sane_arithmetic!(n; n - 1);
    remaining_factors[crate::sane::u_as_usize(n_minus_1)] = BigInt::one();
    for k in (0..n_minus_1).rev() {
        let next = crate::sane::u_as_usize(crate::sane_arithmetic!(k; k + 1));
        let factor_a = crate::sane_arithmetic!(k; k * 2 + 2);
        let factor_b = crate::sane_arithmetic!(k; k * 2 + 3);
        let k_usize = crate::sane::u_as_usize(k);
        remaining_factors[k_usize] = &remaining_factors[next] * factor_a * factor_b;
    }

    // The common denominator Q = (2n-1)! = R_0
    let common_factorial = remaining_factors[0].clone();

    // Compute the common exponent for alignment.
    // Term k has exponent e*(2k+1). We align all to the minimum exponent.
    let n_i64 = i64::from(n);

    let common_exp: i64 = if e < 0 {
        // Last term (k=n-1) has smallest exponent: e*(2n-1)
        let two_n_minus_1 = crate::sane_i64_arithmetic!(n_i64; n_i64 * 2 - 1);
        crate::sane_i64_arithmetic!(e, two_n_minus_1; e * two_n_minus_1)
    } else {
        e // For e >= 0, minimum exponent is e (first term)
    };

    // Shift amounts are always non-negative: for e < 0, shifts decrease from
    // 2|e|*(n-1) to 0; for e > 0, shifts increase from 0 to 2e*(n-1).
    // For e == 0: no shifting (abs_two_e = 0).
    let abs_two_e: u64 = e.unsigned_abs().checked_mul(2).unwrap_or_else(|| {
        crate::detected_computable_would_exhaust_memory!("2|e| overflow in taylor_sin")
    });
    let shift_decreasing = e < 0;

    let mut current_shift: u64 = if shift_decreasing {
        abs_two_e
            .checked_mul(u64::from(n_minus_1))
            .unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "initial shift overflow in taylor_sin"
                )
            })
    } else {
        0
    };

    // Accumulate: P = sum_k { (-1)^k * m^(2k+1) * R_k * 2^(shift_k) }
    let mut numerator = BigInt::zero();
    let mut power_m = m.clone(); // m^(2k+1), starts at m^1

    for (k, remaining_factor) in remaining_factors.iter().enumerate() {
        let mut contribution = if k % 2 == 0 {
            &power_m * remaining_factor
        } else {
            -(&power_m * remaining_factor)
        };

        if current_shift > 0 {
            let shift = usize::try_from(current_shift).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "shift exceeds usize in taylor_sin"
                )
            });
            contribution <<= shift;
        }

        numerator += contribution;

        if k < crate::sane::u_as_usize(n_minus_1) {
            power_m *= &m_sq;
            if shift_decreasing {
                current_shift = current_shift.checked_sub(abs_two_e).unwrap_or_else(|| {
                    crate::detected_computable_would_exhaust_memory!(
                        "shift underflow in taylor_sin"
                    )
                });
            } else {
                current_shift = current_shift.checked_add(abs_two_e).unwrap_or_else(|| {
                    crate::detected_computable_would_exhaust_memory!("shift overflow in taylor_sin")
                });
            }
        }
    }

    // sum = numerator * 2^common_exp / common_factorial
    let (sum_lo, sum_hi) =
        divide_bigint_directed(&numerator, common_exp, &common_factorial, target_precision);

    // Error bound: |x|^(2n+1) / (2n+1)!
    power_m *= &m_sq; // advance to m^(2n+1)
    let two_n = crate::sane_i64_arithmetic!(n_i64; n_i64 * 2);
    let two_n_plus_1 = crate::sane_i64_arithmetic!(two_n; two_n + 1);
    let error_exp = crate::sane_i64_arithmetic!(e, two_n_plus_1; e * two_n_plus_1);

    let error_factorial = &common_factorial * two_n * two_n_plus_1;

    let abs_power_m = BigInt::from(power_m.magnitude().clone());
    let (_, error) =
        divide_bigint_directed(&abs_power_m, error_exp, &error_factorial, target_precision);

    (sum_lo.sub(&error), sum_hi.add(&error))
}

/// Computes `numerator * 2^exponent / denominator` with directed rounding,
/// returning both (floor, ceil) as Binary values.
///
/// This performs a single BigInt division with enough extra precision bits to
/// produce `target_precision` significant bits in the quotient. Unlike
/// `divide_by_factorial_directed` (which computes a reciprocal then multiplies),
/// this function handles the case where numerator and denominator are of comparable
/// magnitude without losing precision.
fn divide_bigint_directed(
    numerator: &BigInt,
    exponent: i64,
    denominator: &BigInt,
    target_precision: U,
) -> (Binary, Binary) {
    use num_integer::Integer;

    if denominator.is_zero() || numerator.is_zero() {
        return (Binary::zero(), Binary::zero());
    }

    // We want to compute numerator * 2^exponent / denominator with target_precision
    // significant bits. To do this as integer arithmetic, shift the numerator left
    // by enough bits so the integer quotient has the desired precision.
    //
    // The quotient |numerator| / |denominator| has roughly
    // (bits(numerator) - bits(denominator)) bits. We need target_precision bits in
    // the result, so shift by: target_precision + bits(denominator) - bits(numerator) + 1
    // (the +1 ensures we don't lose a bit from rounding).
    let num_bits = crate::sane::bits_as_u(numerator.magnitude().bits());
    let den_bits = crate::sane::bits_as_u(denominator.magnitude().bits());
    // We need target_precision significant bits in the quotient.
    // extra_shift = target_precision + max(0, den_bits - num_bits) + 1
    let deficit = den_bits.saturating_sub(num_bits);
    let extra_shift = crate::sane_arithmetic!(target_precision, deficit;
        target_precision + deficit + 1);

    let is_negative = numerator.is_negative();
    let abs_num = numerator.magnitude();
    let abs_den = denominator.magnitude();

    // Shift numerator left by extra_shift bits
    let shifted_num = abs_num << extra_shift;

    // Integer division: quotient = floor(shifted_num / abs_den)
    let (quotient, remainder) = shifted_num.div_rem(abs_den);
    let has_remainder = !remainder.is_zero();

    // The exact value is:
    //   sign * quotient * 2^(exponent - extra_shift) / 1
    //   (with possible +1 to quotient for ceiling)
    //
    // For floor (toward -inf):
    //   positive: use quotient as-is (truncation = floor for positive)
    //   negative: if has_remainder, add 1 to quotient (round away from zero = toward -inf)
    // For ceil (toward +inf):
    //   positive: if has_remainder, add 1 to quotient (round away from zero = toward +inf)
    //   negative: use quotient as-is (truncation = ceil for negative)
    let extra_shift_i64 = i64::from(extra_shift);
    let result_exp = crate::sane_i64_arithmetic!(exponent, extra_shift_i64; exponent - extra_shift_i64);

    let floor_quotient = if is_negative && has_remainder {
        &quotient + 1u32
    } else {
        quotient.clone()
    };
    let ceil_quotient = if !is_negative && has_remainder {
        &quotient + 1u32
    } else {
        quotient
    };

    let floor_mantissa = if is_negative {
        -BigInt::from(floor_quotient)
    } else {
        BigInt::from(floor_quotient)
    };
    let ceil_mantissa = if is_negative {
        -BigInt::from(ceil_quotient)
    } else {
        BigInt::from(ceil_quotient)
    };

    let result_exp_i = I::try_from(result_exp).unwrap_or_else(|_| {
        crate::detected_computable_would_exhaust_memory!(
            "result exponent exceeds I in divide_bigint_directed"
        )
    });

    (
        Binary::new(floor_mantissa, result_exp_i),
        Binary::new(ceil_mantissa, result_exp_i),
    )
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
    target_precision: U,
) -> Binary {
    if factorial.is_zero() {
        return value.clone();
    }

    // Use the caller-provided target precision, ensuring it is at least as large
    // as the input value's mantissa to avoid losing information.
    let value_bits = crate::sane::bits_as_u(value.mantissa().magnitude().bits());
    let precision_bits = target_precision.max(value_bits);

    // Determine rounding direction for reciprocal based on overall rounding and sign of value.
    // For directed rounding toward +/- infinity:
    // - Round UP (+inf): positive values need reciprocal rounded up, negative need it rounded down
    // - Round DOWN (-inf): positive values need reciprocal rounded down, negative need it rounded up
    let is_negative = value.mantissa().is_negative();
    let recip_rounding = match (rounding, is_negative) {
        (RoundingDirection::Up, false) => ReciprocalRounding::Ceil,
        (RoundingDirection::Up, true) => ReciprocalRounding::Floor,
        (RoundingDirection::Down, false) => ReciprocalRounding::Floor,
        (RoundingDirection::Down, true) => ReciprocalRounding::Ceil,
    };

    // Compute 1/|factorial| with directed rounding
    let reciprocal = reciprocal_of_biguint(factorial.magnitude(), precision_bits, recip_rounding);

    // Multiply value by reciprocal
    value.mul(&reciprocal)
}

/// Estimates the number of precision bits in a `Prefix` interval.
///
/// Returns `Some(bits)` where `bits ≈ -log2(width)` for finite bounds with
/// nonzero width. Returns `None` for zero-width (exact) or infinite bounds.
fn estimate_precision_bits(prefix: &Prefix) -> Option<U> {
    let lo = prefix.lower();
    let hi_xb = prefix.upper();
    let (lo_bin, hi_bin) = match (&lo, &hi_xb) {
        (XBinary::Finite(l), XBinary::Finite(h)) => (l, h),
        _ => return None,
    };

    let width = hi_bin.sub(lo_bin);
    if width.mantissa().is_zero() {
        return None;
    }

    // -log2(width) ≈ -(mantissa_bits + exponent)
    let mantissa_bits = crate::sane::bits_as_u(width.mantissa().magnitude().bits());
    let mantissa_bits_i64 = i64::from(mantissa_bits);
    let exp_i64 = i64::from(width.exponent());
    let log2_width = exp_i64.checked_add(mantissa_bits_i64)?;
    let neg_log2 = match log2_width.checked_neg() {
        Some(v) if v >= 0_i64 => v,
        _ => return None,
    };
    U::try_from(neg_log2).ok()
}

// Test helpers - exposed for integration tests
#[cfg(test)]
pub fn taylor_sin_bounds_test(x: &Binary, n: U) -> (Binary, Binary) {
    taylor_sin_bounds(x, n, n.checked_mul(10).expect("n * 10 does not overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computable::Computable;
    use crate::refinement::prefix_width_leq;
    use crate::sane::XI;
    use crate::test_utils::{bin, interval_midpoint_computable, unwrap_finite, xbin};

    fn assert_prefix_compatible_with_expected(
        prefix: &Prefix,
        expected: &Binary,
        tolerance_exp: &XI,
    ) {
        let lower = unwrap_finite(&prefix.lower());
        let upper = unwrap_finite(&prefix.upper());

        assert!(lower <= *expected && *expected <= upper);
        assert!(prefix_width_leq(prefix, tolerance_exp));
    }

    #[test]
    fn sin_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();
        let epsilon = XI::from_i32(-8);
        let bounds = sin_zero
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // sin(0) = 0
        let expected = bin(0, 0);
        assert_prefix_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sin_of_zero_exact() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();

        // sin(0) = 0 exactly — should converge to exact bounds
        let bounds = sin_zero
            .refine_to_default(XI::NegInf)
            .expect("sin(0) should converge to exact zero");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());
        assert_eq!(lower, bin(0, 0), "sin(0) lower bound should be exactly 0");
        assert_eq!(upper, bin(0, 0), "sin(0) upper bound should be exactly 0");
    }

    #[test]
    fn sin_bounds_zero_input_ignores_pi_precision() {
        // sin(0) = 0 regardless of pi's precision. The budget system correctly
        // gives pi an infinite budget when the input is zero (since 0 * pi = 0),
        // so pi may never be refined. sin_prefix must handle this by returning
        // exact zero before consulting pi_prefix.
        let input_prefix = Prefix::from_lower_upper(xbin(0, 0), xbin(0, 0));
        let pi_prefix = Prefix::from_lower_upper(XBinary::NegInf, XBinary::PosInf);

        let result = sin_prefix(&input_prefix, &pi_prefix, 10).expect("sin_prefix should succeed");
        let lower = unwrap_finite(&result.lower());
        let upper = unwrap_finite(&result.upper());
        assert_eq!(
            lower,
            bin(0, 0),
            "sin(0) should be 0 regardless of pi precision"
        );
        assert_eq!(
            upper,
            bin(0, 0),
            "sin(0) should be 0 regardless of pi precision"
        );
    }

    #[test]
    fn sin_of_pi_over_2() {
        // pi/2 ~= 1.5707963...
        // We approximate it as 3217/2048 ~= 1.5708...
        let pi_over_2 = Computable::constant(bin(3217, -11));
        let sin_pi_2 = pi_over_2.sin();
        let epsilon = XI::from_i32(-6);
        let bounds = sin_pi_2
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // sin(pi/2) = 1
        let expected_f64 = (std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        // sin(pi/2) should be very close to 1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_of_pi() {
        // pi ~= 3.14159...
        // We approximate it as 6434/2048 ~= 3.1416...
        let pi_approx = Computable::constant(bin(6434, -11));
        let sin_pi = pi_approx.sin();
        let epsilon = XI::from_i32(-6);
        let bounds = sin_pi
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // sin(pi) ~= 0 (should be close to 0)
        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

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
        let epsilon = XI::from_i32(-6);
        let bounds = sin_neg_pi_2
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // sin(-pi/2) = -1
        let expected_f64 = (-std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        // sin(-pi/2) should be very close to -1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sin_bounds_always_in_minus_one_to_one() {
        // Test with a large value that exercises argument reduction
        let large_value = Computable::constant(bin(100, 0));
        let sin_large = large_value.sin();
        let bounds = sin_large.prefix().expect("bounds should succeed");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

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
        let epsilon = XI::from_i32(-8);
        let bounds = sin_small
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // sin(0.0625) ~= 0.0624593...
        let expected = XBinary::from_f64(0.0625_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn sin_interval_spanning_maximum() {
        // An interval that spans pi/2 (where sin has maximum)
        let computable = interval_midpoint_computable(1, 2); // [1, 2] includes pi/2 ~= 1.57
        let sin_interval = computable.sin();
        let bounds = sin_interval.prefix().expect("bounds should succeed");

        let upper = unwrap_finite(&bounds.upper());

        // The upper bound should be close to 1 since the interval contains pi/2
        assert!(upper >= bin(1, -1)); // Upper bound should be at least 0.5
    }

    #[test]
    fn sin_with_infinite_input_prefix() {
        let unbounded = Computable::new(
            0u32,
            |_| Ok(Prefix::from_lower_upper(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let sin_unbounded = unbounded.sin();
        let bounds = sin_unbounded.prefix().expect("bounds should succeed");

        // sin of unbounded input should be [-1, 1]
        assert_eq!(bounds.lower(), xbin(-1, 0));
        assert_eq!(bounds.upper(), xbin(1, 0));
    }

    #[test]
    fn sin_expression_with_arithmetic() {
        // Test sin(x) + cos-like expression: sin(x)^2 + sin(x + pi/2)^2 should be close to 1
        // Here we just test that sin works in expressions
        let x = Computable::constant(bin(1, 0)); // x = 1
        let sin_x = x.clone().sin();
        let two = Computable::constant(bin(2, 0));
        let expr = sin_x.clone() * two; // 2 * sin(1)

        let epsilon = XI::from_i32(-8);
        let bounds = expr
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        // 2 * sin(1) ~= 2 * 0.8414... ~= 1.6829...
        let expected = XBinary::from_f64(2.0 * 1.0_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn directed_rounding_produces_valid_bounds() {
        // Test that directed rounding produces well-ordered bounds that contain the true value.

        let test_cases = [
            bin(1, -2),  // 0.25
            bin(1, 0),   // 1.0
            bin(3, 0),   // 3.0
            bin(-1, 0),  // -1.0
            bin(5, -1),  // 2.5
            bin(-3, -1), // -1.5
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
                lower,
                upper,
                x
            );

            // Verify bounds are within sin's range [-1, 1]
            assert!(
                lower >= neg_one,
                "Lower bound {} should be >= -1 for x = {}",
                lower,
                x
            );
            assert!(
                upper <= one,
                "Upper bound {} should be <= 1 for x = {}",
                upper,
                x
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
            width10,
            width5
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
        // Verify that the lower bound is <= the upper bound from taylor_sin_bounds
        let x = bin(1, 0); // 1.0
        let n = 5;

        let (lower, upper) = taylor_sin_bounds_test(&x, n);

        // The lower bound should be <= upper bound
        assert!(
            lower <= upper,
            "Lower bound {} should be <= upper bound {}",
            lower,
            upper
        );
    }

    #[test]
    fn sin_of_large_multiple_of_pi() {
        // Test sin(100) which requires significant range reduction
        // This exercises the pi error propagation
        let x = Computable::constant(bin(100, 0)); // 100
        let sin_x = x.sin();
        let epsilon = XI::from_i32(-4);
        let bounds = sin_x
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

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
            lower,
            upper,
            expected_value
        );
    }

    #[test]
    fn sin_pi_bounds_contain_zero() {
        // Use our pi implementation for a more precise test
        use super::super::pi::pi_bounds_at_precision;

        let (pi_lo, pi_hi) = pi_bounds_at_precision(64);
        let pi_mid = pi_lo.add(&pi_hi);
        let pi_approx = Binary::new(
            pi_mid.mantissa().clone(),
            pi_mid.exponent().checked_sub(1).unwrap(),
        );

        let sin_pi = Computable::constant(pi_approx).sin();
        let epsilon = XI::from_i32(-10);
        let bounds = sin_pi
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());
        let zero = bin(0, 0);

        // sin(pi) = 0, bounds should contain zero
        assert!(
            lower <= zero && zero <= upper,
            "sin(pi) bounds [{}, {}] should contain zero",
            lower,
            upper
        );
    }

    #[test]
    fn sin_of_one_to_512_bit_precision() {
        // Verify that the adaptive-precision fix allows sin to converge well
        // beyond the old 64-bit cap. We request 512-bit precision (epsilon = 2^-512).
        // sin(1) ≈ 0.8414709848... is in [-pi/2, pi/2], so it exercises both
        // compute_sin_on_monotonic_interval and divide_by_factorial_directed directly.
        let one = Computable::constant(bin(1, 0));
        let sin_one = one.sin();
        let epsilon = XI::from_i32(-512);
        // Need many Taylor terms for 512-bit accuracy; allow up to 1024 refinement steps.
        let bounds = sin_one
            .refine_to::<1024>(epsilon)
            .expect("refine_to 512-bit precision should succeed");

        // Width must be at most 2^-512
        assert!(
            prefix_width_leq(&bounds, &epsilon),
            "width should be <= 2^-512"
        );

        // The 512-bit midpoint should agree with f64 sin(1) to nearly all 53 mantissa bits.
        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());
        let midpoint = FiniteBounds::new(lower, upper).midpoint();
        let expected_f64 = 1.0_f64.sin();
        let expected_bin =
            unwrap_finite(&XBinary::from_f64(expected_f64).expect("expected value should convert"));
        let diff = if midpoint > expected_bin {
            midpoint.sub(&expected_bin)
        } else {
            expected_bin.sub(&midpoint)
        };
        assert!(
            diff < bin(1, -52),
            "midpoint of 512-bit bounds should agree with f64 sin(1) to ~52 bits, diff = {}",
            diff
        );
    }

    // =====================================================================
    // Tests for the fixed correctness issues
    // =====================================================================

    #[test]
    fn sin_extremely_large_input() {
        // Correctness invariant for large inputs: 2^20 ≈ 1_048_576 (k ≈ 166_886).
        let large = Computable::constant(bin(1, 20)); // 2^20 = 1_048_576
        let sin_large = large.sin();
        let epsilon = XI::from_i32(-4);
        let bounds = sin_large
            .refine_to_default(epsilon)
            .expect("refine_to should succeed for very large input");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);
        assert!(lower >= neg_one, "lower bound {} should be >= -1", lower);
        assert!(upper <= one, "upper bound {} should be <= 1", upper);

        // sin(2^20) ≈ -0.24271... — verify bounds contain this
        let expected_f64 = (1048576.0_f64).sin();
        let expected =
            unwrap_finite(&XBinary::from_f64(expected_f64).expect("expected value should convert"));
        assert!(
            lower <= expected && expected <= upper,
            "sin(2^20) bounds [{}, {}] should contain expected value {}",
            lower,
            upper,
            expected
        );
    }

    #[test]
    fn sin_very_large_input_2_pow_30() {
        // Correctness invariant: 2^30 ≈ 1 billion, k ≈ 170 million
        let large = Computable::constant(bin(1, 30));
        let sin_large = large.sin();
        let epsilon = XI::from_i32(-4);
        let bounds = sin_large
            .refine_to_default(epsilon)
            .expect("refine_to should succeed for 2^30 input");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);
        assert!(lower >= neg_one, "lower bound should be >= -1");
        assert!(upper <= one, "upper bound should be <= 1");
    }

    #[test]
    fn sin_negative_large_input() {
        // Correctness invariant: large negative input -10000
        let x = Computable::constant(bin(-10000, 0));
        let sin_x = x.sin();
        let epsilon = XI::from_i32(-4);
        let bounds = sin_x
            .refine_to_default(epsilon)
            .expect("refine_to should succeed for large negative input");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);
        assert!(lower >= neg_one, "lower bound should be >= -1");
        assert!(upper <= one, "upper bound should be <= 1");

        // sin(-10000) ≈ 0.30561... — verify bounds contain this
        let expected_f64 = (-10000.0_f64).sin();
        let expected =
            unwrap_finite(&XBinary::from_f64(expected_f64).expect("expected value should convert"));
        assert!(
            lower <= expected && expected <= upper,
            "sin(-10000) bounds [{}, {}] should contain expected value {}",
            lower,
            upper,
            expected
        );
    }

    #[test]
    fn sin_midpoint_correctness_uses_lo_bound() {
        // Regression test for sin-midpoint-correctness: verify that the full-period
        // check uses the lower bound of 2π (conservative direction).
        //
        // We directly call sin_prefix with an interval whose width is slightly
        // above the lower bound of 2π. The result should be [-1, 1] because
        // the interval might span a full period.
        use super::super::pi::{pi_bounds_at_precision, two_pi_interval_at_precision};

        let two_pi = two_pi_interval_at_precision(64);
        // Create an interval [0, two_pi_lo + epsilon]: width > two_pi_lo
        let lo = bin(0, 0);
        let hi = two_pi.lo().add(&bin(1, -60));
        let input_prefix = Prefix::from_lower_upper(XBinary::Finite(lo), XBinary::Finite(hi));
        let (pi_lo, pi_hi) = pi_bounds_at_precision(64);
        let pi_prefix = Prefix::from_lower_upper(XBinary::Finite(pi_lo), XBinary::Finite(pi_hi));
        let result = sin_prefix(&input_prefix, &pi_prefix, 5).expect("sin_prefix should succeed");

        // Because the width >= two_pi_lo, the conservative check should trigger
        // and return [-1, 1] (possibly slightly widened by simplification).
        let lower = unwrap_finite(&result.lower());
        let upper = unwrap_finite(&result.upper());

        // The key property: the bounds must be at least as wide as [-1, 1]
        assert!(lower <= bin(-1, 0), "lower bound {} should be <= -1", lower);
        assert!(upper >= bin(1, 0), "upper bound {} should be >= 1", upper);
    }

    #[test]
    fn sin_interval_straddling_both_critical_points() {
        // Correctness invariant: [-2, 2] straddles both +pi/2 and -pi/2.
        let computable = interval_midpoint_computable(-2, 2);
        let sin_x = computable.sin();
        let bounds = sin_x.prefix().expect("bounds should succeed");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        // The interval contains both pi/2 (sin=1) and -pi/2 (sin=-1),
        // so the bounds must cover [-1, 1].
        assert!(
            lower <= bin(-1, 0),
            "lower bound {} should be <= -1 (interval straddles both critical points)",
            lower
        );
        assert!(
            upper >= bin(1, 0),
            "upper bound {} should be >= 1 (interval straddles both critical points)",
            upper
        );
    }

    #[test]
    fn sin_wide_interval_near_period_boundary() {
        // Regression: [-3, 3] straddles both ±pi/2. The old code's Case 4
        // (ContainsMax) fired first, producing a lower bound above -1.
        let computable = interval_midpoint_computable(-3, 3);
        let sin_x = computable.sin();
        let bounds = sin_x.prefix().expect("bounds should succeed");

        let lower = unwrap_finite(&bounds.lower());
        let upper = unwrap_finite(&bounds.upper());

        // Width ~6 covers both ±pi/2, so bounds must include [-1, 1]
        assert!(lower <= bin(-1, 0), "lower bound should be <= -1");
        assert!(upper >= bin(1, 0), "upper bound should be >= 1");
    }
}
