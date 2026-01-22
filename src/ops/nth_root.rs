//! N-th root operation with binary search refinement.
//!
//! This module implements the n-th root operation (x^(1/n)) using:
//! - Binary search (bisection) for guaranteed convergence
//! - Interval arithmetic for provably correct bounds
//!
//! The algorithm maintains an interval [lower, upper] where the true root lies,
//! and refines by bisection: if mid^n <= target, the root is in [mid, upper],
//! otherwise it's in [lower, mid].
//!
//! This module uses the generic binary search helper from [`crate::binary::bisection`],
//! which can be reused for other operations that use bisection (e.g., finding roots
//! of monotonic functions).
//!
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
//! numbers (but aren't completely negative) currently just return (0, ∞) bounds
//! instead of returning a recoverable error that would trigger refinement of the
//! input until the bounds are fully non-negative. This should be fixed to match
//! the behavior described in the README for sqrt.
//!
//! BLOCKED: This is blocked on refactoring the refinement system to use the
//! async/event-driven model described in the README (see TODO in refinement.rs)
//! rather than the current synchronous lock-step model. The recoverable error
//! approach requires the ability for a node to request refinement of its input
//! and receive updates, which the current synchronous model doesn't support.

use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::{
    Binary, Bounds, FiniteBounds, UXBinary, XBinary, margin_from_width, simplify_bounds_if_needed,
};
use crate::binary_utils::bisection::{
    BisectionComparison, NormalizedBisectionResult, NormalizedBounds, bisection_step_normalized,
    midpoint, normalize_bounds,
};
use crate::binary_utils::power::binary_pow;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

/// Precision threshold for triggering bounds simplification.
/// 64 chosen: bisection benefits most from simplification (13% speedup vs disabled).
const PRECISION_SIMPLIFICATION_THRESHOLD: u64 = 64;

/// Loosening fraction for bounds simplification.
/// 3 = loosen by width/8. Benchmarks show margin has minimal performance impact.
const MARGIN_SHIFT: u32 = 3;

/// N-th root operation with binary search refinement.
///
/// Computes x^(1/n) where n is the root degree.
/// For n=2, this is square root; n=3 is cube root, etc.
///
/// # Constraints
/// - For even n: requires x >= 0 (otherwise returns infinite bounds)
/// - For odd n: supports all real x (negative values have negative roots)
pub struct NthRootOp {
    /// The input node whose n-th root we're computing.
    pub inner: Arc<Node>,
    /// The root degree (n in x^(1/n)). Guaranteed to be >= 1 by the type system.
    pub degree: NonZeroU32,
    /// Current bisection state: tracks the interval for the root.
    ///
    /// This is `None` until the first refinement step, which initializes it from
    /// the input bounds. We use `Option` because initialization requires calling
    /// `inner.get_bounds()` which can fail, but node construction (via `nth_root()`)
    /// is not supposed to be fallible. By deferring initialization to the first
    /// `refine_step()` call, we can propagate errors through the normal Result path.
    ///
    /// Each refinement step halves this interval via bisection.
    pub bisection_state: RwLock<Option<BisectionState>>,
}

/// State for the bisection algorithm.
/// Tracks the current interval bounds for the n-th root in normalized form.
#[derive(Clone, Debug)]
pub struct BisectionState {
    /// Current bounds in normalized form.
    pub bounds: NormalizedBounds,
    /// The target value (x) whose n-th root we're computing.
    pub target: Binary,
    /// Whether the result should be negated (for odd roots of negative numbers).
    pub negate_result: bool,
    /// If set, the exact root value (set when bisection hits Exact).
    pub exact_value: Option<Binary>,
}

impl NodeOp for NthRootOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let state = self.bisection_state.read();

        match &*state {
            None => {
                // Return initial conservative bounds based on input
                compute_initial_bounds(&input_bounds, self.degree.get())
            }
            Some(s) => {
                // Return current bisection interval, simplified if needed
                let finite_bounds = {
                    let bounds = if let Some(exact) = &s.exact_value {
                        // Exact match found - point interval with zero width
                        FiniteBounds::point(exact.clone())
                    } else {
                        // Reconstruct bounds from normalized form
                        s.bounds.to_finite_bounds()
                    };
                    if s.negate_result {
                        bounds.interval_neg()
                    } else {
                        bounds
                    }
                };
                let raw_bounds = Bounds::from_lower_and_width(
                    XBinary::Finite(finite_bounds.small().clone()),
                    UXBinary::Finite(finite_bounds.width().clone()),
                );
                // Simplify bounds to reduce precision bloat from bisection
                let margin = margin_from_width(raw_bounds.width(), MARGIN_SHIFT);
                Ok(simplify_bounds_if_needed(
                    &raw_bounds,
                    PRECISION_SIMPLIFICATION_THRESHOLD,
                    &margin,
                ))
            }
        }
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let mut state = self.bisection_state.write();

        match &mut *state {
            None => {
                // Initialize the bisection state from input bounds
                *state = Some(initialize_nth_root_bisection_state(
                    &input_bounds,
                    self.degree.get(),
                )?);
                Ok(true)
            }
            Some(s) => {
                // If we already have an exact value, no need to refine
                if s.exact_value.is_some() {
                    return Ok(false);
                }

                // Perform one bisection step
                let degree = self.degree.get();
                let target = &s.target;
                let result = bisection_step_normalized(&s.bounds, |mid| {
                    let mid_pow = binary_pow(mid, degree);
                    match mid_pow.cmp(target) {
                        std::cmp::Ordering::Less => BisectionComparison::Above,
                        std::cmp::Ordering::Equal => BisectionComparison::Exact,
                        std::cmp::Ordering::Greater => BisectionComparison::Below,
                    }
                });

                match result {
                    NormalizedBisectionResult::Narrowed(new_bounds) => {
                        s.bounds = new_bounds;
                    }
                    NormalizedBisectionResult::Exact(mid) => {
                        s.exact_value = Some(mid);
                    }
                }
                Ok(true)
            }
        }
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Computes initial conservative bounds for the n-th root.
///
/// Computes lower and upper output bounds separately, which allows handling
/// mixed finite/infinite input bounds (e.g., getting a finite lower output
/// bound even when upper input is PosInf).
fn compute_initial_bounds(input_bounds: &Bounds, degree: u32) -> Result<Bounds, ComputableError> {
    let lower_input = input_bounds.small();
    let upper_input = &input_bounds.large();
    let is_even = degree.is_multiple_of(2);

    let lower_output = compute_output_lower_bound(lower_input, is_even, degree)?;
    let upper_output = compute_output_upper_bound(upper_input, is_even, degree)?;

    Ok(Bounds::new(lower_output, upper_output))
}

/// Computes a lower bound for the n-th root output from the lower input bound.
///
/// For odd roots of negative values: cbrt(x) = -cbrt(|x|), so to get a LOWER
/// bound on cbrt(x), we need an UPPER bound on cbrt(|x|), then negate.
fn compute_output_lower_bound(
    lower_input: &XBinary,
    is_even: bool,
    degree: u32,
) -> Result<XBinary, ComputableError> {
    match lower_input {
        XBinary::NegInf => {
            // Lower input is -∞
            Ok(if is_even {
                // Even root: output is non-negative, so lower bound is 0
                XBinary::Finite(Binary::zero())
            } else {
                // Odd root: cbrt(-∞) = -∞
                XBinary::NegInf
            })
        }
        XBinary::PosInf => {
            // Lower input is +∞ - currently unexpected for a lower bound.
            crate::detected_computable_with_infinite_value!("lower input bound is PosInf");
            Ok(XBinary::PosInf)
        }
        XBinary::Finite(lower_bin) => {
            if lower_bin.mantissa().is_negative() {
                if is_even {
                    // Even root of negative: output lower bound is 0
                    // (the actual root computation may error if entirely negative,
                    // but that's checked when we also consider the upper bound)
                    Ok(XBinary::Finite(Binary::zero()))
                } else {
                    // Odd root of negative: cbrt(lower) = -cbrt(|lower|)
                    // Lower bound on output = -upper_bound(cbrt(|lower|))
                    let neg_lower = lower_bin.neg();
                    let lower_root = compute_root_upper_bound(&neg_lower, degree).neg();
                    Ok(XBinary::Finite(lower_root))
                }
            } else {
                // Non-negative input
                let lower_root = compute_root_lower_bound(lower_bin, degree);
                Ok(XBinary::Finite(lower_root))
            }
        }
    }
}

/// Computes an upper bound for the n-th root output from the upper input bound.
///
/// For odd roots of negative values: cbrt(x) = -cbrt(|x|), so to get an UPPER
/// bound on cbrt(x), we need a LOWER bound on cbrt(|x|), then negate.
fn compute_output_upper_bound(
    upper_input: &XBinary,
    is_even: bool,
    degree: u32,
) -> Result<XBinary, ComputableError> {
    match upper_input {
        XBinary::PosInf => Ok(XBinary::PosInf),
        XBinary::NegInf => {
            // Upper input is -∞ - currently unexpected for an upper bound.
            crate::detected_computable_with_infinite_value!("upper input bound is NegInf");
            Ok(XBinary::NegInf)
        }
        XBinary::Finite(upper_bin) => {
            if upper_bin.mantissa().is_negative() {
                if is_even {
                    // Even root of entirely negative interval: no real root
                    return Err(ComputableError::DomainError);
                }
                // Odd root of negative: cbrt(upper) = -cbrt(|upper|)
                // Upper bound on output = -lower_bound(cbrt(|upper|))
                let neg_upper = upper_bin.neg();
                let upper_root = compute_root_lower_bound(&neg_upper, degree).neg();
                Ok(XBinary::Finite(upper_root))
            } else {
                // Non-negative input
                let upper_root = compute_root_upper_bound(upper_bin, degree);
                Ok(XBinary::Finite(upper_root))
            }
        }
    }
}

/// Computes an upper bound for the n-th root of a positive value.
/// Returns a value >= x^(1/n).
fn compute_root_upper_bound(x: &Binary, _degree: u32) -> Binary {
    // Conservative upper bound: max(1, |x|)
    // This is always >= x^(1/n) for x >= 0 and n >= 1
    let one = Binary::new(BigInt::one(), BigInt::zero());
    let abs_x = x.magnitude().to_binary();

    if abs_x > one { abs_x } else { one }
}

/// Computes a lower bound for the n-th root of a positive value.
/// Returns a value <= x^(1/n).
fn compute_root_lower_bound(x: &Binary, _degree: u32) -> Binary {
    // Conservative lower bound: min(1, |x|) for x > 0, 0 otherwise
    if x.mantissa().is_zero() || x.mantissa().is_negative() {
        return Binary::zero();
    }

    let one = Binary::new(BigInt::one(), BigInt::zero());
    if x < &one { x.clone() } else { one }
}

/// Initializes the bisection state for nth root computation.
///
/// Takes the midpoint of input bounds as the target value, then sets up initial
/// bisection bounds to find the nth root of that target.
fn initialize_nth_root_bisection_state(
    input_bounds: &Bounds,
    degree: u32,
) -> Result<BisectionState, ComputableError> {
    let lower = input_bounds.small();
    let upper = &input_bounds.large();

    // Get the target value - use midpoint for intervals, exact for points
    let target = match (lower, upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => midpoint(l, u),
        _ => return Err(ComputableError::InfiniteBounds),
    };

    let is_even = degree.is_multiple_of(2);

    // Handle negative targets for even roots
    if is_even && target.mantissa().is_negative() {
        return Err(ComputableError::DomainError);
    }

    // For odd roots of negative values, compute root of |target| and negate
    let (actual_target, negate_result) = if !is_even && target.mantissa().is_negative() {
        (target.neg(), true)
    } else {
        (target.clone(), false)
    };

    // Initial bounds for bisection: [0 or small, max(1, target)]
    let one = Binary::new(BigInt::one(), BigInt::zero());

    let bisection_lower = if actual_target.mantissa().is_zero() {
        Binary::zero()
    } else if actual_target < one {
        // For 0 < target < 1, the root is > target, so use target as lower bound
        actual_target.clone()
    } else {
        // For target >= 1, the root is <= target, so use 1 as lower bound
        one.clone()
    };

    let bisection_upper = if actual_target.mantissa().is_zero() {
        Binary::zero()
    } else if actual_target < one {
        // For 0 < target < 1, the root is < 1, so use 1 as upper bound
        one
    } else {
        // For target >= 1, the root is <= target, so use target as upper bound
        actual_target.clone()
    };

    // Normalize bounds once at initialization to ensure bisection automatically
    // selects shortest representations at each step
    let initial_bounds = FiniteBounds::new(bisection_lower, bisection_upper);
    let normalized = normalize_bounds(&initial_bounds)?;

    // Extract mantissa and exponent from normalized bounds.
    // Use width's exponent since it's always correct (even when lower is zero,
    // which normalizes to exponent 0).
    let exponent = normalized.width().exponent().clone();
    let normalized_lower = normalized.small();

    // If lower is zero, mantissa is 0 regardless of exponent.
    // Otherwise, we need to ensure the mantissa is at the width's exponent.
    let mantissa = if normalized_lower.mantissa().is_zero() {
        BigInt::zero()
    } else {
        // Lower should already be at the correct exponent from bounds_from_normalized
        normalized_lower.mantissa().clone()
    };

    Ok(BisectionState {
        bounds: NormalizedBounds::new(mantissa, exponent),
        target: actual_target,
        negate_result,
        exact_value: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::UBinary;
    use crate::computable::Computable;
    use crate::test_utils::{
        bin, interval_noop_computable, ubin, unwrap_finite, unwrap_finite_uxbinary,
    };

    /// Helper to create NonZeroU32 from a literal in tests.
    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("test degree must be non-zero")
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

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected,
            lower,
            upper
        );
        assert!(
            width <= *epsilon,
            "Width {} should be <= epsilon {}",
            width,
            epsilon
        );
    }

    #[test]
    fn sqrt_of_4() {
        // sqrt(4) = 2
        let four = Computable::constant(bin(4, 0));
        let sqrt_four = four.nth_root(nz(2));
        let epsilon = ubin(1, -8);
        let bounds = sqrt_four
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_2() {
        // sqrt(2) ~= 1.414...
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(nz(2));
        let epsilon = ubin(1, -8);
        let bounds = sqrt_two
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_8() {
        // cbrt(8) = 2
        let eight = Computable::constant(bin(8, 0));
        let cbrt_eight = eight.nth_root(nz(3));
        let epsilon = ubin(1, -8);
        let bounds = cbrt_eight
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_negative_8() {
        // cbrt(-8) = -2
        let neg_eight = Computable::constant(bin(-8, 0));
        let cbrt_neg_eight = neg_eight.nth_root(nz(3));
        let epsilon = ubin(1, -8);
        let bounds = cbrt_neg_eight
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(-2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn fourth_root_of_16() {
        // 16^(1/4) = 2
        let sixteen = Computable::constant(bin(16, 0));
        let fourth_root = sixteen.nth_root(nz(4));
        let epsilon = ubin(1, -8);
        let bounds = fourth_root
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_half() {
        // sqrt(0.5) ~= 0.707...
        let half = Computable::constant(bin(1, -1));
        let sqrt_half = half.nth_root(nz(2));
        let epsilon = ubin(1, -8);
        let bounds = sqrt_half
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 0.5_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn nth_root_in_expression() {
        // Test that nth_root works in expressions: sqrt(2) + cbrt(8) = sqrt(2) + 2
        let sqrt_2 = Computable::constant(bin(2, 0)).nth_root(nz(2));
        let cbrt_8 = Computable::constant(bin(8, 0)).nth_root(nz(3));
        let sum = sqrt_2 + cbrt_8;

        let epsilon = ubin(1, -8);
        let bounds = sum
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt() + 2.0;
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_zero() {
        // sqrt(0) = 0
        let zero = Computable::constant(bin(0, 0));
        let sqrt_zero = zero.nth_root(nz(2));
        let bounds = sqrt_zero.bounds().expect("bounds should succeed");

        let expected = bin(0, 0);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sqrt_of_interval_overlapping_zero() {
        // Test even root of a Computable with bounds overlapping zero: [-1, 4]
        // The sqrt should have bounds [0, upper] (since sqrt is only defined for non-negative)
        let interval = interval_noop_computable(-1, 4);
        let sqrt_interval = interval.nth_root(nz(2));
        let bounds = sqrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Lower output bound should be 0 (since input overlaps negative)
        assert_eq!(lower, bin(0, 0));
        // Upper output bound should be >= sqrt(4) = 2
        assert!(upper >= bin(2, 0));
    }

    #[test]
    fn cbrt_of_interval_overlapping_zero() {
        // Test odd root of a Computable with bounds overlapping zero: [-8, 27]
        // cbrt(-8) = -2, cbrt(27) = 3, so output should be approximately [-2, 3]
        let interval = interval_noop_computable(-8, 27);
        let cbrt_interval = interval.nth_root(nz(3));
        let bounds = cbrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Lower output bound should be <= cbrt(-8) = -2
        assert!(lower <= bin(-2, 0));
        // Upper output bound should be >= cbrt(27) = 3
        assert!(upper >= bin(3, 0));
    }
}
