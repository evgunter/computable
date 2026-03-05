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
//! BLOCKED: This requires node-initiated refinement — the ability for a node's
//! `refine_step` to return a recoverable error requesting that the coordinator
//! refine a specific input before retrying. The current model doesn't support
//! this: the coordinator decides which refiners to step, and nodes cannot signal
//! "my input bounds are too wide, refine them first."

use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::{Binary, Bounds, FiniteBounds, UXBinary, XBinary};
use crate::binary_utils::bisection::{
    PrefixBisectionResult, PrefixBounds, bisection_step_normalized, midpoint, normalize_bounds,
};
use crate::binary_utils::power::binary_pow;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;

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
    /// This is `None` until the first `compute_bounds()` call, which initializes
    /// it from the input bounds. We use `Option` because initialization requires
    /// calling `inner.get_bounds()` which can fail, but node construction (via
    /// `nth_root()`) is not supposed to be fallible. By deferring initialization
    /// to the first `compute_bounds()` call (which returns `Result`), we can
    /// propagate errors through the normal Result path.
    ///
    /// Each refinement step halves this interval via bisection.
    pub bisection_state: RwLock<Option<BisectionState>>,
}

/// State for the bisection algorithm.
/// Tracks the current interval bounds for the n-th root in prefix form.
#[derive(Clone, Debug)]
pub struct BisectionState {
    /// Current bounds in prefix form.
    pub bounds: PrefixBounds,
    /// The target value (x) whose n-th root we're computing.
    pub target: Binary,
    /// Whether the result should be negated (for odd roots of negative numbers).
    pub negate_result: bool,
    /// If set, the exact root value (set when bisection hits Exact).
    pub exact_value: Option<Binary>,
}

impl NodeOp for NthRootOp {
    fn compute_bounds(&self) -> Result<Prefix, ComputableError> {
        let input_bounds = self.inner.get_bounds_as_bounds()?;

        // Fast path: read lock to check if already initialized.
        {
            let state = self.bisection_state.read();
            if let Some(s) = &*state {
                return Ok(Prefix::from(&bounds_from_bisection_state(s)));
            }
        }
        // Slow path: upgrade to write lock and initialize.
        // Double-check after acquiring write lock (another thread may have initialized).
        let mut state = self.bisection_state.write();
        if let Some(s) = &*state {
            return Ok(Prefix::from(&bounds_from_bisection_state(s)));
        }
        let s = initialize_nth_root_bisection_state(&input_bounds, self.degree.get())?;
        let bounds = bounds_from_bisection_state(&s);
        *state = Some(s);
        Ok(Prefix::from(&bounds))
    }

    fn refine_step(&self, _precision_bits: usize) -> Result<bool, ComputableError> {
        // Ensure bisection state is initialized (compute_bounds is always called
        // before refine_step by the coordinator, but be defensive).
        {
            let state = self.bisection_state.read();
            if state.is_none() {
                drop(state);
                // Trigger initialization via compute_bounds.
                self.compute_bounds()?;
            }
        }

        let mut state = self.bisection_state.write();
        let s = match state.as_mut() {
            Some(s) => s,
            None => return Err(ComputableError::InfiniteBounds),
        };

        // If we already have an exact value, no need to refine
        if s.exact_value.is_some() {
            return Ok(false);
        }

        // Perform one bisection step
        let degree = self.degree.get();
        let target = &s.target;
        let result =
            bisection_step_normalized(&s.bounds, |mid| binary_pow(mid, degree).cmp(target));

        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                s.bounds = new_bounds;
            }
            PrefixBisectionResult::Exact(mid) => {
                s.exact_value = Some(mid);
            }
        }
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }

    /// Sensitivity of x^(1/n): derivative = (1/n) · x^((1-n)/n).
    /// Max |derivative| at smallest input x = a: (1/n) · a^((1-n)/n).
    /// Child budget = target · n · a^((n-1)/n).
    ///
    /// We approximate a^((n-1)/n) ≈ a, which is conservative (budget too
    /// loose) for a ≥ 1 and slightly tight for a < 1. This avoids needing
    /// to compute an nth root inside the budget function.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        use crate::binary::UBinary;
        use num_bigint::BigUint;

        let n = self.degree.get();
        if n == 1 {
            return target_width.clone();
        }
        let min_abs = match self.inner.cached_bounds_as_bounds() {
            Some(b) => {
                let (lo, hi) = b.abs();
                std::cmp::min(lo, hi)
            }
            None => return target_width.clone(),
        };
        let n_ux = UXBinary::Finite(UBinary::new(BigUint::from(n), BigInt::zero()));
        target_width.mul(&n_ux).mul(&min_abs)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        self.degree.get() > 1
    }
}

/// Extracts bounds from an initialized bisection state.
fn bounds_from_bisection_state(s: &BisectionState) -> Bounds {
    let finite_bounds = {
        let bounds = if let Some(exact) = &s.exact_value {
            FiniteBounds::point(exact.clone())
        } else {
            s.bounds.to_finite_bounds()
        };
        if s.negate_result {
            bounds.interval_neg()
        } else {
            bounds
        }
    };
    Bounds::from_lower_and_width(
        XBinary::Finite(finite_bounds.small().clone()),
        UXBinary::Finite(finite_bounds.width().clone()),
    )
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
        // Lower should already be at the correct exponent from normalize_bounds
        normalized_lower.mantissa().clone()
    };

    Ok(BisectionState {
        bounds: PrefixBounds::new(mantissa, exponent),
        target: actual_target,
        negate_result,
        exact_value: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computable::Computable;
    use crate::prefix::Prefix;
    use crate::refinement::{XUsize, prefix_width_leq};
    use crate::test_utils::{bin, interval_noop_computable, unwrap_finite};

    /// Helper to create NonZeroU32 from a literal in tests.
    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("test degree must be non-zero")
    }

    fn assert_bounds_compatible_with_expected(
        prefix: &Prefix,
        expected: &Binary,
        tolerance_exp: &XUsize,
    ) {
        let bounds = Bounds::from(prefix);
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let upper = unwrap_finite(&upper_xb);

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected,
            lower,
            upper
        );
        assert!(
            prefix_width_leq(prefix, tolerance_exp),
            "Bounds width should be <= tolerance",
        );
    }

    #[test]
    fn sqrt_of_4() {
        let four = Computable::constant(bin(4, 0));
        let sqrt_four = four.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let prefix = sqrt_four
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        assert_bounds_compatible_with_expected(&prefix, &bin(2, 0), &epsilon);
    }

    #[test]
    fn sqrt_of_2() {
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let prefix = sqrt_two
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        let expected_f64 = 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);
        assert_bounds_compatible_with_expected(&prefix, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_8() {
        let eight = Computable::constant(bin(8, 0));
        let cbrt_eight = eight.nth_root(nz(3));
        let epsilon = XUsize::Finite(8);
        let prefix = cbrt_eight
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        assert_bounds_compatible_with_expected(&prefix, &bin(2, 0), &epsilon);
    }

    #[test]
    fn cbrt_of_negative_8() {
        let neg_eight = Computable::constant(bin(-8, 0));
        let cbrt_neg_eight = neg_eight.nth_root(nz(3));
        let epsilon = XUsize::Finite(8);
        let prefix = cbrt_neg_eight
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        assert_bounds_compatible_with_expected(&prefix, &bin(-2, 0), &epsilon);
    }

    #[test]
    fn fourth_root_of_16() {
        let sixteen = Computable::constant(bin(16, 0));
        let fourth_root = sixteen.nth_root(nz(4));
        let epsilon = XUsize::Finite(8);
        let prefix = fourth_root
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        assert_bounds_compatible_with_expected(&prefix, &bin(2, 0), &epsilon);
    }

    #[test]
    fn sqrt_of_half() {
        let half = Computable::constant(bin(1, -1));
        let sqrt_half = half.nth_root(nz(2));
        let epsilon = XUsize::Finite(8);
        let prefix = sqrt_half
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        let expected_f64 = 0.5_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);
        assert_bounds_compatible_with_expected(&prefix, &expected, &epsilon);
    }

    #[test]
    fn nth_root_in_expression() {
        let sqrt_2 = Computable::constant(bin(2, 0)).nth_root(nz(2));
        let cbrt_8 = Computable::constant(bin(8, 0)).nth_root(nz(3));
        let sum = sqrt_2 + cbrt_8;
        let epsilon = XUsize::Finite(8);
        let prefix = sum
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        let expected_f64 = 2.0_f64.sqrt() + 2.0_f64;
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);
        assert_bounds_compatible_with_expected(&prefix, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sqrt_zero = zero.nth_root(nz(2));
        let prefix = sqrt_zero.bounds().expect("bounds should succeed");
        let bounds = Bounds::from(&prefix);
        let expected = bin(0, 0);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn sqrt_of_interval_overlapping_zero() {
        let interval = interval_noop_computable(-1, 4);
        let sqrt_interval = interval.nth_root(nz(2));
        let prefix = sqrt_interval.bounds().expect("bounds should succeed");
        let bounds = Bounds::from(&prefix);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        assert!(lower <= bin(1, 0), "lower {} should be <= 1", lower);
        assert!(upper >= bin(1, 0), "upper {} should be >= 1", upper);
    }

    #[test]
    fn cbrt_of_interval_overlapping_zero() {
        let interval = interval_noop_computable(-8, 27);
        let cbrt_interval = interval.nth_root(nz(3));
        let prefix = cbrt_interval.bounds().expect("bounds should succeed");
        let bounds = Bounds::from(&prefix);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        assert!(lower <= bin(2, 0), "lower {} should be <= 2", lower);
        assert!(upper >= bin(2, 0), "upper {} should be >= 2", upper);
    }
}
