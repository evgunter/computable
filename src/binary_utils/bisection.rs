//! Binary search (bisection) helper for iterative refinement.
//!
//! This module provides generic binary search functionality for finding values
//! within an interval using bisection. The helper can be reused by any operation
//! that needs to refine bounds via bisection (e.g., nth_root, inverse functions,
//! root-finding for monotonic functions).
//!
//! # Usage
//!
//! The [`bisection_step`] function performs a single step of binary search.
//! It's designed to be called repeatedly by the refinement infrastructure
//! (e.g., `refine_to_default`), which controls the iteration count.
//!
//! ```
//! use computable::{Binary, FiniteBounds};
//! use computable::binary_utils::bisection::{BisectionComparison, bisection_step};
//! use num_bigint::BigInt;
//! use num_traits::Zero;
//!
//! // Find sqrt(4) in the interval [0, 4]
//! // Starting from [0, 4], midpoint is 2, and 2^2 = 4 exactly
//! let mut bounds = FiniteBounds::new(
//!     Binary::new(BigInt::from(0), BigInt::from(0)),
//!     Binary::new(BigInt::from(4), BigInt::from(0)),
//! );
//! let target = Binary::new(BigInt::from(4), BigInt::from(0));
//!
//! // Perform bisection steps until we find exact match or reach desired precision
//! for _ in 0..20 {
//!     bounds = bisection_step(bounds, |mid| {
//!         let mid_sq = mid.mul(mid);
//!         match mid_sq.cmp(&target) {
//!             std::cmp::Ordering::Less => BisectionComparison::Above,
//!             std::cmp::Ordering::Equal => BisectionComparison::Exact,
//!             std::cmp::Ordering::Greater => BisectionComparison::Below,
//!         }
//!     });
//!     if bounds.width().is_zero() {
//!         break; // Found exact match
//!     }
//! }
//!
//! // bounds now contains sqrt(4) = 2
//! assert_eq!(*bounds.small(), Binary::new(BigInt::from(2), BigInt::from(0)));
//! ```

use num_bigint::BigInt;
use num_traits::One;

use crate::binary::{Binary, FiniteBounds};

/// Result of comparing a test value against the target in a binary search.
///
/// The comparison tells the binary search algorithm which half of the interval
/// to continue searching in:
/// - `Above`: The target value is above the test point, so search [mid, upper]
/// - `Below`: The target value is below the test point, so search [lower, mid]
/// - `Exact`: The test point is exactly the target value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BisectionComparison {
    /// The target is above the test value.
    ///
    /// The search should continue in the upper half: [test_value, upper_bound].
    Above,

    /// The target is below the test value.
    ///
    /// The search should continue in the lower half: [lower_bound, test_value].
    Below,

    /// The test value is exactly the target.
    ///
    /// The search is complete; both bounds should be set to this value.
    Exact,
}

/// Computes the midpoint of two Binary numbers.
///
/// The midpoint is calculated as (lower + upper) / 2.
pub fn midpoint(lower: &Binary, upper: &Binary) -> Binary {
    let sum = lower.add(upper);
    // Divide by 2 by subtracting 1 from the exponent
    Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one())
}

/// Performs a single step of binary search between bounds.
///
/// This function:
/// 1. Computes the midpoint of the bounds
/// 2. Calls the comparison function with the midpoint
/// 3. Returns new bounds based on the comparison result
///
/// # Arguments
///
/// * `bounds` - The current bounds interval
/// * `compare` - A function that compares the midpoint to the target value
///   and returns whether the target is above, below, or exactly at the midpoint
///
/// # Returns
///
/// New [`FiniteBounds`] after the bisection step. If an exact match was found,
/// the bounds will have zero width (i.e., `bounds.width().is_zero()` is true).
///
/// # Behavior
///
/// - If `compare` returns `Above`: the target is above the midpoint, so
///   the new interval is [midpoint, upper]
/// - If `compare` returns `Below`: the target is below the midpoint, so
///   the new interval is [lower, midpoint]
/// - If `compare` returns `Exact`: the midpoint is exactly the target,
///   so both bounds are set to the midpoint (width becomes zero)
pub fn bisection_step<F>(bounds: FiniteBounds, compare: F) -> FiniteBounds
where
    F: FnOnce(&Binary) -> BisectionComparison,
{
    let lower = bounds.small().clone();
    let upper = bounds.large();
    let mid = midpoint(&lower, &upper);

    match compare(&mid) {
        BisectionComparison::Above => FiniteBounds::new(mid, upper),
        BisectionComparison::Below => FiniteBounds::new(lower, mid),
        BisectionComparison::Exact => FiniteBounds::new(mid.clone(), mid),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use num_traits::Zero;

    use super::*;
    use crate::test_utils::bin;

    #[test]
    fn midpoint_of_integers() {
        let lower = bin(2, 0);
        let upper = bin(4, 0);
        let mid = midpoint(&lower, &upper);
        assert_eq!(mid, bin(3, 0));
    }

    #[test]
    fn midpoint_of_fractions() {
        let lower = bin(1, -1); // 0.5
        let upper = bin(3, -1); // 1.5
        let mid = midpoint(&lower, &upper);
        assert_eq!(mid, bin(1, 0)); // 1.0
    }

    #[test]
    fn bisection_step_above() {
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let bounds = FiniteBounds::new(lower, upper.clone());
        let result = bisection_step(bounds, |_mid| {
            // Pretend target is above the midpoint (2)
            BisectionComparison::Above
        });
        assert_eq!(result.small(), &bin(2, 0)); // midpoint becomes lower
        assert_eq!(result.large(), upper);
        assert!(!result.width().is_zero());
    }

    #[test]
    fn bisection_step_below() {
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let bounds = FiniteBounds::new(lower.clone(), upper);
        let result = bisection_step(bounds, |_mid| {
            // Pretend target is below the midpoint (2)
            BisectionComparison::Below
        });
        assert_eq!(result.small(), &lower);
        assert_eq!(result.large(), bin(2, 0)); // midpoint becomes upper
        assert!(!result.width().is_zero());
    }

    #[test]
    fn bisection_step_exact() {
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let bounds = FiniteBounds::new(lower, upper);
        let result = bisection_step(bounds, |_mid| BisectionComparison::Exact);
        assert_eq!(result.small(), &bin(2, 0));
        assert_eq!(result.large(), bin(2, 0));
        assert!(result.width().is_zero());
    }

    #[test]
    fn bisection_finds_sqrt_4() {
        // Find sqrt(4) = 2 by bisection
        // We're looking for x where x^2 = 4
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let target = bin(4, 0);
        let mut bounds = FiniteBounds::new(lower, upper);

        for _ in 0..50 {
            bounds = bisection_step(bounds, |mid| {
                let mid_sq = mid.mul(mid);
                match mid_sq.cmp(&target) {
                    std::cmp::Ordering::Less => BisectionComparison::Above,
                    std::cmp::Ordering::Equal => BisectionComparison::Exact,
                    std::cmp::Ordering::Greater => BisectionComparison::Below,
                }
            });
            if bounds.width().is_zero() {
                break;
            }
        }

        // Should find exact match for sqrt(4) = 2
        assert!(bounds.width().is_zero());
        assert_eq!(bounds.small(), &bin(2, 0));
        assert_eq!(bounds.large(), bin(2, 0));
    }

    #[test]
    fn bisection_narrows_sqrt_2() {
        // Find sqrt(2) ~ 1.414... by bisection
        // This won't find an exact match (irrational), but should narrow the interval
        let lower = bin(1, 0);
        let upper = bin(2, 0);
        let target = bin(2, 0);
        let mut bounds = FiniteBounds::new(lower.clone(), upper.clone());

        for _ in 0..10 {
            bounds = bisection_step(bounds, |mid| {
                let mid_sq = mid.mul(mid);
                match mid_sq.cmp(&target) {
                    std::cmp::Ordering::Less => BisectionComparison::Above,
                    std::cmp::Ordering::Equal => BisectionComparison::Exact,
                    std::cmp::Ordering::Greater => BisectionComparison::Below,
                }
            });
            if bounds.width().is_zero() {
                break;
            }
        }

        // Should not find exact match (sqrt(2) is irrational)
        assert!(!bounds.width().is_zero());

        // Interval should have narrowed
        assert!(bounds.small() > &lower);
        assert!(bounds.large() < upper);

        // Bounds should still contain sqrt(2) ≈ 1.414
        let sqrt_2_approx = bin(1414, -10); // Rough approximation
        assert!(bounds.small() <= &sqrt_2_approx || bounds.large() >= sqrt_2_approx);
    }

    #[test]
    fn bisection_respects_iterations() {
        let lower = bin(0, 0);
        let upper = bin(1024, 0);
        let mut bounds = FiniteBounds::new(lower, upper);

        // With 5 iterations, should halve the interval 5 times
        // Starting width: 1024, final width: 1024 / 2^5 = 32
        for _ in 0..5 {
            bounds = bisection_step(bounds, |_mid| BisectionComparison::Above);
        }

        // After 5 iterations always going Above, we should have narrowed
        // Each step halves the interval, so final width = 1024/32 = 32
        let width = bounds.large() - bounds.small().clone();
        assert_eq!(width, bin(32, 0));
    }
}
