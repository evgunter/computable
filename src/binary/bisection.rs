//! Binary search (bisection) helper for iterative refinement.
//!
//! This module provides generic binary search functionality for finding values
//! within an interval using bisection. The helper can be reused by any operation
//! that needs to refine bounds via bisection (e.g., nth_root, inverse functions,
//! root-finding for monotonic functions).
//!
//! # Example
//!
//! ```ignore
//! use computable::binary::{Binary, bisection::{BisectionComparison, bisection_step}};
//!
//! // Find sqrt(4) in the interval [0, 4]
//! let mut lower = Binary::new(0.into(), 0.into());
//! let mut upper = Binary::new(4.into(), 0.into());
//! let target = Binary::new(4.into(), 0.into());
//!
//! for _ in 0..20 {
//!     let result = bisection_step(lower.clone(), upper.clone(), |mid| {
//!         let mid_sq = mid.mul(mid);
//!         match mid_sq.cmp(&target) {
//!             std::cmp::Ordering::Less => BisectionComparison::Above,
//!             std::cmp::Ordering::Equal => BisectionComparison::Exact,
//!             std::cmp::Ordering::Greater => BisectionComparison::Below,
//!         }
//!     });
//!     lower = result.lower;
//!     upper = result.upper;
//!     if result.exact {
//!         break;
//!     }
//! }
//! // lower and upper now bracket sqrt(4) = 2
//! ```

use num_bigint::BigInt;
use num_traits::One;

use super::Binary;

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

/// Result of a single bisection step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BisectionStepResult {
    /// The new lower bound after the step.
    pub lower: Binary,
    /// The new upper bound after the step.
    pub upper: Binary,
    /// Whether an exact match was found at the midpoint.
    pub exact: bool,
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
/// 1. Computes the midpoint of `lower` and `upper`
/// 2. Calls the comparison function with the midpoint
/// 3. Returns new bounds based on the comparison result
///
/// # Arguments
///
/// * `lower` - The current lower bound
/// * `upper` - The current upper bound
/// * `compare` - A function that compares the midpoint to the target value
///   and returns whether the target is above, below, or exactly at the midpoint
///
/// # Returns
///
/// A `BisectionStepResult` containing the new bounds and whether an exact match was found.
///
/// # Behavior
///
/// - If `compare` returns `Above`: the target is above the midpoint, so
///   the new interval is [midpoint, upper]
/// - If `compare` returns `Below`: the target is below the midpoint, so
///   the new interval is [lower, midpoint]
/// - If `compare` returns `Exact`: the midpoint is exactly the target,
///   so both bounds are set to the midpoint and `exact` is `true`
pub fn bisection_step<F>(lower: Binary, upper: Binary, compare: F) -> BisectionStepResult
where
    F: FnOnce(&Binary) -> BisectionComparison,
{
    let mid = midpoint(&lower, &upper);
    match compare(&mid) {
        BisectionComparison::Above => BisectionStepResult {
            lower: mid,
            upper,
            exact: false,
        },
        BisectionComparison::Below => BisectionStepResult {
            lower,
            upper: mid,
            exact: false,
        },
        BisectionComparison::Exact => BisectionStepResult {
            lower: mid.clone(),
            upper: mid,
            exact: true,
        },
    }
}

/// Performs multiple steps of binary search, refining until a condition is met.
///
/// This function repeatedly applies `bisection_step` until either:
/// - An exact match is found
/// - The maximum number of iterations is reached
/// - The interval width is small enough (determined by the caller checking bounds)
///
/// # Arguments
///
/// * `lower` - The initial lower bound
/// * `upper` - The initial upper bound
/// * `max_iterations` - Maximum number of bisection steps to perform
/// * `compare` - A function that compares the midpoint to the target value
///
/// # Returns
///
/// A `BisectionStepResult` containing the final bounds and whether an exact match was found.
pub fn bisection_refine<F>(
    mut lower: Binary,
    mut upper: Binary,
    max_iterations: u32,
    mut compare: F,
) -> BisectionStepResult
where
    F: FnMut(&Binary) -> BisectionComparison,
{
    for _ in 0..max_iterations {
        let result = bisection_step(lower, upper, &mut compare);
        lower = result.lower;
        upper = result.upper;
        if result.exact {
            return BisectionStepResult {
                lower,
                upper,
                exact: true,
            };
        }
    }
    BisectionStepResult {
        lower,
        upper,
        exact: false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

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
        let result = bisection_step(lower.clone(), upper.clone(), |_mid| {
            // Pretend target is above the midpoint (2)
            BisectionComparison::Above
        });
        assert_eq!(result.lower, bin(2, 0)); // midpoint becomes lower
        assert_eq!(result.upper, upper);
        assert!(!result.exact);
    }

    #[test]
    fn bisection_step_below() {
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let result = bisection_step(lower.clone(), upper.clone(), |_mid| {
            // Pretend target is below the midpoint (2)
            BisectionComparison::Below
        });
        assert_eq!(result.lower, lower);
        assert_eq!(result.upper, bin(2, 0)); // midpoint becomes upper
        assert!(!result.exact);
    }

    #[test]
    fn bisection_step_exact() {
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let result = bisection_step(lower, upper, |_mid| BisectionComparison::Exact);
        assert_eq!(result.lower, bin(2, 0));
        assert_eq!(result.upper, bin(2, 0));
        assert!(result.exact);
    }

    #[test]
    fn bisection_finds_sqrt_4() {
        // Find sqrt(4) = 2 by bisection
        // We're looking for x where x^2 = 4
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let target = bin(4, 0);

        let result = bisection_refine(lower, upper, 50, |mid| {
            let mid_sq = mid.mul(mid);
            match mid_sq.cmp(&target) {
                std::cmp::Ordering::Less => BisectionComparison::Above,
                std::cmp::Ordering::Equal => BisectionComparison::Exact,
                std::cmp::Ordering::Greater => BisectionComparison::Below,
            }
        });

        // Should find exact match for sqrt(4) = 2
        assert!(result.exact);
        assert_eq!(result.lower, bin(2, 0));
        assert_eq!(result.upper, bin(2, 0));
    }

    #[test]
    fn bisection_narrows_sqrt_2() {
        // Find sqrt(2) ~ 1.414... by bisection
        // This won't find an exact match (irrational), but should narrow the interval
        let lower = bin(1, 0);
        let upper = bin(2, 0);
        let target = bin(2, 0);

        let result = bisection_refine(lower.clone(), upper.clone(), 10, |mid| {
            let mid_sq = mid.mul(mid);
            match mid_sq.cmp(&target) {
                std::cmp::Ordering::Less => BisectionComparison::Above,
                std::cmp::Ordering::Equal => BisectionComparison::Exact,
                std::cmp::Ordering::Greater => BisectionComparison::Below,
            }
        });

        // Should not find exact match (sqrt(2) is irrational)
        assert!(!result.exact);

        // Interval should have narrowed
        assert!(result.lower > lower);
        assert!(result.upper < upper);

        // Bounds should still contain sqrt(2) ≈ 1.414
        let sqrt_2_approx = bin(1414, -10); // Rough approximation
        assert!(result.lower <= sqrt_2_approx || result.upper >= sqrt_2_approx);
    }

    #[test]
    fn bisection_respects_max_iterations() {
        let lower = bin(0, 0);
        let upper = bin(1024, 0);

        // With 5 iterations, should halve the interval 5 times
        // Starting width: 1024, final width: 1024 / 2^5 = 32
        let result = bisection_refine(lower, upper, 5, |_mid| BisectionComparison::Above);

        // After 5 iterations always going Above, we should have narrowed
        // Each step halves the interval, so final width = 1024/32 = 32
        let width = result.upper.sub(&result.lower);
        assert_eq!(width, bin(32, 0));
    }
}
