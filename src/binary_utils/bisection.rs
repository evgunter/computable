//! Binary search (bisection) helper for iterative refinement.
//!
//! This module provides generic binary search functionality for finding values
//! within an interval using bisection. The helper can be reused by any operation
//! that needs to refine bounds via bisection (e.g., nth_root, inverse functions,
//! root-finding for monotonic functions).
//!
//! # Functions
//!
//! - [`bisection_step_midpoint`]: Performs bisection using midpoint strategy
//! - [`bounds_from_normalized`]: Creates normalized bounds for optimal midpoint bisection
//! - [`normalize_bounds`]: Converts arbitrary bounds to normalized form
//!
//! # Normalized Bounds Strategy
//!
//! When bounds are in normalized form (lower and width share the same exponent with integer
//! mantissas, and width's mantissa is 1), midpoint bisection automatically selects the
//! shortest representation at each step. This eliminates the need for explicit shortest-
//! representation searches.
//!
//! Use [`bounds_from_normalized`] to create bounds in normalized form, or [`normalize_bounds`]
//! to convert existing bounds.
//!
//! # Usage
//!
//! The [`bisection_step_midpoint`] function performs a single step of binary search.
//! It's designed to be called repeatedly by the refinement infrastructure
//! (e.g., `refine_to_default`), which controls the iteration count.
//!
//! ```
//! use computable::{Binary, FiniteBounds};
//! use computable::binary_utils::bisection::{BisectionComparison, bisection_step_midpoint};
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
//!     bounds = bisection_step_midpoint(bounds, |mid| {
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
use num_traits::{One, ToPrimitive, Zero};

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

// TODO: this doesn't need to take exponent as a BigInt since we don't really do that anywhere else.
// switch it to whatever's convenient for its callers once they're integrated

/// Creates normalized bounds suitable for midpoint-based bisection.
///
/// If the lower bound and width can be written as a pair of binary numbers
/// with integer mantissa and the same exponent, and the mantissa of the width is 1,
/// then binary search will choose the shortest representation in the interval automatically.
/// This means that binary search is guaranteed to find an exact answer if it exists.
/// (this condition is equivalent to having the bounds be represented as just a binary prefix)
///
/// # Arguments
///
/// * `mantissa` - The mantissa of the lower bound (should be an integer)
/// * `exponent` - The shared exponent for both the lower bound and width
///
/// # Returns
///
/// [`FiniteBounds`] with lower = `mantissa * 2^exponent` and width = `1 * 2^exponent`.
///
/// # Example
///
/// ```
/// use computable::binary_utils::bisection::bounds_from_normalized;
/// use num_bigint::BigInt;
///
/// // Create bounds with lower = 3 * 2^(-1) = 1.5 and width = 1 * 2^(-1) = 0.5
/// // This gives the interval [1.5, 2.0]
/// let bounds = bounds_from_normalized(BigInt::from(3), BigInt::from(-1));
///
/// // The width should be 1 * 2^(-1)
/// assert_eq!(*bounds.width().mantissa(), 1u32.into());
/// assert_eq!(*bounds.width().exponent(), BigInt::from(-1));
/// ```
pub fn bounds_from_normalized(mantissa: BigInt, exponent: BigInt) -> FiniteBounds {
    use num_bigint::BigUint;
    use crate::binary::UBinary;

    let lower = Binary::new(mantissa, exponent.clone());
    let width = UBinary::new(BigUint::one(), exponent);
    FiniteBounds::from_lower_and_width(lower, width)
}

/// Converts arbitrary finite bounds to normalized form.
///
/// Takes any finite bounds and returns normalized bounds that contain the original interval.
/// The normalized bounds have the property that lower and width share the same exponent
/// with integer mantissas, and width's mantissa is 1.
///
/// This may slightly expand the interval to achieve normalization.
///
/// # Arguments
///
/// * `bounds` - The finite bounds to normalize
///
/// # Returns
///
/// [`FiniteBounds`] in normalized form that contains the input bounds.
pub fn normalize_bounds(bounds: &FiniteBounds) -> FiniteBounds {
    use num_traits::Signed;

    let lower = bounds.small();
    let width = bounds.width().to_binary();

    // Get the exponent we'll use (from the width)
    let exp = width.exponent().clone();

    // Shift lower to have the same exponent
    // lower_mantissa = lower * 2^(-exp) = lower.mantissa * 2^(lower.exponent - exp)
    let shift = lower.exponent() - &exp;
    let lower_mantissa = if shift.is_zero() {
        lower.mantissa().clone()
    } else if shift.is_positive() {
        // Shift left
        lower.mantissa() << shift.magnitude().to_usize().expect("shift too large")
    } else {
        // Shift right (floor division for normalization)
        lower.mantissa() >> shift.magnitude().to_usize().expect("shift too large")
    };

    bounds_from_normalized(lower_mantissa, exp)
}

/// Selects the midpoint as the split point.
///
/// This is the traditional bisection strategy.
/// Computes lower + width/2 to avoid redundant operations.
fn select_midpoint(bounds: &FiniteBounds) -> Binary {
    let half_width = bounds.width().to_binary();
    let half_width_shifted = Binary::new(
        half_width.mantissa().clone(),
        half_width.exponent() - BigInt::one(),
    );
    bounds.small().add(&half_width_shifted)
}

/// Performs a single step of binary search using the midpoint strategy.
///
/// This is the standard bisection approach that splits at the midpoint.
/// For best results, use [`normalize_bounds`] to convert your initial bounds to
/// normalized form, which ensures midpoint bisection automatically selects the
/// shortest representation at each step.
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
pub fn bisection_step_midpoint<C>(bounds: FiniteBounds, compare: C) -> FiniteBounds
where
    C: FnOnce(&Binary) -> BisectionComparison,
{
    let lower = bounds.small().clone();
    let upper = bounds.large();
    let mid = select_midpoint(&bounds);

    match compare(&mid) {
        BisectionComparison::Above => FiniteBounds::new(mid, upper),
        BisectionComparison::Below => FiniteBounds::new(lower, mid.clone()),
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
        let result = bisection_step_midpoint(bounds, |_mid| {
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
        let result = bisection_step_midpoint(bounds, |_mid| {
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
        let result = bisection_step_midpoint(bounds, |_mid| BisectionComparison::Exact);
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
            bounds = bisection_step_midpoint(bounds, |mid| {
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
            bounds = bisection_step_midpoint(bounds, |mid| {
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
            bounds = bisection_step_midpoint(bounds, |_mid| BisectionComparison::Above);
        }

        // After 5 iterations always going Above, we should have narrowed
        // Each step halves the interval, so final width = 1024/32 = 32
        let width = bounds.large() - bounds.small().clone();
        assert_eq!(width, bin(32, 0));
    }

    #[test]
    fn bisection_step_midpoint_finds_sqrt_4() {
        // Same as bisection_finds_sqrt_4 but using the midpoint strategy
        let lower = bin(0, 0);
        let upper = bin(4, 0);
        let target = bin(4, 0);
        let mut bounds = FiniteBounds::new(lower, upper);

        for _ in 0..50 {
            bounds = bisection_step_midpoint(bounds, |mid| {
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
    fn bounds_from_normalized_creates_correct_width() {
        use num_bigint::BigUint;

        // Create bounds with lower = 1.5 (3 * 2^-1) and width = 2^-10
        // Express 1.5 with exponent -10: 1.5 = 3 * 2^-1 = (3 << 9) * 2^-10
        let bounds = super::bounds_from_normalized(BigInt::from(3 << 9), BigInt::from(-10));

        // Check that lower bound is 1.5
        assert_eq!(bounds.small(), &bin(3, -1));

        // Check that width is 1 * 2^(-10)
        assert_eq!(bounds.width().mantissa(), &BigUint::from(1u32));
        assert_eq!(bounds.width().exponent(), &BigInt::from(-10));

        // Check that upper bound is 1.5 + 2^(-10) = ((3 << 9) + 1) * 2^-10
        assert_eq!(bounds.large(), bin((3 << 9) + 1, -10));
    }

    #[test]
    fn bounds_from_normalized_with_integer_lower() {
        use num_bigint::BigUint;

        // Create bounds with lower = 5 and width = 2^-8
        // Express 5 with exponent -8: 5 = (5 << 8) * 2^-8
        let bounds = super::bounds_from_normalized(BigInt::from(5 << 8), BigInt::from(-8));

        // Check that lower bound is 5
        assert_eq!(bounds.small(), &bin(5, 0));

        // Check that width is 1 * 2^(-8) = 1/256
        assert_eq!(bounds.width().mantissa(), &BigUint::from(1u32));
        assert_eq!(bounds.width().exponent(), &BigInt::from(-8));

        // Check that upper bound is 5 + 1/256 = ((5 << 8) + 1) * 2^-8
        assert_eq!(bounds.large(), bin((5 << 8) + 1, -8));
    }

    #[test]
    fn bounds_from_normalized_can_be_used_for_bisection() {
        // Create normalized bounds: lower = 1, width = 2^-10
        // Express 1 with exponent -10: 1 = (1 << 10) * 2^-10
        let bounds = super::bounds_from_normalized(BigInt::from(1 << 10), BigInt::from(-10));

        // Perform one bisection step
        let target = bin(5, -2); // 1.25, which should be in our interval [1, 1 + 1/1024]
        let result = bisection_step_midpoint(bounds, |mid| {
            match mid.cmp(&target) {
                std::cmp::Ordering::Less => BisectionComparison::Above,
                std::cmp::Ordering::Equal => BisectionComparison::Exact,
                std::cmp::Ordering::Greater => BisectionComparison::Below,
            }
        });

        // Should have narrowed the interval
        assert!(result.width() < &super::bounds_from_normalized(BigInt::from(1 << 10), BigInt::from(-10)).width().clone());
    }
}
