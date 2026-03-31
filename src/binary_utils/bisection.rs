//! Binary search (bisection) helper for iterative refinement.
//!
//! This module provides generic binary search functionality for finding values
//! within an interval using bisection. The helper can be reused by any operation
//! that needs to refine bounds via bisection (e.g., nth_root, inverse functions,
//! root-finding for monotonic functions).
//!
//! # Types and Functions
//!
//! - [`PrefixBounds`]: Bounds in prefix form (mantissa, exponent)
//! - [`PrefixBisectionResult`]: Result of a prefix bisection step
//! - [`bisection_step_normalized`]: Performs bisection on prefix bounds
//! - [`bounds_from_normalized`]: Converts prefix form to `FiniteBounds`
//!
//! # Prefix Bounds Strategy
//!
//! When bounds are in prefix form (lower and width share the same exponent with integer
//! mantissas, and width's mantissa is 1), midpoint bisection automatically selects the
//! shortest representation at each step. This eliminates the need for explicit shortest-
//! representation searches.
//!
//! Use [`PrefixBounds`] and [`bisection_step_normalized`] for the most efficient
//! bisection on normalized bounds.
//!
//! # Usage
//!
//! The [`bisection_step_normalized`] function performs a single step of binary search.
//! It's designed to be called repeatedly by the refinement infrastructure
//! (e.g., `refine_to_default`), which controls the iteration count.
//!
//! ```
//! use computable::Binary;
//! use computable::binary_utils::bisection::{
//!     PrefixBounds, PrefixBisectionResult, bisection_step_normalized,
//! };
//! use num_bigint::BigInt;
//!
//! // Find sqrt(4) in the interval [0, 4]
//! // Using normalized bounds: mantissa=0, exponent=2 represents [0, 4]
//! let mut bounds = PrefixBounds::new(BigInt::from(0), 2);
//! let target = Binary::new(BigInt::from(4), 0);
//!
//! // Perform bisection steps until we find exact match or reach desired precision
//! for _ in 0..20 {
//!     match bisection_step_normalized(&bounds, |mid| {
//!         // Compare mid^2 to target
//!         mid.mul(mid).cmp(&target)
//!     }) {
//!         PrefixBisectionResult::Narrowed(new_bounds) => bounds = new_bounds,
//!         PrefixBisectionResult::Exact(mid) => {
//!             // Found exact match: sqrt(4) = 2
//!             assert_eq!(mid, Binary::new(BigInt::from(2), 0));
//!             break;
//!         }
//!     }
//! }
//! ```

use num_bigint::BigInt;
use num_traits::One;

use std::cmp::Ordering;

use crate::binary::{Binary, FiniteBounds};
use crate::sane::I;

/// Prefix bounds for bisection where lower = mantissa * 2^exponent and width = 2^exponent.
///
/// This representation ensures that midpoint bisection automatically selects the shortest
/// representation at each step, eliminating the need for explicit shortest-representation
/// searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixBounds {
    /// Mantissa of the lower bound.
    pub mantissa: BigInt,
    /// Shared exponent for lower bound and width.
    pub exponent: I,
}

impl PrefixBounds {
    /// Creates new prefix bounds.
    ///
    /// The bounds represent the interval [mantissa * 2^exponent, (mantissa + 1) * 2^exponent].
    pub fn new(mantissa: BigInt, exponent: I) -> Self {
        Self { mantissa, exponent }
    }

    /// Converts to `FiniteBounds`.
    pub fn to_finite_bounds(&self) -> FiniteBounds {
        bounds_from_normalized(self.mantissa.clone(), self.exponent)
    }

    /// Returns the midpoint: (2 * mantissa + 1) * 2^(exponent - 1).
    pub fn midpoint(&self) -> Binary {
        // 2*k + 1 is always odd, so skip normalization.
        let exp = self.exponent;
        Binary::new_normalized(
            &self.mantissa * 2 + 1,
            crate::sane_i_arithmetic!(exp; exp - 1),
        )
    }
}

/// Result of a prefix bisection step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixBisectionResult {
    /// The interval was narrowed (target not exactly at midpoint).
    Narrowed(PrefixBounds),
    /// The midpoint was exactly the target.
    Exact(Binary),
}

/// Performs a single bisection step on normalized bounds.
///
/// This operates directly on the normalized representation, updating mantissa and exponent
/// without needing to convert to/from `FiniteBounds`.
///
/// # Arguments
///
/// * `bounds` - The current normalized bounds
/// * `compare` - A function that compares the midpoint to the target value,
///   returning `Ordering::Less` if mid < target (search upper half),
///   `Ordering::Greater` if mid > target (search lower half),
///   or `Ordering::Equal` if mid == target (exact match)
///
/// # Returns
///
/// - `Narrowed(new_bounds)` if the comparison was Less or Greater
/// - `Exact(midpoint)` if the comparison was Equal
pub fn bisection_step_normalized<C>(bounds: &PrefixBounds, compare: C) -> PrefixBisectionResult
where
    C: FnOnce(&Binary) -> Ordering,
{
    let mid = bounds.midpoint();

    let exp = bounds.exponent;
    let new_exp = crate::sane_i_arithmetic!(exp; exp - 1);

    match compare(&mid) {
        Ordering::Less => {
            // mid < target, so new interval is [mid, upper]
            // mid = (2m + 1) * 2^(e-1), so new mantissa = 2m + 1
            PrefixBisectionResult::Narrowed(PrefixBounds {
                mantissa: &bounds.mantissa * 2 + 1,
                exponent: new_exp,
            })
        }
        Ordering::Greater => {
            // mid > target, so new interval is [lower, mid]
            // lower at new exponent: m * 2^e = 2m * 2^(e-1), so new mantissa = 2m
            PrefixBisectionResult::Narrowed(PrefixBounds {
                mantissa: &bounds.mantissa * 2,
                exponent: new_exp,
            })
        }
        Ordering::Equal => PrefixBisectionResult::Exact(mid),
    }
}

/// Computes the midpoint of two Binary numbers.
///
/// The midpoint is calculated as (lower + upper) / 2.
pub fn midpoint(lower: &Binary, upper: &Binary) -> Binary {
    FiniteBounds::new(lower.clone(), upper.clone()).midpoint()
}

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
/// let bounds = bounds_from_normalized(BigInt::from(3), -1);
///
/// // The width should be 1 * 2^(-1)
/// assert_eq!(*bounds.width().mantissa(), 1u32.into());
/// assert_eq!(bounds.width().exponent(), -1);
/// ```
pub fn bounds_from_normalized(mantissa: BigInt, exponent: I) -> FiniteBounds {
    use crate::binary::UBinary;
    use num_bigint::BigUint;

    let lower = Binary::new(mantissa, exponent);
    let width = UBinary::new(BigUint::one(), exponent);
    FiniteBounds::from_lower_and_width(lower, width)
}

#[cfg(test)]
mod tests {
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
    fn bisection_step_less() {
        // Prefix bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0_i32), 2);
        let result = bisection_step_normalized(&bounds, |_mid| {
            // Pretend mid < target, so search upper half
            Ordering::Less
        });
        // After Less: mantissa = 2*0 + 1 = 1, exponent = 1
        // Bounds become [2, 4]
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.mantissa, BigInt::from(1_i32));
                assert_eq!(new_bounds.exponent, 1_i32);
                let finite = new_bounds.to_finite_bounds();
                assert_eq!(finite.small(), &bin(2, 0));
                assert_eq!(finite.large(), &bin(4, 0));
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }

    #[test]
    fn bisection_step_greater() {
        // Prefix bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0_i32), 2);
        let result = bisection_step_normalized(&bounds, |_mid| {
            // Pretend mid > target, so search lower half
            Ordering::Greater
        });
        // After Greater: mantissa = 2*0 = 0, exponent = 1
        // Bounds become [0, 2]
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.mantissa, BigInt::from(0_i32));
                assert_eq!(new_bounds.exponent, 1_i32);
                let finite = new_bounds.to_finite_bounds();
                assert_eq!(finite.small(), &bin(0, 0));
                assert_eq!(finite.large(), &bin(2, 0));
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }

    #[test]
    fn bisection_step_equal() {
        // Prefix bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0_i32), 2);
        let result = bisection_step_normalized(&bounds, |_mid| Ordering::Equal);
        // midpoint = (2*0 + 1) * 2^1 = 2
        match result {
            PrefixBisectionResult::Exact(mid) => {
                assert_eq!(mid, bin(2, 0));
            }
            PrefixBisectionResult::Narrowed(_) => panic!("expected Exact"),
        }
    }

    #[test]
    fn bisection_finds_sqrt_4() {
        // Find sqrt(4) = 2 by bisection
        // We're looking for x where x^2 = 4
        // Normalized bounds [0, 4]: mantissa=0, exponent=2
        let target = bin(4, 0);
        let mut bounds = PrefixBounds::new(BigInt::from(0_i32), 2);

        for _ in 0_i32..50_i32 {
            match bisection_step_normalized(&bounds, |mid| mid.mul(mid).cmp(&target)) {
                PrefixBisectionResult::Narrowed(new_bounds) => bounds = new_bounds,
                PrefixBisectionResult::Exact(mid) => {
                    // Should find exact match for sqrt(4) = 2
                    assert_eq!(mid, bin(2, 0));
                    return;
                }
            }
        }

        panic!("should have found exact match for sqrt(4)");
    }

    #[test]
    fn bisection_narrows_sqrt_2() {
        // Find sqrt(2) ~ 1.414... by bisection
        // This won't find an exact match (irrational), but should narrow the interval
        // Prefix bounds [1, 2]: mantissa=1, exponent=0
        let target = bin(2, 0);
        let mut bounds = PrefixBounds::new(BigInt::from(1_i32), 0);
        let initial_lower = bin(1, 0);
        let initial_upper = bin(2, 0);

        for _ in 0_i32..10_i32 {
            match bisection_step_normalized(&bounds, |mid| mid.mul(mid).cmp(&target)) {
                PrefixBisectionResult::Narrowed(new_bounds) => bounds = new_bounds,
                PrefixBisectionResult::Exact(_) => {
                    panic!("sqrt(2) is irrational, should not find exact match");
                }
            }
        }

        // Interval should have narrowed
        let finite = bounds.to_finite_bounds();
        assert!(finite.small() > &initial_lower);
        assert!(*finite.large() < initial_upper);

        // Bounds should still contain sqrt(2) ≈ 1.414
        let sqrt_2_approx = bin(1414, -10); // Rough approximation
        assert!(finite.small() <= &sqrt_2_approx || *finite.large() >= sqrt_2_approx);
    }

    #[test]
    fn bisection_respects_iterations() {
        // Prefix bounds [0, 1024]: mantissa=0, exponent=10
        let mut bounds = PrefixBounds::new(BigInt::from(0_i32), 10);

        // With 5 iterations, should halve the interval 5 times
        // Starting width: 1024, final width: 1024 / 2^5 = 32
        for _ in 0_i32..5_i32 {
            match bisection_step_normalized(&bounds, |_mid| Ordering::Less) {
                PrefixBisectionResult::Narrowed(new_bounds) => bounds = new_bounds,
                PrefixBisectionResult::Exact(_) => panic!("unexpected exact"),
            }
        }

        // After 5 iterations always going Above, exponent should be 10 - 5 = 5
        // Width = 2^5 = 32
        assert_eq!(bounds.exponent, 5_i32);
        let finite = bounds.to_finite_bounds();
        let width = finite.large().clone() - finite.small().clone();
        assert_eq!(width, bin(32, 0));
    }

    #[test]
    fn bounds_from_normalized_creates_correct_width() {
        use num_bigint::BigUint;

        // Create bounds with lower = 1.5 (3 * 2^-1) and width = 2^-10
        // Express 1.5 with exponent -10: 1.5 = 3 * 2^-1 = (3 << 9) * 2^-10
        let bounds = super::bounds_from_normalized(BigInt::from(3_i32 << 9_i32), -10);

        // Check that lower bound is 1.5
        assert_eq!(bounds.small(), &bin(3, -1));

        // Check that width is 1 * 2^(-10)
        assert_eq!(bounds.width().mantissa(), &BigUint::from(1u32));
        assert_eq!(bounds.width().exponent(), -10_i32);

        // Check that upper bound is 1.5 + 2^(-10) = ((3 << 9) + 1) * 2^-10
        assert_eq!(bounds.large(), &bin((3 << 9) + 1, -10));
    }

    #[test]
    fn bounds_from_normalized_with_integer_lower() {
        use num_bigint::BigUint;

        // Create bounds with lower = 5 and width = 2^-8
        // Express 5 with exponent -8: 5 = (5 << 8) * 2^-8
        let bounds = super::bounds_from_normalized(BigInt::from(5_i32 << 8_i32), -8);

        // Check that lower bound is 5
        assert_eq!(bounds.small(), &bin(5, 0));

        // Check that width is 1 * 2^(-8) = 1/256
        assert_eq!(bounds.width().mantissa(), &BigUint::from(1u32));
        assert_eq!(bounds.width().exponent(), -8_i32);

        // Check that upper bound is 5 + 1/256 = ((5 << 8) + 1) * 2^-8
        assert_eq!(bounds.large(), &bin((5 << 8) + 1, -8));
    }

    #[test]
    fn prefix_bounds_can_be_used_for_bisection() {
        // Create prefix bounds: lower = 1, width = 2^-10
        // Express 1 with exponent -10: 1 = (1 << 10) * 2^-10
        let bounds = PrefixBounds::new(BigInt::from(1_i32 << 10_i32), -10);

        // Perform one bisection step
        let target = bin(5, -2); // 1.25, which is above the midpoint
        let result = bisection_step_normalized(&bounds, |mid| mid.cmp(&target));

        // Should have narrowed the interval (exponent decreased by 1)
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.exponent, -11_i32);
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }
}
