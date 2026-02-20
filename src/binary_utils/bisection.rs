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
//! - [`normalize_bounds`]: Converts arbitrary bounds to prefix form
//! - [`normalize_finite_to_bounds`]: Converts finite bounds to `Bounds` via prefix normalization
//!
//! # Prefix Bounds Strategy
//!
//! When bounds are in prefix form (lower and width share the same exponent with integer
//! mantissas, and width's mantissa is 1), midpoint bisection automatically selects the
//! shortest representation at each step. This eliminates the need for explicit shortest-
//! representation searches.
//!
//! Use [`PrefixBounds`] and [`bisection_step_normalized`] for the most efficient
//! bisection on normalized bounds, or [`normalize_bounds`] to convert existing bounds.
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
//! let mut bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(2));
//! let target = Binary::new(BigInt::from(4), BigInt::from(0));
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
//!             assert_eq!(mid, Binary::new(BigInt::from(2), BigInt::from(0)));
//!             break;
//!         }
//!     }
//! }
//! ```

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};

use std::cmp::Ordering;

use crate::binary::{Binary, Bounds, FiniteBounds, UXBinary, XBinary};

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
    pub exponent: BigInt,
}

impl PrefixBounds {
    /// Creates new normalized bounds.
    ///
    /// The bounds represent the interval [mantissa * 2^exponent, (mantissa + 1) * 2^exponent].
    pub fn new(mantissa: BigInt, exponent: BigInt) -> Self {
        Self { mantissa, exponent }
    }

    /// Converts to `FiniteBounds`.
    pub fn to_finite_bounds(&self) -> FiniteBounds {
        bounds_from_normalized(self.mantissa.clone(), self.exponent.clone())
    }

    /// Returns the midpoint: (2 * mantissa + 1) * 2^(exponent - 1).
    pub fn midpoint(&self) -> Binary {
        Binary::new(&self.mantissa * 2 + 1, self.exponent.clone() - 1)
    }
}

/// Result of a normalized bisection step.
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

    match compare(&mid) {
        Ordering::Less => {
            // mid < target, so new interval is [mid, upper]
            // mid = (2m + 1) * 2^(e-1), so new mantissa = 2m + 1
            PrefixBisectionResult::Narrowed(PrefixBounds {
                mantissa: &bounds.mantissa * 2 + 1,
                exponent: bounds.exponent.clone() - 1,
            })
        }
        Ordering::Greater => {
            // mid > target, so new interval is [lower, mid]
            // lower at new exponent: m * 2^e = 2m * 2^(e-1), so new mantissa = 2m
            PrefixBisectionResult::Narrowed(PrefixBounds {
                mantissa: &bounds.mantissa * 2,
                exponent: bounds.exponent.clone() - 1,
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
/// let bounds = bounds_from_normalized(BigInt::from(3), BigInt::from(-1));
///
/// // The width should be 1 * 2^(-1)
/// assert_eq!(*bounds.width().mantissa(), 1u32.into());
/// assert_eq!(*bounds.width().exponent(), BigInt::from(-1));
/// ```
pub fn bounds_from_normalized(mantissa: BigInt, exponent: BigInt) -> FiniteBounds {
    use crate::binary::UBinary;
    use num_bigint::BigUint;

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
/// [`Result`] containing [`FiniteBounds`] in normalized form that contains the input bounds,
/// or a [`ComputableError::InfiniteBounds`] if the exponent shift is too large.
///
/// # Errors
///
/// Returns [`ComputableError::InfiniteBounds`] if the exponent shift required for normalization
/// is too large to represent (doesn't fit in `usize`).
pub fn normalize_bounds(
    bounds: &FiniteBounds,
) -> Result<FiniteBounds, crate::error::ComputableError> {
    use num_traits::Signed;

    let lower = bounds.small();
    let width_ubinary = bounds.width();

    // The exponent must be large enough that the width fits in one unit:
    // 2^e >= width, so e >= log2(width)
    // For width = m * 2^exp where m has b bits: log2(width) < exp + b
    // We use e = exp + b to ensure 2^e > width (or 2^e >= width if m is a power of 2)
    //
    // However, if width mantissa is 1 AND lower is representable at width's exponent
    // (i.e., already normalized), we can use that exponent directly.
    use num_bigint::BigUint;
    use num_traits::One;
    let width_bits = width_ubinary.mantissa().bits();

    let is_already_normalized = *width_ubinary.mantissa() == BigUint::one()
        && (lower.exponent() == width_ubinary.exponent() || lower.mantissa().is_zero()); // Zero is compatible with any exponent

    let target_exp = if is_already_normalized {
        width_ubinary.exponent().clone()
    } else {
        // Add 1 extra to account for rounding when flooring the lower bound
        width_ubinary.exponent() + BigInt::from(width_bits as i64) + BigInt::one()
    };

    // Floor the lower bound to this exponent
    // lower_floored = floor(lower / 2^target_exp) * 2^target_exp
    let shift = lower.exponent() - &target_exp;
    let lower_mantissa = if shift.is_zero() {
        lower.mantissa().clone()
    } else if shift.is_positive() {
        // Shift left (no rounding needed)
        let shift_amount = shift
            .magnitude()
            .to_usize()
            .ok_or(crate::error::ComputableError::InfiniteBounds)?;
        lower.mantissa() << shift_amount
    } else {
        // Shift right (floor toward -∞)
        // For negative numbers, arithmetic right shift rounds toward -∞
        // For positive numbers, it also rounds toward -∞ (rounds down)
        let shift_amount = shift
            .magnitude()
            .to_usize()
            .ok_or(crate::error::ComputableError::InfiniteBounds)?;
        lower.mantissa() >> shift_amount
    };

    Ok(bounds_from_normalized(lower_mantissa, target_exp))
}

/// Precision threshold (total mantissa bits of both endpoints) above which
/// normalization to prefix form is applied. Below this threshold, bounds are
/// returned as-is to avoid the ~4x width expansion that normalization entails.
///
/// 64 bits is chosen because:
/// - It's large enough to avoid normalizing coarse early-refinement bounds
/// - It's small enough to prevent significant precision bloat in long refinements
const NORMALIZATION_PRECISION_THRESHOLD: u64 = 64;

/// Normalizes finite bounds to `Bounds`, handling edge cases where prefix form isn't possible.
///
/// Prefix-form intervals `[m×2^e, (m+1)×2^e]` cannot represent:
/// - **Zero-width intervals**: normalization would expand them to width 2^e
/// - **Zero-crossing intervals**: no prefix interval can span zero
///   (for m=-1 the upper is 0; for m=0 the lower is 0)
///
/// Additionally, normalization is skipped when the total mantissa precision is
/// below [`NORMALIZATION_PRECISION_THRESHOLD`] to avoid unnecessary interval
/// expansion during early refinement steps.
///
/// In these cases, the bounds are returned unchanged. This is fine because:
/// - Zero-width bounds are already minimal (no precision to accumulate)
/// - Zero-crossing bounds from sin/etc. have few mantissa bits (e.g., [-1, 1])
/// - Low-precision bounds don't suffer from precision bloat
pub fn normalize_finite_to_bounds(
    bounds: &FiniteBounds,
) -> Result<Bounds, crate::error::ComputableError> {
    use num_traits::Signed;

    let lower_bits = bounds.small().mantissa().magnitude().bits();
    let upper_bits = bounds.large().mantissa().magnitude().bits();
    let total_precision = lower_bits + upper_bits;

    let can_normalize = total_precision > NORMALIZATION_PRECISION_THRESHOLD
        && !bounds.width().mantissa().is_zero()
        && !(bounds.small().mantissa().is_negative() && bounds.large().mantissa().is_positive());

    if can_normalize {
        let normalized = normalize_bounds(bounds)?;
        Ok(Bounds::from_lower_and_width(
            XBinary::Finite(normalized.small().clone()),
            UXBinary::Finite(normalized.width().clone()),
        ))
    } else {
        Ok(Bounds::from_lower_and_width(
            XBinary::Finite(bounds.small().clone()),
            UXBinary::Finite(bounds.width().clone()),
        ))
    }
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
        // Normalized bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(2));
        let result = bisection_step_normalized(&bounds, |_mid| {
            // Pretend mid < target, so search upper half
            Ordering::Less
        });
        // After Less: mantissa = 2*0 + 1 = 1, exponent = 1
        // Bounds become [2, 4]
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.mantissa, BigInt::from(1));
                assert_eq!(new_bounds.exponent, BigInt::from(1));
                let finite = new_bounds.to_finite_bounds();
                assert_eq!(finite.small(), &bin(2, 0));
                assert_eq!(finite.large(), bin(4, 0));
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }

    #[test]
    fn bisection_step_greater() {
        // Normalized bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(2));
        let result = bisection_step_normalized(&bounds, |_mid| {
            // Pretend mid > target, so search lower half
            Ordering::Greater
        });
        // After Greater: mantissa = 2*0 = 0, exponent = 1
        // Bounds become [0, 2]
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.mantissa, BigInt::from(0));
                assert_eq!(new_bounds.exponent, BigInt::from(1));
                let finite = new_bounds.to_finite_bounds();
                assert_eq!(finite.small(), &bin(0, 0));
                assert_eq!(finite.large(), bin(2, 0));
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }

    #[test]
    fn bisection_step_equal() {
        // Normalized bounds [0, 4]: mantissa=0, exponent=2
        let bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(2));
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
        let mut bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(2));

        for _ in 0..50 {
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
        // Normalized bounds [1, 2]: mantissa=1, exponent=0
        let target = bin(2, 0);
        let mut bounds = PrefixBounds::new(BigInt::from(1), BigInt::from(0));
        let initial_lower = bin(1, 0);
        let initial_upper = bin(2, 0);

        for _ in 0..10 {
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
        assert!(finite.large() < initial_upper);

        // Bounds should still contain sqrt(2) ≈ 1.414
        let sqrt_2_approx = bin(1414, -10); // Rough approximation
        assert!(finite.small() <= &sqrt_2_approx || finite.large() >= sqrt_2_approx);
    }

    #[test]
    fn bisection_respects_iterations() {
        // Normalized bounds [0, 1024]: mantissa=0, exponent=10
        let mut bounds = PrefixBounds::new(BigInt::from(0), BigInt::from(10));

        // With 5 iterations, should halve the interval 5 times
        // Starting width: 1024, final width: 1024 / 2^5 = 32
        for _ in 0..5 {
            match bisection_step_normalized(&bounds, |_mid| Ordering::Less) {
                PrefixBisectionResult::Narrowed(new_bounds) => bounds = new_bounds,
                PrefixBisectionResult::Exact(_) => panic!("unexpected exact"),
            }
        }

        // After 5 iterations always going Above, exponent should be 10 - 5 = 5
        // Width = 2^5 = 32
        assert_eq!(bounds.exponent, BigInt::from(5));
        let finite = bounds.to_finite_bounds();
        let width = finite.large() - finite.small().clone();
        assert_eq!(width, bin(32, 0));
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
    fn normalized_bounds_can_be_used_for_bisection() {
        // Create normalized bounds: lower = 1, width = 2^-10
        // Express 1 with exponent -10: 1 = (1 << 10) * 2^-10
        let bounds = PrefixBounds::new(BigInt::from(1 << 10), BigInt::from(-10));

        // Perform one bisection step
        let target = bin(5, -2); // 1.25, which is above the midpoint
        let result = bisection_step_normalized(&bounds, |mid| mid.cmp(&target));

        // Should have narrowed the interval (exponent decreased by 1)
        match result {
            PrefixBisectionResult::Narrowed(new_bounds) => {
                assert_eq!(new_bounds.exponent, BigInt::from(-11));
            }
            PrefixBisectionResult::Exact(_) => panic!("expected Narrowed"),
        }
    }

    #[test]
    fn normalize_bounds_contains_original_simple() {
        use num_bigint::BigUint;

        // Simple case: [1, 2]
        let original = FiniteBounds::new(bin(1, 0), bin(2, 0));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Normalized bounds should contain original bounds
        assert!(normalized.small() <= original.small());
        assert!(normalized.large() >= original.large());

        // Normalized bounds should have unit width mantissa
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
    }

    #[test]
    fn normalize_bounds_contains_original_fractional() {
        use num_bigint::BigUint;

        // Fractional bounds: [0.25, 0.75] = [1 * 2^-2, 3 * 2^-2]
        let original = FiniteBounds::new(bin(1, -2), bin(3, -2));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Normalized bounds should contain original bounds
        assert!(normalized.small() <= original.small());
        assert!(normalized.large() >= original.large());

        // Normalized bounds should have unit width mantissa
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
    }

    #[test]
    fn normalize_bounds_contains_original_mixed_exponents() {
        use num_bigint::BigUint;

        // Bounds with different exponents: [5 * 2^0, 11 * 2^-1] = [5, 5.5]
        let original = FiniteBounds::new(bin(5, 0), bin(11, -1));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Normalized bounds should contain original bounds
        assert!(normalized.small() <= original.small());
        assert!(normalized.large() >= original.large());

        // Normalized bounds should have unit width mantissa
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
    }

    #[test]
    fn normalize_bounds_contains_original_large_mantissas() {
        use num_bigint::BigUint;

        // Bounds with large mantissas: [123 * 2^-5, 125 * 2^-5]
        let original = FiniteBounds::new(bin(123, -5), bin(125, -5));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Normalized bounds should contain original bounds
        assert!(normalized.small() <= original.small());
        assert!(normalized.large() >= original.large());

        // Normalized bounds should have unit width mantissa
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
    }

    #[test]
    fn normalize_bounds_contains_original_negative() {
        use num_bigint::BigUint;

        // Negative bounds: [-3, -1]
        let original = FiniteBounds::new(bin(-3, 0), bin(-1, 0));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Normalized bounds should contain original bounds
        assert!(normalized.small() <= original.small());
        assert!(normalized.large() >= original.large());

        // Normalized bounds should have unit width mantissa
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
    }

    #[test]
    fn normalize_bounds_preserves_already_normalized() {
        use num_bigint::BigUint;

        // Already normalized: [5 * 2^-3, 6 * 2^-3] with width = 1 * 2^-3
        let original = FiniteBounds::new(bin(5, -3), bin(6, -3));
        let normalized = normalize_bounds(&original).expect("normalization failed");

        // Should be exactly equal for already-normalized bounds
        assert_eq!(normalized.small(), original.small());
        assert_eq!(normalized.large(), original.large());
        assert_eq!(normalized.width().mantissa(), &BigUint::from(1u32));
        assert_eq!(normalized.width().exponent(), &BigInt::from(-3));
    }

    #[test]
    fn normalize_bounds_is_idempotent() {
        // Test that normalize_bounds(normalize_bounds(x)) == normalize_bounds(x)

        // Test with various bounds
        let test_cases = vec![
            FiniteBounds::new(bin(1, 0), bin(4, 0)),       // [1, 4]
            FiniteBounds::new(bin(123, -5), bin(125, -5)), // fractional
            FiniteBounds::new(bin(-10, 0), bin(-5, 0)),    // negative
            FiniteBounds::new(bin(7, -2), bin(9, -2)),     // mixed
        ];

        for original in test_cases {
            let normalized_once = normalize_bounds(&original).expect("first normalization failed");
            let normalized_twice =
                normalize_bounds(&normalized_once).expect("second normalization failed");

            // Normalizing twice should give the same result as normalizing once
            assert_eq!(
                normalized_once.small(),
                normalized_twice.small(),
                "Idempotency failed for lower bound of {:?}",
                original
            );
            assert_eq!(
                normalized_once.large(),
                normalized_twice.large(),
                "Idempotency failed for upper bound of {:?}",
                original
            );
        }
    }

    // =========================================================================
    // Tests for normalize_finite_to_bounds
    // =========================================================================

    #[test]
    fn normalize_finite_to_bounds_skips_low_precision() {
        // Bounds with small mantissas (total bits well below the 64-bit threshold)
        // should be returned unchanged.
        let original = FiniteBounds::new(bin(3, 0), bin(5, 0)); // 2 + 3 = 5 bits total
        let result = normalize_finite_to_bounds(&original).expect("should succeed");

        // Result should exactly match the original bounds
        assert_eq!(result.small(), &XBinary::Finite(original.small().clone()));
        assert_eq!(result.large(), XBinary::Finite(original.hi()));
    }

    #[test]
    fn normalize_finite_to_bounds_skips_zero_width() {
        // Zero-width (point) intervals should be returned unchanged regardless of
        // precision, because normalization would expand them.
        let point = bin(1, -100); // very high precision but zero width
        let original = FiniteBounds::point(point.clone());
        let result = normalize_finite_to_bounds(&original).expect("should succeed");

        assert_eq!(result.small(), &XBinary::Finite(original.small().clone()));
        assert_eq!(result.large(), XBinary::Finite(original.hi()));
    }

    #[test]
    fn normalize_finite_to_bounds_skips_zero_crossing() {
        // Intervals that cross zero cannot be represented in prefix form.
        // Use large mantissas to exceed the precision threshold.
        let lo = bin(-((1i64 << 40) + 1), -40); // negative, ~41 bits
        let hi = bin((1i64 << 40) + 1, -40); // positive, ~41 bits
        let original = FiniteBounds::new(lo, hi);

        let result = normalize_finite_to_bounds(&original).expect("should succeed");

        // Should return unchanged because the interval crosses zero
        assert_eq!(result.small(), &XBinary::Finite(original.small().clone()));
        assert_eq!(result.large(), XBinary::Finite(original.hi()));
    }

    #[test]
    fn normalize_finite_to_bounds_normalizes_high_precision() {
        // High-precision positive bounds should be normalized.
        // Create bounds with ~40 bits per endpoint (80 total, > 64 threshold).
        let lo = bin((1i64 << 39) + 1, -50); // ~40 bits mantissa
        let hi = bin((1i64 << 39) + 3, -50); // ~40 bits mantissa
        let original = FiniteBounds::new(lo, hi);

        let result = normalize_finite_to_bounds(&original).expect("should succeed");

        // The normalized result should contain the original bounds
        let result_lo = match result.small() {
            XBinary::Finite(b) => b,
            _ => panic!("expected finite lower"),
        };
        let result_hi = match &result.large() {
            XBinary::Finite(b) => b.clone(),
            _ => panic!("expected finite upper"),
        };

        assert!(
            result_lo <= original.small(),
            "normalized lower {} should be <= original lower {}",
            result_lo,
            original.small()
        );
        assert!(
            result_hi >= original.hi(),
            "normalized upper {} should be >= original upper {}",
            result_hi,
            original.hi()
        );
    }

    #[test]
    fn normalize_finite_to_bounds_normalizes_high_precision_negative() {
        // High-precision negative bounds should also be normalized.
        let lo = bin(-((1i64 << 39) + 3), -50);
        let hi = bin(-((1i64 << 39) + 1), -50);
        let original = FiniteBounds::new(lo, hi);

        let result = normalize_finite_to_bounds(&original).expect("should succeed");

        let result_lo = match result.small() {
            XBinary::Finite(b) => b,
            _ => panic!("expected finite lower"),
        };
        let result_hi = match &result.large() {
            XBinary::Finite(b) => b.clone(),
            _ => panic!("expected finite upper"),
        };

        assert!(
            result_lo <= original.small(),
            "normalized lower {} should be <= original lower {}",
            result_lo,
            original.small()
        );
        assert!(
            result_hi >= original.hi(),
            "normalized upper {} should be >= original upper {}",
            result_hi,
            original.hi()
        );
    }
}
