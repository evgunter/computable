//! Binary number representations for exact arithmetic.
//!
//! This module provides various binary number types for exact computation:
//!
//! - [`Binary`]: Signed exact binary number `mantissa * 2^exponent`
//! - [`XBinary`]: Extended signed binary with +/-infinity
//! - [`UBinary`]: Unsigned binary number (non-negative)
//! - [`UXBinary`]: Extended unsigned binary with +infinity
//!
//! The module also provides:
//! - [`BinaryError`] and [`XBinaryError`]: Error types for binary operations
//! - [`ReciprocalRounding`]: Rounding modes for reciprocal computation
//! - [`reciprocal_rounded_abs_extended`]: Reciprocal computation function
//!
//! # Architecture
//!
//! The types are organized as two parallel families:
//!
//! ```text
//! Signed:                      Unsigned:
//! Binary (finite)              UBinary (finite)
//!    │                            │
//!    └──► XBinary (±∞)            └──► UXBinary (+∞)
//!
//! Binary ←──────────────────────► UBinary
//!       (conversions: try_from_binary, to_binary)
//! ```
//!
//! - `XBinary` is `enum { NegInf, Finite(Binary), PosInf }`
//! - `UXBinary` is `enum { Finite(UBinary), PosInf }`
//! - `Binary` and `UBinary` are independent structs with different mantissa types
//!
//! All types maintain a canonical representation where the mantissa is odd
//! (unless the value is zero).

mod binary_impl;
mod display;
mod error;
mod reciprocal;
mod shift;
mod shortest;
mod ubinary;
mod uxbinary;
mod xbinary;

// Re-export all public types
pub use binary_impl::Binary;
pub use error::{BinaryError, XBinaryError};
pub use reciprocal::{ReciprocalRounding, reciprocal_of_biguint, reciprocal_rounded_abs_extended};
pub use shortest::{
    margin_from_width, shortest_binary_in_finite_bounds, shortest_xbinary_in_bounds,
    simplify_bounds_if_needed,
};
pub use ubinary::UBinary;
pub use uxbinary::UXBinary;
pub use xbinary::XBinary;

// BigInt/BigUint trait implementations for ordered_pair compatibility
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

use crate::ordered_pair::{AbsDistance, AddWidth, Interval, Unsigned};

/// Bounds on a computable number: lower and upper bounds as XBinary values.
/// The width is stored as UXBinary to guarantee non-negativity through the type system.
///
/// This type enforces the invariant from the formalism that bounds widths are
/// always nonnegative (elements of D_inf where the value is >= 0).
pub type Bounds = Interval<XBinary, UXBinary>;

/// Finite bounds on a value: lower and upper bounds as Binary values.
///
/// Unlike [`Bounds`], this type guarantees that both bounds are finite
/// (no infinities). This is useful for algorithms like bisection that
/// require finite intervals, and for interval arithmetic in computations
/// like pi and sin.
///
// TODO: Investigate code deduplication between FiniteBounds and Bounds. Both types
// are Interval<T, W> with different type parameters and have similar interval arithmetic
// needs. Consider whether the interval_add, interval_sub, interval_neg, scale_positive,
// scale_bigint, midpoint, and comparison methods could be generalized to work on any
// Interval<T, W> where T and W satisfy appropriate trait bounds.
pub type FiniteBounds = Interval<Binary, UBinary>;

//=============================================================================
// Interval arithmetic methods for FiniteBounds
//=============================================================================

impl FiniteBounds {
    /// Creates a point interval [x, x] with zero width.
    pub fn point(x: Binary) -> Self {
        Self::from_lower_and_width(x, UBinary::zero())
    }

    /// Returns the lower bound of the interval.
    ///
    /// This is a convenience alias for `small()`.
    pub fn lo(&self) -> &Binary {
        self.small()
    }

    /// Returns the upper bound of the interval.
    ///
    /// This is a convenience method that computes `lower + width`.
    pub fn hi(&self) -> Binary {
        self.large()
    }

    /// Returns the width of the interval as a Binary (for compatibility with existing code).
    pub fn width_as_binary(&self) -> Binary {
        self.width().to_binary()
    }

    /// Interval addition: [a,b] + [c,d] = [a+c, b+d]
    ///
    /// Width of result = width(self) + width(other)
    pub fn interval_add(&self, other: &Self) -> Self {
        let new_lower = self.lo().add(other.lo());
        let new_width = self.width().add(other.width());
        Self::from_lower_and_width(new_lower, new_width)
    }

    /// Interval subtraction: [a,b] - [c,d] = [a-d, b-c]
    ///
    /// Note: The result lower bound is `self.lo - other.hi`,
    /// and the result upper bound is `self.hi - other.lo`.
    /// Width of result = width(self) + width(other)
    pub fn interval_sub(&self, other: &Self) -> Self {
        // [a, b] - [c, d] = [a - d, b - c]
        // lower = a - d = self.lo - other.hi = self.lo - (other.lo + other.width)
        let other_hi = other.hi();
        let new_lower = self.lo().sub(&other_hi);
        // width = (b - c) - (a - d) = b - c - a + d = (b - a) + (d - c) = width(self) + width(other)
        let new_width = self.width().add(other.width());
        Self::from_lower_and_width(new_lower, new_width)
    }

    /// Interval negation: -[a,b] = [-b, -a]
    pub fn interval_neg(&self) -> Self {
        // -[a, b] = [-b, -a]
        // new_lower = -b = -(a + width) = -a - width
        // new_upper = -a
        // new_width = -a - (-b) = b - a = original width
        let new_lower = self.hi().neg();
        Self::from_lower_and_width(new_lower, self.width().clone())
    }

    /// Interval multiplication by a non-negative scalar k: k * [a,b] = [k*a, k*b]
    pub fn scale_positive(&self, k: &UBinary) -> Self {
        // k * [a, b] = [k*a, k*b]
        // width = k*b - k*a = k * (b - a) = k * width
        let k_binary = k.to_binary();
        let new_lower = self.lo().mul(&k_binary);
        let new_width = self.width().mul(k);
        Self::from_lower_and_width(new_lower, new_width)
    }

    /// Interval multiplication by a BigInt (can be negative).
    ///
    /// If k >= 0: k * [a,b] = [k*a, k*b]
    /// If k < 0: k * [a,b] = [k*b, k*a] = -|k| * [-b, -a] = -|k| * neg([a,b])
    pub fn scale_bigint(&self, k: &BigInt) -> Self {
        use num_traits::Signed;

        let abs_k = UBinary::new(k.magnitude().clone(), BigInt::zero());

        if k.is_negative() {
            // k * [a,b] = -|k| * [a,b] = |k| * (-[a,b]) = |k| * [-b, -a]
            self.interval_neg().scale_positive(&abs_k)
        } else {
            self.scale_positive(&abs_k)
        }
    }

    /// Returns the midpoint of the interval: lo + width/2
    pub fn midpoint(&self) -> Binary {
        let width = self.width().to_binary();
        let half_width = Binary::new(width.mantissa().clone(), width.exponent() - BigInt::one());
        self.lo().add(&half_width)
    }

    /// Checks if this interval contains a point.
    pub fn contains(&self, point: &Binary) -> bool {
        self.lo() <= point && *point <= self.hi()
    }

    /// Checks if this interval is entirely less than another.
    pub fn entirely_less_than(&self, other: &Self) -> bool {
        self.hi() < *other.lo()
    }

    /// Checks if this interval is entirely greater than another.
    pub fn entirely_greater_than(&self, other: &Self) -> bool {
        *self.lo() > other.hi()
    }

    /// Checks if this interval overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        !(self.entirely_less_than(other) || self.entirely_greater_than(other))
    }

    /// Returns the join (smallest enclosing interval) of two intervals.
    ///
    /// `[a, b].join([c, d]) = [min(a, c), max(b, d)]`
    ///
    /// This is the lattice join operation: the smallest interval that contains
    /// both inputs. Note that if the intervals are disjoint, the result includes
    /// points in neither original interval (i.e., this is the convex hull).
    pub fn join(&self, other: &Self) -> Self {
        let min_lo = std::cmp::min(self.lo(), other.lo()).clone();
        let max_hi = std::cmp::max(self.hi(), other.hi());
        Self::new(min_lo, max_hi)
    }

    /// Returns the intersection of two intervals, if non-empty.
    ///
    /// `[a, b].intersection([c, d]) = [max(a, c), min(b, d)]` if non-empty
    ///
    /// Returns `None` if the intervals don't overlap.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.overlaps(other) {
            return None;
        }
        let max_lo = std::cmp::max(self.lo(), other.lo()).clone();
        let min_hi = std::cmp::min(self.hi(), other.hi());
        Some(Self::new(max_lo, min_hi))
    }
}

impl Unsigned for BigUint {}

impl AbsDistance<BigInt, BigUint> for BigInt {
    fn abs_distance(self, other: BigInt) -> BigUint {
        (self - other).magnitude().clone()
    }
}

impl AddWidth<BigInt, BigUint> for BigInt {
    fn add_width(self, width: BigUint) -> Self {
        self + BigInt::from(width)
    }
}

#[cfg(test)]
mod integration_tests {
    //! Integration tests that verify cross-module functionality.

    use super::*;
    use crate::test_utils::{bin, xbin};

    #[test]
    fn bounds_reject_invalid_order() {
        let lower = xbin(1, 0);
        let upper = xbin(-1, 0);
        let result = Bounds::new_checked(lower, upper);
        assert!(result.is_err());
    }

    #[test]
    fn binary_to_ubinary_to_xbinary_roundtrip() {
        let original = bin(7, 3);
        let ubinary = UBinary::try_from_binary(&original).expect("should succeed");
        let back = ubinary.to_binary();
        assert_eq!(original, back);
    }

    #[test]
    fn uxbinary_xbinary_conversion() {
        use num_bigint::BigUint;

        let ub = UBinary::new(BigUint::from(5u32), BigInt::from(2));
        let uxb = UXBinary::Finite(ub);
        let xb = XBinary::from(uxb);

        if let XBinary::Finite(binary) = xb {
            assert_eq!(binary.mantissa(), &BigInt::from(5));
            assert_eq!(binary.exponent(), &BigInt::from(2));
        } else {
            panic!("expected finite value");
        }
    }

    #[test]
    fn bounds_from_lower_and_width_constructs_correctly() {
        let lower = xbin(5, 0);
        let width = UXBinary::Finite(UBinary::new(
            num_bigint::BigUint::from(3u32),
            BigInt::from(0),
        ));

        let bounds = Bounds::from_lower_and_width(lower.clone(), width.clone());

        assert_eq!(bounds.small(), &xbin(5, 0));
        assert_eq!(bounds.width(), &width);
        assert_eq!(bounds.large(), xbin(8, 0));
    }

    #[test]
    fn bounds_from_lower_and_width_matches_new() {
        let lower = xbin(10, 0);
        let upper = xbin(25, 0);

        let via_new = Bounds::new(lower.clone(), upper.clone());
        let via_from_lower_and_width =
            Bounds::from_lower_and_width(lower.clone(), via_new.width().clone());

        assert_eq!(via_new, via_from_lower_and_width);
    }

    #[test]
    fn bounds_from_lower_and_width_zero_width() {
        let lower = xbin(42, 0);
        let width = UXBinary::Finite(UBinary::new(
            num_bigint::BigUint::from(0u32),
            BigInt::from(0),
        ));

        let bounds = Bounds::from_lower_and_width(lower.clone(), width);

        assert_eq!(bounds.small(), &xbin(42, 0));
        assert_eq!(bounds.large(), xbin(42, 0));
    }

    #[test]
    fn finite_bounds_interval_subtraction() {
        // Test that [a,b] - [c,d] = [a-d, b-c]
        let a = FiniteBounds::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteBounds::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = a.interval_sub(&b);
        // [1, 2] - [3, 5] = [1-5, 2-3] = [-4, -1]
        assert_eq!(result.lo(), &bin(-4, 0));
        assert_eq!(result.hi(), bin(-1, 0));
    }

    #[test]
    fn finite_bounds_interval_negation() {
        let a = FiniteBounds::new(bin(1, 0), bin(3, 0)); // [1, 3]
        let neg_a = a.interval_neg(); // [-3, -1]
        assert_eq!(neg_a.lo(), &bin(-3, 0));
        assert_eq!(neg_a.hi(), bin(-1, 0));
    }

    #[test]
    fn finite_bounds_join_overlapping() {
        // Test join of overlapping intervals
        let a = FiniteBounds::new(bin(1, 0), bin(4, 0)); // [1, 4]
        let b = FiniteBounds::new(bin(3, 0), bin(6, 0)); // [3, 6]

        let result = a.join(&b);
        // [1, 4].join([3, 6]) = [1, 6]
        assert_eq!(result.lo(), &bin(1, 0));
        assert_eq!(result.hi(), bin(6, 0));
    }

    #[test]
    fn finite_bounds_join_disjoint() {
        // Test join of disjoint intervals (convex hull)
        let a = FiniteBounds::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteBounds::new(bin(5, 0), bin(7, 0)); // [5, 7]

        let result = a.join(&b);
        // [1, 2].join([5, 7]) = [1, 7] (convex hull)
        assert_eq!(result.lo(), &bin(1, 0));
        assert_eq!(result.hi(), bin(7, 0));
    }

    #[test]
    fn finite_bounds_join_nested() {
        // Test join where one interval contains the other
        let outer = FiniteBounds::new(bin(1, 0), bin(10, 0)); // [1, 10]
        let inner = FiniteBounds::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = outer.join(&inner);
        // [1, 10].join([3, 5]) = [1, 10]
        assert_eq!(result.lo(), &bin(1, 0));
        assert_eq!(result.hi(), bin(10, 0));
    }

    #[test]
    fn finite_bounds_intersection_overlapping() {
        // Test intersection of overlapping intervals
        let a = FiniteBounds::new(bin(1, 0), bin(4, 0)); // [1, 4]
        let b = FiniteBounds::new(bin(3, 0), bin(6, 0)); // [3, 6]

        let result = a.intersection(&b).expect("should overlap");
        // [1, 4] ∩ [3, 6] = [3, 4]
        assert_eq!(result.lo(), &bin(3, 0));
        assert_eq!(result.hi(), bin(4, 0));
    }

    #[test]
    fn finite_bounds_intersection_disjoint() {
        // Test intersection of disjoint intervals
        let a = FiniteBounds::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteBounds::new(bin(5, 0), bin(7, 0)); // [5, 7]

        let result = a.intersection(&b);
        assert!(
            result.is_none(),
            "disjoint intervals should have no intersection"
        );
    }

    #[test]
    fn finite_bounds_intersection_nested() {
        // Test intersection where one interval contains the other
        let outer = FiniteBounds::new(bin(1, 0), bin(10, 0)); // [1, 10]
        let inner = FiniteBounds::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = outer.intersection(&inner).expect("should overlap");
        // [1, 10] ∩ [3, 5] = [3, 5]
        assert_eq!(result.lo(), &bin(3, 0));
        assert_eq!(result.hi(), bin(5, 0));
    }

    #[test]
    fn finite_bounds_intersection_touching() {
        // Test intersection of intervals that touch at a point
        let a = FiniteBounds::new(bin(1, 0), bin(3, 0)); // [1, 3]
        let b = FiniteBounds::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = a.intersection(&b).expect("should touch at a point");
        // [1, 3] ∩ [3, 5] = [3, 3]
        assert_eq!(result.lo(), &bin(3, 0));
        assert_eq!(result.hi(), bin(3, 0));
    }
}
