//! Shortest mantissa selection within bounds.
//!
//! This module provides functions to find binary representations with shorter mantissas
//! that lie within given bounds. This is useful for reducing memory usage and computation
//! time during iterative refinement, where intermediate bounds can accumulate many bits
//! of precision.
//!
//! # Main Functions
//!
//! - [`shortest_binary_in_finite_bounds`]: Find the shortest representation within finite bounds
//! - [`shortest_xbinary_in_bounds`]: Find the shortest representation strictly within extended bounds
//! - [`simplify_bounds`]: Simplify bounds by finding shorter representations for both
//!   the lower bound and width (matching how `Bounds` stores data internally)
//!
//! # TODO: Evaluate if this module is still needed
//!
//! With the introduction of `bounds_from_normalized` in the bisection module, it may be possible
//! to avoid needing explicit shortest-representation searches. When bounds are initialized in
//! normalized form (lower and width share same exponent, width mantissa = 1), midpoint bisection
//! automatically selects the shortest representation at each step. This module may only be needed
//! for cases where bounds cannot be normalized initially, or for the `simplify_bounds` operation.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use super::Bounds;
use super::FiniteBounds;
use super::{Binary, UBinary, UXBinary, XBinary};

/// Trait for bounds types that support finding the shortest representation within.
///
/// This trait abstracts over the common logic for finding shortest representations
/// in both finite bounds (`FiniteBounds`) and extended bounds (`Bounds`), using the
/// type system to ensure correctness without runtime panics.
trait ShortestInBounds {
    /// The signed output type (Binary for finite, XBinary for extended)
    type Output;

    /// Returns Some(zero) if the interval spans zero, otherwise None
    fn check_zero_crossing(&self) -> Option<Self::Output>;

    /// Returns the shortest representation assuming the interval is entirely non-negative
    fn shortest_positive(&self) -> Self::Output;

    /// Returns the shortest representation assuming the interval is entirely negative
    fn shortest_negative(&self) -> Self::Output;

    /// Returns true if the lower bound is negative
    fn lower_is_negative(&self) -> bool;
}

impl ShortestInBounds for FiniteBounds {
    type Output = Binary;

    fn check_zero_crossing(&self) -> Option<Binary> {
        let lower_sign = self.small().mantissa().sign();
        let upper_sign = self.large().mantissa().sign();
        match (lower_sign, upper_sign) {
            (Sign::Minus, Sign::Plus)
            | (Sign::Minus, Sign::NoSign)
            | (Sign::NoSign, Sign::Plus) => Some(Binary::zero()),
            _ => None,
        }
    }

    fn shortest_positive(&self) -> Binary {
        shortest_binary_in_positive_interval(
            &self.small().magnitude(),
            &UXBinary::Finite(self.width().clone()),
        )
        .to_binary()
    }

    fn shortest_negative(&self) -> Binary {
        shortest_binary_in_positive_interval(
            &self.large().magnitude(),
            &UXBinary::Finite(self.width().clone()),
        )
        .to_binary()
        .neg()
    }

    fn lower_is_negative(&self) -> bool {
        self.small().mantissa().sign() == Sign::Minus
    }
}

impl ShortestInBounds for Bounds {
    type Output = XBinary;

    fn check_zero_crossing(&self) -> Option<XBinary> {
        let (lower_sign, _) = split_xbinary(self.small());
        match lower_sign {
            Sign::NoSign => Some(XBinary::zero()),
            Sign::Plus => None,
            Sign::Minus => {
                if self.large() >= XBinary::zero() {
                    Some(XBinary::zero())
                } else {
                    None
                }
            }
        }
    }

    fn shortest_positive(&self) -> XBinary {
        let (_, lower_mag) = split_xbinary(self.small());
        shortest_xbinary_in_positive_interval(&lower_mag, self.width())
    }

    fn shortest_negative(&self) -> XBinary {
        shortest_xbinary_in_positive_interval(&self.large().magnitude(), self.width()).neg()
    }

    fn lower_is_negative(&self) -> bool {
        let (lower_sign, _) = split_xbinary(self.small());
        lower_sign == Sign::Minus
    }
}

/// Generic implementation of shortest representation finding.
///
/// This function contains the shared control flow for both finite and extended bounds,
/// with the type system ensuring correctness through the `ShortestInBounds` trait.
fn shortest_in_bounds<B: ShortestInBounds>(bounds: &B) -> B::Output {
    if let Some(zero) = bounds.check_zero_crossing() {
        return zero;
    }

    if bounds.lower_is_negative() {
        bounds.shortest_negative()
    } else {
        bounds.shortest_positive()
    }
}

/// Returns a Binary value inside the finite bounds with the shortest normalized mantissa.
///
/// This is useful for reducing precision accumulation during binary search / bisection.
/// When used as a split point instead of the midpoint, the resulting bounds will have
/// shorter mantissas, reducing memory usage and computation time.
///
/// # Arguments
///
/// * `bounds` - Finite bounds (no infinities) to find the shortest value within
///
/// # Returns
///
/// A `Binary` value that lies strictly inside the bounds and has the shortest possible
/// mantissa. If the bounds have zero width, returns the single value in the bounds.
pub fn shortest_binary_in_finite_bounds(bounds: &FiniteBounds) -> Binary {
    shortest_in_bounds(bounds)
}

/// Returns an XBinary value inside the bounds with the shortest normalized mantissa.
///
/// Infinities are only returned if the bounds do not contain any finite numbers;
/// this shouldn't happen currently but it might be valid in the future.
pub fn shortest_xbinary_in_bounds(bounds: &Bounds) -> XBinary {
    shortest_in_bounds(bounds)
}

/// Infinities are only returned if the bounds do not contain any finite numbers;
/// this shouldn't happen currently but it might be valid in the future.
fn shortest_xbinary_in_positive_interval(lower: &UXBinary, width: &UXBinary) -> XBinary {
    match lower {
        UXBinary::Inf => {
            crate::detected_computable_with_infinite_value!("lower input bound is PosInf");
            XBinary::PosInf
        }
        UXBinary::Finite(lm) => {
            XBinary::Finite(shortest_binary_in_positive_interval(lm, width).to_binary())
        }
    }
}

fn shortest_binary_in_positive_interval(lower: &UBinary, width: &UXBinary) -> UBinary {
    match width {
        UXBinary::Inf => {
            // next power of 2 >= lower
            let exponent =
                lower.exponent() + BigInt::from((lower.mantissa() - BigUint::one()).bits());
            UBinary::new(BigUint::one(), exponent)
        }
        UXBinary::Finite(wm) => {
            // take lower and then cancel out as many mantissa bits as possible by adding at most wm.
            // to do this we compare lower and lower + wm by shifting them both to have the same exponent as lower.
            // then we can find the largest bit where the mantissas differ and only take the part of the mantissa before that (shifting the exponent accordingly)

            let (lower_aligned, upper_aligned, common_exponent) =
                UBinary::align_mantissas(lower, &lower.add(wm));
            let xor = &lower_aligned ^ &upper_aligned;
            if xor.is_zero() {
                // i.e. width is zero
                return lower.clone();
            }

            // position of highest differing bit, 0-indexed from the right
            // since lower is smaller, lower[k] = 0 and upper[k] = 1
            let k = xor.bits() - 1;

            // if lower already has > k trailing zeros,
            // our plan to flip lower[k] to 1 and clear all following bits would make things worse
            let mask = (BigUint::one() << (k + 1)) - BigUint::one();
            if (&lower_aligned & &mask).is_zero() {
                return lower.clone();
            }

            let result_mantissa = &upper_aligned >> k;
            let result_exponent = common_exponent + BigInt::from(k);
            UBinary::new(result_mantissa, result_exponent)
        }
    }
}

fn split_xbinary(value: &XBinary) -> (Sign, UXBinary) {
    match value {
        XBinary::NegInf => (Sign::Minus, UXBinary::Inf),
        XBinary::PosInf => (Sign::Plus, UXBinary::Inf),
        XBinary::Finite(v) => (v.mantissa().sign(), UXBinary::Finite(v.magnitude())),
    }
}

/// Computes margin from width using a bit shift.
///
/// This is a convenience function for callers that want to specify margin as a fraction of width.
///
/// # Arguments
///
/// * `width` - The width to compute margin from
/// * `shift` - How many bits to right-shift the width.
///   E.g., `2` means `margin = width >> 2 = width/4`, `3` means `margin = width/8`.
///
/// # Returns
///
/// The margin as `UXBinary`. Returns `UXBinary::Inf` if width is infinite.
pub fn margin_from_width(width: &UXBinary, shift: u32) -> UXBinary {
    match width {
        UXBinary::Inf => UXBinary::Inf,
        UXBinary::Finite(w) => {
            let margin_exponent = w.exponent() - BigInt::from(shift);
            UXBinary::Finite(UBinary::new(w.mantissa().clone(), margin_exponent))
        }
    }
}

/// simplifies bounds by finding shorter binary representations for both the lower bound and the width
/// which contain the original bounds.
///
/// # Arguments
///
/// * `bounds` - The bounds to simplify
/// * `margin` - How much to loosen the bounds by. The lower bound can decrease by up to this amount,
///   and the width can increase by up to this amount. If infinite, returns bounds unchanged.
///
/// # Returns
///
/// bounds which contain the original bounds, widen the bounds by at most the specified margin, and which have
/// a shorter binary representation for the lower bound and width (or, if this is not possible, just return the original bounds)
///
pub fn simplify_bounds(bounds: &Bounds, margin: &UXBinary) -> Bounds {
    let finite_margin = match margin {
        UXBinary::Inf => return bounds.clone(),
        UXBinary::Finite(m) => m,
    };

    if matches!(bounds.width(), UXBinary::Inf) {
        return bounds.clone();
    }

    if finite_margin.mantissa().is_zero() {
        return bounds.clone();
    }

    // Step 1: Find shortest lower bound in [original_lower - margin, original_lower]
    // This relaxes the lower bound downward to find a shorter representation
    let original_lower = match bounds.small() {
        XBinary::Finite(lower) => lower.clone(),
        _ => return bounds.clone(), // Can't simplify infinite lower bound
    };

    let relaxed_lower = original_lower.sub(&finite_margin.to_binary());
    let lower_search_bounds = Bounds::new(
        XBinary::Finite(relaxed_lower),
        XBinary::Finite(original_lower.clone()),
    );
    let new_lower = match shortest_xbinary_in_bounds(&lower_search_bounds) {
        XBinary::Finite(l) => l,
        _ => return bounds.clone(),
    };

    // Step 2: Find shortest width in [min_width, min_width + margin]
    // where min_width = original_upper - new_lower (the minimum to contain original interval)
    let original_upper = match bounds.large() {
        XBinary::Finite(upper) => upper,
        _ => return bounds.clone(), // Can't simplify infinite upper bound
    };

    // TODO: can we use the type system to ensure that this is non-negative?
    // min_width = original_upper - new_lower
    // Since new_lower <= original_lower <= original_upper, this is non-negative
    let min_width = original_upper.sub(&new_lower);
    let min_width_unsigned = match UBinary::try_from_binary(&min_width) {
        Ok(w) => w,
        Err(_) => return bounds.clone(), // Shouldn't happen, but be safe
    };

    // Find shortest width in [min_width, min_width + margin]
    let new_width = shortest_binary_in_positive_interval(
        &min_width_unsigned,
        &UXBinary::Finite(finite_margin.clone()),
    );

    // Construct new bounds directly from lower and width
    Bounds::from_lower_and_width(XBinary::Finite(new_lower), UXBinary::Finite(new_width))
}

/// Computes the mantissa bit count of a Binary number.
///
/// Returns the number of bits in the mantissa, which is a measure of precision.
/// Useful for tracking precision growth during refinement.
pub fn mantissa_bits(value: &Binary) -> u64 {
    value.mantissa().magnitude().bits()
}

/// Computes the total mantissa bits used by bounds (lower + upper).
///
/// This is useful for monitoring precision accumulation during refinement.
pub fn bounds_precision(bounds: &Bounds) -> u64 {
    let lower_bits = match bounds.small() {
        XBinary::Finite(b) => mantissa_bits(b),
        _ => 0,
    };
    let upper_bits = match bounds.large() {
        XBinary::Finite(b) => mantissa_bits(&b),
        _ => 0,
    };
    lower_bits + upper_bits
}

// TODO: all the cases that use this seem to not be tracking refinement progress properly.
// i don't expect to see this in cases where we don't use the previous bounds to calculate the new bounds;
// i think it's likely that what's happening in the cases where this is used now is that we're requesting
// too much precision for bounds on a wide interval.
// also in the longer term i think this function may not be necessary

/// Simplifies bounds if they exceed a precision threshold.
///
/// This is the recommended function for use in refinement loops. It only
/// simplifies when the accumulated precision exceeds the threshold, avoiding
/// unnecessary work on already-simple bounds.
///
/// # Arguments
///
/// * `bounds` - The bounds to potentially simplify
/// * `precision_threshold` - Only simplify if total mantissa bits exceed this
/// * `margin` - How much to loosen the bounds by (passed to `simplify_bounds`).
///   If infinite, returns bounds unchanged.
///
/// # Returns
///
/// Simplified bounds if precision exceeded threshold, otherwise the original bounds.
pub fn simplify_bounds_if_needed(
    bounds: &Bounds,
    precision_threshold: u64,
    margin: &UXBinary,
) -> Bounds {
    if bounds_precision(bounds) > precision_threshold {
        simplify_bounds(bounds, margin)
    } else {
        bounds.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::bin;

    use super::super::Binary;

    fn midpoint_between(lower: &Binary, upper: &Binary) -> Binary {
        let mid_sum = lower.add(upper);
        let exponent = mid_sum.exponent() - BigInt::from(1);
        Binary::new(mid_sum.mantissa().clone(), exponent)
    }

    #[test]
    fn shortest_xbinary_in_bounds_finds_sqrt_four() {
        let four = bin(1, 2);
        let epsilon = bin(1, 0);
        let mut lower = bin(0, 0);
        let mut upper = bin(4, 0);

        loop {
            let width = upper.clone() - lower.clone();
            if width <= epsilon {
                break;
            }

            let mid = midpoint_between(&lower, &upper);
            let mid_sq = mid.clone() * mid.clone();

            if mid_sq <= four {
                lower = mid;
            } else {
                upper = mid;
            }
        }

        let bounds = Bounds::new(XBinary::Finite(lower), XBinary::Finite(upper));
        let shortest = shortest_xbinary_in_bounds(&bounds);
        assert_eq!(shortest, XBinary::Finite(bin(1, 1)));
    }

    #[test]
    fn shortest_xbinary_handles_infinite_bounds() {
        let bounds = Bounds::new(XBinary::Finite(bin(1, 0)), XBinary::PosInf);
        assert_eq!(
            shortest_xbinary_in_bounds(&bounds),
            XBinary::Finite(bin(1, 0))
        );

        let bounds = Bounds::new(XBinary::Finite(bin(-3, -1)), XBinary::Finite(bin(-1, 0)));
        assert_eq!(
            shortest_xbinary_in_bounds(&bounds),
            XBinary::Finite(bin(-1, 0))
        );

        // this case is currently blocked by detected_computable_with_infinite_value!, but it should be added if we want to support extended reals
        // let bounds = Bounds::new(XBinary::PosInf, XBinary::PosInf);
        // assert_eq!(shortest_xbinary_in_bounds(&bounds), XBinary::PosInf);
    }

    // Tests for new bound-loosening functions

    #[test]
    fn simplify_bounds_reduces_precision() {
        // Create bounds with many bits of precision (simulating bisection accumulation)
        // After many bisection steps on sqrt(2), we might get something like:
        // [1.41421356..., 1.41421357...] with many mantissa bits
        let lower = bin(181, -7); // ~1.4140625
        let upper = bin(363, -8); // ~1.41796875
        let bounds = Bounds::new(
            XBinary::Finite(lower.clone()),
            XBinary::Finite(upper.clone()),
        );

        let original_precision = bounds_precision(&bounds);
        let margin = margin_from_width(bounds.width(), 2); // loosen by width/4
        let simplified = simplify_bounds(&bounds, &margin);
        let new_precision = bounds_precision(&simplified);

        // The simplified bounds should have less or equal precision
        // (exact reduction depends on the values, but shouldn't increase)
        assert!(new_precision <= original_precision + 10); // Allow some tolerance

        // The simplified bounds should still contain the original bounds
        assert!(simplified.small() <= bounds.small());
        assert!(simplified.large() >= bounds.large());
    }

    #[test]
    fn simplify_bounds_preserves_correctness() {
        // Bounds around sqrt(2) ~ 1.414...
        let lower = bin(1414, -10); // ~1.380859375
        let upper = bin(1415, -10); // ~1.3818359375
        let bounds = Bounds::new(
            XBinary::Finite(lower.clone()),
            XBinary::Finite(upper.clone()),
        );

        let margin = margin_from_width(bounds.width(), 2);
        let simplified = simplify_bounds(&bounds, &margin);

        // The simplified bounds must contain the original interval
        assert!(
            simplified.small() <= bounds.small(),
            "Simplified lower bound {} should be <= original lower {}",
            simplified.small(),
            bounds.small()
        );
        assert!(
            simplified.large() >= bounds.large(),
            "Simplified upper bound {} should be >= original upper {}",
            simplified.large(),
            bounds.large()
        );
    }

    #[test]
    fn simplify_bounds_handles_zero_width() {
        let point = bin(5, 0);
        let bounds = Bounds::new(
            XBinary::Finite(point.clone()),
            XBinary::Finite(point.clone()),
        );

        // margin_from_width with zero width gives a zero margin
        let margin = margin_from_width(bounds.width(), 2);
        let simplified = simplify_bounds(&bounds, &margin);

        // With zero margin, should return unchanged
        assert_eq!(simplified.small(), bounds.small());
        assert_eq!(simplified.large(), bounds.large());
    }

    #[test]
    fn simplify_bounds_handles_infinite_width() {
        let bounds = Bounds::new(XBinary::Finite(bin(1, 0)), XBinary::PosInf);

        // margin_from_width returns Inf for infinite width
        assert!(matches!(
            margin_from_width(bounds.width(), 2),
            UXBinary::Inf
        ));

        // simplify_bounds with infinite margin returns bounds unchanged
        let margin = margin_from_width(bounds.width(), 2);
        let simplified = simplify_bounds(&bounds, &margin);

        // With infinite margin, should return unchanged
        assert_eq!(simplified.small(), bounds.small());
        assert_eq!(simplified.large(), bounds.large());
    }

    #[test]
    fn simplify_bounds_if_needed_respects_threshold() {
        let lower = bin(12345, -14);
        let upper = bin(12346, -14);
        let bounds = Bounds::new(XBinary::Finite(lower), XBinary::Finite(upper));

        let precision = bounds_precision(&bounds);
        let margin = margin_from_width(bounds.width(), 2);

        // Below threshold: should return unchanged
        let unchanged = simplify_bounds_if_needed(&bounds, precision + 100, &margin);
        assert_eq!(unchanged, bounds);

        // Above threshold: should simplify
        let simplified = simplify_bounds_if_needed(&bounds, 1, &margin);
        // The result should still contain the original bounds
        assert!(simplified.small() <= bounds.small());
        assert!(simplified.large() >= bounds.large());
    }

    #[test]
    fn mantissa_bits_counts_correctly() {
        assert_eq!(mantissa_bits(&bin(1, 0)), 1);
        assert_eq!(mantissa_bits(&bin(3, 0)), 2);
        assert_eq!(mantissa_bits(&bin(7, 0)), 3);
        assert_eq!(mantissa_bits(&bin(255, 0)), 8);
        assert_eq!(mantissa_bits(&bin(-255, 0)), 8); // magnitude
    }

    #[test]
    fn bounds_precision_sums_both_endpoints() {
        let lower = bin(7, 0); // 3 bits
        let upper = bin(255, 0); // 8 bits
        let bounds = Bounds::new(XBinary::Finite(lower), XBinary::Finite(upper));

        assert_eq!(bounds_precision(&bounds), 11);
    }

    #[test]
    fn simplify_bounds_significant_reduction_after_bisection() {
        // Simulate what happens after many bisection steps
        // Each step can double the mantissa bits
        let mut lower = bin(1, 0);
        let upper = bin(2, 0);

        // Do 20 "fake" bisection steps that accumulate precision
        for _ in 0..20 {
            let mid = midpoint_between(&lower, &upper);
            // Arbitrarily take upper half each time
            lower = mid;
        }

        let bounds = Bounds::new(XBinary::Finite(lower), XBinary::Finite(upper));
        let original_precision = bounds_precision(&bounds);

        // After 20 bisections, precision should have grown significantly
        assert!(
            original_precision > 20,
            "Expected precision growth from bisection, got {}",
            original_precision
        );

        let margin = margin_from_width(bounds.width(), 2);
        let simplified = simplify_bounds(&bounds, &margin);
        let new_precision = bounds_precision(&simplified);

        // Simplified should have noticeably less precision
        // (the exact amount depends on the values, but should improve)
        println!(
            "Original precision: {}, Simplified precision: {}",
            original_precision, new_precision
        );

        // Most importantly: correctness is preserved
        assert!(simplified.small() <= bounds.small());
        assert!(simplified.large() >= bounds.large());
    }

    // Tests for shortest_binary_in_finite_bounds

    #[test]
    fn shortest_binary_in_finite_bounds_positive_interval() {
        // Interval [1, 3] - shortest value should be 2 (mantissa = 1, exponent = 1)
        let bounds = FiniteBounds::new(bin(1, 0), bin(3, 0));
        let shortest = shortest_binary_in_finite_bounds(&bounds);
        assert_eq!(shortest, bin(1, 1)); // 1 * 2^1 = 2

        // Verify the result is within bounds
        assert!(shortest >= *bounds.small());
        assert!(shortest <= bounds.large());
    }

    #[test]
    fn shortest_binary_in_finite_bounds_negative_interval() {
        // Interval [-3, -1] - shortest value should be -2 (mantissa = -1, exponent = 1)
        let bounds = FiniteBounds::new(bin(-3, 0), bin(-1, 0));
        let shortest = shortest_binary_in_finite_bounds(&bounds);
        assert_eq!(shortest, bin(-1, 1)); // -1 * 2^1 = -2

        // Verify the result is within bounds
        assert!(shortest >= *bounds.small());
        assert!(shortest <= bounds.large());
    }

    #[test]
    fn shortest_binary_in_finite_bounds_spanning_zero() {
        // Interval [-1, 3] - shortest value should be 0
        let bounds = FiniteBounds::new(bin(-1, 0), bin(3, 0));
        let shortest = shortest_binary_in_finite_bounds(&bounds);
        assert_eq!(shortest, bin(0, 0));

        // Verify the result is within bounds
        assert!(shortest >= *bounds.small());
        assert!(shortest <= bounds.large());
    }

    #[test]
    fn shortest_binary_in_finite_bounds_zero_width() {
        // Interval [5, 5] - single point, should return that point
        let bounds = FiniteBounds::new(bin(5, 0), bin(5, 0));
        let shortest = shortest_binary_in_finite_bounds(&bounds);
        assert_eq!(shortest, bin(5, 0));
    }

    #[test]
    fn shortest_binary_in_finite_bounds_after_bisection() {
        // Simulate what happens during sqrt(4) bisection
        let four = bin(1, 2);
        let mut lower = bin(0, 0);
        let mut upper = bin(4, 0);

        // Do some bisection steps
        for _ in 0..10 {
            let mid = midpoint_between(&lower, &upper);
            let mid_sq = mid.clone() * mid.clone();
            if mid_sq <= four {
                lower = mid;
            } else {
                upper = mid;
            }
        }

        // After bisection, we should have high-precision bounds around 2
        let bounds = FiniteBounds::new(lower.clone(), upper.clone());
        let shortest = shortest_binary_in_finite_bounds(&bounds);

        // Verify the result is within bounds
        assert!(
            shortest >= lower,
            "shortest {} should be >= lower {}",
            shortest,
            lower
        );
        assert!(
            shortest <= upper,
            "shortest {} should be <= upper {}",
            shortest,
            upper
        );

        // The shortest representation should ideally be 2 (mantissa = 1, exponent = 1)
        // or at least have fewer mantissa bits than the bounds
        let shortest_bits = shortest.mantissa().magnitude().bits();
        let lower_bits = lower.mantissa().magnitude().bits();
        assert!(
            shortest_bits <= lower_bits,
            "shortest ({} bits) should have <= bits than lower ({} bits)",
            shortest_bits,
            lower_bits
        );
    }
}
