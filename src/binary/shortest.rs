//! Shortest mantissa selection within bounds.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use super::Bounds;
use super::{UBinary, UXBinary, XBinary};

// TODO: use these functions to make binary-search-based refinement not need to represent intervals that have so many bits of precision
// TODO: use similar functions to make other refinement strategies not need to represent intervals that have so many bits of precision
// by loosening bounds (this will require something other than plain shortest_xbinary_in_bounds since the loosened bounds would obviously not
// lie within the original bounds; but perhaps lower bound - width/4 and upper bound + width/4 would be reasonable?
// or even something that uses epsilon rather than just width?)

/// Returns an XBinary value inside the bounds with the shortest normalized mantissa.
/// infinities are only returned if the bounds do not contain any finite numbers; this shouldn't happen currently but it might be valid in the future.
pub fn shortest_xbinary_in_bounds(bounds: &Bounds) -> XBinary {
    // NOTE: Bounds stores lower + width with UXBinary, so intervals like (-inf, -1]
    // cannot be represented because the width is +inf and large() becomes +inf.
    let (lower_sign, lower_mag) = split_xbinary(bounds.small());
    match lower_sign {
        Sign::NoSign => XBinary::zero(),
        Sign::Plus => shortest_xbinary_in_positive_interval(&lower_mag, bounds.width()),
        Sign::Minus => {
            if bounds.large() >= XBinary::zero() {
                XBinary::zero()
            } else {
                shortest_xbinary_in_positive_interval(&bounds.large().magnitude(), bounds.width()).neg()
            }
        }
    }
}

/// infinities are only returned if the bounds do not contain any finite numbers; this shouldn't happen currently but it might be valid in the future.
fn shortest_xbinary_in_positive_interval(lower: &UXBinary, width: &UXBinary) -> XBinary {
    match lower {
        UXBinary::Inf => {
            // This debug_assert is here because nothing currently produces this case, so
            // hitting it likely indicates a bug. However, this could become a valid case
            // if we later support computations in the extended reals where +∞ bounds are
            // meaningful. If that feature is added, this assertion should be removed.
            debug_assert!(false, "lower input bound is PosInf - unexpected but may be valid for extended reals");
            XBinary::PosInf
        }
        UXBinary::Finite(lm) => XBinary::Finite(shortest_binary_in_positive_interval(lm, width).to_binary())
    }
}

fn shortest_binary_in_positive_interval(lower: &UBinary, width: &UXBinary) -> UBinary {
    match width {
        UXBinary::Inf => {
                // next power of 2 >= lower
                let exponent = lower.exponent() + BigInt::from((lower.mantissa() - BigUint::one()).bits());
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
        XBinary::Finite(v) => (v.mantissa().sign(), UXBinary::Finite(v.magnitude()))
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

        // this case is currently blocked by the debug_assert, but it should be added if we want to support extended reals
        // let bounds = Bounds::new(XBinary::PosInf, XBinary::PosInf);
        // assert_eq!(shortest_xbinary_in_bounds(&bounds), XBinary::PosInf);
    }
}
