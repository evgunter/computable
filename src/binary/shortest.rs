//! Shortest mantissa selection within bounds.

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use super::shift;
use super::Bounds;
use super::{Binary, UBinary, UXBinary, XBinary};

// TODO: use these functions to make binary-search-based refinement not need to represent intervals that have so many bits of precision
// TODO: use similar functions to make other refinement strategies not need to represent intervals that have so many bits of precision
// by loosening bounds (this will require something other than plain shortest_xbinary_in_bounds since the loosened bounds would obviously not
// lie within the original bounds; but perhaps lower bound - width/4 and upper bound + width/4 would be reasonable?
// or even something that uses epsilon rather than just width?)

/// Returns an XBinary value inside the bounds with the shortest normalized mantissa.
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

fn shortest_xbinary_in_positive_interval(lower: &UXBinary, width: &UXBinary) -> XBinary {
    match lower {
        UXBinary::PosInf => XBinary::PosInf,
        UXBinary::Finite(lm) => XBinary::Finite(shortest_binary_in_positive_interval(&lm, width).to_binary())
    }
}

fn shortest_binary_in_positive_interval(lower: &UBinary, width: &UXBinary) -> UBinary {
    match width {
        UXBinary::PosInf => {
            let exponent = lower.exponent() + BigInt::from(lower.mantissa().bits());
            UBinary::new(BigUint::one(), exponent)
        }
        UXBinary::Finite(wm) => {
            // TODO: we want to take lower and then cancel out as many mantissa bits as possible by adding at most wm.
            // to do this we compare lower and lower + wm by shifting them both to have the same exponent as lower.
            // (which we can do using the implementation that's already in binary/, though we may need to refactor out the shift logic to a separate function)
            // then we can find the largest bit where the mantissas differ and only take the part of the mantissa before that (shifting the exponent accordingly)
            todo!()
        }
    }
}


fn split_xbinary(value: &XBinary) -> (Sign, UXBinary) {
    match value {
        XBinary::NegInf => (Sign::Minus, UXBinary::PosInf),
        XBinary::PosInf => (Sign::Plus, UXBinary::PosInf),
        XBinary::Finite(v) => (v.mantissa().sign(), UXBinary::Finite(v.magnitude()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::test_utils::bin;

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

        let bounds = Bounds::new(XBinary::PosInf, XBinary::PosInf);
        assert_eq!(shortest_xbinary_in_bounds(&bounds), XBinary::PosInf);
    }
}
