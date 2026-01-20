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
            // then we can find the largest bit where the mantissas differ and only take the part of the mantissa before that (shifting the exponent accordingly)
            todo!()
        }
    }
}

fn shortest_power_of_two_at_least(lower: &Binary) -> Option<Binary> {
    let exponent = ceil_log2_binary(lower)?;
    Some(Binary::new(BigInt::one(), exponent))
}

fn shortest_power_of_two_at_most(upper: &Binary) -> Option<Binary> {
    let magnitude = upper.neg();
    let exponent = ceil_log2_binary(&magnitude)?;
    Some(Binary::new(BigInt::from(-1), exponent))
}

fn ceil_log2_binary(value: &Binary) -> Option<BigInt> {
    if !value.mantissa().is_positive() {
        return None;
    }
    if value.mantissa().is_zero() {
        return Some(BigInt::zero());
    }

    let mantissa_bits = value.mantissa().bits();
    let floor_log2_mantissa = mantissa_bits.saturating_sub(1);
    let base = value.exponent() + BigInt::from(floor_log2_mantissa);

    if value.mantissa() == &BigInt::one() {
        Some(base)
    } else {
        Some(base + BigInt::one())
    }
}

fn max_power_of_two_divisor(lower: &BigInt, upper: &BigInt) -> u64 {
    let mut low = 0u64;
    let mut high = upper.bits().saturating_sub(1);

    while low < high {
        let mid = (low + high).div_ceil(2);
        if has_multiple_pow2(lower, upper, mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

fn has_multiple_pow2(lower: &BigInt, upper: &BigInt, shift: u64) -> bool {
    div_ceil_pow2(lower, shift) <= div_floor_pow2(upper, shift)
}

fn div_floor_pow2(value: &BigInt, shift: u64) -> BigInt {
    if shift == 0 || value.is_zero() {
        return value.clone();
    }

    let bits = value.bits();
    if shift >= bits {
        return BigInt::zero();
    }

    value >> shift
}

fn div_ceil_pow2(value: &BigInt, shift: u64) -> BigInt {
    if shift == 0 || value.is_zero() {
        return value.clone();
    }

    let bits = value.bits();
    if shift >= bits {
        return BigInt::one();
    }

    let floor = value >> shift;
    let trailing_zeros = value.trailing_zeros().unwrap_or(0);
    if trailing_zeros >= shift {
        floor
    } else {
        floor + 1
    }
}

fn shift_left_bigint(value: &BigInt, shift: &BigUint) -> BigInt {
    if shift.is_zero() {
        return value.clone();
    }
    let chunk_limit = BigUint::from(usize::MAX);
    shift::shift_mantissa_chunked::<BigInt>(value, shift, &chunk_limit)
}

fn split_xbinary(value: &XBinary) -> (Sign, UXBinary) {
    match value {
        XBinary::NegInf => (Sign::Minus, UXBinary::PosInf),
        XBinary::PosInf => (Sign::Plus, UXBinary::PosInf),
        XBinary::Finite(binary) => {
            let mantissa = binary.mantissa();
            if mantissa.is_zero() {
                (Sign::NoSign, UXBinary::zero())
            } else {
                let magnitude = UBinary::new(mantissa.magnitude().clone(), binary.exponent().clone());
                if mantissa.is_positive() {
                    (Sign::Plus, UXBinary::Finite(magnitude))
                } else {
                    (Sign::Minus, UXBinary::Finite(magnitude))
                }
            }
        }
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
