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
mod ubinary;
mod uxbinary;
mod xbinary;

// Re-export all public types
pub use binary_impl::Binary;
pub use error::{BinaryError, XBinaryError};
pub use reciprocal::{reciprocal_rounded_abs_extended, ReciprocalRounding};
pub use ubinary::UBinary;
pub use uxbinary::UXBinary;
pub use xbinary::XBinary;

// BigInt/BigUint trait implementations for ordered_pair compatibility
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::ordered_pair::{AbsDistance, AddWidth, Bounds, Unsigned};

/// Returns a binary value inside the bounds with the shortest normalized mantissa.
pub fn shortest_binary_in_bounds(bounds: &Bounds) -> Option<Binary> {
    let lower = bounds.small();
    let upper = bounds.large();
    let zero = XBinary::zero();

    if lower <= &zero && &zero <= &upper {
        return Some(Binary::zero());
    }

    match (lower, &upper) {
        (XBinary::Finite(lower), XBinary::Finite(upper)) => {
            if lower.mantissa().is_positive() {
                shortest_binary_in_positive_interval(lower, upper)
            } else if upper.mantissa().is_negative() {
                let lower_pos = upper.neg();
                let upper_pos = lower.neg();
                shortest_binary_in_positive_interval(&lower_pos, &upper_pos)
                    .map(|value| value.neg())
            } else {
                None
            }
        }
        (XBinary::Finite(lower), XBinary::PosInf) => {
            if lower.mantissa().is_positive() {
                shortest_power_of_two_at_least(lower)
            } else {
                None
            }
        }
        (XBinary::NegInf, XBinary::Finite(upper)) => {
            if upper.mantissa().is_negative() {
                shortest_power_of_two_at_most(upper)
            } else {
                None
            }
        }
        (XBinary::NegInf, XBinary::PosInf) => Some(Binary::zero()),
        (XBinary::PosInf, XBinary::PosInf) | (XBinary::NegInf, XBinary::NegInf) => None,
        _ => None,
    }
}

fn shortest_binary_in_positive_interval(lower: &Binary, upper: &Binary) -> Option<Binary> {
    if !lower.mantissa().is_positive() || !upper.mantissa().is_positive() {
        return None;
    }

    // Scale to a shared exponent so we can reason about integer multiples of powers of two.
    let base = if lower.exponent() <= upper.exponent() {
        lower.exponent().clone()
    } else {
        upper.exponent().clone()
    };
    let lower_shift = BigUint::try_from(lower.exponent() - &base).ok()?;
    let upper_shift = BigUint::try_from(upper.exponent() - &base).ok()?;
    let lower_int = shift_left_bigint(lower.mantissa(), &lower_shift);
    let upper_int = shift_left_bigint(upper.mantissa(), &upper_shift);

    let max_pow2 = max_power_of_two_divisor(&lower_int, &upper_int);
    let mut mantissa = div_ceil_pow2(&lower_int, max_pow2);
    let mantissa_max = div_floor_pow2(&upper_int, max_pow2);
    if mantissa > mantissa_max {
        return None;
    }
    if mantissa.is_even() {
        mantissa += 1;
        if mantissa > mantissa_max {
            return None;
        }
    }

    let exponent = base + BigInt::from(max_pow2);
    Some(Binary::new(mantissa, exponent))
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
        let mid = (low + high + 1) / 2;
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

    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ordered_pair::Bounds;
    use num_bigint::BigInt;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn xbin(mantissa: i64, exponent: i64) -> XBinary {
        XBinary::Finite(bin(mantissa, exponent))
    }

    fn midpoint_between(lower: &Binary, upper: &Binary) -> Binary {
        let mid_sum = lower.add(upper);
        let exponent = mid_sum.exponent() - BigInt::from(1);
        Binary::new(mid_sum.mantissa().clone(), exponent)
    }

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
    fn shortest_binary_in_bounds_finds_sqrt_four() {
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
        let shortest = shortest_binary_in_bounds(&bounds).expect("shortest should exist");
        assert_eq!(shortest, bin(1, 1));
    }
}
