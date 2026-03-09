//! Unsigned binary number implementation.
//!
//! This module provides `UBinary`, an unsigned variant of `Binary` for representing
//! non-negative values like interval widths.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Shl, Shr};

use num_bigint::BigUint;
use num_traits::Zero;

use crate::ordered_pair::Unsigned;

use super::binary_impl::Binary;
use super::error::BinaryError;
use super::shift::shift_mantissa_chunked;

/// Unsigned binary number represented as `mantissa * 2^exponent` where mantissa >= 0.
///
/// The mantissa is normalized to be odd unless the value is zero.
/// This type is used for representing non-negative quantities like interval widths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UBinary {
    mantissa: BigUint,
    exponent: i64,
}

impl UBinary {
    /// Creates a new UBinary number, normalizing the representation.
    pub fn new(mantissa: BigUint, exponent: i64) -> Self {
        Self::normalize(mantissa, exponent)
    }

    /// Creates a UBinary from components that are already in canonical form.
    pub(crate) fn new_normalized(mantissa: BigUint, exponent: i64) -> Self {
        debug_assert!(
            mantissa.is_zero() && exponent == 0
                || !mantissa.is_zero() && mantissa.trailing_zeros() == Some(0),
            "new_normalized: mantissa must be odd (or both zero)"
        );
        Self { mantissa, exponent }
    }

    /// Returns the zero value.
    pub fn zero() -> Self {
        Self {
            mantissa: BigUint::zero(),
            exponent: 0_i64,
        }
    }

    /// Returns a reference to the mantissa.
    pub fn mantissa(&self) -> &BigUint {
        &self.mantissa
    }

    /// Returns the exponent.
    pub fn exponent(&self) -> i64 {
        self.exponent
    }

    /// Creates a UBinary from a Binary, returning an error if the mantissa is negative.
    pub fn try_from_binary(binary: &Binary) -> Result<Self, BinaryError> {
        use num_traits::Signed;

        if binary.mantissa().is_negative() {
            return Err(BinaryError::NegativeMantissa);
        }
        if binary.mantissa().is_zero() {
            return Ok(Self::zero());
        }
        let mantissa = binary.mantissa().magnitude().clone();
        Ok(Self::new_normalized(mantissa, binary.exponent()))
    }

    /// Converts this unsigned binary to a signed binary.
    pub fn to_binary(&self) -> Binary {
        use num_bigint::BigInt;
        if self.mantissa.is_zero() {
            return Binary::zero();
        }
        Binary::new_normalized(BigInt::from(self.mantissa.clone()), self.exponent)
    }

    /// Adds two UBinary numbers.
    pub fn add(&self, other: &Self) -> Self {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other);
        Self::normalize(lhs + rhs, exponent)
    }

    /// Multiplies two UBinary numbers.
    pub fn mul(&self, other: &Self) -> Self {
        if self.mantissa.is_zero() || other.mantissa.is_zero() {
            return Self::zero();
        }
        let exponent = self.exponent.checked_add(other.exponent).unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in UBinary::mul")
        });
        let mantissa = &self.mantissa * &other.mantissa;
        Self::new_normalized(mantissa, exponent)
    }

    /// Conservative floor division: returns a value <= self / other.
    ///
    /// The result is a lower bound on the true quotient, with precision
    /// proportional to the divisor's mantissa size.
    ///
    /// # Precondition
    ///
    /// The denominator must be nonzero.
    pub fn div_floor(&self, other: &Self) -> Self {
        debug_assert!(
            !other.mantissa.is_zero(),
            "div_floor: denominator must be nonzero"
        );
        if self.mantissa.is_zero() {
            return Self::zero();
        }
        // a / b = (m_a * 2^e_a) / (m_b * 2^e_b) = (m_a / m_b) * 2^(e_a - e_b)
        // Shift numerator left until quotient has enough bits for a
        // meaningful result (at least 1 bit, i.e. quotient >= 1).
        let extra_bits = other.mantissa.bits();
        let shifted_num = &self.mantissa << extra_bits;
        let quotient = shifted_num / &other.mantissa;
        let exponent = self
            .exponent
            .checked_sub(other.exponent)
            .and_then(|e| {
                e.checked_sub(i64::try_from(extra_bits).unwrap_or_else(|_| {
                    crate::detected_computable_would_exhaust_memory!(
                        "extra_bits exceeds i64 in UBinary::div_floor"
                    )
                }))
            })
            .unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in UBinary::div_floor"
                )
            });
        Self::new(quotient, exponent)
    }

    /// Normalizes the representation by factoring out powers of 2 from the mantissa.
    fn normalize(mut mantissa: BigUint, mut exponent: i64) -> Self {
        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: 0_i64,
            };
        }

        if let Some(tz_u64) = mantissa.trailing_zeros() {
            let tz = crate::sane::bits_as_usize(tz_u64);
            mantissa >>= tz;
            exponent = exponent
                .checked_add(i64::try_from(tz).unwrap_or_else(|_| {
                    crate::detected_computable_would_exhaust_memory!(
                        "trailing zeros exceeds i64 in UBinary::normalize"
                    )
                }))
                .unwrap_or_else(|| {
                    crate::detected_computable_would_exhaust_memory!(
                        "exponent overflow in UBinary::normalize"
                    )
                });
        }

        Self { mantissa, exponent }
    }

    /// Aligns the mantissas of two UBinary numbers to a common exponent.
    /// Returns (lhs_mantissa, rhs_mantissa, common_exponent) where both mantissas
    /// are shifted to the minimum exponent of the two inputs.
    pub fn align_mantissas(lhs: &Self, rhs: &Self) -> (BigUint, BigUint, i64) {
        let exponent = std::cmp::min(lhs.exponent, rhs.exponent);
        let lhs_shift = lhs.exponent.abs_diff(exponent);
        let rhs_shift = rhs.exponent.abs_diff(exponent);
        let lhs_mantissa = if lhs_shift == 0 {
            lhs.mantissa.clone()
        } else {
            let shift_usize = usize::try_from(lhs_shift).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "shift exceeds usize in UBinary::align_mantissas"
                )
            });
            crate::assert_sane_computation_size!(shift_usize);
            &lhs.mantissa << shift_usize
        };
        let rhs_mantissa = if rhs_shift == 0 {
            rhs.mantissa.clone()
        } else {
            let shift_usize = usize::try_from(rhs_shift).unwrap_or_else(|_| {
                crate::detected_computable_would_exhaust_memory!(
                    "shift exceeds usize in UBinary::align_mantissas"
                )
            });
            crate::assert_sane_computation_size!(shift_usize);
            &rhs.mantissa << shift_usize
        };
        (lhs_mantissa, rhs_mantissa, exponent)
    }
    /// Shifts a mantissa left by the given amount, handling large shifts.
    #[allow(dead_code)]
    fn shift_mantissa(mantissa: &BigUint, shift: &BigUint) -> BigUint {
        if shift.is_zero() {
            return mantissa.clone();
        }
        shift_mantissa_chunked::<BigUint>(mantissa, shift, usize::MAX)
    }
}

impl Ord for UBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use the same comparison logic as Binary but with unsigned mantissas
        let self_binary = self.to_binary();
        let other_binary = other.to_binary();
        self_binary.cmp(&other_binary)
    }
}

impl PartialOrd for UBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl num_traits::Zero for UBinary {
    fn zero() -> Self {
        UBinary::zero()
    }

    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }
}

impl Add for UBinary {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        UBinary::add(&self, &rhs)
    }
}

impl Mul for UBinary {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        UBinary::mul(&self, &rhs)
    }
}

impl Shl<u32> for UBinary {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // shifting = exponent adjustment
    fn shl(self, rhs: u32) -> Self::Output {
        if self.mantissa.is_zero() {
            return self;
        }
        Self::new_normalized(
            self.mantissa,
            self.exponent.checked_add(i64::from(rhs)).unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in UBinary::shl"
                )
            }),
        )
    }
}

impl Shr<u32> for UBinary {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // shifting = exponent adjustment
    fn shr(self, rhs: u32) -> Self::Output {
        if self.mantissa.is_zero() {
            return self;
        }
        Self::new_normalized(
            self.mantissa,
            self.exponent.checked_sub(i64::from(rhs)).unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!(
                    "exponent overflow in UBinary::shr"
                )
            }),
        )
    }
}

impl Unsigned for UBinary {}

impl fmt::Display for UBinary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::display::format_ubinary_display(f, &self.mantissa, self.exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{bin, ubin};
    use num_bigint::BigInt;

    #[test]
    fn ubinary_normalizes_even_mantissa() {
        let value = ubin(8, 0);
        assert_eq!(value.mantissa(), &BigUint::from(1u32));
        assert_eq!(value.exponent(), 3_i64);
    }

    #[test]
    fn ubinary_zero_uses_zero_exponent() {
        let value = UBinary::new(BigUint::zero(), 42_i64);
        assert_eq!(value.mantissa(), &BigUint::zero());
        assert_eq!(value.exponent(), 0_i64);
    }

    #[test]
    fn ubinary_ordering_works() {
        let one = ubin(1, 0);
        let two = ubin(1, 1);
        let half = ubin(1, -1);
        assert!(two > one);
        assert!(one > half);
    }

    #[test]
    fn ubinary_add_works() {
        let one = ubin(1, 0);
        let half = ubin(1, -1);
        let sum = one + half;
        let expected = ubin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn ubinary_try_from_binary_works() {
        let positive = bin(5, 2);
        let result = UBinary::try_from_binary(&positive);
        assert!(result.is_ok());
        let ubinary = result.expect("should succeed");
        assert_eq!(ubinary.mantissa(), &BigUint::from(5u32));
        assert_eq!(ubinary.exponent(), 2_i64);

        let negative = bin(-5, 2);
        let neg_result = UBinary::try_from_binary(&negative);
        assert!(neg_result.is_err());
    }

    #[test]
    fn ubinary_to_binary_works() {
        let ubinary = ubin(7, 3);
        let binary = ubinary.to_binary();
        assert_eq!(binary.mantissa(), &BigInt::from(7_i32));
        assert_eq!(binary.exponent(), 3_i64);
    }

    #[test]
    fn ubinary_mul_works() {
        let two = ubin(1, 1);
        let three = ubin(3, 0);
        let product = two * three;
        let expected = ubin(3, 1); // 2 * 3 = 6 = 3 * 2^1
        assert_eq!(product, expected);
    }
}
