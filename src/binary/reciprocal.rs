//! Reciprocal computation for binary numbers.
//!
//! This module provides functions for computing reciprocals of extended binary numbers
//! with controlled precision and rounding.

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::One;

use super::binary_impl::Binary;
use crate::sane::U;

/// Result of dividing `2^precision_bits` by a denominator, keeping both quotient and remainder.
///
/// This allows extending to higher precision cheaply via [`extend_reciprocal`]:
/// only the new quotient bits need to be computed, using the stored remainder.
pub struct ReciprocalWithRemainder {
    pub quotient: BigUint,
    pub remainder: BigUint,
    pub precision_bits: U,
}

/// Computes `floor(2^precision_bits / denom)` and the remainder.
pub fn reciprocal_with_remainder(denom: &BigUint, precision_bits: U) -> ReciprocalWithRemainder {
    let numerator = BigUint::one() << (crate::sane::u_as_usize(precision_bits));
    let (quotient, remainder) = numerator.div_rem(denom);
    ReciprocalWithRemainder {
        quotient,
        remainder,
        precision_bits,
    }
}

/// Extends a previous reciprocal to higher precision using the stored remainder.
///
/// If `2^P = q*d + r`, then `2^(P+δ) = (q << δ)*d + (r << δ)`.
/// We only need to divide `(r << δ)` by `d` to get the new quotient bits.
pub fn extend_reciprocal(
    prev: &ReciprocalWithRemainder,
    denom: &BigUint,
    new_precision: U,
) -> ReciprocalWithRemainder {
    debug_assert!(new_precision >= prev.precision_bits);
    #[allow(clippy::arithmetic_side_effects)] // guarded by debug_assert above
    let delta = new_precision - prev.precision_bits;
    let (extra_q, new_r) = (&prev.remainder << (crate::sane::u_as_usize(delta))).div_rem(denom);
    let new_q = (&prev.quotient << (crate::sane::u_as_usize(delta))) + extra_q;
    ReciprocalWithRemainder {
        quotient: new_q,
        remainder: new_r,
        precision_bits: new_precision,
    }
}

/// Specifies the rounding direction for reciprocal computation.
#[derive(Clone, Copy, Debug)]
pub enum ReciprocalRounding {
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceil,
}

/// Computes 1/denominator where denominator is a positive integer (BigUint).
///
/// Returns `mantissa * 2^(-precision_bits)` where:
/// - For `Floor`: `mantissa = floor(2^precision_bits / denominator)`
/// - For `Ceil`: `mantissa = ceil(2^precision_bits / denominator)`
///
/// Using `BigUint` enforces positivity through the type system.
///
/// # Arguments
/// * `denominator` - A positive BigUint to take the reciprocal of
/// * `precision_bits` - Number of bits of precision for the computation
/// * `rounding` - Whether to round toward floor or ceiling
pub fn reciprocal_of_biguint(
    denominator: &BigUint,
    precision_bits: U,
    rounding: ReciprocalRounding,
) -> Binary {
    let numerator = BigUint::one() << (crate::sane::u_as_usize(precision_bits));
    let quotient = match rounding {
        ReciprocalRounding::Floor => numerator.div_floor(denominator),
        ReciprocalRounding::Ceil => {
            (&numerator + denominator - BigUint::one()).div_floor(denominator)
        }
    };
    let precision_i64 = i64::from(precision_bits);
    let exponent = precision_i64.checked_neg().unwrap_or_else(|| {
        crate::detected_computable_would_exhaust_memory!(
            "exponent overflow in reciprocal_of_biguint"
        )
    });
    Binary::new(BigInt::from(quotient), exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_of_biguint_basic() {
        let denom = BigUint::from(5u32);
        let result = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Floor);

        // 2^8 / 5 = 256 / 5 = 51 (floor)
        assert_eq!(result.mantissa(), &BigInt::from(51_i32));
        assert_eq!(result.exponent(), -8_i64);
    }

    #[test]
    fn reciprocal_of_biguint_rounding_modes() {
        let denom = BigUint::from(3u32);

        let floor = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Floor);
        let ceil = reciprocal_of_biguint(&denom, 8, ReciprocalRounding::Ceil);

        // 2^8 / 3 = 85.33..., floor=85, ceil=86
        assert_eq!(floor.mantissa(), &BigInt::from(85_i32));
        assert_eq!(floor.exponent(), -8_i64);
        assert_eq!(ceil.mantissa(), &BigInt::from(43_i32));
        assert_eq!(ceil.exponent(), -7_i64); // 43 * 2^-7 = 86 * 2^-8
        assert!(ceil > floor);
    }

    #[test]
    fn reciprocal_with_remainder_basic() {
        let denom = BigUint::from(3u32);
        let r = reciprocal_with_remainder(&denom, 8);
        // 256 / 3 = 85 rem 1
        assert_eq!(r.quotient, BigUint::from(85u32));
        assert_eq!(r.remainder, BigUint::from(1u32));
        assert_eq!(r.precision_bits, 8);
    }

    #[test]
    fn extend_reciprocal_matches_fresh() {
        let denom = BigUint::from(7u32);
        let initial = reciprocal_with_remainder(&denom, 16);
        let extended = extend_reciprocal(&initial, &denom, 64);
        let fresh = reciprocal_with_remainder(&denom, 64);
        assert_eq!(extended.quotient, fresh.quotient);
        assert_eq!(extended.remainder, fresh.remainder);
    }
}
