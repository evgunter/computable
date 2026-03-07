//! Shift utilities for binary mantissa operations.
//!
//! This module provides chunked shift operations that handle arbitrarily large
//! shift amounts by breaking them into primitive-sized chunks.

use std::ops::ShlAssign;

use num_bigint::{BigInt, BigUint};

/// Marker trait for arbitrary-precision integer types whose arithmetic
/// (including `<<=`) cannot overflow. Implemented only for `BigInt` and
/// `BigUint`, so `shift_mantissa_chunked` is restricted to those types.
pub(crate) trait ArbitraryPrecision: Clone + ShlAssign<usize> {}
impl ArbitraryPrecision for BigInt {}
impl ArbitraryPrecision for BigUint {}

/// Performs a left shift of a mantissa by a potentially large amount.
///
/// BigInt only implements shifts for primitive integers, so this function
/// chunks large shifts into smaller operations that fit within `usize`.
///
/// # Arguments
/// * `mantissa` - The value to shift
/// * `shift` - The shift amount (can be arbitrarily large)
/// * `chunk_limit` - Maximum shift per chunk (injected for testability)
///
/// # Panics
/// Panics via `detected_computable_would_exhaust_memory!` if `shift` exceeds
/// `MAX_COMPUTATION_BITS`, since the result would be too large to fit in memory.
pub(crate) fn shift_mantissa_chunked<M>(mantissa: &M, shift: &BigUint, chunk_limit: usize) -> M
where
    M: ArbitraryPrecision,
{
    let shift_usize: usize = shift.try_into().unwrap_or_else(|_| {
        crate::detected_computable_would_exhaust_memory!("shift by extreme exponent")
    });
    crate::assert_sane_computation_size!(shift_usize);
    let mut shifted = mantissa.clone();
    let mut remaining = shift_usize;
    while remaining > 0 {
        let chunk = remaining.min(chunk_limit);
        #[allow(clippy::arithmetic_side_effects)] // M: ArbitraryPrecision — only BigInt/BigUint
        {
            shifted <<= chunk;
        }
        // remaining >= chunk by construction of min, so this cannot underflow.
        remaining = remaining.wrapping_sub(chunk);
    }
    shifted
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::Zero;

    #[test]
    fn shift_mantissa_chunks_large_shift() {
        let mantissa = BigInt::from(1_i32);
        let shift = BigUint::from(128u32);
        let chunk_limit = 64_usize;
        let chunked = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, chunk_limit);
        let expected = &mantissa << 128usize;
        assert_eq!(chunked, expected);
    }

    #[test]
    fn shift_zero_returns_original() {
        let mantissa = BigInt::from(42_i32);
        let shift = BigUint::zero();
        let chunk_limit = 64_usize;
        let result = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, chunk_limit);
        assert_eq!(result, mantissa);
    }

    #[test]
    fn shift_within_single_chunk() {
        let mantissa = BigInt::from(1_i32);
        let shift = BigUint::from(10u32);
        let chunk_limit = 64_usize;
        let result = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, chunk_limit);
        assert_eq!(result, BigInt::from(1024_i32)); // 1 << 10 = 1024
    }
}
