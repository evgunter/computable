//! Shift utilities for binary mantissa operations.
//!
//! This module provides chunked shift operations that handle arbitrarily large
//! shift amounts by breaking them into primitive-sized chunks.

use std::ops::ShlAssign;

use num_bigint::BigUint;
use num_traits::Zero;

/// Maximum shift amount in bits before we consider it memory-exhausting.
///
/// A shift of 2^32 bits produces a number requiring ~512 MB just for storage,
/// and intermediate computations would need far more. We panic explicitly rather
/// than attempting the allocation and hitting an OOM.
const MAX_SHIFT_BITS: u64 = 1u64 << 32;

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
/// `MAX_SHIFT_BITS`, since the result would be too large to fit in memory.
pub(crate) fn shift_mantissa_chunked<M>(mantissa: &M, shift: &BigUint, chunk_limit: &BigUint) -> M
where
    M: Clone + ShlAssign<usize>,
{
    if shift > &BigUint::from(MAX_SHIFT_BITS) {
        crate::detected_computable_would_exhaust_memory!(
            "shift by extreme exponent"
        );
    }
    let mut shifted = mantissa.clone();
    let mut remaining = shift.clone();
    let chunk_limit_usize = {
        let digits = chunk_limit.to_u64_digits();
        match digits.first() {
            Some(value) => *value as usize,
            None => 0,
        }
    };
    while remaining > BigUint::zero() {
        let chunk_usize = if &remaining > chunk_limit {
            chunk_limit_usize
        } else {
            let digits = remaining.to_u64_digits();
            match digits.first() {
                Some(value) => *value as usize,
                None => 0,
            }
        };
        shifted <<= chunk_usize;
        remaining -= BigUint::from(chunk_usize);
    }
    shifted
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn shift_mantissa_chunks_large_shift() {
        let mantissa = BigInt::from(1);
        let shift = BigUint::from(128u32);
        let chunk_limit = BigUint::from(64u32);
        let chunked = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, &chunk_limit);
        let expected = &mantissa << 128usize;
        assert_eq!(chunked, expected);
    }

    #[test]
    fn shift_zero_returns_original() {
        let mantissa = BigInt::from(42);
        let shift = BigUint::zero();
        let chunk_limit = BigUint::from(64u32);
        let result = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, &chunk_limit);
        assert_eq!(result, mantissa);
    }

    #[test]
    fn shift_within_single_chunk() {
        let mantissa = BigInt::from(1);
        let shift = BigUint::from(10u32);
        let chunk_limit = BigUint::from(64u32);
        let result = shift_mantissa_chunked::<BigInt>(&mantissa, &shift, &chunk_limit);
        assert_eq!(result, BigInt::from(1024)); // 1 << 10 = 1024
    }
}
