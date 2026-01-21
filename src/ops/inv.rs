//! Multiplicative inverse operation with precision-based refinement.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::Zero;
use parking_lot::RwLock;

use crate::binary::{
    margin_from_width, reciprocal_rounded_abs_extended, simplify_bounds_if_needed, Bounds, ReciprocalRounding, XBinary,
};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

/// Precision threshold for triggering bounds simplification.
/// 128 chosen: 12% faster than 64 due to lower overhead in precision-doubling refinement.
const PRECISION_SIMPLIFICATION_THRESHOLD: u64 = 128;

/// margin parameter for bounds simplification.
/// 3 = loosen by width/8. Benchmarks show margin has minimal performance impact.
const MARGIN_SHIFT: u32 = 3;

/// Initial precision bits to start with for inv refinement.
/// Starting at a reasonable value avoids unnecessary early iterations.
pub const INV_INITIAL_PRECISION_BITS: i64 = 4;

// TODO: Improve inv() precision strategy. Currently precision_bits starts at 0 and
// doubles on each refine_step. This is simple but potentially inefficient:
// - For a given epsilon, we don't know how many bits are needed upfront
// - Each step recomputes the reciprocal from scratch at the new precision
// Consider: adaptive precision based on current bounds width, or Newton-Raphson iteration.

/// Inverse (reciprocal) operation with precision-based refinement.
pub struct InvOp {
    pub inner: Arc<Node>,
    pub precision_bits: RwLock<BigInt>,
}

impl NodeOp for InvOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        let raw_bounds = reciprocal_bounds(&existing, &self.precision_bits.read())?;
        // Simplify bounds to reduce precision bloat from high-precision reciprocal computation
        let margin = margin_from_width(raw_bounds.width(), MARGIN_SHIFT);
        Ok(simplify_bounds_if_needed(
            &raw_bounds,
            PRECISION_SIMPLIFICATION_THRESHOLD,
            &margin,
        ))
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut precision = self.precision_bits.write();
        // Double precision each step for O(log n) convergence.
        // If precision is 0, start with initial value to bootstrap.
        // Once the TODO above is implemented (reusing precision calculation state),
        // this should be changed back to linear increment to avoid unnecessary
        // computation to higher precision than requested.
        if precision.is_zero() {
            *precision = BigInt::from(INV_INITIAL_PRECISION_BITS);
        } else {
            *precision *= 2;
        }
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Computes reciprocal bounds for an interval.
fn reciprocal_bounds(bounds: &Bounds, precision_bits: &BigInt) -> Result<Bounds, ComputableError> {
    let lower = bounds.small();
    let upper = bounds.large();
    let zero = XBinary::zero();
    if lower <= &zero && upper >= zero {
        return Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf));
    }

    let (lower_bound, upper_bound) = if upper < zero {
        let lower_bound = reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            ReciprocalRounding::Ceil,
        )?
        .neg();
        let upper_bound = reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            ReciprocalRounding::Floor,
        )?
        .neg();
        (lower_bound, upper_bound)
    } else {
        let lower_bound = reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            ReciprocalRounding::Floor,
        )?;
        let upper_bound = reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            ReciprocalRounding::Ceil,
        )?;
        (lower_bound, upper_bound)
    };

    // TODO: can the type system ensure that the bounds remain ordered?
    Bounds::new_checked(lower_bound, upper_bound).map_err(|_| ComputableError::InvalidBoundsOrder)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use crate::binary::{Binary, Bounds, UBinary, XBinary};
    use crate::test_utils::{ubin, unwrap_finite, unwrap_finite_uxbinary, interval_midpoint_computable};

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    #[test]
    fn inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            Bounds::new(XBinary::NegInf, XBinary::PosInf)
        );
    }

    #[test]
    fn inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = ubin(1, -8);
        let bounds = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected_binary = XBinary::from_f64(1.0 / 3.0)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected_value, &epsilon);
    }
}
