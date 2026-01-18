//! Multiplicative inverse operation with precision-based refinement.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::Zero;
use parking_lot::RwLock;

use crate::binary::{reciprocal_rounded_abs_extended, ReciprocalRounding, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::ordered_pair::Bounds;

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
        reciprocal_bounds(&existing, &self.precision_bits.read())
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

    Bounds::new_checked(lower_bound, upper_bound).map_err(|_| ComputableError::InvalidBoundsOrder)
}
