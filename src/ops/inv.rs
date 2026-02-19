//! Multiplicative inverse operation with precision-based refinement.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use parking_lot::RwLock;

use crate::binary::{
    Bounds, FiniteBounds, ReciprocalRounding, UXBinary, XBinary,
    reciprocal_rounded_abs_extended,
};
use crate::binary_utils::bisection::normalize_bounds;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

/// Initial precision bits to start with for inv refinement when input bounds are infinite.
/// Starting at a reasonable value avoids unnecessary early iterations.
pub const INV_INITIAL_PRECISION_BITS: i64 = 8;

/// Inverse (reciprocal) operation with precision-based refinement.
pub struct InvOp {
    pub inner: Arc<Node>,
    /// Precision bits for reciprocal computation. `None` means not yet initialized.
    pub precision_bits: RwLock<Option<BigInt>>,
}

impl NodeOp for InvOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        let precision = self.precision_bits.read();
        let raw_bounds = reciprocal_bounds(&existing, precision.as_ref())?;
        // Normalize to prefix form to prevent precision accumulation.
        // Infinite width or infinite endpoints can't be normalized — return as-is.
        if matches!(raw_bounds.width(), UXBinary::Inf) {
            return Ok(raw_bounds);
        }
        match (raw_bounds.small(), &raw_bounds.large()) {
            (XBinary::Finite(lo), XBinary::Finite(hi)) => {
                let finite = FiniteBounds::new(lo.clone(), hi.clone());
                let normalized = normalize_bounds(&finite)?;
                Ok(Bounds::from_lower_and_width(
                    XBinary::Finite(normalized.small().clone()),
                    UXBinary::Finite(normalized.width().clone()),
                ))
            }
            _ => Ok(raw_bounds),
        }
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut precision = self.precision_bits.write();

        match precision.as_ref() {
            None => {
                // First step: estimate initial precision from input bounds.
                // If input has finite bounds, use the input's precision as a starting point.
                // This avoids wasting iterations on low precision when input is already precise.
                let input_bounds = self.inner.get_bounds()?;
                let initial_precision = estimate_initial_precision(&input_bounds);
                *precision = Some(BigInt::from(initial_precision));
            }
            Some(current) => {
                // Double precision each step for O(log n) convergence to high precision.
                // This is more efficient than linear increment for reaching high precision
                // targets, as the reciprocal computation cost grows with precision.
                //
                // TODO: If state reuse is implemented (e.g., Newton-Raphson iteration that
                // builds on previous results), this should be changed to linear increment
                // to avoid computing to higher precision than requested.
                *precision = Some(current * 2);
            }
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

/// Estimates an initial precision based on the input bounds.
///
/// The idea is to start at a precision that's appropriate for the input:
/// - If the input has finite bounds with width W, we estimate precision as -log2(W)
/// - If the input has infinite bounds, we use the default initial precision
///
/// This avoids wasting iterations on low precision when the input is already precise.
/// Combined with the doubling strategy, this ensures we start at a reasonable point
/// and quickly reach high precision targets.
fn estimate_initial_precision(input_bounds: &Bounds) -> i64 {
    // Try to get effective precision from input width
    if let UXBinary::Finite(width) = input_bounds.width() {
        // The width is mantissa * 2^exponent
        // Effective precision is roughly -exponent (ignoring mantissa for simplicity)
        // We use this as a starting point, with a minimum floor
        if let Some(exp) = width.exponent().to_i64() {
            // Precision ≈ -exponent, clamped to reasonable range
            let estimated = (-exp).max(INV_INITIAL_PRECISION_BITS);
            return estimated;
        }
    }

    // Fall back to default for infinite bounds or extreme exponents
    INV_INITIAL_PRECISION_BITS
}

/// Computes reciprocal bounds for an interval.
///
/// If `precision_bits` is `None`, the precision has not been initialized yet, so
/// we return the widest possible bounds ((-inf, +inf) for intervals containing zero,
/// or the appropriate infinite bounds for strictly positive/negative intervals).
fn reciprocal_bounds(
    bounds: &Bounds,
    precision_bits: Option<&BigInt>,
) -> Result<Bounds, ComputableError> {
    let lower = bounds.small();
    let upper = bounds.large();
    let zero = XBinary::zero();
    if lower <= &zero && upper >= zero {
        return Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf));
    }

    // If precision is not yet initialized, return infinite bounds in the appropriate direction
    let Some(precision) = precision_bits else {
        return if upper < zero {
            Ok(Bounds::new(XBinary::NegInf, XBinary::zero()))
        } else {
            Ok(Bounds::new(XBinary::zero(), XBinary::PosInf))
        };
    };

    let (lower_bound, upper_bound) = if upper < zero {
        let lower_bound =
            reciprocal_rounded_abs_extended(&upper, precision, ReciprocalRounding::Ceil)?.neg();
        let upper_bound =
            reciprocal_rounded_abs_extended(lower, precision, ReciprocalRounding::Floor)?.neg();
        (lower_bound, upper_bound)
    } else {
        let lower_bound =
            reciprocal_rounded_abs_extended(&upper, precision, ReciprocalRounding::Floor)?;
        let upper_bound =
            reciprocal_rounded_abs_extended(lower, precision, ReciprocalRounding::Ceil)?;
        (lower_bound, upper_bound)
    };

    // TODO: can the type system ensure that the bounds remain ordered?
    Bounds::new_checked(lower_bound, upper_bound).map_err(|_| ComputableError::InvalidBoundsOrder)
}

#[cfg(test)]
mod tests {
    use crate::binary::{Binary, Bounds, UBinary, XBinary};
    use crate::test_utils::{
        interval_midpoint_computable, ubin, unwrap_finite, unwrap_finite_uxbinary,
    };

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
        assert_eq!(bounds, Bounds::new(XBinary::NegInf, XBinary::PosInf));
    }

    #[test]
    fn inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = ubin(1, -8);
        let bounds = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected_binary =
            XBinary::from_f64(1.0 / 3.0).expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected_value, &epsilon);
    }
}
