//! Arithmetic operations: negation, addition, and multiplication.

use std::sync::Arc;

use crate::binary::{Bounds, UXBinary, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::sane::XIsize;

/// Negation operation.
pub struct NegOp {
    pub inner: Arc<Node>,
}

impl NodeOp for NegOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        // Fast path: exact input → exact output.
        if existing.width().is_zero() {
            return Ok(Bounds::point(existing.small().neg()));
        }
        let lower = existing.small().neg();
        let upper = existing.large().neg();
        Ok(Bounds::new_checked(upper, lower)?)
    }

    fn refine_step(&self, _target_width_exp: XIsize) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// Negation preserves width exactly.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        target_width.clone()
    }
}

/// Addition operation.
pub struct AddOp {
    pub left: Arc<Node>,
    pub right: Arc<Node>,
}

impl NodeOp for AddOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let left_bounds = self.left.get_bounds()?;
        let right_bounds = self.right.get_bounds()?;
        // Fast path: when both inputs are exact (zero width), the sum is
        // exact. Skip the redundant upper-bound computation and the
        // width derivation in new_checked.
        if left_bounds.width().is_zero() && right_bounds.width().is_zero() {
            let sum = left_bounds.small().add_lower(right_bounds.small());
            return Ok(Bounds::point(sum));
        }
        let lower = left_bounds.small().add_lower(right_bounds.small());
        let upper = left_bounds.large().add_upper(&right_bounds.large());
        Ok(Bounds::new_checked(lower, upper)?)
    }

    fn refine_step(&self, _target_width_exp: XIsize) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// w_out = w_left + w_right, so each child gets half the target.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_index: usize) -> UXBinary {
        target_width.clone() >> 1u32
    }
}

/// Multiplication operation.
pub struct MulOp {
    pub left: Arc<Node>,
    pub right: Arc<Node>,
}

impl NodeOp for MulOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let left_bounds = self.left.get_bounds()?;
        let right_bounds = self.right.get_bounds()?;
        // Fast path: when both inputs are exact, only one multiplication
        // is needed (instead of 4 endpoint products + 6 comparisons).
        if left_bounds.width().is_zero() && right_bounds.width().is_zero() {
            let product = left_bounds.small().mul(right_bounds.small());
            return Ok(Bounds::point(product));
        }

        let zero = XBinary::zero();
        let left_lower = left_bounds.small();
        let left_upper = left_bounds.large();
        let right_lower = right_bounds.small();
        let right_upper = right_bounds.large();

        let left_non_neg = *left_lower >= zero;
        let left_non_pos = left_upper <= zero;
        let right_non_neg = *right_lower >= zero;
        let right_non_pos = right_upper <= zero;

        let (min, max) = if left_non_neg {
            if right_non_neg {
                // [a,b] >= 0, [c,d] >= 0 => [a*c, b*d]
                (left_lower.mul(right_lower), left_upper.mul(&right_upper))
            } else if right_non_pos {
                // [a,b] >= 0, [c,d] <= 0 => [b*c, a*d]
                (left_upper.mul(right_lower), left_lower.mul(&right_upper))
            } else {
                // [a,b] >= 0, right mixed => [b*c, b*d]
                (left_upper.mul(right_lower), left_upper.mul(&right_upper))
            }
        } else if left_non_pos {
            if right_non_neg {
                // [a,b] <= 0, [c,d] >= 0 => [a*d, b*c]
                (left_lower.mul(&right_upper), left_upper.mul(right_lower))
            } else if right_non_pos {
                // [a,b] <= 0, [c,d] <= 0 => [b*d, a*c]
                (left_upper.mul(&right_upper), left_lower.mul(right_lower))
            } else {
                // [a,b] <= 0, right mixed => [a*d, a*c]
                (left_lower.mul(&right_upper), left_lower.mul(right_lower))
            }
        } else {
            // left mixed (a < 0, b > 0)
            if right_non_neg {
                // left mixed, [c,d] >= 0 => [a*d, b*d]
                (left_lower.mul(&right_upper), left_upper.mul(&right_upper))
            } else if right_non_pos {
                // left mixed, [c,d] <= 0 => [b*c, a*c]
                (left_upper.mul(right_lower), left_lower.mul(right_lower))
            } else {
                // Both mixed: need all 4 products
                let a_d = left_lower.mul(&right_upper);
                let b_c = left_upper.mul(right_lower);
                let a_c = left_lower.mul(right_lower);
                let b_d = left_upper.mul(&right_upper);
                (a_d.min(b_c), a_c.max(b_d))
            }
        };

        Ok(Bounds::new_checked(min, max)?)
    }

    fn refine_step(&self, _target_width_exp: XIsize) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// w_out ≈ |a| · w_b + |b| · w_a.
    /// Child a gets target / (2·max_abs(b)), child b gets target / (2·max_abs(a)).
    fn child_demand_budget(&self, target_width: &UXBinary, child_index: usize) -> UXBinary {
        let sibling = if child_index == 0 {
            &self.right
        } else {
            &self.left
        };
        let sibling_max_abs = match sibling.cached_bounds() {
            Some(b) => {
                let (lo, hi) = b.abs();
                std::cmp::max(lo, hi)
            }
            None => return target_width.clone(), // unknown bounds → conservative pass-through
        };
        (target_width.clone() >> 1u32).div_floor(&sibling_max_abs)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::Bounds;
    use crate::test_utils::{interval_midpoint_computable, xbin};

    #[test]
    fn add_combines_bounds() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left + right;
        let sum_bounds = sum.bounds().expect("bounds should succeed");
        assert_eq!(sum_bounds, Bounds::new(xbin(1, 0), xbin(5, 0)));
    }

    #[test]
    fn sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(diff_bounds, Bounds::new(xbin(2, 0), xbin(5, 0)));
    }

    #[test]
    fn neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-3, 0), xbin(-1, 0)));
    }

    #[test]
    fn mul_combines_bounds_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(2, 0), xbin(12, 0)));
    }

    #[test]
    fn mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-12, 0), xbin(-2, 0)));
    }

    #[test]
    fn mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-10, 0), xbin(15, 0)));
    }

    #[test]
    fn mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-8, 0), xbin(12, 0)));
    }
}
