//! Arithmetic operations: negation, addition, and multiplication.

use std::sync::Arc;

use crate::binary::{Bounds, UXBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

/// Negation operation.
pub struct NegOp {
    pub inner: Arc<Node>,
}

impl NodeOp for NegOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        let lower = existing.small().neg();
        let upper = existing.large().neg();
        Ok(Bounds::new_checked(upper, lower)?)
    }

    fn refine_step(&self, _target_width_exp: i64) -> Result<bool, ComputableError> {
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
        let lower = left_bounds.small().add_lower(right_bounds.small());
        let upper = left_bounds.large().add_upper(&right_bounds.large());
        Ok(Bounds::new_checked(lower, upper)?)
    }

    fn refine_step(&self, _target_width_exp: i64) -> Result<bool, ComputableError> {
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
        let left_lower = left_bounds.small();
        let left_upper = left_bounds.large();
        let right_lower = right_bounds.small();
        let right_upper = right_bounds.large();

        let ll_rl = left_lower.mul(right_lower);
        let ll_ru = left_lower.mul(&right_upper);
        let lu_rl = left_upper.mul(right_lower);
        let lu_ru = left_upper.mul(&right_upper);

        let min = ll_rl
            .clone()
            .min(ll_ru.clone())
            .min(lu_rl.clone())
            .min(lu_ru.clone());
        let max = ll_rl.max(ll_ru).max(lu_rl).max(lu_ru);

        Ok(Bounds::new_checked(min, max)?)
    }

    fn refine_step(&self, _target_width_exp: i64) -> Result<bool, ComputableError> {
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
