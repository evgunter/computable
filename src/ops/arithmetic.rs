//! Arithmetic operations: negation, addition, and multiplication.

use std::sync::Arc;

use crate::binary::UXBinary;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;

/// Negation operation.
pub struct NegOp {
    pub inner: Arc<Node>,
}

impl NodeOp for NegOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let existing = self.inner.get_prefix()?;
        let lower = existing.lower().neg();
        let upper = existing.upper().neg();
        Ok(Prefix::from_lower_upper(upper, lower))
    }

    fn refine_step(&self, _precision_bits: usize) -> Result<bool, ComputableError> {
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
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let left_prefix = self.left.get_prefix()?;
        let right_prefix = self.right.get_prefix()?;
        let lower = left_prefix.lower().add_lower(&right_prefix.lower());
        let upper = left_prefix.upper().add_upper(&right_prefix.upper());
        Ok(Prefix::from_lower_upper(lower, upper))
    }

    fn refine_step(&self, _precision_bits: usize) -> Result<bool, ComputableError> {
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
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let left_prefix = self.left.get_prefix()?;
        let right_prefix = self.right.get_prefix()?;
        let left_lower = left_prefix.lower();
        let left_upper = left_prefix.upper();
        let right_lower = right_prefix.lower();
        let right_upper = right_prefix.upper();

        let ll_rl = left_lower.mul(&right_lower);
        let ll_ru = left_lower.mul(&right_upper);
        let lu_rl = left_upper.mul(&right_lower);
        let lu_ru = left_upper.mul(&right_upper);

        let min = ll_rl
            .clone()
            .min(ll_ru.clone())
            .min(lu_rl.clone())
            .min(lu_ru.clone());
        let max = ll_rl.max(ll_ru).max(lu_rl).max(lu_ru);

        Ok(Prefix::from_lower_upper(min, max))
    }

    fn refine_step(&self, _precision_bits: usize) -> Result<bool, ComputableError> {
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
        let sibling_max_abs = match sibling.cached_prefix() {
            Some(p) => {
                let (lo, hi) = p.abs();
                std::cmp::max(lo, hi)
            }
            None => return target_width.clone(), // unknown prefix → conservative pass-through
        };
        (target_width.clone() >> 1u32).div_floor(&sibling_max_abs)
    }

    fn budget_depends_on_prefix(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{assert_bounds_contain, interval_midpoint_computable, xbin};

    #[test]
    fn add_combines_prefix() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left + right;
        let prefix = sum.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(1, 0), &xbin(5, 0));
    }

    #[test]
    fn sub_combines_prefix() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let prefix = diff.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(2, 0), &xbin(5, 0));
    }

    #[test]
    fn neg_flips_prefix() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let prefix = negated.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(-3, 0), &xbin(-1, 0));
    }

    #[test]
    fn mul_combines_prefix_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let prefix = product.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(2, 0), &xbin(12, 0));
    }

    #[test]
    fn mul_combines_prefix_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let prefix = product.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(-12, 0), &xbin(-2, 0));
    }

    #[test]
    fn mul_combines_prefix_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let prefix = product.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(-10, 0), &xbin(15, 0));
    }

    #[test]
    fn mul_combines_prefix_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let prefix = product.prefix().expect("bounds should succeed");
        assert_bounds_contain(&prefix, &xbin(-8, 0), &xbin(12, 0));
    }
}
