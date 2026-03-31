//! Arithmetic operations: negation, addition, and multiplication.

use std::sync::Arc;

use crate::binary::{UXBinary, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::XI;

/// Negation operation.
pub struct NegOp {
    pub(crate) inner: Arc<Node>,
}

impl NegOp {
    pub fn new(inner: Arc<Node>) -> Self {
        Self { inner }
    }
}

impl NodeOp for NegOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let inner_prefix = self.inner.get_prefix()?;
        // Negate lower and upper, then swap (negation reverses order)
        let lower = inner_prefix.lower().neg();
        let upper = inner_prefix.upper().neg();
        // After negation, the old upper becomes the new lower
        Ok(Prefix::from_lower_upper(upper, lower))
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// Negation preserves width exactly.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_idx: bool) -> UXBinary {
        target_width.clone()
    }
}

/// Addition operation.
pub struct AddOp {
    pub(crate) left: Arc<Node>,
    pub(crate) right: Arc<Node>,
}

impl AddOp {
    pub fn new(left: Arc<Node>, right: Arc<Node>) -> Self {
        Self { left, right }
    }
}

impl NodeOp for AddOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let left_prefix = self.left.get_prefix()?;
        let right_prefix = self.right.get_prefix()?;
        let left_lower = left_prefix.lower();
        let left_upper = left_prefix.upper();
        let right_lower = right_prefix.lower();
        let right_upper = right_prefix.upper();
        // Fast path: when both inputs are exact (zero width), the sum is exact.
        if left_prefix.width_exponent() == XI::NegInf && right_prefix.width_exponent() == XI::NegInf
        {
            let sum = left_lower.add_lower(&right_lower);
            return Ok(Prefix::from_lower_upper(sum.clone(), sum));
        }
        let lower = left_lower.add_lower(&right_lower);
        let upper = left_upper.add_upper(&right_upper);
        Ok(Prefix::from_lower_upper(lower, upper))
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }

    /// w_out = w_left + w_right, so each child gets half the target.
    fn child_demand_budget(&self, target_width: &UXBinary, _child_idx: bool) -> UXBinary {
        target_width.clone() >> 1u32
    }
}

/// Multiplication operation.
pub struct MulOp {
    pub(crate) left: Arc<Node>,
    pub(crate) right: Arc<Node>,
}

impl MulOp {
    pub fn new(left: Arc<Node>, right: Arc<Node>) -> Self {
        Self { left, right }
    }
}

impl NodeOp for MulOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let left_prefix = self.left.get_prefix()?;
        let right_prefix = self.right.get_prefix()?;
        let left_lower = left_prefix.lower();
        let left_upper = left_prefix.upper();
        let right_lower = right_prefix.lower();
        let right_upper = right_prefix.upper();
        // Fast path: when both inputs are exact, only one multiplication is needed.
        if left_prefix.width_exponent() == XI::NegInf && right_prefix.width_exponent() == XI::NegInf
        {
            let product = left_lower.mul(&right_lower);
            return Ok(Prefix::from_lower_upper(product.clone(), product));
        }

        let zero = XBinary::zero();

        let left_non_neg = left_lower >= zero;
        let left_non_pos = left_upper <= zero;
        let right_non_neg = right_lower >= zero;
        let right_non_pos = right_upper <= zero;

        let (min, max) = if left_non_neg {
            if right_non_neg {
                (left_lower.mul(&right_lower), left_upper.mul(&right_upper))
            } else if right_non_pos {
                (left_upper.mul(&right_lower), left_lower.mul(&right_upper))
            } else {
                (left_upper.mul(&right_lower), left_upper.mul(&right_upper))
            }
        } else if left_non_pos {
            if right_non_neg {
                (left_lower.mul(&right_upper), left_upper.mul(&right_lower))
            } else if right_non_pos {
                (left_upper.mul(&right_upper), left_lower.mul(&right_lower))
            } else {
                (left_lower.mul(&right_upper), left_lower.mul(&right_lower))
            }
        } else {
            // left mixed (a < 0, b > 0)
            if right_non_neg {
                (left_lower.mul(&right_upper), left_upper.mul(&right_upper))
            } else if right_non_pos {
                (left_upper.mul(&right_lower), left_lower.mul(&right_lower))
            } else {
                // Both mixed: need all 4 products
                let a_d = left_lower.mul(&right_upper);
                let b_c = left_upper.mul(&right_lower);
                let a_c = left_lower.mul(&right_lower);
                let b_d = left_upper.mul(&right_upper);
                (a_d.min(b_c), a_c.max(b_d))
            }
        };

        Ok(Prefix::from_lower_upper(min, max))
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
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
    fn child_demand_budget(&self, target_width: &UXBinary, child_idx: bool) -> UXBinary {
        let sibling = if child_idx { &self.left } else { &self.right };
        let sibling_max_abs = match sibling.cached_prefix() {
            Some(p) => {
                let (lo, hi) = p.abs();
                std::cmp::max(lo, hi)
            }
            None => return target_width.clone(), // unknown prefix → conservative pass-through
        };
        (target_width.clone() >> 1u32).div_floor(&sibling_max_abs)
    }

    fn budget_depends_on_bounds(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::prefix::Prefix;
    use crate::test_utils::{interval_midpoint_computable, xbin};

    #[test]
    fn add_combines_bounds() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left + right;
        let prefix = sum.prefix().expect("prefix should succeed");
        assert_eq!(prefix, Prefix::from_lower_upper(xbin(1, 0), xbin(5, 0)));
    }

    #[test]
    fn sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let prefix = diff.prefix().expect("prefix should succeed");
        assert_eq!(prefix, Prefix::from_lower_upper(xbin(2, 0), xbin(5, 0)));
    }

    #[test]
    fn neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let prefix = negated.prefix().expect("prefix should succeed");
        assert_eq!(prefix, Prefix::from_lower_upper(xbin(-3, 0), xbin(-1, 0)));
    }

    #[test]
    fn mul_combines_bounds_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let prefix = product.prefix().expect("prefix should succeed");
        assert_eq!(prefix, Prefix::from_lower_upper(xbin(2, 0), xbin(12, 0)));
    }

    #[test]
    fn mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let prefix = product.prefix().expect("prefix should succeed");
        assert_eq!(prefix, Prefix::from_lower_upper(xbin(-12, 0), xbin(-2, 0)));
    }

    #[test]
    fn mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let prefix = product.prefix().expect("prefix should succeed");
        // Prefix may widen bounds (power-of-2 width rounding), so check containment
        assert!(prefix.lower() <= xbin(-10, 0));
        assert!(prefix.upper() >= xbin(15, 0));
    }

    #[test]
    fn mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let prefix = product.prefix().expect("prefix should succeed");
        assert!(prefix.lower() <= xbin(-8, 0));
        assert!(prefix.upper() >= xbin(12, 0));
    }
}
