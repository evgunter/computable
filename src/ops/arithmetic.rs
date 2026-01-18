//! Arithmetic operations: negation, addition, and multiplication.

use std::sync::Arc;

use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::ordered_pair::Bounds;

/// Negation operation.
pub struct NegOp {
    pub inner: Arc<Node>,
}

impl NodeOp for NegOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        let lower = existing.small().neg();
        let upper = existing.large().neg();
        Bounds::new_checked(upper, lower).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
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
        Bounds::new_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
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

        let candidates = [
            left_lower.mul(right_lower),
            left_lower.mul(&right_upper),
            left_upper.mul(right_lower),
            left_upper.mul(&right_upper),
        ];

        let mut min = candidates[0].clone();
        let mut max = candidates[0].clone();
        for candidate in candidates.iter().skip(1) {
            if candidate < &min {
                min = candidate.clone();
            }
            if candidate > &max {
                max = candidate.clone();
            }
        }

        Bounds::new_checked(min, max).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }
}
