//! Base operation that wraps user-defined leaf nodes.

use std::sync::Arc;

use crate::binary::{Bounds, UXBinary};
use crate::error::ComputableError;
use crate::node::{BaseNode, BoundsAccess, Node, NodeOp};
use crate::sane::{U, XI};

/// Operation that wraps a user-defined base node.
pub struct BaseOp {
    pub base: Arc<dyn BaseNode>,
}

impl NodeOp for BaseOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        BoundsAccess::get_bounds(self.base.as_ref())
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
        self.base.refine()?;
        let bounds = BoundsAccess::get_bounds(self.base.as_ref())?;
        if bounds.small() == bounds.large() {
            return Ok(false);
        }
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }

    fn child_demand_budget(&self, _target_width: &UXBinary, _child_index: U) -> UXBinary {
        unreachable!("BaseOp has no children")
    }
}
