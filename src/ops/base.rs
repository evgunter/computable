//! Base operation that wraps user-defined leaf nodes.

use std::sync::Arc;

use crate::binary::Bounds;
use crate::error::ComputableError;
use crate::node::{BaseNode, BoundsAccess, Node, NodeOp};

/// Operation that wraps a user-defined base node.
pub struct BaseOp {
    pub base: Arc<dyn BaseNode>,
}

impl NodeOp for BaseOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        BoundsAccess::get_bounds(self.base.as_ref())
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        self.base.refine()?;
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }
}
