//! Base operation that wraps user-defined leaf nodes.

use std::sync::Arc;

use crate::binary::UXBinary;
use crate::error::ComputableError;
use crate::node::{BaseNode, Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::XI;

/// Operation that wraps a user-defined base node.
pub struct BaseOp {
    base: Arc<dyn BaseNode>,
}

impl BaseOp {
    pub fn new(base: Arc<dyn BaseNode>) -> Self {
        Self { base }
    }
}

impl NodeOp for BaseOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        self.base.get_prefix()
    }

    fn refine_step(&self, _target_width_exp: XI) -> Result<bool, ComputableError> {
        self.base.refine()?;
        let prefix = self.base.get_prefix()?;
        if prefix.width_exponent() == XI::NegInf {
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

    fn child_demand_budget(&self, _target_width: &UXBinary, _child_idx: bool) -> UXBinary {
        unreachable!("BaseOp has no children")
    }
}
