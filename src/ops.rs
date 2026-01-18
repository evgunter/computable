//! Operations for the computation graph.
//!
//! This module contains all the NodeOp implementations:
//! - `BaseOp`: Wraps user-defined base nodes
//! - `NegOp`, `AddOp`, `MulOp`: Arithmetic operations
//! - `InvOp`: Multiplicative inverse with precision refinement
//! - `SinOp`: Sine function using Taylor series

mod arithmetic;
mod base;
mod inv;
pub mod sin;

pub use arithmetic::{AddOp, MulOp, NegOp};
pub use base::BaseOp;
pub use inv::InvOp;
pub use sin::SinOp;
