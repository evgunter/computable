//! Operations for the computation graph.
//!
//! This module contains all the NodeOp implementations:
//! - `BaseOp`: Wraps user-defined base nodes
//! - `NegOp`, `AddOp`, `MulOp`: Arithmetic operations
//! - `InvOp`: Multiplicative inverse with precision refinement
//! - `SinOp`: Sine function using Taylor series
//! - `NthRootOp`: N-th root using binary search

mod arithmetic;
mod base;
mod inv;
pub mod nth_root;
pub mod sin;

pub use arithmetic::{AddOp, MulOp, NegOp};
pub use base::BaseOp;
pub use inv::InvOp;
pub use nth_root::NthRootOp;
pub use sin::SinOp;
