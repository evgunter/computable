//! Utility functions and algorithms for working with Binary numbers.
//!
//! This module contains algorithms that operate on [`Binary`](crate::binary::Binary)
//! and related types but are not part of the core binary number representation.
//!
//! # Modules
//!
//! - [`bisection`]: Binary search helper for iterative refinement
//! - [`power`]: Integer power operations for Binary and XBinary

// TODO: Migrate from mod.rs to the newer Rust convention of using binary_utils.rs
// with `#[path]` attributes or directory-named files.

pub mod bisection;
pub mod power;
