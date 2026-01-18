//! Utility functions and algorithms for working with Binary numbers.
//!
//! This module contains algorithms that operate on [`Binary`](crate::binary::Binary)
//! and related types but are not part of the core binary number representation.
//!
//! # Modules
//!
//! - [`bisection`]: Binary search helper for iterative refinement
//!
//! TODO: Investigate if any pure-Binary operations from `ops/` should be moved here.
//! Candidates might include helper functions that don't depend on Node/Computable
//! infrastructure (e.g., power functions, polynomial evaluation).

pub mod bisection;
