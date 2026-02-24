//! Computable real numbers with provable correctness.
//!
//! This crate provides a framework for exact real arithmetic using interval refinement.
//! Numbers are represented as computations that can be refined to arbitrary precision
//! while maintaining provably correct bounds.
//!
//! # Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`binary`]: Arbitrary-precision binary numbers (mantissa + exponent representation)
//! - [`ordered_pair`]: Interval types with bounds checking (Bounds, Interval)
//! - [`error`]: Error types for computable operations
//! - [`node`]: Computation graph infrastructure (Node, NodeOp traits)
//! - [`ops`]: Arithmetic and transcendental operations (add, mul, inv, sin, etc.)
//! - [`refinement`]: Parallel refinement infrastructure
//! - [`computable`]: The main Computable type
//!
//! # Example
//!
//! ```
//! use computable::{Computable, Binary};
//! use num_bigint::{BigInt, BigUint};
//!
//! // Create a constant
//! let x = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
//!
//! // Arithmetic operations
//! let y = x.clone() + x.clone();
//! let z = y * x;
//!
//! // Get current bounds
//! let bounds = z.bounds().unwrap();
//! ```

// Forbid panic-related lints in non-test code (tests can still use expect/panic/unwrap)
#![cfg_attr(not(test), forbid(clippy::expect_used))]
#![cfg_attr(not(test), forbid(clippy::panic))]
#![cfg_attr(not(test), forbid(clippy::unwrap_used))]
// Lib-only lint: examples and benches need to print
#![deny(clippy::print_stdout)]
// Promote shared lints (from Cargo.toml [lints.clippy]) to deny for the lib crate.
// Examples and benches keep them as non-blocking warnings.
#![deny(
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::dbg_macro,
    clippy::print_stderr,
    clippy::unimplemented,
    clippy::wildcard_enum_match_arm,
    clippy::let_underscore_must_use,
    clippy::arithmetic_side_effects,
    clippy::impl_trait_in_params,
    clippy::field_scoped_visibility_modifiers,
    clippy::as_conversions,
    clippy::lossy_float_literal,
    clippy::default_numeric_fallback,
    clippy::map_err_ignore,
    clippy::missing_asserts_for_indexing,
    clippy::undocumented_unsafe_blocks
)]

// External modules (already exist)
mod binary;
pub mod binary_utils;
mod concurrency;
mod ordered_pair;

// New internal modules
mod computable;
mod error;
mod node;
mod ops;
mod refinement;
mod sane;

// Test utilities module (only compiled in test mode)
#[cfg(test)]
pub mod test_utils;

// Re-export public API
pub use binary::Bounds;
pub use binary::{Binary, BinaryError, FiniteBounds, UBinary, UXBinary, XBinary, XBinaryError};
pub use computable::{Computable, DEFAULT_INV_MAX_REFINES, DEFAULT_MAX_REFINEMENT_ITERATIONS};
pub use error::ComputableError;
pub use ops::{pi, pi_bounds_at_precision};
pub use ordered_pair::{Interval, IntervalError};
pub use sane::{MAX_COMPUTATION_BITS, Sane};
