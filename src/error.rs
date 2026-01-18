//! Error types for computable operations.
//!
//! TODO: Standardize how we handle mathematically impossible cases throughout the
//! codebase. Options include:
//! - `debug_assert!(false, ...)` - only panics in debug builds, silent in release
//! - `panic!(...)` - always panics
//! - `unreachable!()` - semantically clearer but same as panic
//! - Return an error variant (e.g., `ComputableError::InternalError`)
//! Currently we use `debug_assert!` in some places (e.g., impossible bounds ordering
//! in nth_root), but this should be consistent across the codebase.

use crate::binary::BinaryError;
use std::fmt;

/// Errors that can occur during computable operations and refinement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputableError {
    /// Epsilon must be positive for refinement.
    NonpositiveEpsilon,
    /// Computed bounds are not in correct order (lower > upper).
    InvalidBoundsOrder,
    /// Refinement produced worse bounds than before.
    BoundsWorsened,
    /// Refinement did not change the state.
    StateUnchanged,
    /// Cannot refine bounds to exclude a particular value.
    ExcludedValueUnreachable,
    /// The refinement coordination channel was closed unexpectedly.
    RefinementChannelClosed,
    /// Maximum refinement iterations reached without meeting precision.
    MaxRefinementIterations { max: usize },
    /// Error from binary number operations.
    Binary(BinaryError),
    /// Input is outside the domain of the operation (e.g., negative for even roots).
    DomainError,
    /// Input bounds are infinite where finite bounds are required.
    InfiniteBounds,
}

impl fmt::Display for ComputableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonpositiveEpsilon => write!(f, "epsilon must be positive"),
            Self::InvalidBoundsOrder => write!(f, "computed bounds are not ordered"),
            Self::BoundsWorsened => write!(f, "refinement produced worse bounds"),
            Self::StateUnchanged => write!(f, "refinement did not change state"),
            Self::ExcludedValueUnreachable => write!(f, "cannot refine bounds to exclude value"),
            Self::RefinementChannelClosed => {
                write!(f, "refinement coordination channel closed")
            }
            Self::MaxRefinementIterations { max } => {
                write!(f, "maximum refinement iterations ({max}) reached")
            }
            Self::Binary(err) => write!(f, "{err}"),
            Self::DomainError => write!(f, "input is outside the domain of the operation"),
            Self::InfiniteBounds => write!(f, "input bounds are infinite where finite bounds are required"),
        }
    }
}

impl std::error::Error for ComputableError {}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}
