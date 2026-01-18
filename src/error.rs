//! Error types for computable operations.

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
        }
    }
}

impl std::error::Error for ComputableError {}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}
