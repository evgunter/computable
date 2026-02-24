//! Error types for computable operations.
//!
//! # Handling Mathematically Impossible Cases
//!
//! The codebase uses the following conventions for cases that should be mathematically
//! impossible given the invariants of the types involved:
//!
//! - **`unreachable!(...)`**: Use for cases that are truly impossible given the current
//!   type invariants. Always include a TODO comment about investigating whether the type
//!   system could prevent the case from being representable in the first place.
//!
//! See [`crate::sane`] for the `sane_arithmetic!` macro, `Sane` newtype, and diagnostic
//! macros (`detected_computable_with_infinite_value!`, `detected_computable_would_exhaust_memory!`).

use crate::binary::BinaryError;
use crate::ordered_pair::IntervalError;

use std::fmt;

/// Errors that can occur during computable operations and refinement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputableError {
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
            Self::InfiniteBounds => write!(
                f,
                "input bounds are infinite where finite bounds are required"
            ),
        }
    }
}

impl std::error::Error for ComputableError {}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

impl From<IntervalError> for ComputableError {
    fn from(error: IntervalError) -> Self {
        match error {
            IntervalError::InvalidOrder => Self::InvalidBoundsOrder,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_error_converts_to_computable_error() {
        let binary_err = BinaryError::NegativeMantissa;
        let computable_err: ComputableError = binary_err.into();
        assert!(matches!(
            computable_err,
            ComputableError::Binary(BinaryError::NegativeMantissa)
        ));
    }

    #[test]
    fn computable_error_implements_std_error() {
        let err: &dyn std::error::Error = &ComputableError::DomainError;
        // Verify it implements the Error trait by calling source()
        assert!(err.source().is_none());
    }
}
