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
//! - **`detected_computable_with_infinite_value!(...)`**: Use for cases where code encounters
//!   infinite values that are currently unexpected but might become valid in the future
//!   (e.g., if we add extended real number support). This macro wraps `debug_assert!(false, ...)`
//!   to provide a consistent way to flag these cases.

use crate::binary::BinaryError;

/// Macro to flag unexpected but potentially valid extended reals cases.
///
/// This is used to detect cases where code encounters infinite values that
/// shouldn't occur currently but might become valid if we later support
/// computations in the extended reals.
///
/// In debug builds, this triggers a panic to help identify bugs early.
/// In release builds, this is a no-op.
///
/// # Arguments
///
/// * `$msg` - A description of what case was encountered (e.g., "lower input bound is PosInf")
///
/// # Example
///
/// ```ignore
/// detected_computable_with_infinite_value!("lower input bound is PosInf");
/// ```
#[macro_export]
macro_rules! detected_computable_with_infinite_value {
    ($msg:expr) => {
        debug_assert!(
            false,
            concat!($msg, " - unexpected but may be valid for extended reals")
        )
    };
}
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
