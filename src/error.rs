//! Error types for computable operations.
//!
//! # Handling Mathematically Impossible Cases
//!
//! This codebase follows a consistent approach for handling cases that should be
//! mathematically impossible (e.g., a lower bound being +∞, or bounds being in
//! wrong order when an algorithm guarantees correct ordering):
//!
//! ## For functions returning `Result<T, ComputableError>`
//!
//! Use [`ComputableError::InternalError`] with a descriptive message. This variant
//! indicates a bug in the code (invariant violation) rather than a user error.
//!
//! ```ignore
//! return Err(ComputableError::InternalError("lower bound cannot be +∞".into()));
//! ```
//!
//! ## For functions NOT returning `Result`
//!
//! Use `unreachable!()` with a descriptive message. This panics in both debug and
//! release builds, ensuring bugs are never silently ignored.
//!
//! ```ignore
//! unreachable!("bounds are not ordered: this indicates a bug");
//! ```
//!
//! ## Why NOT `debug_assert!`
//!
//! We avoid `debug_assert!(false, ...)` for impossible cases because it only panics
//! in debug builds. In release builds, the code silently continues with potentially
//! incorrect results, which violates our correctness guarantees.

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
    /// Internal error indicating a bug in the implementation.
    ///
    /// This error is used for mathematically impossible cases that indicate
    /// an invariant violation. If you encounter this error, please file a bug report.
    InternalError(String),
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
            Self::InternalError(msg) => write!(f, "internal error (bug): {msg}"),
        }
    }
}

impl std::error::Error for ComputableError {}

/// Creates a [`ComputableError::InternalError`] with a formatted message.
///
/// Use this macro for cases that should be mathematically impossible,
/// indicating a bug in the implementation.
///
/// # Example
///
/// ```ignore
/// use computable::internal_error;
///
/// fn example() -> Result<(), ComputableError> {
///     Err(internal_error!("lower bound {} cannot exceed upper bound {}", lower, upper))
/// }
/// ```
#[macro_export]
macro_rules! internal_error {
    ($($arg:tt)*) => {
        $crate::error::ComputableError::InternalError(format!($($arg)*))
    };
}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}
