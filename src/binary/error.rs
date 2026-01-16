//! Error types for binary number operations.
//!
//! This module contains error types used across the binary number implementations:
//! - `BinaryError`: Errors from basic binary operations
//! - `XBinaryError`: Errors from extended binary operations (includes NaN handling)

use std::fmt;

/// Errors that can occur during binary number operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    /// Exponent overflow occurred during reciprocal computation.
    ReciprocalOverflow,
    /// Cannot create an unsigned binary from a negative mantissa.
    NegativeMantissa,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReciprocalOverflow => write!(f, "exponent overflow during reciprocal"),
            Self::NegativeMantissa => {
                write!(f, "cannot create unsigned binary from negative mantissa")
            }
        }
    }
}

impl std::error::Error for BinaryError {}

/// Errors that can occur during extended binary operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XBinaryError {
    /// Cannot convert NaN to XBinary.
    Nan,
    /// An underlying binary error occurred.
    Binary(BinaryError),
}

impl fmt::Display for XBinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => write!(f, "cannot convert NaN to XBinary"),
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for XBinaryError {}

impl From<BinaryError> for XBinaryError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_error_display() {
        assert_eq!(
            BinaryError::ReciprocalOverflow.to_string(),
            "exponent overflow during reciprocal"
        );
        assert_eq!(
            BinaryError::NegativeMantissa.to_string(),
            "cannot create unsigned binary from negative mantissa"
        );
    }

    #[test]
    fn xbinary_error_display() {
        assert_eq!(XBinaryError::Nan.to_string(), "cannot convert NaN to XBinary");
        assert_eq!(
            XBinaryError::Binary(BinaryError::ReciprocalOverflow).to_string(),
            "exponent overflow during reciprocal"
        );
    }

    #[test]
    fn binary_error_converts_to_xbinary_error() {
        let binary_err = BinaryError::NegativeMantissa;
        let xbinary_err: XBinaryError = binary_err.into();
        assert!(matches!(xbinary_err, XBinaryError::Binary(BinaryError::NegativeMantissa)));
    }
}
