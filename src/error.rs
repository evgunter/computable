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
//!
//! - **`detected_computable_would_exhaust_memory!(...)`**: Use for cases where the numbers
//!   involved are so large that instantiating them would cause an out-of-memory condition.
//!   An explicit panic is preferable to an OOM crash, so we make an exception to our no-panics
//!   policy. Unlike the infinite-value macro, this always panics (not just in debug builds)
//!   because there is no safe fallback.
//!
//! - **`sane_arithmetic!(var1, var2, ...; expr)`**: Use for integer arithmetic on values
//!   representing computation sizes (precision bits, term counts, bit lengths). Each
//!   guard variable is shadowed as a [`Sane`] newtype whose operators use `checked_*`
//!   internally, panicking on overflow. The result is unwrapped back to `usize` and
//!   checked against `MAX_COMPUTATION_BITS`.

/// Maximum reasonable computation size in bits. A computation requiring more than
/// ~2^32 bits of precision would need ~512 MB just to store one number, and intermediate
/// results would require far more. Guaranteed `<= usize::MAX` on all platforms.
pub const MAX_COMPUTATION_BITS: usize = if usize::BITS >= 32 {
    #[allow(clippy::as_conversions)] // safe: branch guards usize::BITS >= 32
    {
        u32::MAX as usize
    }
} else {
    usize::MAX
};

use crate::binary::BinaryError;
use crate::ordered_pair::IntervalError;

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
/// ```should_panic
/// computable::detected_computable_with_infinite_value!("lower input bound is PosInf");
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
/// Macro to flag operations that would exhaust memory if attempted.
///
/// Some computations involve numbers so large that instantiating them would
/// cause an out-of-memory condition. An explicit panic with a clear message
/// is preferable to an OOM crash, so we make an exception to our no-panics
/// policy for these cases.
///
/// Unlike `detected_computable_with_infinite_value!` (which uses `debug_assert`
/// because the code has a reasonable fallback), this macro always panics because
/// there is no safe way to continue — attempting to proceed would OOM.
///
/// # Arguments
///
/// * `$msg` - A description of what case was encountered
///
/// # Example
///
/// ```should_panic
/// computable::detected_computable_would_exhaust_memory!("shift by 2^64 bits");
/// ```
#[macro_export]
macro_rules! detected_computable_would_exhaust_memory {
    ($msg:expr) => {
        panic!(concat!($msg, " - would exhaust memory if attempted"))
    };
}

/// Asserts that a computation size parameter (precision bits, term count, bit length,
/// etc.) is within reasonable bounds for memory.
///
/// Integer arithmetic on these values (e.g., `precision + 10` or `n * 3`) could
/// theoretically overflow, but if the value is large enough to overflow a `usize`
/// or `u64`, the computation would exhaust memory long before reaching that point.
/// This macro makes that reasoning explicit: call it on operands before doing
/// arithmetic, then `#[allow(clippy::arithmetic_side_effects)]` the arithmetic itself.
///
/// # Arguments
///
/// * `$val` - An integer value representing a computation size
///
/// # Panics
///
/// Panics via `detected_computable_would_exhaust_memory!` if the value exceeds
/// `MAX_COMPUTATION_BITS` (2^32).
///
/// # Example
///
/// ```should_panic
/// computable::assert_sane_computation_size!(usize::MAX);
/// ```
#[macro_export]
macro_rules! assert_sane_computation_size {
    ($val:expr) => {
        // $val must be usize. If it exceeds MAX_COMPUTATION_BITS, the
        // computation would exhaust memory, so we panic early.
        let __val: usize = $val;
        if __val > $crate::MAX_COMPUTATION_BITS {
            $crate::detected_computable_would_exhaust_memory!(concat!(
                stringify!($val),
                " exceeds MAX_COMPUTATION_BITS"
            ));
        }
    };
}

/// Guards one or more `usize` variables and evaluates an arithmetic expression using
/// checked arithmetic via [`Sane`].
///
/// Each guard variable is shadowed as a [`Sane`] newtype whose `+`, `-`, `*`, `/`
/// operators use `checked_*` internally, panicking via
/// [`detected_computable_would_exhaust_memory!`] on overflow. The result is unwrapped
/// back to `usize` and checked against [`MAX_COMPUTATION_BITS`].
///
/// # Syntax
///
/// ```text
/// sane_arithmetic!(var1, var2, ...; expression)
/// ```
///
/// Guards must be identifiers (not arbitrary expressions).
///
/// # Example
///
/// ```
/// use computable::sane_arithmetic;
///
/// let num_terms: usize = 10;
/// let exponent = sane_arithmetic!(num_terms; 2 * num_terms + 1);
/// assert_eq!(exponent, 21);
/// ```
#[macro_export]
macro_rules! sane_arithmetic {
    ($($guard:ident),+ ; $expr:expr) => {{
        $(
            #[allow(clippy::shadow_reuse)]
            let $guard = $crate::Sane($guard);
        )+
        let $crate::Sane(__result) = { $expr };
        $crate::assert_sane_computation_size!(__result);
        __result
    }};
}

/// Converts a `u64` bit count (e.g. from `BigUint::bits()`) to `usize`,
/// panicking if the value exceeds `MAX_COMPUTATION_BITS`.
///
/// This centralizes the one unavoidable `u64 -> usize` platform cast that
/// arises because `num_bigint::BigUint::bits()` returns `u64` but shift
/// operations and precision parameters use `usize`.
///
/// # Panics
///
/// Panics via `detected_computable_would_exhaust_memory!` if `bits` exceeds
/// `MAX_COMPUTATION_BITS`.
pub fn bits_as_usize(bits: u64) -> usize {
    // MAX_COMPUTATION_BITS <= usize::MAX by construction, so this single check
    // guarantees both "won't exhaust memory" and "fits in usize".
    #[allow(clippy::as_conversions)] // usize -> u64: always widens or is a no-op
    let max = MAX_COMPUTATION_BITS as u64;
    if bits > max {
        detected_computable_would_exhaust_memory!("bit count exceeds MAX_COMPUTATION_BITS");
    }
    #[allow(clippy::as_conversions)] // safe: bits <= MAX_COMPUTATION_BITS <= usize::MAX
    {
        bits as usize
    }
}

/// Subtracts one from a `NonZeroUsize`, returning the result as `usize`.
///
/// This is trivially correct: `NonZeroUsize` guarantees `>= 1`, so `- 1 >= 0`.
pub fn sub_one(n: std::num::NonZeroUsize) -> usize {
    #[allow(clippy::arithmetic_side_effects)]
    {
        n.get() - 1
    }
}

/// Newtype for checked arithmetic on computation-size values.
///
/// Overloads `+`, `-`, `*`, `/` to use `checked_*` internally, panicking via
/// [`detected_computable_would_exhaust_memory!`] on overflow. This makes it
/// impossible to silently overflow when doing arithmetic on precision bits,
/// term counts, etc.
///
/// Created automatically by the [`sane_arithmetic!`] macro — not intended for
/// direct construction outside of that macro.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sane(pub usize);

// Compile-time assertions:
// - usize ≤ 64 bits: our overflow reasoning assumes checked_* catches all overflows
// - usize ≥ 32 bits: u32 literals can be converted to usize with `as`
const _: () = assert!(usize::BITS <= 64);
const _: () = assert!(u32::BITS <= usize::BITS);

/// Implements checked arithmetic operators for [`Sane`].
///
/// Each invocation generates three impls: `Sane op Sane`, `Sane op u32`, and
/// `u32 op Sane`. The `u32` variants convert via `as usize` (safe per
/// compile-time assert `u32::BITS <= usize::BITS`).
macro_rules! impl_sane_binary_op {
    ($trait:ident, $method:ident, $checked:ident, $msg:literal) => {
        impl core::ops::$trait for Sane {
            type Output = Sane;
            #[inline]
            fn $method(self, rhs: Sane) -> Sane {
                match self.0.$checked(rhs.0) {
                    Some(r) => Sane(r),
                    None => crate::detected_computable_would_exhaust_memory!($msg),
                }
            }
        }

        impl core::ops::$trait<u32> for Sane {
            type Output = Sane;
            #[inline]
            fn $method(self, rhs: u32) -> Sane {
                #[allow(clippy::as_conversions)]
                // safe: compile-time assert guarantees u32 fits in usize
                core::ops::$trait::$method(self, Sane(rhs as usize))
            }
        }

        impl core::ops::$trait<Sane> for u32 {
            type Output = Sane;
            #[inline]
            fn $method(self, rhs: Sane) -> Sane {
                #[allow(clippy::as_conversions)]
                // safe: compile-time assert guarantees u32 fits in usize
                core::ops::$trait::$method(Sane(self as usize), rhs)
            }
        }
    };
}

impl_sane_binary_op!(Add, add, checked_add, "Sane addition overflow");
impl_sane_binary_op!(Sub, sub, checked_sub, "Sane subtraction underflow");
impl_sane_binary_op!(Mul, mul, checked_mul, "Sane multiplication overflow");
impl_sane_binary_op!(Div, div, checked_div, "Sane division by zero");

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

    #[test]
    fn detected_computable_with_infinite_value_macro_compiles() {
        // Verifies the macro is a no-op in release mode (debug_assertions disabled).
        #[cfg(not(debug_assertions))]
        {
            detected_computable_with_infinite_value!("test message");
        }
    }

    #[test]
    #[should_panic(expected = "test message")]
    #[cfg(debug_assertions)]
    fn detected_computable_with_infinite_value_macro_panics_in_debug() {
        detected_computable_with_infinite_value!("test message");
    }

    #[test]
    #[should_panic(expected = "test message")]
    fn detected_computable_would_exhaust_memory_macro_panics() {
        detected_computable_would_exhaust_memory!("test message");
    }

    #[test]
    fn sane_arithmetic_basic_ops() {
        assert_eq!(Sane(3) + Sane(4), Sane(7));
        assert_eq!(Sane(10) - Sane(3), Sane(7));
        assert_eq!(Sane(5) * Sane(6), Sane(30));
        assert_eq!(Sane(20) / Sane(4), Sane(5));
    }

    #[test]
    fn sane_arithmetic_u32_cross_ops() {
        assert_eq!(Sane(3) + 4_u32, Sane(7));
        assert_eq!(4_u32 + Sane(3), Sane(7));
        assert_eq!(Sane(10) - 3_u32, Sane(7));
        assert_eq!(2_u32 * Sane(5), Sane(10));
        assert_eq!(Sane(5) * 2_u32, Sane(10));
        assert_eq!(Sane(20) / 4_u32, Sane(5));
    }

    #[test]
    fn sane_arithmetic_macro_works() {
        let n: usize = 10;
        let result = crate::sane_arithmetic!(n; 2 * n + 1);
        assert_eq!(result, 21_usize);
    }

    #[test]
    #[should_panic(expected = "Sane multiplication overflow")]
    fn sane_mul_overflow_panics() {
        let _ = Sane(usize::MAX) * Sane(2);
    }

    #[test]
    #[should_panic(expected = "Sane subtraction underflow")]
    fn sane_sub_underflow_panics() {
        let _ = Sane(3) - Sane(5);
    }

    #[test]
    #[should_panic(expected = "Sane division by zero")]
    fn sane_div_by_zero_panics() {
        let _ = Sane(10) / Sane(0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_COMPUTATION_BITS")]
    fn sane_arithmetic_macro_rejects_large_result() {
        let n: usize = MAX_COMPUTATION_BITS;
        // n + 1 doesn't overflow usize, but exceeds MAX_COMPUTATION_BITS
        let _ = crate::sane_arithmetic!(n; n + 1);
    }
}
