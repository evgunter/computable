//! Checked arithmetic for computation-size values (precision bits, term counts, bit lengths).
//!
//! # Fixed-width computation types
//!
//! All computation sizes use [`U`] (`u32`) rather than platform-dependent `usize`.
//! This provides compile-time guarantees about the maximum representable value
//! (`u32::MAX` ≈ 4 billion bits of precision, or ~512 MB per number).
//!
//! # Conventions
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
//!   internally, panicking on overflow. The result is unwrapped back to [`U`].

/// The unsigned integer type for computation sizes (precision bits, term counts,
/// bit lengths, node IDs, etc.). Fixed at `u32` for compile-time size guarantees.
///
/// A computation requiring more than ~2^32 bits of precision would need ~512 MB
/// just to store one number, and intermediate results would require far more.
pub type U = u32;

/// The signed integer type corresponding to [`U`], used for width/magnitude exponents
/// inside [`XI`]. Fixed at `i32`.
pub type I = i32;

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

/// Guards one or more [`U`] variables and evaluates an arithmetic expression using
/// checked arithmetic via [`Sane`].
///
/// Each guard variable is shadowed as a [`Sane`] newtype whose `+`, `-`, `*`, `/`
/// operators use `checked_*` internally, panicking via
/// [`detected_computable_would_exhaust_memory!`] on overflow. The result is unwrapped
/// back to [`U`].
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
/// let num_terms: u32 = 10;
/// let exponent = sane_arithmetic!(num_terms; 2 * num_terms + 1);
/// assert_eq!(exponent, 21_u32);
/// ```
#[macro_export]
macro_rules! sane_arithmetic {
    ($($guard:ident),+ ; $expr:expr) => {{
        $(
            #[allow(clippy::shadow_reuse)]
            let $guard = $crate::Sane($guard);
        )+
        let $crate::Sane(__result) = { $expr };
        __result
    }};
}

/// Converts a `u64` bit count (e.g. from `BigUint::bits()`) to [`U`],
/// panicking if the value exceeds `U::MAX`.
///
/// This centralizes the one unavoidable `u64 -> U` cast that arises because
/// `num_bigint::BigUint::bits()` returns `u64` but shift operations and
/// precision parameters use [`U`].
///
/// # Panics
///
/// Panics via `detected_computable_would_exhaust_memory!` if `bits` exceeds
/// `U::MAX`.
/// Compile-time guarantee that `usize` is at least as wide as [`U`] (`u32`),
/// so `u32 as usize` is always a lossless widening or identity conversion.
const _: () = assert!(size_of::<usize>() >= size_of::<U>());

/// Converts a [`U`] (`u32`) to `usize` for use at BigInt shift boundaries.
///
/// BigInt shift operators require `usize`, but all computation sizes use [`U`].
/// The const assertion above guarantees this is lossless on any supported platform.
pub fn u_as_usize(v: U) -> usize {
    #[allow(clippy::as_conversions)]
    {
        v as usize
    }
}

pub fn bits_as_u(bits: u64) -> U {
    match U::try_from(bits) {
        Ok(v) => v,
        Err(_) => detected_computable_would_exhaust_memory!("bit count exceeds U::MAX"),
    }
}

/// Subtracts one from a `NonZeroU32`, returning the result as [`U`].
///
/// This is trivially correct: `NonZeroU32` guarantees `>= 1`, so `- 1 >= 0`.
pub fn sub_one(n: std::num::NonZeroU32) -> U {
    #[allow(clippy::arithmetic_side_effects)]
    {
        n.get() - 1
    }
}

/// Subtracts one from a `NonZeroU64`, returning the result as `u64`.
///
/// This is trivially correct: `NonZeroU64` guarantees `>= 1`, so `- 1 >= 0`.
pub fn sub_one_u64(n: std::num::NonZeroU64) -> u64 {
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
pub struct Sane(pub U);

/// Implements checked arithmetic operators for [`Sane`].
///
/// Each invocation generates three impls: `Sane op Sane`, `Sane op u32`, and
/// `u32 op Sane`. The `u32` variants construct `Sane` directly since [`U`] is `u32`.
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
                core::ops::$trait::$method(self, Sane(rhs))
            }
        }

        impl core::ops::$trait<Sane> for u32 {
            type Output = Sane;
            #[inline]
            fn $method(self, rhs: Sane) -> Sane {
                core::ops::$trait::$method(Sane(self), rhs)
            }
        }
    };
}

impl_sane_binary_op!(Add, add, checked_add, "Sane addition overflow");
impl_sane_binary_op!(Sub, sub, checked_sub, "Sane subtraction underflow");
impl_sane_binary_op!(Mul, mul, checked_mul, "Sane multiplication overflow");
impl_sane_binary_op!(Div, div, checked_div, "Sane division by zero");

/// Signed integer extended with ±infinity, used for width/magnitude exponents.
///
/// - `PosInf`: 2^(+inf) = unbounded
/// - `NegInf`: 2^(-inf) = 0 (exact)
/// - `Finite(e)`: normal 2^e
///
/// Uses [`I`] (`i32`) — the signed counterpart of [`U`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XI {
    NegInf,
    Finite(I),
    PosInf,
}

impl PartialOrd for XI {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for XI {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::NegInf, Self::NegInf) => Ordering::Equal,
            (Self::NegInf, _) => Ordering::Less,
            (_, Self::NegInf) => Ordering::Greater,
            (Self::PosInf, Self::PosInf) => Ordering::Equal,
            (Self::PosInf, _) => Ordering::Greater,
            (_, Self::PosInf) => Ordering::Less,
            (Self::Finite(a), Self::Finite(b)) => a.cmp(b),
        }
    }
}

impl XI {
    /// Interprets this width exponent as a precision requirement in bits.
    ///
    /// - `NegInf` (width = 0, exact) → `XU::Inf` (infinite precision needed)
    /// - `Finite(e)` with `e <= 0` → `XU::Finite(|e|)` (|e| bits of precision)
    /// - `Finite(e)` with `e > 0` → `XU::Finite(0)` (coarse target, no precision)
    /// - `PosInf` (width = ∞) → `XU::Finite(0)` (unbounded, no precision)
    pub fn to_precision_bits(self) -> XU {
        match self {
            Self::NegInf => XU::Inf,
            Self::Finite(e) if e <= 0 => {
                // e is in range I::MIN..=0, so unsigned_abs() fits in U.
                #[allow(clippy::arithmetic_side_effects)]
                let abs = e.unsigned_abs();
                XU::Finite(abs)
            }
            Self::Finite(_) | Self::PosInf => XU::Finite(0),
        }
    }
}

/// A [`U`] extended with positive infinity, analogous to `UXBinary`.
///
/// When used as a tolerance exponent: `Finite(n)` means epsilon = 2^(-n),
/// `Inf` means epsilon = 0 (exact convergence required).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XU {
    Finite(U),
    Inf,
}

impl PartialOrd for XU {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for XU {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Inf, Self::Inf) => Ordering::Equal,
            (Self::Inf, _) => Ordering::Greater,
            (_, Self::Inf) => Ordering::Less,
            (Self::Finite(a), Self::Finite(b)) => a.cmp(b),
        }
    }
}

impl std::ops::Neg for XU {
    type Output = XI;

    /// Negates to produce an `XI`: `Finite(n)` → `XI::Finite(-(n as I))`,
    /// `Inf` → `XI::NegInf`.
    fn neg(self) -> XI {
        match self {
            Self::Inf => XI::NegInf,
            Self::Finite(n) => {
                // I::try_from(n) fails when n > I::MAX. In that case the
                // negated value would be < I::MIN, so map to NegInf.
                match I::try_from(n) {
                    Ok(signed) => match signed.checked_neg() {
                        Some(negated) => XI::Finite(negated),
                        None => XI::NegInf,
                    },
                    Err(_) => XI::NegInf,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn sane_arithmetic_macro_works() {
        let n: U = 10;
        let result = crate::sane_arithmetic!(n; 2 * n + 1);
        assert_eq!(result, 21_u32);
    }

    #[test]
    #[should_panic(expected = "Sane multiplication overflow")]
    fn sane_mul_overflow_panics() {
        let _ = Sane(U::MAX) * Sane(2);
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
    #[should_panic(expected = "Sane addition overflow")]
    fn sane_arithmetic_macro_rejects_large_result() {
        let n: U = U::MAX;
        // n + 1 overflows U (u32), caught by checked_add in Sane
        let _ = crate::sane_arithmetic!(n; n + 1);
    }

    // --- XI tests ---

    #[test]
    fn xi_ordering() {
        assert!(XI::NegInf < XI::Finite(-1000));
        assert!(XI::Finite(-1000) < XI::Finite(0));
        assert!(XI::Finite(0) < XI::Finite(1000));
        assert!(XI::Finite(1000) < XI::PosInf);
        assert_eq!(XI::NegInf, XI::NegInf);
        assert_eq!(XI::PosInf, XI::PosInf);
    }

    #[test]
    fn xi_to_precision_bits() {
        assert_eq!(XI::Finite(0).to_precision_bits(), XU::Finite(0));
        assert_eq!(XI::Finite(-10).to_precision_bits(), XU::Finite(10));
        assert_eq!(XI::Finite(-100).to_precision_bits(), XU::Finite(100));
        // Positive exponents: coarse target, no precision needed
        assert_eq!(XI::Finite(5).to_precision_bits(), XU::Finite(0));
        // NegInf = exact = infinite precision
        assert_eq!(XI::NegInf.to_precision_bits(), XU::Inf);
        // PosInf = unbounded = no precision needed
        assert_eq!(XI::PosInf.to_precision_bits(), XU::Finite(0));
    }

    // --- XU tests ---

    #[test]
    fn xu_ordering() {
        assert!(XU::Finite(0) < XU::Finite(10));
        assert!(XU::Finite(10) < XU::Inf);
        assert_eq!(XU::Inf, XU::Inf);
    }

    #[test]
    fn xu_neg() {
        assert_eq!(-XU::Inf, XI::NegInf);
        assert_eq!(-XU::Finite(0), XI::Finite(0));
        assert_eq!(-XU::Finite(42), XI::Finite(-42));
    }
}
