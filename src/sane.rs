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

/// The signed integer type corresponding to [`U`], used for exponents in
/// [`Binary`]/[`UBinary`] and width exponents in [`XI`]. Fixed at `i32`.
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

/// Guards one or more [`I`] variables and evaluates an arithmetic expression using
/// checked arithmetic via [`SaneI`].
///
/// Each guard variable is shadowed as a [`SaneI`] newtype whose `+`, `-`, `*`, `/`,
/// and unary `-` operators use `checked_*` internally, panicking via
/// [`detected_computable_would_exhaust_memory!`] on overflow. The result is unwrapped
/// back to [`I`].
///
/// # Example
///
/// ```
/// use computable::sane_i_arithmetic;
///
/// let exponent: i32 = -10;
/// let shift: i32 = 3;
/// let result = sane_i_arithmetic!(exponent, shift; exponent - shift);
/// assert_eq!(result, -13_i32);
/// ```
#[macro_export]
macro_rules! sane_i_arithmetic {
    ($($guard:ident),+ ; $expr:expr) => {{
        $(
            #[allow(clippy::shadow_reuse)]
            let $guard = $crate::SaneI($guard);
        )+
        #[allow(clippy::default_numeric_fallback)]
        let $crate::SaneI(__result) = { $expr };
        __result
    }};
}

/// Guards one or more `i64` variables and evaluates an arithmetic expression using
/// checked arithmetic via [`SaneI64`].
///
/// Each guard variable is shadowed as a [`SaneI64`] newtype whose `+`, `-`, `*`, `/`,
/// and unary `-` operators use `checked_*` internally, panicking via
/// [`detected_computable_would_exhaust_memory!`] on overflow. The result is unwrapped
/// back to `i64`.
///
/// # Example
///
/// ```
/// use computable::sane_i64_arithmetic;
///
/// let a: i64 = 1_000_000;
/// let b: i64 = 2;
/// let result = sane_i64_arithmetic!(a, b; a * b + 1);
/// assert_eq!(result, 2_000_001_i64);
/// ```
#[macro_export]
macro_rules! sane_i64_arithmetic {
    ($($guard:ident),+ ; $expr:expr) => {{
        $(
            #[allow(clippy::shadow_reuse)]
            let $guard = $crate::SaneI64($guard);
        )+
        #[allow(clippy::default_numeric_fallback)]
        let $crate::SaneI64(__result) = { $expr };
        __result
    }};
}

/// Converts a [`U`] (`u32`) to [`I`] (`i32`), panicking if the value
/// exceeds `I::MAX`. Used when computation-size values must be combined
/// with signed exponents.
pub fn u_as_i(v: U) -> I {
    match I::try_from(v) {
        Ok(r) => r,
        Err(_) => detected_computable_would_exhaust_memory!("value exceeds I::MAX"),
    }
}

/// Converts a `usize` to [`I`] (`i32`), panicking if the value
/// exceeds `I::MAX`. Used when sizes from BigInt operations must be
/// combined with signed exponents.
pub fn usize_as_i(v: usize) -> I {
    match I::try_from(v) {
        Ok(r) => r,
        Err(_) => detected_computable_would_exhaust_memory!("usize value exceeds I::MAX"),
    }
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

/// Converts a `u64` bit count to [`I`] (`i32`), panicking if the value
/// exceeds `I::MAX`. Used in width-exponent computations where bit counts
/// must be combined with signed exponents.
pub fn bits_as_i(bits: u64) -> I {
    match I::try_from(bits) {
        Ok(v) => v,
        Err(_) => detected_computable_would_exhaust_memory!("bit count exceeds I::MAX"),
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

/// Newtype for checked arithmetic on [`I`] (`i32`) exponent values.
///
/// Overloads `+`, `-`, `*`, `/`, and unary `-` to use `checked_*` internally,
/// panicking via [`detected_computable_would_exhaust_memory!`] on overflow.
///
/// Created automatically by the [`sane_i_arithmetic!`] macro.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SaneI(pub I);

/// Newtype for checked arithmetic on `i64` values.
///
/// Used in Taylor series computations (e.g., sin) where exponent arithmetic
/// is widened to `i64` for intermediate precision.
///
/// Created automatically by the [`sane_i64_arithmetic!`] macro.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SaneI64(pub i64);

/// Implements checked arithmetic operators for a signed sane newtype.
///
/// Generates impls for: `$name op $name`, `$name op $inner`, `$inner op $name`.
macro_rules! impl_sane_signed_binary_op {
    ($name:ident, $inner:ty, $trait:ident, $method:ident, $checked:ident, $msg:literal) => {
        impl core::ops::$trait for $name {
            type Output = $name;
            #[inline]
            fn $method(self, rhs: $name) -> $name {
                match self.0.$checked(rhs.0) {
                    Some(r) => $name(r),
                    None => crate::detected_computable_would_exhaust_memory!($msg),
                }
            }
        }

        impl core::ops::$trait<$inner> for $name {
            type Output = $name;
            #[inline]
            fn $method(self, rhs: $inner) -> $name {
                core::ops::$trait::$method(self, $name(rhs))
            }
        }

        impl core::ops::$trait<$name> for $inner {
            type Output = $name;
            #[inline]
            fn $method(self, rhs: $name) -> $name {
                core::ops::$trait::$method($name(self), rhs)
            }
        }
    };
}

impl_sane_signed_binary_op!(SaneI, i32, Add, add, checked_add, "SaneI addition overflow");
impl_sane_signed_binary_op!(SaneI, i32, Sub, sub, checked_sub, "SaneI subtraction overflow");
impl_sane_signed_binary_op!(SaneI, i32, Mul, mul, checked_mul, "SaneI multiplication overflow");
impl_sane_signed_binary_op!(SaneI, i32, Div, div, checked_div, "SaneI division error");

impl core::ops::Neg for SaneI {
    type Output = SaneI;
    #[inline]
    fn neg(self) -> SaneI {
        match self.0.checked_neg() {
            Some(r) => SaneI(r),
            None => crate::detected_computable_would_exhaust_memory!("SaneI negation overflow"),
        }
    }
}

impl_sane_signed_binary_op!(SaneI64, i64, Add, add, checked_add, "SaneI64 addition overflow");
impl_sane_signed_binary_op!(SaneI64, i64, Sub, sub, checked_sub, "SaneI64 subtraction overflow");
impl_sane_signed_binary_op!(SaneI64, i64, Mul, mul, checked_mul, "SaneI64 multiplication overflow");
impl_sane_signed_binary_op!(SaneI64, i64, Div, div, checked_div, "SaneI64 division error");

impl core::ops::Neg for SaneI64 {
    type Output = SaneI64;
    #[inline]
    fn neg(self) -> SaneI64 {
        match self.0.checked_neg() {
            Some(r) => SaneI64(r),
            None => crate::detected_computable_would_exhaust_memory!("SaneI64 negation overflow"),
        }
    }
}

/// Sign of an [`XI`] value, following `num_bigint::Sign` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Pos,
    Neg,
}

impl std::ops::Neg for Sign {
    type Output = Sign;
    fn neg(self) -> Sign {
        match self {
            Self::Pos => Self::Neg,
            Self::Neg => Self::Pos,
        }
    }
}

/// Signed integer extended with ±infinity, used for width/magnitude exponents.
///
/// - `PosInf`: 2^(+inf) = unbounded
/// - `NegInf`: 2^(-inf) = 0 (exact)
/// - `Finite { sign, magnitude }`: normal 2^(±magnitude)
///
/// Uses sign-magnitude representation ([`Sign`] + [`U`]) so that negation
/// is always a simple sign flip with no overflow risk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XI {
    NegInf,
    Finite { sign: Sign, magnitude: U },
    PosInf,
}

impl XI {
    /// Creates a finite `XI` from an `i32`. Canonicalizes zero to `Sign::Pos`.
    pub fn from_i32(v: i32) -> Self {
        let sign = if v >= 0_i32 { Sign::Pos } else { Sign::Neg };
        Self::Finite {
            sign,
            magnitude: v.unsigned_abs(),
        }
    }

    /// Returns the value as `i64`, or `None` for infinities.
    ///
    /// Always succeeds for `Finite` since sign + `u32` fits in `i64`.
    pub fn to_i64(self) -> Option<i64> {
        match self {
            Self::Finite { sign, magnitude } => {
                let mag = i64::from(magnitude);
                Some(match sign {
                    Sign::Pos => mag,
                    #[allow(clippy::arithmetic_side_effects)]
                    Sign::Neg => -mag,
                })
            }
            Self::NegInf | Self::PosInf => None,
        }
    }

    /// Extracts the finite value as `i64`.
    ///
    /// For infinity variants, panics — caller must handle those separately.
    pub fn finite_i64(self) -> i64 {
        self.to_i64().unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!(
                "infinite exponent where finite expected"
            )
        })
    }

    /// Interprets this width exponent as a precision requirement in bits.
    ///
    /// - `NegInf` (width = 0, exact) → `XU::Inf` (infinite precision needed)
    /// - Negative or zero exponent → `XU::Finite(magnitude)` (bits of precision)
    /// - Positive exponent → `XU::Finite(0)` (coarse target, no precision)
    /// - `PosInf` (width = ∞) → `XU::Finite(0)` (unbounded, no precision)
    pub fn to_precision_bits(self) -> XU {
        match self {
            Self::NegInf => XU::Inf,
            Self::Finite {
                sign: Sign::Neg,
                magnitude,
            } => XU::Finite(magnitude),
            Self::Finite {
                sign: Sign::Pos, ..
            }
            | Self::PosInf => XU::Finite(0),
        }
    }
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
            (
                Self::Finite {
                    sign: s1,
                    magnitude: m1,
                },
                Self::Finite {
                    sign: s2,
                    magnitude: m2,
                },
            ) => match (s1, s2) {
                (Sign::Pos, Sign::Neg) => Ordering::Greater,
                (Sign::Neg, Sign::Pos) => Ordering::Less,
                (Sign::Pos, Sign::Pos) => m1.cmp(m2),
                (Sign::Neg, Sign::Neg) => m2.cmp(m1),
            },
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

impl From<XU> for XI {
    /// Converts `XU` to `XI`: `Finite(n)` → positive `XI`, `Inf` → `PosInf`.
    fn from(xu: XU) -> Self {
        match xu {
            XU::Inf => XI::PosInf,
            XU::Finite(n) => XI::Finite {
                sign: Sign::Pos,
                magnitude: n,
            },
        }
    }
}

impl std::ops::Neg for XI {
    type Output = XI;

    /// Negates by flipping the sign: `PosInf` ↔ `NegInf`, sign flip for finite.
    /// Always safe — no arithmetic overflow possible.
    fn neg(self) -> XI {
        match self {
            Self::PosInf => Self::NegInf,
            Self::NegInf => Self::PosInf,
            Self::Finite { magnitude: 0, .. } => self,
            Self::Finite { sign, magnitude } => Self::Finite {
                sign: -sign,
                magnitude,
            },
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
        assert!(XI::NegInf < XI::from_i32(-1000));
        assert!(XI::from_i32(-1000) < XI::from_i32(0));
        assert!(XI::from_i32(0) < XI::from_i32(1000));
        assert!(XI::from_i32(1000) < XI::PosInf);
        assert_eq!(XI::NegInf, XI::NegInf);
        assert_eq!(XI::PosInf, XI::PosInf);
    }

    // --- XU tests ---

    #[test]
    fn xu_ordering() {
        assert!(XU::Finite(0) < XU::Finite(10));
        assert!(XU::Finite(10) < XU::Inf);
        assert_eq!(XU::Inf, XU::Inf);
    }

    #[test]
    fn xu_to_xi_conversion() {
        assert_eq!(XI::from(XU::Inf), XI::PosInf);
        assert_eq!(XI::from(XU::Finite(0)), XI::from_i32(0));
        assert_eq!(XI::from(XU::Finite(42)), XI::from_i32(42));
    }

    #[test]
    fn xi_neg() {
        assert_eq!(-XI::PosInf, XI::NegInf);
        assert_eq!(-XI::NegInf, XI::PosInf);
        assert_eq!(-XI::from_i32(0), XI::from_i32(0));
        assert_eq!(-XI::from_i32(42), XI::from_i32(-42));
        assert_eq!(-XI::from(XU::Finite(42)), XI::from_i32(-42));
        assert_eq!(-XI::from(XU::Inf), XI::NegInf);
    }

    #[test]
    fn bits_as_i_converts_valid() {
        assert_eq!(bits_as_i(0), 0_i32);
        assert_eq!(bits_as_i(42), 42_i32);
        assert_eq!(bits_as_i(u64::try_from(i32::MAX).unwrap()), i32::MAX);
    }

    #[test]
    #[should_panic(expected = "bit count exceeds I::MAX")]
    fn bits_as_i_panics_on_overflow() {
        let _ = bits_as_i(u64::try_from(i32::MAX).unwrap() + 1);
    }

    // --- SaneI tests ---

    #[test]
    fn sane_i_arithmetic_macro_works() {
        let a: I = -10;
        let b: I = 3;
        let result = crate::sane_i_arithmetic!(a, b; a + b);
        assert_eq!(result, -7_i32);
    }

    #[test]
    fn sane_i_arithmetic_with_literals() {
        let exp: I = -10;
        let result = crate::sane_i_arithmetic!(exp; exp - 1);
        assert_eq!(result, -11_i32);
    }

    #[test]
    fn sane_i_neg_works() {
        let a: I = 42;
        let result = crate::sane_i_arithmetic!(a; -a);
        assert_eq!(result, -42_i32);
    }

    #[test]
    #[should_panic(expected = "SaneI addition overflow")]
    fn sane_i_add_overflow_panics() {
        let _ = SaneI(I::MAX) + SaneI(1);
    }

    #[test]
    #[should_panic(expected = "SaneI subtraction overflow")]
    fn sane_i_sub_overflow_panics() {
        let _ = SaneI(I::MIN) - SaneI(1);
    }

    #[test]
    #[should_panic(expected = "SaneI negation overflow")]
    fn sane_i_neg_overflow_panics() {
        let _ = -SaneI(I::MIN);
    }

    // --- SaneI64 tests ---

    #[test]
    fn sane_i64_arithmetic_macro_works() {
        let a: i64 = 1_000_000;
        let b: i64 = 2;
        let result = crate::sane_i64_arithmetic!(a, b; a * b + 1);
        assert_eq!(result, 2_000_001_i64);
    }

    #[test]
    #[should_panic(expected = "SaneI64 multiplication overflow")]
    fn sane_i64_mul_overflow_panics() {
        let _ = SaneI64(i64::MAX) * SaneI64(2);
    }

    // --- u_as_i / usize_as_i tests ---

    #[test]
    fn u_as_i_converts_valid() {
        assert_eq!(u_as_i(0), 0_i32);
        assert_eq!(u_as_i(42), 42_i32);
    }

    #[test]
    #[should_panic(expected = "value exceeds I::MAX")]
    fn u_as_i_panics_on_overflow() {
        let _ = u_as_i(U::MAX);
    }

    #[test]
    fn usize_as_i_converts_valid() {
        assert_eq!(usize_as_i(0), 0_i32);
        assert_eq!(usize_as_i(100), 100_i32);
    }

    #[test]
    #[should_panic(expected = "usize value exceeds I::MAX")]
    fn usize_as_i_panics_on_overflow() {
        let _ = usize_as_i(usize::MAX);
    }
}
