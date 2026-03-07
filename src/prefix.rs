//! Prefix representation of computable number precision.
//!
//! A [`Prefix`] represents the known precision of a computable number as a binary prefix.
//! This is conceptually cleaner than general interval types because it directly encodes
//! "the known bits" and prevents mantissa bloat during refinement.
//!
//! # Types
//!
//! - [`XExponent`]: Exponent extended with ±infinity for width/magnitude bounds
//! - [`Prefix`]: Known precision as either a single-sign interval or a zero-crossing interval

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::binary::{Binary, UBinary, UXBinary, XBinary};
use crate::binary_utils::bisection::PrefixBounds;

/// Exponent extended with ±infinity, used for width/magnitude bounds.
///
/// - `PosInf`: 2^(+inf) = unbounded
/// - `NegInf`: 2^(-inf) = 0 (exact)
/// - `Finite(e)`: normal 2^e
///
/// Uses `i64` (not `BigInt`) because width_exponent represents precision — how many bits
/// are known. `i64` handles up to 9.2 × 10^18 bits, far beyond any realistic computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XExponent {
    NegInf,
    Finite(i64),
    PosInf,
}

impl PartialOrd for XExponent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for XExponent {
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

impl XExponent {
    /// Returns the maximum of two exponents.
    pub fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

/// Known precision of a computable number as a binary prefix.
///
/// For `Finite`: `inner` is the interval endpoint closest to zero.
/// - If inner >= 0: interval is `[inner, inner + 2^width_exponent]` (extends toward +inf)
/// - If inner < 0: interval is `[inner - 2^width_exponent, inner]` (extends toward -inf)
///
/// This "extend away from zero" convention means the width always increases
/// the magnitude, so negation of a Prefix just negates `inner`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prefix {
    /// Single-sign interval. `inner` is endpoint closest to zero.
    /// Width extends away from zero by 2^width_exponent.
    Finite {
        inner: Binary,
        width_exponent: XExponent,
    },

    /// Interval spanning (or near) zero: `[-2^neg_exponent, 2^pos_exponent]`.
    /// Also handles fully unbounded: `{ neg: PosInf, pos: PosInf } = (-inf, +inf)`.
    /// No mantissa — near zero / unbounded we only track magnitude bounds.
    ZeroCrossing {
        neg_exponent: XExponent,
        pos_exponent: XExponent,
    },
}

impl Prefix {
    /// Returns the fully unbounded prefix `(-inf, +inf)`.
    pub fn unbounded() -> Self {
        Self::ZeroCrossing {
            neg_exponent: XExponent::PosInf,
            pos_exponent: XExponent::PosInf,
        }
    }

    /// Returns an exact prefix (zero width) for a given value.
    pub fn exact(value: Binary) -> Self {
        Self::Finite {
            inner: value,
            width_exponent: XExponent::NegInf,
        }
    }

    /// Returns the lower bound as an `XBinary`.
    pub fn lower(&self) -> XBinary {
        match self {
            Self::Finite {
                inner,
                width_exponent,
            } => {
                if inner.mantissa().is_negative() {
                    // Negative inner: interval extends toward -inf
                    // lower = inner - 2^width_exponent
                    match width_exponent {
                        XExponent::NegInf => XBinary::Finite(inner.clone()),
                        XExponent::PosInf => XBinary::NegInf,
                        XExponent::Finite(e) => {
                            let width = Binary::new(BigInt::one(), BigInt::from(*e));
                            XBinary::Finite(inner.sub(&width))
                        }
                    }
                } else {
                    // Non-negative inner: inner IS the lower bound
                    XBinary::Finite(inner.clone())
                }
            }
            Self::ZeroCrossing {
                neg_exponent,
                pos_exponent: _,
            } => match neg_exponent {
                XExponent::PosInf => XBinary::NegInf,
                XExponent::NegInf => XBinary::Finite(Binary::zero()),
                XExponent::Finite(e) => {
                    // lower = -2^e
                    XBinary::Finite(Binary::new(-BigInt::one(), BigInt::from(*e)))
                }
            },
        }
    }

    /// Returns the upper bound as an `XBinary`.
    pub fn upper(&self) -> XBinary {
        match self {
            Self::Finite {
                inner,
                width_exponent,
            } => {
                if inner.mantissa().is_negative() {
                    // Negative inner: inner IS the upper bound (closest to zero)
                    XBinary::Finite(inner.clone())
                } else {
                    // Non-negative inner: interval extends toward +inf
                    // upper = inner + 2^width_exponent
                    match width_exponent {
                        XExponent::NegInf => XBinary::Finite(inner.clone()),
                        XExponent::PosInf => XBinary::PosInf,
                        XExponent::Finite(e) => {
                            let width = Binary::new(BigInt::one(), BigInt::from(*e));
                            XBinary::Finite(inner.add(&width))
                        }
                    }
                }
            }
            Self::ZeroCrossing {
                neg_exponent: _,
                pos_exponent,
            } => match pos_exponent {
                XExponent::PosInf => XBinary::PosInf,
                XExponent::NegInf => XBinary::Finite(Binary::zero()),
                XExponent::Finite(e) => {
                    // upper = 2^e
                    XBinary::Finite(Binary::new(BigInt::one(), BigInt::from(*e)))
                }
            },
        }
    }

    /// Returns the width of the interval as `UXBinary`.
    pub fn width(&self) -> UXBinary {
        match self {
            Self::Finite {
                inner: _,
                width_exponent,
            } => match width_exponent {
                XExponent::NegInf => UXBinary::Finite(UBinary::zero()),
                XExponent::PosInf => UXBinary::Inf,
                XExponent::Finite(e) => {
                    UXBinary::Finite(UBinary::new(1u32.into(), BigInt::from(*e)))
                }
            },
            Self::ZeroCrossing {
                neg_exponent,
                pos_exponent,
            } => {
                // Width = 2^neg_exp + 2^pos_exp
                let neg_width = xexponent_to_uxbinary(neg_exponent);
                let pos_width = xexponent_to_uxbinary(pos_exponent);
                neg_width.add(&pos_width)
            }
        }
    }

    /// Returns the width exponent — the smallest `e` such that `2^e >= width`.
    ///
    /// For `Finite`, this is the stored exponent. For `ZeroCrossing`, the width
    /// is `2^neg + 2^pos`, so the exponent is `max(neg, pos) + 1` when both
    /// sides contribute nonzero width, or just the nonzero side's exponent
    /// when the other is `NegInf` (zero contribution).
    pub fn width_exponent(&self) -> XExponent {
        match self {
            Self::Finite {
                inner: _,
                width_exponent,
            } => *width_exponent,
            Self::ZeroCrossing {
                neg_exponent,
                pos_exponent,
            } => {
                match (neg_exponent, pos_exponent) {
                    (XExponent::NegInf, other) | (other, XExponent::NegInf) => *other,
                    (XExponent::PosInf, _) | (_, XExponent::PosInf) => XExponent::PosInf,
                    (XExponent::Finite(a), XExponent::Finite(b)) => {
                        // width = 2^a + 2^b > 2^max(a,b), so ceil is max + 1.
                        // Saturate to avoid overflow (would require astronomic exponents).
                        let m = (*a).max(*b);
                        XExponent::Finite(m.saturating_add(1))
                    }
                }
            }
        }
    }

    /// Converts this prefix to a `Bounds` (Interval<XBinary, UXBinary>).
    pub fn to_bounds(&self) -> crate::binary::Bounds {
        crate::binary::Bounds::from_lower_and_width(self.lower(), self.width())
    }

    /// Constructs a `Prefix` from lower and upper `XBinary` bounds.
    ///
    /// This normalizes the bounds to prefix representation, handling:
    /// - Infinite bounds → appropriate ZeroCrossing/Finite variants
    /// - Zero-crossing finite bounds → ZeroCrossing with magnitude exponents
    /// - Single-sign finite bounds → Finite with inner closest to zero
    pub fn from_lower_upper(lower: XBinary, upper: XBinary) -> Self {
        match (&lower, &upper) {
            (XBinary::NegInf, XBinary::PosInf) => Self::unbounded(),
            (XBinary::NegInf, XBinary::Finite(hi)) => {
                let pos_exp = if hi.mantissa().is_positive() {
                    magnitude_exponent(hi)
                } else if hi.mantissa().is_zero() {
                    XExponent::NegInf
                } else {
                    return Self::Finite {
                        inner: hi.clone(),
                        width_exponent: XExponent::PosInf,
                    };
                };
                Self::ZeroCrossing {
                    neg_exponent: XExponent::PosInf,
                    pos_exponent: pos_exp,
                }
            }
            (XBinary::Finite(lo), XBinary::PosInf) => {
                let neg_exp = if lo.mantissa().is_negative() {
                    magnitude_exponent(&lo.neg())
                } else if lo.mantissa().is_zero() {
                    XExponent::NegInf
                } else {
                    return Self::Finite {
                        inner: lo.clone(),
                        width_exponent: XExponent::PosInf,
                    };
                };
                Self::ZeroCrossing {
                    neg_exponent: neg_exp,
                    pos_exponent: XExponent::PosInf,
                }
            }
            (XBinary::NegInf, XBinary::NegInf) | (XBinary::PosInf, XBinary::PosInf) => {
                crate::detected_computable_with_infinite_value!(
                    "both bounds are the same infinity"
                );
                Self::unbounded()
            }
            (XBinary::PosInf, _) | (_, XBinary::NegInf) => {
                crate::detected_computable_with_infinite_value!(
                    "invalid bounds ordering with infinities"
                );
                Self::unbounded()
            }
            (XBinary::Finite(lo), XBinary::Finite(hi)) => finite_bounds_to_prefix(lo, hi),
        }
    }

    /// Returns the absolute value (magnitude) of each bound as a pair.
    ///
    /// For prefix with bounds [lower, upper], returns (|lower|, |upper|).
    pub fn abs(&self) -> (UXBinary, UXBinary) {
        (self.lower().magnitude(), self.upper().magnitude())
    }

    /// Returns true if this prefix contains the given point.
    pub fn contains(&self, point: &XBinary) -> bool {
        let lo = self.lower();
        let hi = self.upper();
        lo <= *point && *point <= hi
    }
}

/// Converts an `XExponent` to `UXBinary` (2^exponent).
fn xexponent_to_uxbinary(exp: &XExponent) -> UXBinary {
    match exp {
        XExponent::NegInf => UXBinary::Finite(UBinary::zero()),
        XExponent::PosInf => UXBinary::Inf,
        XExponent::Finite(e) => UXBinary::Finite(UBinary::new(1u32.into(), BigInt::from(*e))),
    }
}

// =========================================================================
// Conversions
// =========================================================================

/// Converts finite lower/upper bounds to a Prefix.
fn finite_bounds_to_prefix(lower: &Binary, upper: &Binary) -> Prefix {
    let lower_negative = lower.mantissa().is_negative();
    let upper_positive = upper.mantissa().is_positive();
    let lower_zero = lower.mantissa().is_zero();
    let upper_zero = upper.mantissa().is_zero();

    // Zero-width (exact)
    if lower == upper {
        return Prefix::exact(lower.clone());
    }

    // Zero-crossing: lower < 0 and upper > 0
    if lower_negative && upper_positive {
        let neg_mag = magnitude_exponent(&lower.neg());
        let pos_mag = magnitude_exponent(upper);
        return Prefix::ZeroCrossing {
            neg_exponent: neg_mag,
            pos_exponent: pos_mag,
        };
    }

    // Lower is zero, upper positive: zero-crossing with exact lower
    if lower_zero && upper_positive {
        let pos_mag = magnitude_exponent(upper);
        return Prefix::ZeroCrossing {
            neg_exponent: XExponent::NegInf,
            pos_exponent: pos_mag,
        };
    }

    // Lower is negative, upper is zero: zero-crossing with exact upper
    if lower_negative && upper_zero {
        let neg_mag = magnitude_exponent(&lower.neg());
        return Prefix::ZeroCrossing {
            neg_exponent: neg_mag,
            pos_exponent: XExponent::NegInf,
        };
    }

    // Both zero
    if lower_zero && upper_zero {
        return Prefix::exact(Binary::zero());
    }

    // Single-sign interval: compute inner (closest to zero) and width
    if !lower_negative {
        // Both non-negative, lower is closest to zero
        let width = upper.sub(lower);
        let we = width_to_xexponent(&width);
        Prefix::Finite {
            inner: lower.clone(),
            width_exponent: we,
        }
    } else {
        // Both negative, upper (less negative) is closest to zero
        let width = upper.sub(lower);
        let we = width_to_xexponent(&width);
        Prefix::Finite {
            inner: upper.clone(),
            width_exponent: we,
        }
    }
}

/// Computes the magnitude exponent of a positive Binary value.
///
/// Returns the smallest exponent e such that `|value| <= 2^e`.
/// This is `ceil(log2(|value|))` when value is not a power of 2,
/// and `log2(|value|)` when it is.
fn magnitude_exponent(value: &Binary) -> XExponent {
    // value = mantissa * 2^exponent (mantissa is odd after normalization)
    // |value| = |mantissa| * 2^exponent
    // bits(|mantissa|) = floor(log2(|mantissa|)) + 1
    //
    // If |mantissa| is a power of 2 (i.e. |mantissa| = 2^(bits-1)):
    //   log2(|value|) = (bits-1) + exponent
    // Otherwise:
    //   ceil(log2(|value|)) = bits + exponent
    //
    // But Binary normalizes mantissa to be odd, so |mantissa| is a power of 2
    // only when |mantissa| = 1 (since 1 is the only odd power of 2).
    let mantissa_bits = match std::num::NonZeroU64::new(value.mantissa().magnitude().bits()) {
        None => return XExponent::NegInf,
        Some(nz) => nz,
    };
    let is_power_of_two = *value.mantissa().magnitude() == num_bigint::BigUint::one();

    let mantissa_bits_i64 = i64::try_from(mantissa_bits.get())
        .unwrap_or_else(|_| crate::detected_computable_would_exhaust_memory!("mantissa too large"));
    let exponent_i64 = value
        .exponent()
        .to_i64()
        .unwrap_or_else(|| crate::detected_computable_would_exhaust_memory!("exponent too large"));

    let effective_bits = if is_power_of_two {
        i64::try_from(crate::sane::sub_one_u64(mantissa_bits)).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!("mantissa too large")
        })
    } else {
        mantissa_bits_i64
    };
    let result = effective_bits
        .checked_add(exponent_i64)
        .unwrap_or_else(|| crate::detected_computable_would_exhaust_memory!("exponent overflow"));
    XExponent::Finite(result)
}

/// Computes the XExponent for a width (positive Binary value).
///
/// Returns the smallest e such that 2^e >= width.
fn width_to_xexponent(width: &Binary) -> XExponent {
    // width = mantissa * 2^exponent, mantissa is odd and positive
    // If mantissa == 1, then width = 2^exponent exactly
    // Otherwise width > 2^(bits-1+exponent), so we need ceil(log2(width)) = bits + exponent
    let mantissa_bits = match std::num::NonZeroU64::new(width.mantissa().magnitude().bits()) {
        None => return XExponent::NegInf,
        Some(nz) => nz,
    };
    let bits_minus_1 = crate::sane::sub_one_u64(mantissa_bits);
    let is_power_of_two =
        *width.mantissa().magnitude() == (num_bigint::BigUint::one() << bits_minus_1);
    let exponent_i64 = width
        .exponent()
        .to_i64()
        .unwrap_or_else(|| crate::detected_computable_would_exhaust_memory!("exponent too large"));

    if is_power_of_two {
        // width = 2^(bits-1) * 2^exponent = 2^(bits-1+exponent)
        let bits_minus_1_i64 = i64::try_from(bits_minus_1).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!("mantissa too large")
        });
        let result = bits_minus_1_i64
            .checked_add(exponent_i64)
            .unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!("exponent overflow")
            });
        XExponent::Finite(result)
    } else {
        // Need ceil: bits + exponent
        let bits_i64 = i64::try_from(mantissa_bits.get()).unwrap_or_else(|_| {
            crate::detected_computable_would_exhaust_memory!("mantissa too large")
        });
        let result = bits_i64.checked_add(exponent_i64).unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow")
        });
        XExponent::Finite(result)
    }
}

impl From<&PrefixBounds> for Prefix {
    /// Converts a bisection `PrefixBounds` to `Prefix`.
    ///
    /// PrefixBounds represents `[mantissa * 2^exponent, (mantissa + 1) * 2^exponent]`.
    fn from(pb: &PrefixBounds) -> Self {
        let lower = Binary::new(pb.mantissa.clone(), pb.exponent.clone());
        let upper_mantissa = &pb.mantissa + BigInt::one();
        let upper = Binary::new(upper_mantissa, pb.exponent.clone());

        // Determine if zero-crossing
        if lower.mantissa().is_negative() && upper.mantissa().is_positive() {
            // Zero-crossing
            let neg_mag = magnitude_exponent(&lower.neg());
            let pos_mag = magnitude_exponent(&upper);
            Prefix::ZeroCrossing {
                neg_exponent: neg_mag,
                pos_exponent: pos_mag,
            }
        } else if lower.mantissa().is_zero() || upper.mantissa().is_zero() {
            // One endpoint is zero
            finite_bounds_to_prefix(&lower, &upper)
        } else {
            // Single-sign: convert exponent
            let we = pb.exponent.to_i64().unwrap_or_else(|| {
                crate::detected_computable_would_exhaust_memory!("exponent too large")
            });
            if lower.mantissa().is_negative() {
                Prefix::Finite {
                    inner: upper,
                    width_exponent: XExponent::Finite(we),
                }
            } else {
                Prefix::Finite {
                    inner: lower,
                    width_exponent: XExponent::Finite(we),
                }
            }
        }
    }
}

impl From<PrefixBounds> for Prefix {
    fn from(pb: PrefixBounds) -> Self {
        Self::from(&pb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{bin, xbin};

    #[test]
    fn unbounded_prefix() {
        let p = Prefix::unbounded();
        assert_eq!(p.lower(), XBinary::NegInf);
        assert_eq!(p.upper(), XBinary::PosInf);
        assert_eq!(p.width(), UXBinary::Inf);
        assert_eq!(p.width_exponent(), XExponent::PosInf);
    }

    #[test]
    fn exact_positive() {
        let p = Prefix::exact(bin(3, 0));
        assert_eq!(p.lower(), xbin(3, 0));
        assert_eq!(p.upper(), xbin(3, 0));
        assert_eq!(p.width(), UXBinary::Finite(UBinary::zero()));
        assert_eq!(p.width_exponent(), XExponent::NegInf);
    }

    #[test]
    fn exact_negative() {
        let p = Prefix::exact(bin(-3, 0));
        assert_eq!(p.lower(), xbin(-3, 0));
        assert_eq!(p.upper(), xbin(-3, 0));
        assert_eq!(p.width(), UXBinary::Finite(UBinary::zero()));
    }

    #[test]
    fn exact_zero() {
        let p = Prefix::exact(Binary::zero());
        assert_eq!(p.lower(), XBinary::Finite(Binary::zero()));
        assert_eq!(p.upper(), XBinary::Finite(Binary::zero()));
    }

    #[test]
    fn finite_positive_interval() {
        // [3, 3 + 2^(-10)] → Finite { inner: 3, width_exponent: Finite(-10) }
        let p = Prefix::Finite {
            inner: bin(3, 0),
            width_exponent: XExponent::Finite(-10),
        };
        assert_eq!(p.lower(), xbin(3, 0));
        let expected_upper = bin(3, 0).add(&bin(1, -10));
        assert_eq!(p.upper(), XBinary::Finite(expected_upper));
        assert!(p.contains(&xbin(3, 0)));
    }

    #[test]
    fn finite_negative_interval() {
        // [-5, -3] → Finite { inner: -3, width_exponent: Finite(1) }
        // because width = 2 = 2^1, inner = -3 (closest to zero)
        let p = Prefix::Finite {
            inner: bin(-3, 0),
            width_exponent: XExponent::Finite(1),
        };
        // lower = -3 - 2^1 = -5
        assert_eq!(p.lower(), xbin(-5, 0));
        // upper = -3 (inner)
        assert_eq!(p.upper(), xbin(-3, 0));
        assert!(p.contains(&xbin(-4, 0)));
    }

    #[test]
    fn finite_with_posinf_width_positive() {
        // [3, +inf) → Finite { inner: 3, width_exponent: PosInf }
        let p = Prefix::Finite {
            inner: bin(3, 0),
            width_exponent: XExponent::PosInf,
        };
        assert_eq!(p.lower(), xbin(3, 0));
        assert_eq!(p.upper(), XBinary::PosInf);
    }

    #[test]
    fn finite_with_posinf_width_negative() {
        // (-inf, -3] → Finite { inner: -3, width_exponent: PosInf }
        let p = Prefix::Finite {
            inner: bin(-3, 0),
            width_exponent: XExponent::PosInf,
        };
        assert_eq!(p.lower(), XBinary::NegInf);
        assert_eq!(p.upper(), xbin(-3, 0));
    }

    #[test]
    fn zero_crossing_symmetric() {
        // [-0.5, 0.5] = ZeroCrossing { neg: Finite(-1), pos: Finite(-1) }
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::Finite(-1),
            pos_exponent: XExponent::Finite(-1),
        };
        // lower = -2^(-1) = -0.5
        assert_eq!(p.lower(), xbin(-1, -1));
        // upper = 2^(-1) = 0.5
        assert_eq!(p.upper(), xbin(1, -1));
        assert!(p.contains(&XBinary::Finite(Binary::zero())));
    }

    #[test]
    fn zero_crossing_asymmetric() {
        // [-2, 1] ≈ ZeroCrossing { neg: Finite(1), pos: Finite(0) }
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::Finite(1),
            pos_exponent: XExponent::Finite(0),
        };
        // lower = -2^1 = -2
        assert_eq!(p.lower(), xbin(-1, 1));
        // upper = 2^0 = 1
        assert_eq!(p.upper(), xbin(1, 0));
    }

    #[test]
    fn zero_crossing_half_unbounded() {
        // (-inf, 1] = ZeroCrossing { neg: PosInf, pos: Finite(0) }
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::PosInf,
            pos_exponent: XExponent::Finite(0),
        };
        assert_eq!(p.lower(), XBinary::NegInf);
        assert_eq!(p.upper(), xbin(1, 0));
    }

    // =========================================================================
    // Conversion tests
    // =========================================================================

    #[test]
    fn from_lower_upper_unbounded() {
        let p = Prefix::from_lower_upper(XBinary::NegInf, XBinary::PosInf);
        assert_eq!(p, Prefix::unbounded());
    }

    #[test]
    fn from_lower_upper_exact() {
        let p = Prefix::from_lower_upper(xbin(3, 0), xbin(3, 0));
        assert_eq!(p, Prefix::exact(bin(3, 0)));
    }

    #[test]
    fn from_lower_upper_positive_interval() {
        // [2, 4] → width = 2 = 2^1
        let p = Prefix::from_lower_upper(xbin(2, 0), xbin(4, 0));
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                // inner should be 2 (closest to zero)
                assert_eq!(*inner, bin(2, 0));
                // width_exponent should be Finite(1) since width = 2 = 2^1
                assert_eq!(*width_exponent, XExponent::Finite(1));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
        // Should contain original endpoints
        assert!(p.contains(&xbin(2, 0)));
        assert!(p.contains(&xbin(4, 0)));
    }

    #[test]
    fn from_lower_upper_negative_interval() {
        // [-4, -2] → inner = -2 (closest to zero), width = 2 = 2^1
        let p = Prefix::from_lower_upper(xbin(-4, 0), xbin(-2, 0));
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                assert_eq!(*inner, bin(-2, 0));
                assert_eq!(*width_exponent, XExponent::Finite(1));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
        assert!(p.contains(&xbin(-4, 0)));
        assert!(p.contains(&xbin(-2, 0)));
    }

    #[test]
    fn from_lower_upper_zero_crossing() {
        // [-2, 3]
        let p = Prefix::from_lower_upper(xbin(-2, 0), xbin(3, 0));
        match &p {
            Prefix::ZeroCrossing {
                neg_exponent,
                pos_exponent,
            } => {
                // |-2| = 2 = 2^1, so neg_exponent = Finite(1)
                assert_eq!(*neg_exponent, XExponent::Finite(1));
                // 3 < 2^2, so pos_exponent = Finite(2)
                assert_eq!(*pos_exponent, XExponent::Finite(2));
            }
            _ => panic!("expected ZeroCrossing, got {:?}", p),
        }
        assert!(p.contains(&xbin(-2, 0)));
        assert!(p.contains(&xbin(3, 0)));
    }

    #[test]
    fn from_lower_upper_half_infinite_positive() {
        // [3, +inf)
        let p = Prefix::from_lower_upper(xbin(3, 0), XBinary::PosInf);
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                assert_eq!(*inner, bin(3, 0));
                assert_eq!(*width_exponent, XExponent::PosInf);
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
    }

    #[test]
    fn from_bounds_half_infinite_negative() {
        // (-inf, -3] — test via direct Prefix construction.
        let p = Prefix::Finite {
            inner: bin(-3, 0),
            width_exponent: XExponent::PosInf,
        };
        assert_eq!(p.lower(), XBinary::NegInf);
        assert_eq!(p.upper(), xbin(-3, 0));
    }

    #[test]
    fn lower_upper_roundtrip_finite() {
        let prefix = Prefix::from_lower_upper(xbin(5, 0), xbin(8, 0));
        // Prefix may widen (width rounded up to power of 2)
        // but should contain the original endpoints
        assert!(prefix.lower() <= xbin(5, 0));
        assert!(prefix.upper() >= xbin(8, 0));
    }

    #[test]
    fn lower_upper_roundtrip_exact() {
        let prefix = Prefix::from_lower_upper(xbin(42, 0), xbin(42, 0));
        assert_eq!(prefix.lower(), xbin(42, 0));
        assert_eq!(prefix.upper(), xbin(42, 0));
    }

    #[test]
    fn lower_upper_roundtrip_unbounded() {
        let prefix = Prefix::from_lower_upper(XBinary::NegInf, XBinary::PosInf);
        assert_eq!(prefix.lower(), XBinary::NegInf);
        assert_eq!(prefix.upper(), XBinary::PosInf);
    }

    #[test]
    fn from_prefix_bounds() {
        // PrefixBounds: [3 * 2^(-2), 4 * 2^(-2)] = [0.75, 1.0]
        let pb = PrefixBounds::new(BigInt::from(3_i32), BigInt::from(-2_i32));
        let p = Prefix::from(&pb);
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                // inner = lower = 0.75 (positive, closest to zero)
                assert_eq!(*inner, bin(3, -2));
                assert_eq!(*width_exponent, XExponent::Finite(-2));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
    }

    #[test]
    fn from_prefix_bounds_negative() {
        // PrefixBounds: [-5 * 2^0, -4 * 2^0] = [-5, -4]
        let pb = PrefixBounds::new(BigInt::from(-5_i32), BigInt::from(0_i32));
        let p = Prefix::from(&pb);
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                // inner = upper = -4 (closest to zero)
                assert_eq!(*inner, bin(-4, 0));
                assert_eq!(*width_exponent, XExponent::Finite(0));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
    }

    #[test]
    fn xexponent_ordering() {
        assert!(XExponent::NegInf < XExponent::Finite(-1000));
        assert!(XExponent::Finite(-1000) < XExponent::Finite(0));
        assert!(XExponent::Finite(0) < XExponent::Finite(1000));
        assert!(XExponent::Finite(1000) < XExponent::PosInf);
    }

    #[test]
    fn width_exponent_for_finite() {
        let p = Prefix::Finite {
            inner: bin(3, 0),
            width_exponent: XExponent::Finite(-10),
        };
        assert_eq!(p.width_exponent(), XExponent::Finite(-10));
    }

    #[test]
    fn width_exponent_for_zero_crossing() {
        // Asymmetric: width = 2^3 + 2^5 = 40, ceil(log2(40)) = 6 = max(3,5) + 1
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::Finite(3),
            pos_exponent: XExponent::Finite(5),
        };
        assert_eq!(p.width_exponent(), XExponent::Finite(6));
    }

    #[test]
    fn width_exponent_for_zero_crossing_symmetric() {
        // Symmetric: width = 2^4 + 2^4 = 2^5, exponent = 5 = 4 + 1
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::Finite(4),
            pos_exponent: XExponent::Finite(4),
        };
        assert_eq!(p.width_exponent(), XExponent::Finite(5));
    }

    #[test]
    fn width_exponent_for_zero_crossing_one_side_zero() {
        // One side NegInf (zero contribution): width = 0 + 2^3 = 2^3
        let p = Prefix::ZeroCrossing {
            neg_exponent: XExponent::NegInf,
            pos_exponent: XExponent::Finite(3),
        };
        assert_eq!(p.width_exponent(), XExponent::Finite(3));
    }

    #[test]
    fn contains_check() {
        let p = Prefix::Finite {
            inner: bin(3, 0),
            width_exponent: XExponent::Finite(0),
        };
        // Interval is [3, 4]
        assert!(p.contains(&xbin(3, 0)));
        assert!(p.contains(&xbin(4, 0)));
        assert!(p.contains(&xbin(7, -1))); // 3.5
        assert!(!p.contains(&xbin(5, 0)));
        assert!(!p.contains(&xbin(2, 0)));
    }

    #[test]
    fn from_lower_upper_non_power_of_two_width() {
        // [1, 4] → width = 3, which is not a power of 2
        // width_to_xexponent should round up: ceil(log2(3)) = 2
        let p = Prefix::from_lower_upper(xbin(1, 0), xbin(4, 0));
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                assert_eq!(*inner, bin(1, 0));
                assert_eq!(*width_exponent, XExponent::Finite(2));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
        // Verify containment
        assert!(p.contains(&xbin(1, 0)));
        assert!(p.contains(&xbin(4, 0)));
    }

    #[test]
    fn from_lower_upper_fractional() {
        // [0.5, 1.5] → width = 1 = 2^0
        let p = Prefix::from_lower_upper(xbin(1, -1), xbin(3, -1));
        match &p {
            Prefix::Finite {
                inner,
                width_exponent,
            } => {
                assert_eq!(*inner, bin(1, -1));
                assert_eq!(*width_exponent, XExponent::Finite(0));
            }
            _ => panic!("expected Finite, got {:?}", p),
        }
        assert!(p.contains(&xbin(1, -1)));
        assert!(p.contains(&xbin(3, -1)));
    }
}
