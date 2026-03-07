//! Lightweight interval arithmetic on finite Binary values.
//!
//! [`FiniteInterval`] is a `pub(crate)` `(lower, upper)` wrapper used by pi/sin
//! Taylor series accumulation and bisection helpers. Unlike the generic
//! `Interval<S, W>`, it stores both endpoints directly and is not part of the
//! public API.

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::binary::{Binary, UBinary};

/// A finite interval `[lower, upper]` of `Binary` values.
///
/// Stores both endpoints directly (not lower + width). Used internally for
/// interval arithmetic in Taylor series (pi, sin) and bisection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FiniteInterval {
    lower: Binary,
    upper: Binary,
}

impl FiniteInterval {
    /// Creates a new interval, ordering the two values so `lower <= upper`.
    pub fn new(a: Binary, b: Binary) -> Self {
        if a <= b {
            Self { lower: a, upper: b }
        } else {
            Self { lower: b, upper: a }
        }
    }

    /// Creates a point interval `[x, x]`.
    pub fn point(x: Binary) -> Self {
        Self {
            lower: x.clone(),
            upper: x,
        }
    }

    /// Creates an interval from a lower bound and unsigned width.
    ///
    /// The upper bound is computed as `lower + width`.
    pub fn from_lower_and_width(lower: Binary, width: UBinary) -> Self {
        let upper = lower.add(&width.to_binary());
        Self { lower, upper }
    }

    /// Returns the lower bound.
    pub fn lo(&self) -> &Binary {
        &self.lower
    }

    /// Returns the upper bound (cloned for ownership).
    pub fn hi(&self) -> Binary {
        self.upper.clone()
    }

    /// Alias for `lo()` (compatibility with code that used `Interval::small()`).
    pub fn small(&self) -> &Binary {
        &self.lower
    }

    /// Returns the width as `UBinary`.
    pub fn width(&self) -> UBinary {
        let diff = self.upper.sub(&self.lower);
        UBinary::try_from_binary(&diff).unwrap_or_else(|_| UBinary::zero())
    }

    /// Interval addition: `[a,b] + [c,d] = [a+c, b+d]`.
    pub fn interval_add(&self, other: &Self) -> Self {
        Self {
            lower: self.lower.add(other.lo()),
            upper: self.upper.add(&other.upper),
        }
    }

    /// Interval subtraction: `[a,b] - [c,d] = [a-d, b-c]`.
    pub fn interval_sub(&self, other: &Self) -> Self {
        Self {
            lower: self.lower.sub(&other.upper),
            upper: self.upper.sub(other.lo()),
        }
    }

    /// Interval negation: `-[a,b] = [-b, -a]`.
    pub fn interval_neg(&self) -> Self {
        Self {
            lower: self.upper.neg(),
            upper: self.lower.neg(),
        }
    }

    /// Interval multiplication by a non-negative scalar: `k * [a,b] = [k*a, k*b]`.
    pub fn scale_positive(&self, k: &UBinary) -> Self {
        let k_binary = k.to_binary();
        Self {
            lower: self.lower.mul(&k_binary),
            upper: self.upper.mul(&k_binary),
        }
    }

    /// Interval multiplication by a `BigInt` (can be negative).
    ///
    /// If `k >= 0`: `k * [a,b] = [k*a, k*b]`
    /// If `k < 0`: `k * [a,b] = [k*b, k*a]`
    pub fn scale_bigint(&self, k: &BigInt) -> Self {
        use num_traits::Signed;

        let abs_k = UBinary::new(k.magnitude().clone(), BigInt::zero());

        if k.is_negative() {
            self.interval_neg().scale_positive(&abs_k)
        } else {
            self.scale_positive(&abs_k)
        }
    }

    /// Returns the midpoint: `(lower + upper) / 2`.
    pub fn midpoint(&self) -> Binary {
        let sum = self.lower.add(&self.upper);
        Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one())
    }

    /// Returns the join (smallest enclosing interval) of two intervals.
    ///
    /// `[a, b].join([c, d]) = [min(a, c), max(b, d)]`
    pub fn join(&self, other: &Self) -> Self {
        let min_lo = std::cmp::min(&self.lower, &other.lower).clone();
        let max_hi = std::cmp::max(&self.upper, &other.upper).clone();
        Self {
            lower: min_lo,
            upper: max_hi,
        }
    }
}
