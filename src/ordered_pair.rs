use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

use crate::binary::{xbinary_width, UXBinary, XBinary};

/// TODO: rename to `Interval` and require that `width` is positive.
/// Stores two values ordered so that `large >= small` using a lower bound and width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedPair<T>
where
    T: Add<Output = T> + Sub<Output = T>,
{
    lower: T,
    width: T,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedPairError {
    InvalidOrder,
}

impl fmt::Display for OrderedPairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOrder => write!(f, "first value must be <= second value"),
        }
    }
}

impl std::error::Error for OrderedPairError {}

impl<T> OrderedPair<T>
where
    T: Ord + Add<Output = T> + Sub<Output = T> + Clone,
{
    pub fn new(a: T, b: T) -> Self {
        match a.cmp(&b) {
            Ordering::Less => {
                let width = b.clone() - a.clone();
                Self { lower: a, width }
            }
            Ordering::Equal | Ordering::Greater => {
                let width = a.clone() - b.clone();
                Self { lower: b, width }
            }
        }
    }

    pub fn new_checked(small: T, large: T) -> Result<Self, OrderedPairError> {
        if small > large {
            return Err(OrderedPairError::InvalidOrder);
        }

        let width = large.clone() - small.clone();
        Ok(Self {
            lower: small,
            width,
        })
    }
}

impl<T> OrderedPair<T>
where
    T: Add<Output = T> + Sub<Output = T>,
{
    pub fn small(&self) -> &T {
        &self.lower
    }

    pub fn width(&self) -> &T {
        &self.width
    }

    pub fn large(&self) -> T
    where
        T: Clone,
    {
        self.lower.clone() + self.width.clone()
    }
}

// ============================================================================
// Bounds: Specialized OrderedPair for XBinary with UXBinary width
// ============================================================================

/// Bounds on a computable number: lower and upper bounds as XBinary values.
/// The width is stored as UXBinary to guarantee non-negativity through the type system.
///
/// This type enforces the invariant from the formalism that bounds widths are
/// always nonnegative (elements of D_inf where the value is >= 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bounds {
    lower: XBinary,
    width: UXBinary,
}

impl Bounds {
    /// Creates new bounds, automatically ordering the values so lower <= upper.
    pub fn new(a: XBinary, b: XBinary) -> Self {
        Self { lower: std::cmp::min(a.clone(), b.clone()), width: xbinary_width(&a, &b) }
    }

    /// Creates new bounds with explicit small and large values.
    /// Returns an error if small > large.
    pub fn new_checked(small: XBinary, large: XBinary) -> Result<Self, OrderedPairError> {
        if small > large {
            return Err(OrderedPairError::InvalidOrder);
        }

        let width = xbinary_width(&small, &large);
        Ok(Self {
            lower: small,
            width,
        })
    }

    /// Returns a reference to the lower bound.
    pub fn small(&self) -> &XBinary {
        &self.lower
    }

    /// Returns the width as a UXBinary (type-safe nonnegative value).
    pub fn width(&self) -> &UXBinary {
        &self.width
    }

    /// Returns the width converted to XBinary for compatibility with existing code.
    pub fn width_as_xbinary(&self) -> XBinary {
        self.width.to_xbinary()
    }

    /// Computes and returns the upper bound.
    pub fn large(&self) -> XBinary {
        self.lower.clone() + self.width.to_xbinary()
    }
}
