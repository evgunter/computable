use std::fmt;

use crate::binary::{UXBinary, XBinary};

pub trait AddWidth<T, W> {
    fn add_width(self, width: W) -> T;
}

pub trait SubWidth<T, W> {
    fn sub_width(self, width: W) -> T;
}

pub trait Unsigned {}

/// TODO: require that `width` is positive.
/// Stores two values ordered so that `large >= small` using a lower bound and width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval<T, W>
where
    T: AddWidth<T, W> + SubWidth<T, W>,
    W: Unsigned + PartialOrd,
{
    lower: T,
    width: W,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalError {
    InvalidOrder,
}

impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOrder => write!(f, "first value must be <= second value"),
        }
    }
}

impl std::error::Error for IntervalError {}

pub trait AbsDistance<T, W> {
    fn abs_distance(self, other: T) -> W;
}

impl<T, W> Interval<T, W>
where
    T: Ord + AddWidth<T, W> + SubWidth<T, W> + Clone + AbsDistance<T, W>,
    W: Clone + PartialOrd + Unsigned,
{
    pub fn new(a: T, b: T) -> Self {
        let (lower, larger) = if a <= b { (a, b) } else { (b, a) };
        let width = lower.clone().abs_distance(larger);
        Self { lower, width }
    }

    pub fn new_checked(small: T, large: T) -> Result<Self, IntervalError> {
        if small > large {
            return Err(IntervalError::InvalidOrder);
        }
        let width = small.clone().abs_distance(large);
        Ok(Self {
            lower: small,
            width,
        })
    }

    pub fn small(&self) -> &T {
        &self.lower
    }

    pub fn width(&self) -> &W {
        &self.width
    }

    pub fn large(&self) -> T {
        self.lower.clone().add_width(self.width.clone())
    }
}

/// Bounds on a computable number: lower and upper bounds as XBinary values.
/// The width is stored as UXBinary to guarantee non-negativity through the type system.
///
/// This type enforces the invariant from the formalism that bounds widths are
/// always nonnegative (elements of D_inf where the value is >= 0).
pub type Bounds = Interval<XBinary, UXBinary>;
