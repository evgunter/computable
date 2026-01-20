use std::fmt;

pub trait AddWidth<T, W> {
    fn add_width(self, width: W) -> T;
}

pub trait Unsigned {}

/// Stores two values ordered so that `large >= small` using a lower bound and width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval<T, W>
where
    T: AddWidth<T, W>,
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
    T: Ord + AddWidth<T, W> + Clone + AbsDistance<T, W>,
    W: Clone + PartialOrd + Unsigned + num_traits::Zero,
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

    /// Creates an interval where lower == upper (a point).
    /// This is an optimization that avoids computing abs_distance.
    pub fn point(value: T) -> Self {
        Self {
            lower: value,
            width: W::zero(),
        }
    }

    pub fn small(&self) -> &T {
        &self.lower
    }

    pub fn width(&self) -> &W {
        &self.width
    }

    /// Returns the upper bound, with fast path for zero-width intervals.
    pub fn large(&self) -> T {
        if self.width.is_zero() {
            self.lower.clone()
        } else {
            self.lower.clone().add_width(self.width.clone())
        }
    }
}

