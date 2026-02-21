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

    /// Constructs an interval directly from a lower bound and width.
    ///
    /// This is more efficient than `new` when you already have the width computed,
    /// as it avoids the round-trip of computing upper = lower + width only to have
    /// `new` compute width = upper - lower again.
    ///
    /// # Arguments
    ///
    /// * `lower` - The lower bound of the interval
    /// * `width` - The width of the interval (must be non-negative by the type system)
    pub fn from_lower_and_width(lower: T, width: W) -> Self {
        Self { lower, width }
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

    /// Creates a point interval [x, x] with zero width.
    pub fn point(x: T) -> Self
    where
        W: num_traits::Zero,
    {
        Self::from_lower_and_width(x, W::zero())
    }

    /// Checks if this interval contains a point.
    pub fn contains(&self, point: &T) -> bool {
        self.small() <= point && *point <= self.large()
    }

    /// Checks if this interval is entirely less than another.
    pub fn entirely_less_than(&self, other: &Self) -> bool {
        self.large() < *other.small()
    }

    /// Checks if this interval is entirely greater than another.
    pub fn entirely_greater_than(&self, other: &Self) -> bool {
        *self.small() > other.large()
    }

    /// Checks if this interval overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        !(self.entirely_less_than(other) || self.entirely_greater_than(other))
    }

    /// Returns the join (smallest enclosing interval) of two intervals.
    ///
    /// `[a, b].join([c, d]) = [min(a, c), max(b, d)]`
    ///
    /// This is the lattice join operation: the smallest interval that contains
    /// both inputs. Note that if the intervals are disjoint, the result includes
    /// points in neither original interval (i.e., this is the convex hull).
    pub fn join(&self, other: &Self) -> Self {
        let min_lo = std::cmp::min(self.small(), other.small()).clone();
        let max_hi = std::cmp::max(self.large(), other.large());
        Self::new(min_lo, max_hi)
    }

    /// Returns the intersection of two intervals, if non-empty.
    ///
    /// `[a, b].intersection([c, d]) = [max(a, c), min(b, d)]` if non-empty
    ///
    /// Returns `None` if the intervals don't overlap.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.overlaps(other) {
            return None;
        }
        let max_lo = std::cmp::max(self.small(), other.small()).clone();
        let min_hi = std::cmp::min(self.large(), other.large());
        Some(Self::new(max_lo, min_hi))
    }
}
