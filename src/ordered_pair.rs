use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

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
