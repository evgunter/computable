use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

use num_traits::{CheckedSub, Zero};

/// Stores two values ordered so that `large >= small` using a lower bound and width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedPair<T>
where
    T: Add<Output = T> + Sub<Output = T>,
{
    lower: T,
    width: T,
    width_overflowed: bool,
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
    T: Ord + Add<Output = T> + Sub<Output = T> + Clone + CheckedSub + Zero,
{
    fn width_from_bounds(lower: &T, upper: &T) -> (T, bool) {
        match upper.checked_sub(lower) {
            Some(width) => (width, false),
            None => (T::zero(), true),
        }
    }

    pub fn new(a: T, b: T) -> Self {
        match a.cmp(&b) {
            Ordering::Less => {
                let (width, width_overflowed) = Self::width_from_bounds(&a, &b);
                Self {
                    lower: a,
                    width,
                    width_overflowed,
                }
            }
            Ordering::Equal | Ordering::Greater => {
                let (width, width_overflowed) = Self::width_from_bounds(&b, &a);
                Self {
                    lower: b,
                    width,
                    width_overflowed,
                }
            }
        }
    }

    pub fn new_checked(small: T, large: T) -> Result<Self, OrderedPairError> {
        if small > large {
            return Err(OrderedPairError::InvalidOrder);
        }

        let (width, width_overflowed) = Self::width_from_bounds(&small, &large);
        Ok(Self {
            lower: small,
            width,
            width_overflowed,
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

    pub fn width_overflowed(&self) -> bool {
        self.width_overflowed
    }

    pub fn large(&self) -> T
    where
        T: Clone,
    {
        self.lower.clone() + self.width.clone()
    }
}
