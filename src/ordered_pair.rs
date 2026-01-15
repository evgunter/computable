use std::cmp::Ordering;
use std::fmt;

/// Stores two values ordered so that `large >= small`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedPair<T> {
    large: T,
    small: T,
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

impl<T: Ord> OrderedPair<T> {
    pub fn new(a: T, b: T) -> Self {
        match a.cmp(&b) {
            Ordering::Less => Self { large: b, small: a },
            Ordering::Equal | Ordering::Greater => Self { large: a, small: b },
        }
    }

    pub fn new_checked(small: T, large: T) -> Result<Self, OrderedPairError> {
        if small > large {
            return Err(OrderedPairError::InvalidOrder);
        }

        Ok(Self { large, small })
    }
}

impl<T> OrderedPair<T> {
    pub fn small(&self) -> &T {
        &self.small
    }

    pub fn large(&self) -> &T {
        &self.large
    }
}
