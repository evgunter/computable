use std::cmp::Ordering;
use std::fmt;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Signed, Zero};

mod ordered_pair;

pub use ordered_pair::{ordered_pair_checked, OrderedPair, OrderedPairError};

/// Exponent type for `Binary`; alias to keep the representation flexible.
pub type Exponent = i64;

impl OrderedPair<Exponent> {
    pub fn delta_usize(&self) -> Option<usize> {
        self.large
            .checked_sub(self.small)
            .and_then(|delta| usize::try_from(delta).ok())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    ExponentOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOverflow => write!(f, "exponent overflow during normalization"),
        }
    }
}

impl std::error::Error for BinaryError {}

/// Exact binary number represented as `mantissa * 2^exponent`.
/// `mantissa` is normalized to be odd unless the value is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binary {
    mantissa: BigInt,
    exponent: Exponent,
}

impl Binary {
    pub fn new(mantissa: BigInt, exponent: Exponent) -> Result<Self, BinaryError> {
        Self::normalize(mantissa, exponent)
    }

    pub fn mantissa(&self) -> &BigInt {
        &self.mantissa
    }

    pub fn exponent(&self) -> Exponent {
        self.exponent
    }

    fn normalize(mut mantissa: BigInt, mut exponent: Exponent) -> Result<Self, BinaryError> {
        if mantissa.is_zero() {
            return Ok(Self {
                mantissa,
                exponent: 0,
            });
        }

        while mantissa.is_even() {
            mantissa /= 2;
            exponent = exponent
                .checked_add(1)
                .ok_or(BinaryError::ExponentOverflow)?;
        }

        Ok(Self { mantissa, exponent })
    }

    fn cmp_shifted(
        mantissa: &BigInt,
        exponent: Exponent,
        other: &BigInt,
        other_exp: Exponent,
    ) -> Ordering {
        fn cmp_large_exp(
            large_mantissa: &BigInt,
            small_mantissa: &BigInt,
            pair: OrderedPair<Exponent>,
        ) -> Ordering {
            let shift = pair.delta_usize();

            if let Some(shift) = shift {
                let shifted = large_mantissa << shift;
                shifted.cmp(small_mantissa)
            } else if large_mantissa.is_zero() {
                BigInt::zero().cmp(small_mantissa)
            } else if large_mantissa.is_positive() {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }

        match exponent.cmp(&other_exp) {
            Ordering::Equal => mantissa.cmp(other),
            Ordering::Greater => {
                let pair = OrderedPair::new(exponent, other_exp);
                cmp_large_exp(mantissa, other, pair)
            }
            Ordering::Less => {
                let pair = OrderedPair::new(other_exp, exponent);
                cmp_large_exp(other, mantissa, pair).reverse()
            }
        }
    }
}

impl Ord for Binary {
    fn cmp(&self, other: &Self) -> Ordering {
        Self::cmp_shifted(
            &self.mantissa,
            self.exponent,
            &other.mantissa,
            other.exponent,
        )
    }
}

impl PartialOrd for Binary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type Bounds = OrderedPair<Binary>;

pub fn bounds_lower(bounds: &Bounds) -> &Binary {
    &bounds.small
}

pub fn bounds_upper(bounds: &Bounds) -> &Binary {
    &bounds.large
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_normalizes_even_mantissa() {
        let value = Binary::new(BigInt::from(8), 0).expect("binary should normalize");
        assert_eq!(value.mantissa(), &BigInt::from(1));
        assert_eq!(value.exponent(), 3);
    }

    #[test]
    fn binary_zero_uses_zero_exponent() {
        let value = Binary::new(BigInt::zero(), 42).expect("binary should normalize");
        assert_eq!(value.mantissa(), &BigInt::zero());
        assert_eq!(value.exponent(), 0);
    }

    #[test]
    fn binary_ordering_with_exponents() {
        let one = Binary::new(BigInt::from(1), 0).expect("binary should normalize");
        let half = Binary::new(BigInt::from(1), -1).expect("binary should normalize");
        assert!(one > half);
    }

    #[test]
    fn bounds_reject_invalid_order() {
        let lower = Binary::new(BigInt::from(1), 0).expect("binary should normalize");
        let upper = Binary::new(BigInt::from(-1), 0).expect("binary should normalize");
        let result = ordered_pair_checked(lower, upper);
        assert!(result.is_err());
    }
}
