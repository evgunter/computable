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
    ShiftOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOverflow => write!(f, "exponent overflow during normalization"),
            Self::ShiftOverflow => write!(f, "exponent shift overflow during alignment"),
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

    pub fn add(&self, other: &Self) -> Result<Self, BinaryError> {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other)?;
        Self::normalize(lhs + rhs, exponent)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, BinaryError> {
        let (lhs, rhs, exponent) = Self::align_mantissas(self, other)?;
        Self::normalize(lhs - rhs, exponent)
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

    fn align_mantissas(lhs: &Self, rhs: &Self) -> Result<(BigInt, BigInt, Exponent), BinaryError> {
        let exponent = lhs.exponent.min(rhs.exponent);
        let lhs_shift = lhs
            .exponent
            .checked_sub(exponent)
            .ok_or(BinaryError::ShiftOverflow)?;
        let rhs_shift = rhs
            .exponent
            .checked_sub(exponent)
            .ok_or(BinaryError::ShiftOverflow)?;
        let lhs_mantissa = Self::shift_mantissa(&lhs.mantissa, lhs_shift)?;
        let rhs_mantissa = Self::shift_mantissa(&rhs.mantissa, rhs_shift)?;
        Ok((lhs_mantissa, rhs_mantissa, exponent))
    }

    fn shift_mantissa(mantissa: &BigInt, shift: Exponent) -> Result<BigInt, BinaryError> {
        if shift < 0 {
            return Err(BinaryError::ShiftOverflow);
        }
        let shift = usize::try_from(shift).map_err(|_| BinaryError::ShiftOverflow)?;
        Ok(mantissa << shift)
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputableError {
    NonpositiveEpsilon,
    Binary(BinaryError),
}

impl fmt::Display for ComputableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonpositiveEpsilon => write!(f, "epsilon must be positive"),
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ComputableError {}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

pub struct Computable<X, B, F> {
    state: X,
    bounds: B,
    refine: F,
}

impl<X, B, F> Computable<X, B, F>
where
    B: Fn(&X) -> Bounds,
    F: Fn(X) -> X,
{
    pub fn new(state: X, bounds: B, refine: F) -> Self {
        Self {
            state,
            bounds,
            refine,
        }
    }

    pub fn bounds(&self) -> Bounds {
        (self.bounds)(&self.state)
    }

    pub fn refine_to(mut self, epsilon: Binary) -> Result<(Bounds, Self), ComputableError> {
        if !epsilon.mantissa().is_positive() {
            return Err(ComputableError::NonpositiveEpsilon);
        }

        let mut bounds = (self.bounds)(&self.state);
        while !bounds_width_leq(&bounds, &epsilon)? {
            self.state = (self.refine)(self.state);
            bounds = (self.bounds)(&self.state);
        }

        Ok((bounds, self))
    }
}

fn bounds_width_leq(bounds: &Bounds, epsilon: &Binary) -> Result<bool, BinaryError> {
    let upper = bounds_upper(bounds);
    let lower = bounds_lower(bounds);
    let lower_plus = lower.add(epsilon)?;
    Ok(upper <= &lower_plus)
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

    #[test]
    fn binary_ordering_handles_large_exponent_gaps() {
        let huge_pos = Binary::new(BigInt::from(1), Exponent::MAX).expect("binary should normalize");
        let tiny_pos = Binary::new(BigInt::from(1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_pos > tiny_pos);

        let huge_neg = Binary::new(BigInt::from(-1), Exponent::MAX).expect("binary should normalize");
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_ordering_overflow_path_uses_sign() {
        let huge_pos = Binary::new(BigInt::from(1), Exponent::MAX).expect("binary should normalize");
        let tiny_neg = Binary::new(BigInt::from(-1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_pos > tiny_neg);

        let huge_neg = Binary::new(BigInt::from(-1), Exponent::MAX).expect("binary should normalize");
        let tiny_pos = Binary::new(BigInt::from(1), Exponent::MIN).expect("binary should normalize");
        assert!(huge_neg < tiny_pos);
    }

    #[test]
    fn binary_add_aligns_exponents() {
        let one = Binary::new(BigInt::from(1), 0).expect("binary should normalize");
        let half = Binary::new(BigInt::from(1), -1).expect("binary should normalize");
        let sum = one.add(&half).expect("binary should add");
        let expected = Binary::new(BigInt::from(3), -1).expect("binary should normalize");
        assert_eq!(sum, expected);
    }

    #[test]
    fn binary_sub_handles_negative() {
        let one = Binary::new(BigInt::from(1), 0).expect("binary should normalize");
        let two = Binary::new(BigInt::from(1), 1).expect("binary should normalize");
        let diff = one.sub(&two).expect("binary should subtract");
        let expected = Binary::new(BigInt::from(-1), 0).expect("binary should normalize");
        assert_eq!(diff, expected);
    }

    #[test]
    fn computable_refine_to_rejects_negative_epsilon() {
        #[derive(Clone)]
        struct IntervalState {
            lower: Binary,
            upper: Binary,
        }

        fn bounds(state: &IntervalState) -> Bounds {
            OrderedPair::new(state.lower.clone(), state.upper.clone())
        }

        fn refine(state: IntervalState) -> IntervalState {
            let lower = state.lower;
            IntervalState {
                upper: lower.clone(),
                lower,
            }
        }

        let state = IntervalState {
            lower: Binary::new(BigInt::from(0), 0).expect("binary should normalize"),
            upper: Binary::new(BigInt::from(2), 0).expect("binary should normalize"),
        };
        let computable = Computable::new(state, bounds, refine);
        let epsilon = Binary::new(BigInt::from(-1), 0).expect("binary should normalize");
        let result = computable.refine_to(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        #[derive(Clone)]
        struct IntervalState {
            lower: Binary,
            upper: Binary,
        }

        fn bounds(state: &IntervalState) -> Bounds {
            OrderedPair::new(state.lower.clone(), state.upper.clone())
        }

        fn refine(state: IntervalState) -> IntervalState {
            let lower = state.lower;
            IntervalState {
                upper: lower.clone(),
                lower,
            }
        }

        let state = IntervalState {
            lower: Binary::new(BigInt::from(0), 0).expect("binary should normalize"),
            upper: Binary::new(BigInt::from(2), 0).expect("binary should normalize"),
        };
        let computable = Computable::new(state, bounds, refine);
        let epsilon = Binary::new(BigInt::from(1), -1).expect("binary should normalize");
        let (bounds, refined) = computable
            .refine_to(epsilon)
            .expect("refine_to should succeed");
        assert_eq!(bounds_lower(&bounds), bounds_upper(&bounds));
        assert_eq!(
            refined.bounds(),
            OrderedPair::new(
                Binary::new(BigInt::from(0), 0).expect("binary should normalize"),
                Binary::new(BigInt::from(0), 0).expect("binary should normalize")
            )
        );
    }

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        #[derive(Clone)]
        struct IntervalState {
            lower: Binary,
            upper: Binary,
        }

        fn bounds(state: &IntervalState) -> Bounds {
            OrderedPair::new(state.lower.clone(), state.upper.clone())
        }

        fn refine(state: IntervalState) -> IntervalState {
            let lower = state.lower;
            IntervalState {
                upper: lower.clone(),
                lower,
            }
        }

        let state = IntervalState {
            lower: Binary::new(BigInt::from(0), 0).expect("binary should normalize"),
            upper: Binary::new(BigInt::from(2), 0).expect("binary should normalize"),
        };
        let computable = Computable::new(state, bounds, refine);
        let epsilon = Binary::new(BigInt::from(0), 0).expect("binary should normalize");
        let result = computable.refine_to(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }
}
