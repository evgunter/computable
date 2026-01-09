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
    MultiplicationOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOverflow => write!(f, "exponent overflow during normalization"),
            Self::ShiftOverflow => write!(f, "exponent shift overflow during alignment"),
            Self::MultiplicationOverflow => write!(f, "exponent overflow during multiplication"),
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

    pub fn neg(&self) -> Self {
        if self.mantissa.is_zero() {
            return self.clone();
        }
        Self {
            mantissa: -self.mantissa.clone(),
            exponent: self.exponent,
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self, BinaryError> {
        let exponent = self
            .exponent
            .checked_add(other.exponent)
            .ok_or(BinaryError::MultiplicationOverflow)?;
        let mantissa = &self.mantissa * &other.mantissa;
        Self::normalize(mantissa, exponent)
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
    InvalidBoundsOrder,
    NonImprovingBounds,
    Binary(BinaryError),
}

impl fmt::Display for ComputableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonpositiveEpsilon => write!(f, "epsilon must be positive"),
            Self::InvalidBoundsOrder => write!(f, "computed bounds are not ordered"),
            Self::NonImprovingBounds => write!(f, "refinement did not tighten bounds"),
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
    B: Fn(&X) -> Result<Bounds, ComputableError>,
    F: Fn(X) -> X,
{
    pub fn new(state: X, bounds: B, refine: F) -> Self {
        Self {
            state,
            bounds,
            refine,
        }
    }

    pub fn bounds(&self) -> Result<Bounds, ComputableError> {
        (self.bounds)(&self.state)
    }

    pub fn refine_to(mut self, epsilon: Binary) -> Result<(Bounds, Self), ComputableError> {
        if !epsilon.mantissa().is_positive() {
            return Err(ComputableError::NonpositiveEpsilon);
        }

        let mut bounds = (self.bounds)(&self.state)?;
        while !bounds_width_leq(&bounds, &epsilon)? {
            if bounds_lower(&bounds) != bounds_upper(&bounds) {
                let previous = bounds.clone();
                self.state = (self.refine)(self.state);
                bounds = (self.bounds)(&self.state)?;
                let lower_improved = bounds_lower(&bounds) > bounds_lower(&previous);
                let upper_improved = bounds_upper(&bounds) < bounds_upper(&previous);
                if !(lower_improved || upper_improved) {
                    return Err(ComputableError::NonImprovingBounds);
                }
            } else {
                self.state = (self.refine)(self.state);
                bounds = (self.bounds)(&self.state)?;
            }
        }

        Ok((bounds, self))
    }

    pub fn add<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<(X, Y), impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>, impl Fn((X, Y)) -> (X, Y)>
    where
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: Fn(Y) -> Y,
    {
        // Note: we don't implement std::ops::Add here because the composed type uses `impl Fn`
        // in its return signature, which cannot be named for an associated Output type.
        let Computable {
            state: left_state,
            bounds: left_bounds,
            refine: left_refine,
        } = self;
        let Computable {
            state: right_state,
            bounds: right_bounds,
            refine: right_refine,
        } = other;

        let bounds = move |state: &(X, Y)| -> Result<Bounds, ComputableError> {
            let left = (left_bounds)(&state.0)?;
            let right = (right_bounds)(&state.1)?;
            let lower = bounds_lower(&left).add(bounds_lower(&right))?;
            let upper = bounds_upper(&left).add(bounds_upper(&right))?;
            ordered_pair_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
        };

        let refine = move |state: (X, Y)| -> (X, Y) {
            let (left, right) = state;
            ((left_refine)(left), (right_refine)(right))
        };

        Computable::new((left_state, right_state), bounds, refine)
    }

    pub fn sub<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<(X, Y), impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>, impl Fn((X, Y)) -> (X, Y)>
    where
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: Fn(Y) -> Y,
    {
        let Computable {
            state: left_state,
            bounds: left_bounds,
            refine: left_refine,
        } = self;
        let Computable {
            state: right_state,
            bounds: right_bounds,
            refine: right_refine,
        } = other;

        let bounds = move |state: &(X, Y)| -> Result<Bounds, ComputableError> {
            let left = (left_bounds)(&state.0)?;
            let right = (right_bounds)(&state.1)?;
            let lower = bounds_lower(&left).sub(bounds_upper(&right))?;
            let upper = bounds_upper(&left).sub(bounds_lower(&right))?;
            ordered_pair_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
        };

        let refine = move |state: (X, Y)| -> (X, Y) {
            let (left, right) = state;
            ((left_refine)(left), (right_refine)(right))
        };

        Computable::new((left_state, right_state), bounds, refine)
    }

    pub fn neg(self) -> Computable<X, impl Fn(&X) -> Result<Bounds, ComputableError>, impl Fn(X) -> X> {
        let Computable {
            state,
            bounds,
            refine,
        } = self;

        let bounds = move |state: &X| -> Result<Bounds, ComputableError> {
            let existing = (bounds)(state)?;
            let lower = bounds_lower(&existing).neg();
            let upper = bounds_upper(&existing).neg();
            ordered_pair_checked(upper, lower).map_err(|_| ComputableError::InvalidBoundsOrder)
        };

        Computable::new(state, bounds, refine)
    }

    pub fn mul<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<(X, Y), impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>, impl Fn((X, Y)) -> (X, Y)>
    where
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: Fn(Y) -> Y,
    {
        let Computable {
            state: left_state,
            bounds: left_bounds,
            refine: left_refine,
        } = self;
        let Computable {
            state: right_state,
            bounds: right_bounds,
            refine: right_refine,
        } = other;

        let bounds = move |state: &(X, Y)| -> Result<Bounds, ComputableError> {
            let left = (left_bounds)(&state.0)?;
            let right = (right_bounds)(&state.1)?;
            let left_lower = bounds_lower(&left);
            let left_upper = bounds_upper(&left);
            let right_lower = bounds_lower(&right);
            let right_upper = bounds_upper(&right);

            let candidates = [
                left_lower.mul(right_lower)?,
                left_lower.mul(right_upper)?,
                left_upper.mul(right_lower)?,
                left_upper.mul(right_upper)?,
            ];

            let mut min = candidates[0].clone();
            let mut max = candidates[0].clone();
            for candidate in candidates.iter().skip(1) {
                if candidate < &min {
                    min = candidate.clone();
                }
                if candidate > &max {
                    max = candidate.clone();
                }
            }

            ordered_pair_checked(min, max).map_err(|_| ComputableError::InvalidBoundsOrder)
        };

        let refine = move |state: (X, Y)| -> (X, Y) {
            let (left, right) = state;
            ((left_refine)(left), (right_refine)(right))
        };

        Computable::new((left_state, right_state), bounds, refine)
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

    type IntervalState = Bounds;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), exponent).expect("binary should normalize")
    }

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn interval_refine(state: IntervalState) -> IntervalState {
        let mid = state
            .small
            .add(&state.large)
            .expect("binary should add");
        let exponent = mid
            .exponent()
            .checked_sub(1)
            .expect("midpoint exponent should not underflow");
        let mid = Binary::new(mid.mantissa().clone(), exponent)
            .expect("binary should normalize");
        OrderedPair::new(mid.clone(), mid)
    }

    fn interval_refine_strict(state: IntervalState) -> IntervalState {
        let mid = state
            .small
            .add(&state.large)
            .expect("binary should add");
        let exponent = mid
            .exponent()
            .checked_sub(1)
            .expect("midpoint exponent should not underflow");
        let mid = Binary::new(mid.mantissa().clone(), exponent)
            .expect("binary should normalize");
        OrderedPair::new(state.small.clone(), mid)
    }

    fn interval_computable(
        lower: i64,
        upper: i64,
    ) -> Computable<
        IntervalState,
        impl Fn(&IntervalState) -> Result<Bounds, ComputableError>,
        impl Fn(IntervalState) -> IntervalState,
    > {
        let state = OrderedPair::new(bin(lower, 0), bin(upper, 0));
        Computable::new(state, |state| Ok(interval_bounds(state)), interval_refine)
    }

    #[test]
    fn binary_normalizes_even_mantissa() {
        let value = bin(8, 0);
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
        let one = bin(1, 0);
        let half = bin(1, -1);
        assert!(one > half);
    }

    #[test]
    fn bounds_reject_invalid_order() {
        let lower = bin(1, 0);
        let upper = bin(-1, 0);
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
        let one = bin(1, 0);
        let half = bin(1, -1);
        let sum = one.add(&half).expect("binary should add");
        let expected = bin(3, -1);
        assert_eq!(sum, expected);
    }

    #[test]
    fn binary_sub_handles_negative() {
        let one = bin(1, 0);
        let two = bin(1, 1);
        let diff = one.sub(&two).expect("binary should subtract");
        let expected = bin(-1, 0);
        assert_eq!(diff, expected);
    }

    #[test]
    fn binary_mul_adds_exponents() {
        let two = bin(1, 1);
        let half = bin(1, -1);
        let product = two.mul(&half).expect("binary should multiply");
        let expected = bin(1, 0);
        assert_eq!(product, expected);
    }

    #[test]
    fn computable_refine_to_rejects_negative_epsilon() {
        let computable = interval_computable(0, 2);
        let epsilon = bin(-1, 0);
        let result = computable.refine_to(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        let computable = interval_computable(0, 2);
        let epsilon = bin(1, -1);
        let (bounds, refined) = computable
            .refine_to(epsilon)
            .expect("refine_to should succeed");
        assert_eq!(bounds_lower(&bounds), bounds_upper(&bounds));
        assert_eq!(
            refined.bounds().expect("bounds should succeed"),
            OrderedPair::new(
                bin(1, 0),
                bin(1, 0)
            )
        );
    }

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        let computable = interval_computable(0, 2);
        let epsilon = bin(0, 0);
        let result = computable.refine_to(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }

    #[test]
    fn computable_refine_to_rejects_non_improving_refine() {
        let state = OrderedPair::new(bin(0, 0), bin(2, 0));
        let computable = Computable::new(state, |state| Ok(interval_bounds(state)), |state| state);
        let epsilon = bin(1, -2);
        let result = computable.refine_to(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonImprovingBounds)
        ));
    }

    #[test]
    fn computable_refine_to_handles_non_meeting_bounds() {
        let state = OrderedPair::new(bin(0, 0), bin(4, 0));
        let computable = Computable::new(
            state,
            |state| Ok(interval_bounds(state)),
            interval_refine_strict,
        );
        let epsilon = bin(1, -1);
        let (bounds, refined) = computable
            .refine_to(epsilon)
            .expect("refine_to should succeed");
        assert!(bounds_lower(&bounds) < bounds_upper(&bounds));
        assert_eq!(refined.bounds().expect("bounds should succeed"), bounds);
    }

    #[test]
    fn computable_add_combines_bounds() {
        let left = interval_computable(0, 2);
        let right = interval_computable(1, 3);

        let sum = left.add(right);
        let sum_bounds = sum.bounds().expect("bounds should succeed");
        assert_eq!(
            sum_bounds,
            OrderedPair::new(
                bin(1, 0),
                bin(5, 0)
            )
        );
    }

    #[test]
    fn computable_sub_combines_bounds() {
        let left = interval_computable(4, 6);
        let right = interval_computable(1, 2);

        let diff = left.sub(right);
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(
            diff_bounds,
            OrderedPair::new(
                bin(2, 0),
                bin(5, 0)
            )
        );
    }

    #[test]
    fn computable_neg_flips_bounds() {
        let value = interval_computable(1, 3);
        let negated = value.neg();
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                bin(-3, 0),
                bin(-1, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_positive() {
        let left = interval_computable(1, 3);
        let right = interval_computable(2, 4);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                bin(2, 0),
                bin(12, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_negative() {
        let left = interval_computable(-3, -1);
        let right = interval_computable(2, 4);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                bin(-12, 0),
                bin(-2, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_mixed() {
        let left = interval_computable(-2, 3);
        let right = interval_computable(4, 5);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                bin(-10, 0),
                bin(15, 0)
            )
        );
    }
}
