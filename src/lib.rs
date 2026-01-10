#![warn(
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used,
)]

use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Float, One, Signed, Zero};

mod ordered_pair;

pub use ordered_pair::{OrderedPair, OrderedPairError};

/// exponent type for `Binary`; alias to keep the representation flexible.
pub type Exponent = i64;

impl OrderedPair<Exponent> {
    pub fn delta_usize(&self) -> Option<usize> {
        self.large()
            .checked_sub(*self.small())
            .and_then(|delta| usize::try_from(delta).ok())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryError {
    ExponentOverflow,
    ShiftOverflow,
    MultiplicationOverflow,
    ReciprocalOverflow,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOverflow => write!(f, "exponent overflow during normalization"),
            Self::ShiftOverflow => write!(f, "exponent shift overflow during alignment"),
            Self::MultiplicationOverflow => write!(f, "exponent overflow during multiplication"),
            Self::ReciprocalOverflow => write!(f, "exponent overflow during reciprocal"),
        }
    }
}

impl std::error::Error for BinaryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtendedBinaryError {
    Nan,
    Binary(BinaryError),
}

impl fmt::Display for ExtendedBinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => write!(f, "cannot convert NaN to ExtendedBinary"),
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ExtendedBinaryError {}

impl From<BinaryError> for ExtendedBinaryError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

/// exact binary number represented as `mantissa * 2^exponent`.
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

    pub fn zero() -> Self {
        Self {
            mantissa: BigInt::zero(),
            exponent: 0,
        }
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
        let shift_amount = usize::try_from(shift).map_err(|_| BinaryError::ShiftOverflow)?;
        Ok(mantissa << shift_amount)
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
            let shift_amount_opt = pair.delta_usize();

            if let Some(shift_amount) = shift_amount_opt {
                let shifted = large_mantissa << shift_amount;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtendedBinary {
    NegInf,
    Finite(Binary),
    PosInf,
}

impl ExtendedBinary {
    pub fn zero() -> Self {
        Self::Finite(Binary::zero())
    }

    pub fn neg(&self) -> Self {
        match self {
            Self::NegInf => Self::PosInf,
            Self::PosInf => Self::NegInf,
            Self::Finite(value) => Self::Finite(value.neg()),
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Finite(value) if value.mantissa().is_zero())
    }

    pub fn from_f64(value: f64) -> Result<Self, ExtendedBinaryError> {
        if value.is_nan() {
            return Err(ExtendedBinaryError::Nan);
        }
        if value == 0.0 {
            return Ok(Self::Finite(Binary::zero()));
        }
        if value == f64::INFINITY {
            return Ok(Self::PosInf);
        }
        if value == f64::NEG_INFINITY {
            return Ok(Self::NegInf);
        }
        let (mantissa, exponent, sign) = value.integer_decode();
        let signed_mantissa = BigInt::from(sign) * BigInt::from(mantissa);
        Ok(Self::Finite(Binary::new(
            signed_mantissa,
            Exponent::from(exponent),
        )?))
    }

    fn add_lower(&self, other: &Self) -> Result<Self, ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, _) | (_, NegInf) => Ok(NegInf),
            (PosInf, _) | (_, PosInf) => Ok(PosInf),
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.add(rhs)?)),
        }
    }

    fn add_upper(&self, other: &Self) -> Result<Self, ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (PosInf, _) | (_, PosInf) => Ok(PosInf),
            (NegInf, _) | (_, NegInf) => Ok(NegInf),
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.add(rhs)?)),
        }
    }

    fn mul(&self, other: &Self) -> Result<Self, ComputableError> {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        if self.is_zero() || other.is_zero() {
            return Ok(Finite(Binary::zero()));
        }
        match (self, other) {
            (Finite(lhs), Finite(rhs)) => Ok(Finite(lhs.mul(rhs)?)),
            (Finite(lhs), PosInf) | (PosInf, Finite(lhs)) => {
                if lhs.mantissa().is_positive() {
                    Ok(PosInf)
                } else {
                    Ok(NegInf)
                }
            }
            (Finite(lhs), NegInf) | (NegInf, Finite(lhs)) => {
                if lhs.mantissa().is_positive() {
                    Ok(NegInf)
                } else {
                    Ok(PosInf)
                }
            }
            (PosInf, PosInf) | (NegInf, NegInf) => Ok(PosInf),
            (PosInf, NegInf) | (NegInf, PosInf) => Ok(NegInf),
        }
    }
}

impl Ord for ExtendedBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        use ExtendedBinary::{Finite, NegInf, PosInf};
        match (self, other) {
            (NegInf, NegInf) | (PosInf, PosInf) => Ordering::Equal,
            (NegInf, _) => Ordering::Less,
            (_, NegInf) => Ordering::Greater,
            (PosInf, _) => Ordering::Greater,
            (_, PosInf) => Ordering::Less,
            (Finite(lhs), Finite(rhs)) => lhs.cmp(rhs),
        }
    }
}

impl PartialOrd for ExtendedBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type Bounds = OrderedPair<ExtendedBinary>;

pub fn bounds_lower(bounds: &Bounds) -> &ExtendedBinary {
    bounds.small()
}

pub fn bounds_upper(bounds: &Bounds) -> &ExtendedBinary {
    bounds.large()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputableError {
    NonpositiveEpsilon,
    InvalidBoundsOrder,
    BoundsWorsened,
    StateUnchanged,
    ExcludedValueUnreachable,
    MaxRefinementIterations { max: usize },
    Binary(BinaryError),
}

impl fmt::Display for ComputableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonpositiveEpsilon => write!(f, "epsilon must be positive"),
            Self::InvalidBoundsOrder => write!(f, "computed bounds are not ordered"),
            Self::BoundsWorsened => write!(f, "refinement produced worse bounds"),
            Self::StateUnchanged => write!(f, "refinement did not change state"),
            Self::ExcludedValueUnreachable => write!(f, "cannot refine bounds to exclude value"),
            Self::MaxRefinementIterations { max } => {
                write!(f, "maximum refinement iterations ({max}) reached")
            }
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

type BoundsFn<'a, X> = dyn Fn(&X) -> Result<Bounds, ComputableError> + 'a;

fn validate_refinement_progress<X: Eq>(
    previous_state: &X,
    next_state: &X,
    previous_bounds: &Bounds,
    next_bounds: &Bounds,
) -> Result<(), ComputableError> {
    if previous_state == next_state {
        return Err(ComputableError::StateUnchanged);
    }
    let lower_worsened = bounds_lower(next_bounds) < bounds_lower(previous_bounds);
    let upper_worsened = bounds_upper(next_bounds) > bounds_upper(previous_bounds);
    if lower_worsened || upper_worsened {
        return Err(ComputableError::BoundsWorsened);
    }
    Ok(())
}

fn halve_binary(value: &Binary) -> Result<Binary, BinaryError> {
    let exponent = value
        .exponent()
        .checked_sub(1)
        .ok_or(BinaryError::ExponentOverflow)?;
    Binary::new(value.mantissa().clone(), exponent)
}

fn refine_to_with_step<X, F>(
    state: X,
    bounds: &BoundsFn<'_, X>,
    refine_step: &F,
    epsilon: Binary,
    max_iterations: usize,
) -> Result<(Bounds, X), ComputableError>
where
    X: Eq + Clone,
    F: Fn(X) -> X,
{
    if !epsilon.mantissa().is_positive() {
        return Err(ComputableError::NonpositiveEpsilon);
    }

    let current_bounds = bounds(&state)?;
    refine_to_with_step_inner(
        state,
        current_bounds,
        bounds,
        refine_step,
        &epsilon,
        max_iterations,
        0,
    )
}

fn refine_to_with_step_inner<X, F>(
    state: X,
    bounds: Bounds,
    bounds_fn: &BoundsFn<'_, X>,
    refine_step: &F,
    epsilon: &Binary,
    max_iterations: usize,
    iterations: usize,
) -> Result<(Bounds, X), ComputableError>
where
    X: Eq + Clone,
    F: Fn(X) -> X,
{
    if bounds_width_leq(&bounds, epsilon)? {
        return Ok((bounds, state));
    }
    if iterations >= max_iterations {
        return Err(ComputableError::MaxRefinementIterations { max: max_iterations });
    }
    let previous_state = state.clone();
    let next_state = refine_step(state);
    let next_bounds = bounds_fn(&next_state)?;
    validate_refinement_progress(&previous_state, &next_state, &bounds, &next_bounds)?;
    refine_to_with_step_inner(
        next_state,
        next_bounds,
        bounds_fn,
        refine_step,
        epsilon,
        max_iterations,
        iterations + 1,
    )
}

fn refine_to_with_policy<X, F>(
    state: X,
    bounds: &BoundsFn<'_, X>,
    target: Binary,
    epsilon: Binary,
    max_iterations: usize,
    refine_state: F,
) -> Result<(Bounds, X), ComputableError>
where
    X: Eq + Clone,
    F: for<'a> Fn(X, &'a BoundsFn<'a, X>, &Binary, usize) -> Result<X, ComputableError>,
{
    if !epsilon.mantissa().is_positive() {
        return Err(ComputableError::NonpositiveEpsilon);
    }

    let current_bounds = bounds(&state)?;
    refine_to_with_policy_inner(
        state,
        current_bounds,
        bounds,
        target,
        &epsilon,
        max_iterations,
        0,
        refine_state,
    )
}

fn refine_to_with_policy_inner<X, F>(
    state: X,
    bounds: Bounds,
    bounds_fn: &BoundsFn<'_, X>,
    target: Binary,
    epsilon: &Binary,
    max_iterations: usize,
    iterations: usize,
    refine_state: F,
) -> Result<(Bounds, X), ComputableError>
where
    X: Eq + Clone,
    F: for<'a> Fn(X, &'a BoundsFn<'a, X>, &Binary, usize) -> Result<X, ComputableError>,
{
    if bounds_width_leq(&bounds, epsilon)? {
        return Ok((bounds, state));
    }
    if iterations >= max_iterations {
        return Err(ComputableError::MaxRefinementIterations { max: max_iterations });
    }
    let previous_state = state.clone();
    let next_state = refine_state(state, bounds_fn, &target, max_iterations)?;
    let next_bounds = bounds_fn(&next_state)?;
    validate_refinement_progress(&previous_state, &next_state, &bounds, &next_bounds)?;
    let next_target = halve_binary(&target)?;
    refine_to_with_policy_inner(
        next_state,
        next_bounds,
        bounds_fn,
        next_target,
        epsilon,
        max_iterations,
        iterations + 1,
        refine_state,
    )
}

fn computable_with_refine_step<X, B, R>(
    state: X,
    bounds: B,
    refine_step: R,
) -> Computable<
    X,
    B,
    impl for<'a> Fn(X, &'a BoundsFn<'a, X>, Binary, usize) -> Result<(Bounds, X), ComputableError>,
>
where
    X: Eq + Clone,
    B: Fn(&X) -> Result<Bounds, ComputableError>,
    R: Fn(X) -> X,
{
    let refine_to = move |state: X,
                          bounds: &BoundsFn<'_, X>,
                          epsilon: Binary,
                          max_iterations: usize|
          -> Result<(Bounds, X), ComputableError> {
        refine_to_with_step(state, bounds, &refine_step, epsilon, max_iterations)
    };

    Computable::new(state, bounds, refine_to)
}

pub struct Computable<X, B, F> {
    state: X,
    bounds: B,
    refine_to: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State<X, S> {
    inner_state: X,
    extra_state: S,
}

impl<X, B, F> Computable<X, B, F>
where
    B: Fn(&X) -> Result<Bounds, ComputableError>,
    F: for<'a> Fn(X, &'a BoundsFn<'a, X>, Binary, usize) -> Result<(Bounds, X), ComputableError>,
{
    pub fn new(state: X, bounds: B, refine_to: F) -> Self {
        Self {
            state,
            bounds,
            refine_to,
        }
    }

    pub fn bounds(&self) -> Result<Bounds, ComputableError> {
        (self.bounds)(&self.state)
    }

    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        self,
        epsilon: Binary,
    ) -> Result<(Bounds, Self), ComputableError>
    {
        let Computable {
            state,
            bounds,
            refine_to,
        } = self;
        let (bounds_value, state) =
            refine_to(state, &bounds, epsilon, MAX_REFINEMENT_ITERATIONS)?;

        Ok((
            bounds_value,
            Computable {
                state,
                bounds,
                refine_to,
            },
        ))
    }

    pub fn refine_to_default(self, epsilon: Binary) -> Result<(Bounds, Self), ComputableError>
    {
        self.refine_to::<DEFAULT_MAX_REFINEMENT_ITERATIONS>(epsilon)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(
        self,
    ) -> Computable<
        X,
        impl Fn(&X) -> Result<Bounds, ComputableError>,
        impl for<'a> Fn(
            X,
            &'a BoundsFn<'a, X>,
            Binary,
            usize,
        ) -> Result<(Bounds, X), ComputableError>,
    > {
        let Computable {
            state,
            bounds: base_bounds,
            refine_to: base_refine_to,
        } = self;

        let base_bounds = Rc::new(base_bounds);
        let neg_bounds = {
            let base_bounds = Rc::clone(&base_bounds);
            move |value_state: &X| -> Result<Bounds, ComputableError> {
                let existing = base_bounds.as_ref()(value_state)?;
                let lower = bounds_lower(&existing).neg();
                let upper = bounds_upper(&existing).neg();
                OrderedPair::new_checked(upper, lower)
                    .map_err(|_| ComputableError::InvalidBoundsOrder)
            }
        };

        let refine_to = {
            let base_bounds = Rc::clone(&base_bounds);
            move |value_state: X,
                  bounds: &BoundsFn<'_, X>,
                  epsilon: Binary,
                  max_iterations: usize|
              -> Result<(Bounds, X), ComputableError> {
                let (_, refined_state) = base_refine_to(
                    value_state,
                    base_bounds.as_ref(),
                    epsilon,
                    max_iterations,
                )?;
                let refined_bounds = bounds(&refined_state)?;
                Ok((refined_bounds, refined_state))
            }
        };

        Computable::new(state, neg_bounds, refine_to)
    }

    #[allow(clippy::type_complexity)]
    pub fn inv(
        self,
    ) -> Computable<
        State<X, BigInt>,
        impl Fn(&State<X, BigInt>) -> Result<Bounds, ComputableError>,
        impl Fn(
            State<X, BigInt>,
            &BoundsFn<'_, State<X, BigInt>>,
            Binary,
            usize,
        ) -> Result<(Bounds, State<X, BigInt>), ComputableError>,
    >
    where
        X: Eq + Clone,
    {
        let Computable {
            state,
            bounds: base_bounds,
            refine_to: base_refine_to,
        } = self;

        let base_bounds = Rc::new(base_bounds);
        let bounds = {
            let base_bounds = Rc::clone(&base_bounds);
            move |composed_state: &State<X, BigInt>| -> Result<Bounds, ComputableError> {
                let existing = base_bounds.as_ref()(&composed_state.inner_state)?;
                reciprocal_bounds(&existing, &composed_state.extra_state)
            }
        };

        let refine_to = {
            let base_bounds = Rc::clone(&base_bounds);
            move |composed_state: State<X, BigInt>,
                  bounds: &BoundsFn<'_, State<X, BigInt>>,
                  epsilon: Binary,
                  max_iterations: usize|
                  -> Result<(Bounds, State<X, BigInt>), ComputableError> {
                let target = epsilon.clone();
                refine_to_with_policy(
                    composed_state,
                    bounds,
                    target,
                    epsilon,
                    max_iterations,
                    |state, _bounds, epsilon, max_iterations| {
                        let State {
                            inner_state,
                            extra_state,
                        } = state;
                        let (_, refined_inner) = base_refine_to(
                            inner_state,
                            base_bounds.as_ref(),
                            epsilon.clone(),
                            max_iterations,
                        )?;
                        let next_precision = extra_state + BigInt::one();
                        Ok(State {
                            inner_state: refined_inner,
                            extra_state: next_precision,
                        })
                    },
                )
            }
        };

        Computable::new(
            State {
                inner_state: state,
                extra_state: BigInt::zero(),
            },
            bounds,
            refine_to,
        )
    }

    #[allow(clippy::should_implement_trait)]
    #[allow(clippy::type_complexity)]
    pub fn add<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<
        (X, Y),
        impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>,
        impl Fn(
            (X, Y),
            &BoundsFn<'_, (X, Y)>,
            Binary,
            usize,
        ) -> Result<(Bounds, (X, Y)), ComputableError>,
    >
    where
        X: Eq + Clone,
        Y: Eq + Clone,
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: for<'a> Fn(
            Y,
            &'a BoundsFn<'a, Y>,
            Binary,
            usize,
        ) -> Result<(Bounds, Y), ComputableError>,
    {
        // note: we don't implement std::ops::Add here because the composed type uses `impl Fn`
        // in its return signature, which cannot be named for an associated Output type.
        let Computable {
            state: left_state,
            bounds: left_bounds,
            refine_to: left_refine_to,
        } = self;
        let Computable {
            state: right_state,
            bounds: right_bounds,
            refine_to: right_refine_to,
        } = other;

        let left_bounds = Rc::new(left_bounds);
        let right_bounds = Rc::new(right_bounds);
        let left_refine_to = Rc::new(left_refine_to);
        let right_refine_to = Rc::new(right_refine_to);

        let bounds = {
            let left_bounds = Rc::clone(&left_bounds);
            let right_bounds = Rc::clone(&right_bounds);
            move |state: &(X, Y)| -> Result<Bounds, ComputableError> {
                let left = left_bounds.as_ref()(&state.0)?;
                let right = right_bounds.as_ref()(&state.1)?;
            let lower = bounds_lower(&left).add_lower(bounds_lower(&right))?;
            let upper = bounds_upper(&left).add_upper(bounds_upper(&right))?;
            OrderedPair::new_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
            }
        };

        let refine_to = {
            let left_bounds = Rc::clone(&left_bounds);
            let right_bounds = Rc::clone(&right_bounds);
            move |state: (X, Y),
                  bounds: &BoundsFn<'_, (X, Y)>,
                  epsilon: Binary,
                  max_iterations: usize|
                  -> Result<(Bounds, (X, Y)), ComputableError> {
                let target = halve_binary(&epsilon)?;
                refine_to_with_policy(
                    state,
                    bounds,
                    target,
                    epsilon,
                    max_iterations,
                    |(left, right), _bounds, epsilon, max_iterations| {
                        let (_, left_refined) = left_refine_to(
                            left,
                            left_bounds.as_ref(),
                            epsilon.clone(),
                            max_iterations,
                        )?;
                        let (_, right_refined) = right_refine_to(
                            right,
                            right_bounds.as_ref(),
                            epsilon.clone(),
                            max_iterations,
                        )?;
                        Ok((left_refined, right_refined))
                    },
                )
            }
        };

        Computable::new((left_state, right_state), bounds, refine_to)
    }

    #[allow(clippy::should_implement_trait)]
    #[allow(clippy::type_complexity)]
    pub fn sub<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<
        (X, Y),
        impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>,
        impl Fn(
            (X, Y),
            &BoundsFn<'_, (X, Y)>,
            Binary,
            usize,
        ) -> Result<(Bounds, (X, Y)), ComputableError>,
    >
    where
        X: Eq + Clone,
        Y: Eq + Clone,
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: for<'a> Fn(
            Y,
            &'a BoundsFn<'a, Y>,
            Binary,
            usize,
        ) -> Result<(Bounds, Y), ComputableError>,
    {
        self.add(other.neg())
    }

    #[allow(clippy::should_implement_trait)]
    #[allow(clippy::type_complexity)]
    pub fn mul<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<
        (X, Y),
        impl Fn(&(X, Y)) -> Result<Bounds, ComputableError>,
        impl Fn(
            (X, Y),
            &BoundsFn<'_, (X, Y)>,
            Binary,
            usize,
        ) -> Result<(Bounds, (X, Y)), ComputableError>,
    >
    where
        X: Eq + Clone,
        Y: Eq + Clone,
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: for<'a> Fn(
            Y,
            &'a BoundsFn<'a, Y>,
            Binary,
            usize,
        ) -> Result<(Bounds, Y), ComputableError>,
    {
        let Computable {
            state: left_state,
            bounds: left_bounds,
            refine_to: left_refine_to,
        } = self;
        let Computable {
            state: right_state,
            bounds: right_bounds,
            refine_to: right_refine_to,
        } = other;

        let left_bounds = Rc::new(left_bounds);
        let right_bounds = Rc::new(right_bounds);
        let left_refine_to = Rc::new(left_refine_to);
        let right_refine_to = Rc::new(right_refine_to);

        let bounds = {
            let left_bounds = Rc::clone(&left_bounds);
            let right_bounds = Rc::clone(&right_bounds);
            move |state: &(X, Y)| -> Result<Bounds, ComputableError> {
                let left = left_bounds.as_ref()(&state.0)?;
                let right = right_bounds.as_ref()(&state.1)?;
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

                OrderedPair::new_checked(min, max)
                    .map_err(|_| ComputableError::InvalidBoundsOrder)
            }
        };

        let refine_to = {
            let left_bounds = Rc::clone(&left_bounds);
            let right_bounds = Rc::clone(&right_bounds);
            move |state: (X, Y),
                  bounds: &BoundsFn<'_, (X, Y)>,
                  epsilon: Binary,
                  max_iterations: usize|
                  -> Result<(Bounds, (X, Y)), ComputableError> {
                let target = halve_binary(&epsilon)?;
                refine_to_with_policy(
                    state,
                    bounds,
                    target,
                    epsilon,
                    max_iterations,
                    |(left, right), _bounds, epsilon, max_iterations| {
                        let (_, left_refined) = left_refine_to(
                            left,
                            left_bounds.as_ref(),
                            epsilon.clone(),
                            max_iterations,
                        )?;
                        let (_, right_refined) = right_refine_to(
                            right,
                            right_bounds.as_ref(),
                            epsilon.clone(),
                            max_iterations,
                        )?;
                        Ok((left_refined, right_refined))
                    },
                )
            }
        };

        Computable::new((left_state, right_state), bounds, refine_to)
    }

    #[allow(clippy::should_implement_trait)]
    #[allow(clippy::type_complexity)]
    pub fn div<Y, B2, F2>(
        self,
        other: Computable<Y, B2, F2>,
    ) -> Computable<
        (X, State<Y, BigInt>),
        impl Fn(&(X, State<Y, BigInt>)) -> Result<Bounds, ComputableError>,
        impl Fn(
            (X, State<Y, BigInt>),
            &BoundsFn<'_, (X, State<Y, BigInt>)>,
            Binary,
            usize,
        ) -> Result<(Bounds, (X, State<Y, BigInt>)), ComputableError>,
    >
    where
        X: Eq + Clone,
        Y: Eq + Clone,
        B2: Fn(&Y) -> Result<Bounds, ComputableError>,
        F2: for<'a> Fn(
            Y,
            &'a BoundsFn<'a, Y>,
            Binary,
            usize,
        ) -> Result<(Bounds, Y), ComputableError>,
    {
        self.mul(other.inv())
    }

}

impl Computable<(), (), ()> {
    #[allow(clippy::type_complexity)]
    pub fn constant(
        value: Binary,
    ) -> Computable<
        Binary,
        fn(&Binary) -> Result<Bounds, ComputableError>,
        impl for<'a> Fn(
            Binary,
            &'a BoundsFn<'a, Binary>,
            Binary,
            usize,
        ) -> Result<(Bounds, Binary), ComputableError>,
    > {
        fn bounds(value: &Binary) -> Result<Bounds, ComputableError> {
            Ok(OrderedPair::new(
                ExtendedBinary::Finite(value.clone()),
                ExtendedBinary::Finite(value.clone()),
            ))
        }

        fn refine(value: Binary) -> Binary {
            value
        }

        computable_with_refine_step(value, bounds, refine)
    }
}

#[cfg(debug_assertions)]
pub const DEFAULT_INV_MAX_REFINES: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_INV_MAX_REFINES: usize = 4096;

#[cfg(debug_assertions)]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 4096;

#[derive(Clone, Copy, Debug)]
enum ReciprocalRounding {
    Floor,
    Ceil,
}

fn reciprocal_bounds(bounds: &Bounds, precision_bits: &BigInt) -> Result<Bounds, ComputableError> {
    let lower = bounds_lower(bounds);
    let upper = bounds_upper(bounds);
    let zero = ExtendedBinary::zero();
    if lower <= &zero && upper >= &zero {
        return Ok(OrderedPair::new(
            ExtendedBinary::NegInf,
            ExtendedBinary::PosInf,
        ));
    }

    let (lower_bound, upper_bound) = if upper < &zero {
        let lower_bound =
            reciprocal_rounded_abs_extended(upper, precision_bits, ReciprocalRounding::Ceil)?
                .neg();
        let upper_bound =
            reciprocal_rounded_abs_extended(lower, precision_bits, ReciprocalRounding::Floor)?
                .neg();
        (lower_bound, upper_bound)
    } else {
        let lower_bound =
            reciprocal_rounded_abs_extended(upper, precision_bits, ReciprocalRounding::Floor)?;
        let upper_bound =
            reciprocal_rounded_abs_extended(lower, precision_bits, ReciprocalRounding::Ceil)?;
        (lower_bound, upper_bound)
    };

    OrderedPair::new_checked(lower_bound, upper_bound)
        .map_err(|_| ComputableError::InvalidBoundsOrder)
}

fn reciprocal_rounded_abs_extended(
    value: &ExtendedBinary,
    precision_bits: &BigInt,
    rounding: ReciprocalRounding,
) -> Result<ExtendedBinary, BinaryError> {
    match value {
        ExtendedBinary::Finite(finite_value) => {
            let abs_mantissa = finite_value.mantissa().abs();
            let shift_bits = precision_bits - BigInt::from(finite_value.exponent());
            let quotient = if shift_bits.is_negative() {
                match rounding {
                    ReciprocalRounding::Floor => BigInt::zero(),
                    ReciprocalRounding::Ceil => BigInt::one(),
                }
            } else {
                let shift = precision_bits_to_usize(&shift_bits)?;
                let numerator = BigInt::one() << shift;
                match rounding {
                    ReciprocalRounding::Floor => numerator.div_floor(&abs_mantissa),
                    ReciprocalRounding::Ceil => numerator.div_ceil(&abs_mantissa),
                }
            };
            let exponent = reciprocal_exponent(precision_bits)?;
            Ok(ExtendedBinary::Finite(Binary::new(quotient, exponent)?))
        }
        ExtendedBinary::NegInf | ExtendedBinary::PosInf => {
            Ok(ExtendedBinary::Finite(Binary::zero()))
        }
    }
}

fn reciprocal_exponent(precision_bits: &BigInt) -> Result<Exponent, BinaryError> {
    let precision = precision_bits_to_exponent(precision_bits)?;
    precision
        .checked_neg()
        .ok_or(BinaryError::ReciprocalOverflow)
}

fn precision_bits_to_usize(precision_bits: &BigInt) -> Result<usize, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    usize::try_from(precision_bits).map_err(|_| BinaryError::ReciprocalOverflow)
}

fn precision_bits_to_exponent(precision_bits: &BigInt) -> Result<Exponent, BinaryError> {
    if precision_bits.is_negative() {
        return Err(BinaryError::ReciprocalOverflow);
    }
    Exponent::try_from(precision_bits).map_err(|_| BinaryError::ReciprocalOverflow)
}

fn bounds_width_leq(bounds: &Bounds, epsilon: &Binary) -> Result<bool, BinaryError> {
    let upper = bounds_upper(bounds);
    let lower = bounds_lower(bounds);
    let (ExtendedBinary::Finite(upper_bound), ExtendedBinary::Finite(lower_bound)) =
        (upper, lower)
    else {
        return Ok(false);
    };
    let width = upper_bound.sub(lower_bound)?;
    Ok(&width <= epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    type IntervalState = Bounds;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), exponent).expect("binary should normalize")
    }

    fn ext(mantissa: i64, exponent: i64) -> ExtendedBinary {
        ExtendedBinary::Finite(bin(mantissa, exponent))
    }

    fn unwrap_finite(value: &ExtendedBinary) -> Binary {
        match value {
            ExtendedBinary::Finite(value) => value.clone(),
            ExtendedBinary::NegInf | ExtendedBinary::PosInf => {
                panic!("expected finite extended binary")
            }
        }
    }

    fn const_computable(
        value: Binary,
    ) -> Computable<
        Binary,
        fn(&Binary) -> Result<Bounds, ComputableError>,
        impl for<'a> Fn(
            Binary,
            &'a BoundsFn<'a, Binary>,
            Binary,
            usize,
        ) -> Result<(Bounds, Binary), ComputableError>,
    > {
        Computable::constant(value)
    }

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn interval_refine(state: IntervalState) -> IntervalState {
        let mid = unwrap_finite(state.small())
            .add(&unwrap_finite(state.large()))
            .expect("binary should add");
        let exponent = mid
            .exponent()
            .checked_sub(1)
            .expect("midpoint exponent should not underflow");
        let mid = Binary::new(mid.mantissa().clone(), exponent)
            .expect("binary should normalize");
        OrderedPair::new(ExtendedBinary::Finite(mid.clone()), ExtendedBinary::Finite(mid))
    }

    fn interval_refine_strict(state: IntervalState) -> IntervalState {
        let mid = unwrap_finite(state.small())
            .add(&unwrap_finite(state.large()))
            .expect("binary should add");
        let exponent = mid
            .exponent()
            .checked_sub(1)
            .expect("midpoint exponent should not underflow");
        let mid = Binary::new(mid.mantissa().clone(), exponent)
            .expect("binary should normalize");
        OrderedPair::new(state.small().clone(), ExtendedBinary::Finite(mid))
    }

    fn interval_midpoint_computable(
        lower: i64,
        upper: i64,
    ) -> Computable<
        IntervalState,
        impl Fn(&IntervalState) -> Result<Bounds, ComputableError>,
        impl for<'a> Fn(
            IntervalState,
            &'a BoundsFn<'a, IntervalState>,
            Binary,
            usize,
        ) -> Result<(Bounds, IntervalState), ComputableError>,
    > {
        let state = OrderedPair::new(ext(lower, 0), ext(upper, 0));
        computable_with_refine_step(state, |state| Ok(interval_bounds(state)), interval_refine)
    }

    fn sqrt2_computable() -> Computable<
        IntervalState,
        impl Fn(&IntervalState) -> Result<Bounds, ComputableError>,
        impl for<'a> Fn(
            IntervalState,
            &'a BoundsFn<'a, IntervalState>,
            Binary,
            usize,
        ) -> Result<(Bounds, IntervalState), ComputableError>,
    > {
        let state = OrderedPair::new(ext(1, 0), ext(2, 0));
        let bounds = |state: &IntervalState| Ok(state.clone());
        let refine = |state: IntervalState| {
            let bounds_sum = unwrap_finite(state.small())
                .add(&unwrap_finite(state.large()))
                .expect("binary should add");
            let exponent = bounds_sum
                .exponent()
                .checked_sub(1)
                .expect("midpoint exponent should not underflow");
            let mid = Binary::new(bounds_sum.mantissa().clone(), exponent)
                .expect("binary should normalize");
            let mid_sq = mid.mul(&mid).expect("binary should multiply");
            let two = bin(1, 1);

            if mid_sq <= two {
                OrderedPair::new(ExtendedBinary::Finite(mid), state.large().clone())
            } else {
                OrderedPair::new(state.small().clone(), ExtendedBinary::Finite(mid))
            }
        };

        computable_with_refine_step(state, bounds, refine)
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
        let result = OrderedPair::new_checked(lower, upper);
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
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(-1, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(1, -1);
        let (bounds, refined) = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected = ext(1, 0);
        let width = unwrap_finite(bounds_upper(&bounds))
            .sub(&unwrap_finite(bounds_lower(&bounds)))
            .expect("binary should subtract");

        assert!(bounds_lower(&bounds) <= &expected && &expected <= bounds_upper(&bounds));
        assert!(width < epsilon);
        let refined_bounds = refined.bounds().expect("bounds should succeed");
        assert!(bounds_lower(&refined_bounds) <= &expected && &expected <= bounds_upper(&refined_bounds));
    }

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(0, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::NonpositiveEpsilon)
        ));
    }

    #[test]
    fn computable_refine_to_rejects_unchanged_state() {
        let state = OrderedPair::new(ext(0, 0), ext(2, 0));
        let computable =
            computable_with_refine_step(state, |state| Ok(interval_bounds(state)), |state| {
                state
            });
        let epsilon = bin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::StateUnchanged)
        ));
    }

    #[test]
    fn computable_refine_to_enforces_max_iterations() {
        let computable = computable_with_refine_step(
            0usize,
            |_| Ok(OrderedPair::new(ExtendedBinary::NegInf, ExtendedBinary::PosInf)),
            |state| state + 1,
        );
        let epsilon = bin(1, -1);
        let result = computable.refine_to::<5>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 5 })
        ));
    }

    #[test]
    fn computable_refine_to_handles_non_meeting_bounds() {
        let state = OrderedPair::new(ext(0, 0), ext(4, 0));
        let computable = computable_with_refine_step(
            state,
            |state| Ok(interval_bounds(state)),
            interval_refine_strict,
        );
        let epsilon = bin(1, -1);
        let (bounds, refined) = computable
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");
        assert!(bounds_lower(&bounds) < bounds_upper(&bounds));
        assert_eq!(refined.bounds().expect("bounds should succeed"), bounds);
    }

    #[test]
    fn computable_refine_to_rejects_worsened_bounds() {
        let state = OrderedPair::new(ext(0, 0), ext(1, 0));
        let computable = computable_with_refine_step(
            state,
            |state| Ok(interval_bounds(state)),
            |state: IntervalState| {
                let worse_upper = unwrap_finite(state.large())
                    .add(&bin(1, 0))
                    .expect("binary should add");
                OrderedPair::new(state.small().clone(), ExtendedBinary::Finite(worse_upper))
            },
        );
        let epsilon = bin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    #[test]
    fn computable_add_combines_bounds() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left.add(right);
        let sum_bounds = sum.bounds().expect("bounds should succeed");
        assert_eq!(
            sum_bounds,
            OrderedPair::new(
                ext(1, 0),
                ext(5, 0)
            )
        );
    }

    #[test]
    fn computable_sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left.sub(right);
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(
            diff_bounds,
            OrderedPair::new(
                ext(2, 0),
                ext(5, 0)
            )
        );
    }

    #[test]
    fn computable_neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = value.neg();
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ext(-3, 0),
                ext(-1, 0)
            )
        );
    }

    #[test]
    fn computable_inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(ExtendedBinary::NegInf, ExtendedBinary::PosInf)
        );
    }

    #[test]
    fn computable_inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = bin(1, -8);
        let (bounds, _) = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let lower = unwrap_finite(bounds_lower(&bounds));
        let upper = unwrap_finite(bounds_upper(&bounds));
        let width = upper.sub(&lower).expect("binary should subtract");
        let expected = ExtendedBinary::from_f64(1.0 / 3.0)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected);

        assert!(lower <= expected && expected <= upper);
        assert!(width <= epsilon);
    }

    #[test]
    fn computable_mul_combines_bounds_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ext(2, 0),
                ext(12, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ext(-12, 0),
                ext(-2, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ext(-10, 0),
                ext(15, 0)
            )
        );
    }

    #[test]
    fn computable_mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left.mul(right);
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ext(-8, 0),
                ext(12, 0)
            )
        );
    }

    #[test]
    fn computable_integration_sqrt2_expression() {
        let one = const_computable(bin(1, 0));
        let expr = sqrt2_computable()
            .add(one)
            .mul(sqrt2_computable().sub(const_computable(bin(1, 0))))
            .add(sqrt2_computable().inv());

        let epsilon = bin(1, -12);
        let (bounds, _) = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds_lower(&bounds));
        let upper = unwrap_finite(bounds_upper(&bounds));
        let expected = 1.0_f64 + 2.0_f64.sqrt().recip();
        let expected_binary = ExtendedBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_binary = unwrap_finite(&expected_binary);
        let eps_binary = epsilon;

        let lower_plus = lower.add(&eps_binary).expect("binary should add");
        let upper_minus = upper.sub(&eps_binary).expect("binary should subtract");

        assert!(lower <= expected_binary && expected_binary <= upper);
        assert!(upper_minus <= expected_binary && expected_binary <= lower_plus);
    }
}
