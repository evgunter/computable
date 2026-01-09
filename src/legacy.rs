use crate::{
    bounds_excludes_value, bounds_lower, bounds_upper, Binary, Bounds, Computable,
    ComputableError, ExtendedBinary, DEFAULT_INV_MAX_REFINES,
};

pub trait LegacyRefinement<X, B, F> {
    fn refine_to_nonzero<const MAX: usize>(self) -> Result<(Bounds, Self), ComputableError>
    where
        Self: Sized;

    fn refine_to_nonzero_default(self) -> Result<(Bounds, Self), ComputableError>
    where
        Self: Sized;

    fn refine_to_exclude<const MAX: usize>(
        self,
        value: Binary,
    ) -> Result<(Bounds, Self), ComputableError>
    where
        Self: Sized;
}

impl<X, B, F> LegacyRefinement<X, B, F> for Computable<X, B, F>
where
    B: Fn(&X) -> Result<Bounds, ComputableError>,
    F: Fn(X) -> X,
{
    fn refine_to_nonzero<const MAX: usize>(mut self) -> Result<(Bounds, Self), ComputableError> {
        let zero = ExtendedBinary::zero();
        for _ in 0..MAX {
            let bounds = (self.bounds)(&self.state)?;
            if bounds_excludes_value(&bounds, &zero) {
                return Ok((bounds, self));
            }

            if bounds_lower(&bounds) == bounds_upper(&bounds) {
                return Err(ComputableError::ExcludedValueUnreachable);
            }

            self.state = (self.refine)(self.state);
            let refined = (self.bounds)(&self.state)?;
            let lower_improved = bounds_lower(&refined) > bounds_lower(&bounds);
            let upper_improved = bounds_upper(&refined) < bounds_upper(&bounds);
            if !(lower_improved || upper_improved) {
                return Err(ComputableError::NonImprovingBounds);
            }
        }

        Err(ComputableError::MaxRefinementIterations { max: MAX })
    }

    fn refine_to_nonzero_default(self) -> Result<(Bounds, Self), ComputableError> {
        self.refine_to_nonzero::<DEFAULT_INV_MAX_REFINES>()
    }

    fn refine_to_exclude<const MAX: usize>(
        mut self,
        value: Binary,
    ) -> Result<(Bounds, Self), ComputableError> {
        let value = ExtendedBinary::Finite(value);
        for _ in 0..MAX {
            let bounds = (self.bounds)(&self.state)?;
            if bounds_excludes_value(&bounds, &value) {
                return Ok((bounds, self));
            }

            if bounds_lower(&bounds) == bounds_upper(&bounds) {
                return Err(ComputableError::ExcludedValueUnreachable);
            }

            self.state = (self.refine)(self.state);
            let refined = (self.bounds)(&self.state)?;
            let lower_improved = bounds_lower(&refined) > bounds_lower(&bounds);
            let upper_improved = bounds_upper(&refined) < bounds_upper(&bounds);
            if !(lower_improved || upper_improved) {
                return Err(ComputableError::NonImprovingBounds);
            }
        }

        Err(ComputableError::MaxRefinementIterations { max: MAX })
    }
}

#[cfg(test)]
mod tests {
    use super::LegacyRefinement;
    use crate::{
        bounds_excludes_value, Binary, Bounds, Computable, ComputableError, ExtendedBinary,
        OrderedPair,
    };
    use num_bigint::BigInt;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), exponent).expect("binary should normalize")
    }

    fn interval_bounds(state: &Bounds) -> Bounds {
        state.clone()
    }

    fn interval_computable(
        lower: i64,
        upper: i64,
    ) -> Computable<
        Bounds,
        impl Fn(&Bounds) -> Result<Bounds, ComputableError>,
        impl Fn(Bounds) -> Bounds,
    > {
        let state = OrderedPair::new(
            ExtendedBinary::Finite(bin(lower, 0)),
            ExtendedBinary::Finite(bin(upper, 0)),
        );
        Computable::new(state, |state| Ok(interval_bounds(state)), |state| state)
    }

    #[test]
    fn computable_refine_to_nonzero_succeeds() {
        let computable = interval_computable(1, 3);
        let (bounds, _) = computable
            .refine_to_nonzero::<4>()
            .expect("refine_to_nonzero should succeed");
        assert!(bounds_excludes_value(&bounds, &ExtendedBinary::zero()));
    }

    #[test]
    fn computable_refine_to_nonzero_fails_when_zero_is_fixed() {
        let state = OrderedPair::new(ExtendedBinary::zero(), ExtendedBinary::zero());
        let computable = Computable::new(state, |state| Ok(interval_bounds(state)), |state| state);
        let result = computable.refine_to_nonzero::<2>();
        assert!(matches!(
            result,
            Err(ComputableError::ExcludedValueUnreachable)
        ));
    }

    #[test]
    fn computable_refine_to_exclude_hits_max_iterations() {
        let state = OrderedPair::new(
            ExtendedBinary::Finite(bin(1, 0)),
            ExtendedBinary::Finite(bin(3, 0)),
        );
        let computable = Computable::new(state, |state| Ok(interval_bounds(state)), |state| state);
        let result = computable.refine_to_exclude::<0>(bin(2, 0));
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 0 })
        ));
    }
}
