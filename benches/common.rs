#![allow(dead_code)]

use computable::{Binary, Bounds, Computable, FiniteBounds, XBinary};

/// Sums terms using a balanced reduction instead of left-associative chaining.
///
/// This keeps the computation graph shallow (O(log n) depth), avoiding deep nesting
/// that can overflow the stack or distort timing by spending most of the runtime
/// walking long expression chains.
pub fn balanced_sum(mut values: Vec<Computable>) -> Computable {
    if values.is_empty() {
        return Computable::constant(Binary::zero());
    }

    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(left + right);
            } else {
                next.push(left);
            }
        }
        values = next;
    }

    values
        .pop()
        .expect("values should contain at least one element")
}

/// Converts an f64 value to a Binary, panicking if the value is not finite.
pub fn binary_from_f64(value: f64) -> Binary {
    match XBinary::from_f64(value) {
        Ok(XBinary::Finite(b)) => b,
        _ => panic!("expected finite f64 value"),
    }
}

/// Tries to extract finite bounds from Bounds, returning None if either endpoint is infinite.
pub fn try_finite_bounds(bounds: &Bounds) -> Option<FiniteBounds> {
    match (bounds.small(), bounds.large()) {
        (XBinary::Finite(lower), XBinary::Finite(upper)) => {
            Some(FiniteBounds::new(lower.clone(), upper))
        }
        _ => None,
    }
}
