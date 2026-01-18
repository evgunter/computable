//! N-th root operation with binary search refinement.
//!
//! This module implements the n-th root operation (x^(1/n)) using:
//! - Binary search (bisection) for guaranteed convergence
//! - Interval arithmetic for provably correct bounds
//!
//! The algorithm maintains an interval [lower, upper] where the true root lies,
//! and refines by bisection: if mid^n <= target, the root is in [mid, upper],
//! otherwise it's in [lower, mid].
//!
//! TODO: Extract a generic binary search helper function that can be reused for
//! other operations that use bisection (e.g., finding roots of monotonic functions).
//! The helper could take: initial bounds, a comparison function, and produce a
//! Computable that refines via bisection.
//!
//! TODO: Add a node type for integer powers of a computable (e.g., x^2, x^3).
//! This would be the "inverse" of nth_root and could be implemented more efficiently
//! than repeated multiplication, especially for computing bounds.
//!
//! TODO: Contra the README, even-degree roots of inputs that overlap with negative
//! numbers (but aren't completely negative) currently just return (0, ∞) bounds
//! instead of returning a recoverable error that would trigger refinement of the
//! input until the bounds are fully non-negative. This should be fixed to match
//! the behavior described in the README for sqrt.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::{Binary, XBinary};
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::binary::Bounds;

/// N-th root operation with binary search refinement.
///
/// Computes x^(1/n) where n is the root degree.
/// For n=2, this is square root; n=3 is cube root, etc.
///
/// # Constraints
/// - For even n: requires x >= 0 (otherwise returns infinite bounds)
/// - For odd n: supports all real x (negative values have negative roots)
pub struct NthRootOp {
    /// The input node whose n-th root we're computing.
    pub inner: Arc<Node>,
    /// The root degree (n in x^(1/n)).
    pub degree: u32,
    /// Current bisection state: tracks the interval for the root.
    /// 
    /// This is `None` until the first refinement step, which initializes it from
    /// the input bounds. We use `Option` because initialization requires calling
    /// `inner.get_bounds()` which can fail, but node construction (via `nth_root()`)
    /// is not supposed to be fallible. By deferring initialization to the first
    /// `refine_step()` call, we can propagate errors through the normal Result path.
    /// 
    /// Each refinement step halves this interval via bisection.
    pub bisection_state: RwLock<Option<BisectionState>>,
}

/// State for the bisection algorithm.
/// Tracks the current interval bounds for the n-th root.
#[derive(Clone, Debug)]
pub struct BisectionState {
    /// Lower bound for the n-th root (finite Binary).
    pub lower: Binary,
    /// Upper bound for the n-th root (finite Binary).
    pub upper: Binary,
    /// The target value (x) whose n-th root we're computing.
    pub target: Binary,
    /// Whether the result should be negated (for odd roots of negative numbers).
    pub negate_result: bool,
}

impl NodeOp for NthRootOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let state = self.bisection_state.read();
        
        match &*state {
            None => {
                // Return initial conservative bounds based on input
                compute_initial_bounds(&input_bounds, self.degree)
            }
            Some(s) => {
                // Return current bisection interval
                let (lower, upper) = if s.negate_result {
                    (s.upper.neg(), s.lower.neg())
                } else {
                    (s.lower.clone(), s.upper.clone())
                };
                Ok(Bounds::new(XBinary::Finite(lower), XBinary::Finite(upper)))
            }
        }
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let mut state = self.bisection_state.write();
        
        match &mut *state {
            None => {
                // Initialize the bisection state from input bounds
                *state = Some(initialize_bisection_state(&input_bounds, self.degree)?);
                Ok(true)
            }
            Some(s) => {
                // Perform one bisection step
                let mid = midpoint(&s.lower, &s.upper);
                let mid_pow = power(&mid, self.degree);
                
                if mid_pow <= s.target {
                    s.lower = mid;
                } else {
                    s.upper = mid;
                }
                Ok(true)
            }
        }
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Computes initial conservative bounds for the n-th root.
/// 
/// Computes lower and upper output bounds separately, which allows handling
/// mixed finite/infinite input bounds (e.g., getting a finite lower output
/// bound even when upper input is PosInf).
fn compute_initial_bounds(input_bounds: &Bounds, degree: u32) -> Result<Bounds, ComputableError> {
    let lower_input = input_bounds.small();
    let upper_input = &input_bounds.large();
    let is_even = degree.is_multiple_of(2);
    
    let lower_output = compute_output_lower_bound(lower_input, is_even, degree)?;
    let upper_output = compute_output_upper_bound(upper_input, is_even, degree)?;
    
    Ok(Bounds::new(lower_output, upper_output))
}

/// Computes a lower bound for the n-th root output from the lower input bound.
/// 
/// For odd roots of negative values: cbrt(x) = -cbrt(|x|), so to get a LOWER
/// bound on cbrt(x), we need an UPPER bound on cbrt(|x|), then negate.
fn compute_output_lower_bound(lower_input: &XBinary, is_even: bool, degree: u32) -> Result<XBinary, ComputableError> {
    match lower_input {
        XBinary::NegInf => {
            // Lower input is -∞
            Ok(if is_even {
                // Even root: output is non-negative, so lower bound is 0
                XBinary::Finite(Binary::zero())
            } else {
                // Odd root: cbrt(-∞) = -∞
                XBinary::NegInf
            })
        }
        XBinary::PosInf => {
            // Lower input is +∞ - this is mathematically impossible for a lower bound
            debug_assert!(false, "lower input bound cannot be PosInf");
            Ok(XBinary::PosInf)
        }
        XBinary::Finite(lower_bin) => {
            if lower_bin.mantissa().is_negative() {
                if is_even {
                    // Even root of negative: output lower bound is 0
                    // (the actual root computation may error if entirely negative,
                    // but that's checked when we also consider the upper bound)
                    Ok(XBinary::Finite(Binary::zero()))
                } else {
                    // Odd root of negative: cbrt(lower) = -cbrt(|lower|)
                    // Lower bound on output = -upper_bound(cbrt(|lower|))
                    let neg_lower = lower_bin.neg();
                    let lower_root = compute_root_upper_bound(&neg_lower, degree).neg();
                    Ok(XBinary::Finite(lower_root))
                }
            } else {
                // Non-negative input
                let lower_root = compute_root_lower_bound(lower_bin, degree);
                Ok(XBinary::Finite(lower_root))
            }
        }
    }
}

/// Computes an upper bound for the n-th root output from the upper input bound.
/// 
/// For odd roots of negative values: cbrt(x) = -cbrt(|x|), so to get an UPPER
/// bound on cbrt(x), we need a LOWER bound on cbrt(|x|), then negate.
fn compute_output_upper_bound(upper_input: &XBinary, is_even: bool, degree: u32) -> Result<XBinary, ComputableError> {
    match upper_input {
        XBinary::PosInf => Ok(XBinary::PosInf),
        XBinary::NegInf => {
            // Upper input is -∞ - this is mathematically impossible for an upper bound
            debug_assert!(false, "upper input bound cannot be NegInf");
            Ok(XBinary::NegInf)
        }
        XBinary::Finite(upper_bin) => {
            if upper_bin.mantissa().is_negative() {
                if is_even {
                    // Even root of entirely negative interval: no real root
                    return Err(ComputableError::DomainError);
                }
                // Odd root of negative: cbrt(upper) = -cbrt(|upper|)
                // Upper bound on output = -lower_bound(cbrt(|upper|))
                let neg_upper = upper_bin.neg();
                let upper_root = compute_root_lower_bound(&neg_upper, degree).neg();
                Ok(XBinary::Finite(upper_root))
            } else {
                // Non-negative input
                let upper_root = compute_root_upper_bound(upper_bin, degree);
                Ok(XBinary::Finite(upper_root))
            }
        }
    }
}

/// Computes an upper bound for the n-th root of a positive value.
/// Returns a value >= x^(1/n).
fn compute_root_upper_bound(x: &Binary, _degree: u32) -> Binary {
    // Conservative upper bound: max(1, |x|)
    // This is always >= x^(1/n) for x >= 0 and n >= 1
    let one = Binary::new(BigInt::one(), BigInt::zero());
    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };
    
    if abs_x > one {
        abs_x
    } else {
        one
    }
}

/// Computes a lower bound for the n-th root of a positive value.
/// Returns a value <= x^(1/n).
fn compute_root_lower_bound(x: &Binary, _degree: u32) -> Binary {
    // Conservative lower bound: min(1, |x|) for x > 0, 0 otherwise
    if x.mantissa().is_zero() || x.mantissa().is_negative() {
        return Binary::zero();
    }
    
    let one = Binary::new(BigInt::one(), BigInt::zero());
    if x < &one {
        x.clone()
    } else {
        one
    }
}

/// Initializes the bisection state from the input bounds.
fn initialize_bisection_state(input_bounds: &Bounds, degree: u32) -> Result<BisectionState, ComputableError> {
    let lower = input_bounds.small();
    let upper = &input_bounds.large();
    
    // Get the target value - use midpoint for intervals, exact for points
    let target = match (lower, upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => midpoint(l, u),
        _ => return Err(ComputableError::InfiniteBounds),
    };
    
    let is_even = degree.is_multiple_of(2);
    
    // Handle negative targets for even roots
    if is_even && target.mantissa().is_negative() {
        return Err(ComputableError::DomainError);
    }
    
    // For odd roots of negative values, compute root of |target| and negate
    let (actual_target, negate_result) = if !is_even && target.mantissa().is_negative() {
        (target.neg(), true)
    } else {
        (target.clone(), false)
    };
    
    // Initial bounds for bisection: [0 or small, max(1, target)]
    let one = Binary::new(BigInt::one(), BigInt::zero());
    
    let bisection_lower = if actual_target.mantissa().is_zero() {
        Binary::zero()
    } else if actual_target < one {
        // For 0 < target < 1, the root is > target, so use target as lower bound
        actual_target.clone()
    } else {
        // For target >= 1, the root is <= target, so use 1 as lower bound
        one.clone()
    };
    
    let bisection_upper = if actual_target.mantissa().is_zero() {
        Binary::zero()
    } else if actual_target < one {
        // For 0 < target < 1, the root is < 1, so use 1 as upper bound
        one
    } else {
        // For target >= 1, the root is <= target, so use target as upper bound
        actual_target.clone()
    };
    
    Ok(BisectionState {
        lower: bisection_lower,
        upper: bisection_upper,
        target: actual_target,
        negate_result,
    })
}

/// Computes the midpoint of two Binary numbers.
fn midpoint(lower: &Binary, upper: &Binary) -> Binary {
    let sum = lower.add(upper);
    // Divide by 2 by subtracting 1 from the exponent
    Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one())
}

/// Computes x^n for a Binary number.
fn power(x: &Binary, n: u32) -> Binary {
    if n == 0 {
        return Binary::new(BigInt::one(), BigInt::zero());
    }
    
    let mut result = x.clone();
    for _ in 1..n {
        result = result.mul(x);
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::UBinary;
    use crate::computable::Computable;
    use crate::test_utils::{bin, ubin, unwrap_finite, unwrap_finite_uxbinary};

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(
            lower <= *expected && *expected <= upper,
            "Expected {} to be in bounds [{}, {}]",
            expected, lower, upper
        );
        assert!(
            width <= *epsilon,
            "Width {} should be <= epsilon {}",
            width, epsilon
        );
    }

    #[test]
    fn sqrt_of_4() {
        // sqrt(4) = 2
        let four = Computable::constant(bin(4, 0));
        let sqrt_four = four.nth_root(2);
        let epsilon = ubin(1, -8);
        let bounds = sqrt_four
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_2() {
        // sqrt(2) ~= 1.414...
        let two = Computable::constant(bin(2, 0));
        let sqrt_two = two.nth_root(2);
        let epsilon = ubin(1, -8);
        let bounds = sqrt_two
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_8() {
        // cbrt(8) = 2
        let eight = Computable::constant(bin(8, 0));
        let cbrt_eight = eight.nth_root(3);
        let epsilon = ubin(1, -8);
        let bounds = cbrt_eight
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn cbrt_of_negative_8() {
        // cbrt(-8) = -2
        let neg_eight = Computable::constant(bin(-8, 0));
        let cbrt_neg_eight = neg_eight.nth_root(3);
        let epsilon = ubin(1, -8);
        let bounds = cbrt_neg_eight
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(-2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn fourth_root_of_16() {
        // 16^(1/4) = 2
        let sixteen = Computable::constant(bin(16, 0));
        let fourth_root = sixteen.nth_root(4);
        let epsilon = ubin(1, -8);
        let bounds = fourth_root
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected = bin(2, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_half() {
        // sqrt(0.5) ~= 0.707...
        let half = Computable::constant(bin(1, -1));
        let sqrt_half = half.nth_root(2);
        let epsilon = ubin(1, -8);
        let bounds = sqrt_half
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 0.5_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn nth_root_in_expression() {
        // Test that nth_root works in expressions: sqrt(2) + cbrt(8) = sqrt(2) + 2
        let sqrt_2 = Computable::constant(bin(2, 0)).nth_root(2);
        let cbrt_8 = Computable::constant(bin(8, 0)).nth_root(3);
        let sum = sqrt_2 + cbrt_8;

        let epsilon = ubin(1, -8);
        let bounds = sum
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let expected_f64 = 2.0_f64.sqrt() + 2.0;
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn sqrt_of_zero() {
        // sqrt(0) = 0
        let zero = Computable::constant(bin(0, 0));
        let sqrt_zero = zero.nth_root(2);
        let bounds = sqrt_zero.bounds().expect("bounds should succeed");

        let expected = bin(0, 0);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected && expected <= upper);
    }

    fn interval_computable(lower: i64, upper: i64) -> Computable {
        let interval_state = Bounds::new(
            XBinary::Finite(bin(lower, 0)),
            XBinary::Finite(bin(upper, 0)),
        );
        Computable::new(
            interval_state,
            |state| Ok(state.clone()),
            |state| state, // No refinement for this test
        )
    }

    #[test]
    fn sqrt_of_interval_overlapping_zero() {
        // Test even root of a Computable with bounds overlapping zero: [-1, 4]
        // The sqrt should have bounds [0, upper] (since sqrt is only defined for non-negative)
        let interval = interval_computable(-1, 4);
        let sqrt_interval = interval.nth_root(2);
        let bounds = sqrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Lower output bound should be 0 (since input overlaps negative)
        assert_eq!(lower, bin(0, 0));
        // Upper output bound should be >= sqrt(4) = 2
        assert!(upper >= bin(2, 0));
    }

    #[test]
    fn cbrt_of_interval_overlapping_zero() {
        // Test odd root of a Computable with bounds overlapping zero: [-8, 27]
        // cbrt(-8) = -2, cbrt(27) = 3, so output should be approximately [-2, 3]
        let interval = interval_computable(-8, 27);
        let cbrt_interval = interval.nth_root(3);
        let bounds = cbrt_interval.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // Lower output bound should be <= cbrt(-8) = -2
        assert!(lower <= bin(-2, 0));
        // Upper output bound should be >= cbrt(27) = 3
        assert!(upper >= bin(3, 0));
    }

    #[test]
    fn power_function() {
        let x = bin(3, 0); // 3
        assert_eq!(power(&x, 2), bin(9, 0)); // 3^2 = 9
        assert_eq!(power(&x, 3), bin(27, 0)); // 3^3 = 27
        assert_eq!(power(&x, 0), bin(1, 0)); // 3^0 = 1
    }

    #[test]
    fn midpoint_function() {
        let lower = bin(2, 0);
        let upper = bin(4, 0);
        let mid = midpoint(&lower, &upper);
        assert_eq!(mid, bin(3, 0));
    }
}
