//! Pi computation using Machin's formula with provably correct bounds.
//!
//! This module implements pi as a Computable number using:
//! - Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239)
//! - Taylor series for arctan with rigorous error bounds
//! - Directed rounding for interval arithmetic
//!
//! ## Key Design Decisions
//!
//! 1. **Full Interval Propagation**: Every intermediate computation tracks [lo, hi] bounds
//! 2. **Directed Rounding**: Lower bounds round down, upper bounds round up
//! 3. **Error Bound Tracking**: Taylor truncation error is computed conservatively
//! 4. **Dynamic Precision**: Pi bounds can be refined to arbitrary precision

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::One;
use parking_lot::RwLock;

use crate::binary::{
    margin_from_width, reciprocal_of_biguint, simplify_bounds_if_needed, Binary, Bounds,
    FiniteBounds, ReciprocalRounding, UBinary, XBinary,
};
use crate::computable::Computable;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};

/// Initial number of Taylor series terms for pi computation.
const INITIAL_PI_TERMS: usize = 10;

/// Precision threshold for triggering bounds simplification.
/// 128 chosen: similar to sin, Taylor series benefits from less frequent simplification.
const PRECISION_SIMPLIFICATION_THRESHOLD: u64 = 128;

/// margin parameter for bounds simplification.
/// 3 = loosen by width/8. Benchmarks show margin has minimal performance impact.
const MARGIN_SHIFT: u32 = 3;

/// Returns pi as a Computable that can be refined to arbitrary precision.
///
/// Uses Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239)
///
/// # Example
///
/// ```
/// use computable::{pi, UBinary};
/// use num_bigint::{BigInt, BigUint};
///
/// let pi_val = pi();
/// let epsilon = UBinary::new(BigUint::from(1u32), BigInt::from(-50));
/// let bounds = pi_val.refine_to_default(epsilon)?;
/// // bounds now contains pi to ~50 bits of precision
/// # Ok::<(), computable::ComputableError>(())
/// ```
pub fn pi() -> Computable {
    let node = Node::new(Arc::new(PiOp {
        num_terms: RwLock::new(INITIAL_PI_TERMS),
    }));
    Computable { node }
}

/// Returns bounds on pi with at least `precision_bits` bits of accuracy.
///
/// This is a helper function for use in sin.rs and other places that need
/// pi bounds without creating a full Computable.
///
/// The returned bounds (pi_lo, pi_hi) satisfy:
/// - pi_lo <= true_pi <= pi_hi
/// - (pi_hi - pi_lo) <= 2^(-precision_bits) approximately
pub fn pi_bounds_at_precision(precision_bits: u64) -> (Binary, Binary) {
    // Compute enough terms to achieve the desired precision.
    // For arctan(1/5), error after n terms is bounded by (1/5)^(2n+1)/(2n+1)
    // For arctan(1/239), error after n terms is bounded by (1/239)^(2n+1)/(2n+1)
    //
    // The dominant error comes from arctan(1/5) since 1/5 > 1/239.
    // We need: 4 * (1/5)^(2n+1)/(2n+1) < 2^(-precision_bits) / 4
    // Approximately: (1/5)^(2n+1) < 2^(-precision_bits) / 16
    // Taking logs: (2n+1) * log2(5) > precision_bits + 4
    // So: n > (precision_bits + 4) / (2 * log2(5)) - 0.5
    // log2(5) ~= 2.32, so: n > (precision_bits + 4) / 4.64 - 0.5
    //
    // We use a conservative estimate with some margin:
    // TODO(correctness): Using f64 for this calculation is not rigorous for a "provably correct"
    // library. Should use integer arithmetic with conservative bounds instead.
    let num_terms = (((precision_bits as f64 + 10.0) / 4.0).ceil() as usize).max(5);
    compute_pi_bounds(num_terms)
}

/// Pi computation operation using Machin's formula.
pub struct PiOp {
    pub num_terms: RwLock<usize>,
}

impl NodeOp for PiOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let num_terms = *self.num_terms.read();
        let (pi_lo, pi_hi) = compute_pi_bounds(num_terms);
        let raw_bounds = Bounds::new_checked(XBinary::Finite(pi_lo), XBinary::Finite(pi_hi))
            .map_err(|_| ComputableError::InvalidBoundsOrder)?;
        // Simplify bounds to reduce precision bloat from high-precision pi computation
        let margin = margin_from_width(raw_bounds.width(), MARGIN_SHIFT);
        Ok(simplify_bounds_if_needed(
            &raw_bounds,
            PRECISION_SIMPLIFICATION_THRESHOLD,
            &margin,
        ))
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();
        // Double the number of terms for faster convergence
        *num_terms = (*num_terms).saturating_mul(2).max(*num_terms + 1);
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Rounding direction for directed rounding in interval arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundDir {
    /// Round toward negative infinity (floor)
    Down,
    /// Round toward positive infinity (ceiling)
    Up,
}

/// Computes bounds on pi using Machin's formula with n Taylor series terms.
///
/// Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239)
/// Therefore: pi = 16*arctan(1/5) - 4*arctan(1/239)
///
/// Returns (lower_bound, upper_bound) where lower_bound <= pi <= upper_bound.
fn compute_pi_bounds(num_terms: usize) -> (Binary, Binary) {
    // Compute arctan(1/5) bounds
    let (atan_5_lo, atan_5_hi) = arctan_recip_bounds(5, num_terms);

    // Compute arctan(1/239) bounds
    let (atan_239_lo, atan_239_hi) = arctan_recip_bounds(239, num_terms);

    // pi = 16*arctan(1/5) - 4*arctan(1/239)
    //
    // For interval arithmetic subtraction: [a,b] - [c,d] = [a-d, b-c]
    // So: pi_lo = 16*atan_5_lo - 4*atan_239_hi
    //     pi_hi = 16*atan_5_hi - 4*atan_239_lo

    let sixteen = Binary::new(BigInt::from(1), BigInt::from(4)); // 2^4 = 16
    let four = Binary::new(BigInt::from(1), BigInt::from(2)); // 2^2 = 4

    // 16 * arctan(1/5) bounds
    let term1_lo = atan_5_lo.mul(&sixteen);
    let term1_hi = atan_5_hi.mul(&sixteen);

    // 4 * arctan(1/239) bounds
    let term2_lo = atan_239_lo.mul(&four);
    let term2_hi = atan_239_hi.mul(&four);

    // Interval subtraction: [term1_lo, term1_hi] - [term2_lo, term2_hi]
    // Result: [term1_lo - term2_hi, term1_hi - term2_lo]
    let pi_lo = term1_lo.sub(&term2_hi);
    let pi_hi = term1_hi.sub(&term2_lo);

    (pi_lo, pi_hi)
}

/// Computes bounds on arctan(1/k) using Taylor series with n terms.
///
/// Taylor series: arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
/// For x = 1/k: arctan(1/k) = 1/k - 1/(3k^3) + 1/(5k^5) - ...
///
/// The series is alternating with decreasing absolute terms for |x| < 1.
/// Error after n terms is bounded by the magnitude of the (n+1)th term:
/// |error| <= |x|^(2n+1) / (2n+1) = 1 / ((2n+1) * k^(2n+1))
///
/// Returns (lower_bound, upper_bound) for arctan(1/k).
fn arctan_recip_bounds(k: u64, num_terms: usize) -> (Binary, Binary) {
    if num_terms == 0 {
        // No terms computed, just return error bound interval centered at 0
        let error = arctan_recip_error_bound(k, 0);
        return (error.neg(), error);
    }

    // Compute partial sum with directed rounding
    let sum_lo = arctan_recip_partial_sum(k, num_terms, RoundDir::Down);
    let sum_hi = arctan_recip_partial_sum(k, num_terms, RoundDir::Up);

    // Compute error bound for truncation (always round up for conservative bound)
    let error = arctan_recip_error_bound(k, num_terms);

    // Final bounds: [sum_lo - error, sum_hi + error]
    (sum_lo.sub(&error), sum_hi.add(&error))
}

/// Computes partial sum of arctan(1/k) Taylor series with directed rounding.
///
/// sum = sum_{i=0}^{n-1} (-1)^i / ((2i+1) * k^(2i+1))
fn arctan_recip_partial_sum(k: u64, num_terms: usize, rounding: RoundDir) -> Binary {
    let mut sum = Binary::zero();
    let k_big = BigInt::from(k);
    let k_squared = &k_big * &k_big;

    // Start with k^1 in denominator
    let mut k_power = k_big.clone(); // k^(2i+1), starts at k^1

    for i in 0..num_terms {
        // Term i: (-1)^i / ((2i+1) * k^(2i+1))
        let coeff = BigInt::from(2 * i + 1); // 2i+1
        let denominator = &coeff * &k_power; // (2i+1) * k^(2i+1)

        // Compute 1/denominator with directed rounding
        // For positive terms (even i): round down for lower, round up for upper
        // For negative terms (odd i): round up for lower, round down for upper
        let is_positive_term = i % 2 == 0;

        let term = divide_one_by_bigint(&denominator, rounding, is_positive_term);
        sum = sum.add(&term);

        // Prepare for next iteration: k^(2i+1) -> k^(2(i+1)+1) = k^(2i+3)
        // Multiply by k^2
        k_power = &k_power * &k_squared;
    }

    sum
}

/// Computes 1/denominator as a Binary with directed rounding.
///
/// Uses the shared `reciprocal_of_biguint` function from the binary module.
///
/// For interval arithmetic, we need to round in the correct direction based on
/// whether this term contributes positively or negatively to the sum.
///
/// - For positive terms: round down gives lower bound, round up gives upper bound
/// - For negative terms: we negate, so round up gives lower bound (more negative), etc.
fn divide_one_by_bigint(denominator: &BigInt, rounding: RoundDir, is_positive_term: bool) -> Binary {
    // TODO(correctness): Fixed 128-bit precision caps the achievable accuracy to ~118 bits.
    // This causes the refinement loop to hang when requesting precision > 118 bits, because
    // the computed width (~2^-119) never reaches epsilon (e.g., 2^-128). Should make precision
    // adaptive based on the requested output precision.
    const PRECISION_BITS: usize = 128;

    // Determine the rounding direction for the reciprocal based on the overall rounding
    // direction and whether the term is positive or negative:
    // - Positive term, Down rounding: floor (to get lower bound)
    // - Positive term, Up rounding: ceil (to get upper bound)
    // - Negative term, Down rounding: ceil then negate (more negative = smaller = lower bound)
    // - Negative term, Up rounding: floor then negate (less negative = larger = upper bound)
    let recip_rounding = match (rounding, is_positive_term) {
        (RoundDir::Down, true) => ReciprocalRounding::Floor,
        (RoundDir::Up, true) => ReciprocalRounding::Ceil,
        (RoundDir::Down, false) => ReciprocalRounding::Ceil,
        (RoundDir::Up, false) => ReciprocalRounding::Floor,
    };

    // Use magnitude() to convert positive BigInt to BigUint.
    // The denominator is always positive (product of positive coefficients and powers).
    let unsigned_result = reciprocal_of_biguint(denominator.magnitude(), PRECISION_BITS, recip_rounding);

    // Apply sign based on whether this is a positive or negative term
    if is_positive_term {
        unsigned_result
    } else {
        unsigned_result.neg()
    }
}

/// Computes error bound for arctan(1/k) Taylor series after n terms.
///
/// Uses the shared `reciprocal_of_biguint` function from the binary module.
///
/// For alternating series with decreasing terms, the error is bounded by
/// the absolute value of the first omitted term:
/// |error| <= 1 / ((2n+1) * k^(2n+1))
///
/// We round UP (ceiling) to get a conservative (safe) error bound.
fn arctan_recip_error_bound(k: u64, num_terms: usize) -> Binary {
    use num_bigint::BigUint;

    // TODO(correctness): Fixed 128-bit precision here has the same limitation as in
    // divide_one_by_bigint: achievable accuracy is capped at ~118 bits, causing refinement
    // to hang for higher precision requests. Should use adaptive precision matching the
    // requested output precision.
    const PRECISION_BITS: usize = 128;

    let exponent = 2 * num_terms + 1;
    let coeff = BigUint::from(exponent as u64); // 2n+1

    let k_big = BigUint::from(k);
    let mut k_power = BigUint::one();
    for _ in 0..exponent {
        k_power *= &k_big;
    }

    let denominator = &coeff * &k_power; // (2n+1) * k^(2n+1)

    // Compute 1/denominator, rounding UP for conservative error bound.
    reciprocal_of_biguint(&denominator, PRECISION_BITS, ReciprocalRounding::Ceil)
}


/// Returns pi as a FiniteBounds interval with specified precision.
pub fn pi_interval_at_precision(precision_bits: u64) -> FiniteBounds {
    let (lo, hi) = pi_bounds_at_precision(precision_bits);
    FiniteBounds::new(lo, hi)
}

/// Returns 2*pi as a FiniteBounds interval with specified precision.
pub fn two_pi_interval_at_precision(precision_bits: u64) -> FiniteBounds {
    use num_bigint::BigUint;

    let pi_interval = pi_interval_at_precision(precision_bits);
    // 2*pi: multiply by 2
    let two = UBinary::new(BigUint::from(1u32), BigInt::from(1)); // 2^1 = 2
    pi_interval.scale_positive(&two)
}

/// Returns pi/2 as a FiniteBounds interval with specified precision.
pub fn half_pi_interval_at_precision(precision_bits: u64) -> FiniteBounds {
    let (pi_lo, pi_hi) = pi_bounds_at_precision(precision_bits);
    // pi/2: divide by 2 (decrement exponent by 1)
    let half_pi_lo = Binary::new(pi_lo.mantissa().clone(), pi_lo.exponent() - BigInt::one());
    let half_pi_hi = Binary::new(pi_hi.mantissa().clone(), pi_hi.exponent() - BigInt::one());
    FiniteBounds::new(half_pi_lo, half_pi_hi)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::UBinary;
    use num_bigint::BigUint;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn ubin(mantissa: u64, exponent: i64) -> UBinary {
        UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
    }

    fn unwrap_finite(x: &XBinary) -> Binary {
        match x {
            XBinary::Finite(b) => b.clone(),
            _ => panic!("expected finite"),
        }
    }

    /// Returns the f64 approximation of pi as a Binary.
    ///
    /// Note: f64 PI = 3.14159265358979311... which is slightly LESS than true pi
    /// (3.14159265358979323...) due to f64 rounding. The difference is ~1.2e-16.
    fn pi_f64_binary() -> Binary {
        Binary::from_f64(std::f64::consts::PI).expect("PI should convert to Binary")
    }

    #[test]
    fn pi_bounds_contain_true_pi() {
        let (pi_lo, pi_hi) = compute_pi_bounds(20);
        let pi_f64 = pi_f64_binary();

        // Check that bounds are ordered correctly
        assert!(pi_lo < pi_hi, "lower bound should be less than upper bound");

        // The upper bound should definitely be >= f64 pi (since f64 pi < true pi < pi_hi)
        assert!(
            pi_hi >= pi_f64,
            "upper bound should be >= f64 pi approximation"
        );

        // The lower bound should be very close to f64 pi. Since f64 has ~53 bits of precision
        // and our pi computation uses 128 bits, the difference between pi_lo and f64_pi
        // should be at most about 2^-52 (the f64 rounding error).
        // We use a generous epsilon of 2^-50 to account for this.
        let f64_error_bound = bin(1, -50);
        let pi_lo_minus_f64 = pi_lo.sub(&pi_f64);
        assert!(
            pi_lo_minus_f64 < f64_error_bound,
            "lower bound should be within 2^-50 of f64 pi approximation"
        );

        // Check that the interval is reasonably tight.
        // With 20 Taylor series terms and 128-bit intermediate precision,
        // we should easily achieve width < 2^-40 (about 12 decimal digits).
        let width = pi_hi.sub(&pi_lo);
        let precision_threshold = bin(1, -40);
        assert!(
            width < precision_threshold,
            "pi bounds with 20 terms should have width < 2^-40"
        );
    }

    #[test]
    fn pi_bounds_refine_to_high_precision() {
        let (pi_lo_5, pi_hi_5) = compute_pi_bounds(5);
        let (pi_lo_20, pi_hi_20) = compute_pi_bounds(20);

        let width_5 = pi_hi_5.sub(&pi_lo_5);
        let width_20 = pi_hi_20.sub(&pi_lo_20);

        assert!(
            width_20 < width_5,
            "more terms should give tighter bounds"
        );
    }

    #[test]
    fn pi_computable_refines() {
        let pi_comp = pi();
        let epsilon = ubin(1, -20); // 2^-20 precision
        let bounds = pi_comp
            .refine_to_default(epsilon.clone())
            .expect("refine should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        let pi_f64 = pi_f64_binary();

        // The upper bound should definitely be >= f64 pi (since f64 pi < true pi < upper)
        assert!(
            upper >= pi_f64,
            "upper bound should be >= f64 pi approximation"
        );

        // The lower bound should be very close to f64 pi. With 2^-20 epsilon precision,
        // the bounds are much looser than f64 precision, so lower should be <= f64 pi.
        // (The refined bounds are simplified/loosened from the raw computation.)
        assert!(
            lower <= pi_f64,
            "lower bound should be <= f64 pi approximation (after simplification)"
        );

        // Check width is within epsilon
        let width = upper.sub(&lower);
        let eps_binary = epsilon.to_binary();
        assert!(width <= eps_binary, "width should be <= epsilon");
    }

    #[test]
    fn pi_bounds_at_precision_helper() {
        const PRECISION_BITS: u64 = 50;
        let (lo, hi) = pi_bounds_at_precision(PRECISION_BITS);
        let width = hi.sub(&lo);

        let threshold = bin(1, -(PRECISION_BITS as i64 - 1));
        assert!(
            width < threshold,
            "{} bits of precision should give width < 2^-{}", PRECISION_BITS, PRECISION_BITS - 1
        );
    }

    #[test]
    fn interval_arithmetic_subtraction() {
        // Test that [a,b] - [c,d] = [a-d, b-c]
        let a = FiniteBounds::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = FiniteBounds::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = a.interval_sub(&b);
        // [1, 2] - [3, 5] = [1-5, 2-3] = [-4, -1]
        assert_eq!(result.lo(), &bin(-4, 0));
        assert_eq!(result.hi(), bin(-1, 0));
    }

    // TODO: should this go with `neg` tests? is this actually needed or redundant?
    #[test]
    fn interval_negation() {
        let a = FiniteBounds::new(bin(1, 0), bin(3, 0)); // [1, 3]
        let neg_a = a.interval_neg(); // [-3, -1]
        assert_eq!(neg_a.lo(), &bin(-3, 0));
        assert_eq!(neg_a.hi(), bin(-1, 0));
    }
}
