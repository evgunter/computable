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
use num_traits::{One, Signed, Zero};
use parking_lot::RwLock;

use crate::binary::{margin_from_width, simplify_bounds_if_needed, Binary, Bounds, XBinary};
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

// TODO: does this have any overlap with the `inv` function? can they be unified or at least share common code?
/// Computes 1/denominator as a Binary with directed rounding.
///
/// For interval arithmetic, we need to round in the correct direction based on
/// whether this term contributes positively or negatively to the sum.
///
/// - For positive terms: round down gives lower bound, round up gives upper bound
/// - For negative terms: we negate, so round up gives lower bound (more negative), etc.
fn divide_one_by_bigint(denominator: &BigInt, rounding: RoundDir, is_positive_term: bool) -> Binary {
    // We compute 2^precision / denominator with appropriate rounding.
    // Result = (2^precision / denominator) * 2^(-precision)
    //
    // Use high precision for intermediate computation
    // TODO(correctness): Fixed 128-bit precision caps the achievable accuracy. If more than ~128
    // bits of pi precision are needed, the rounding errors in this fixed-precision division could
    // accumulate beyond what the error bound accounts for. Should make precision adaptive based
    // on the requested output precision.
    let precision_bits: u64 = 128;

    let numerator = BigInt::one() << precision_bits as usize;
    let (quot, rem) = num_integer::Integer::div_rem(&numerator, denominator);

    // Determine if we need to round up (add 1 to quotient)
    let has_remainder = !rem.is_zero();

    // Effective rounding direction considering sign of term:
    // - Positive term, Down rounding: truncate (floor)
    // - Positive term, Up rounding: round up (ceil)
    // - Negative term, Down rounding: round up in magnitude (more negative = smaller)
    // - Negative term, Up rounding: truncate in magnitude (less negative = larger)
    let should_round_up = has_remainder
        && match (rounding, is_positive_term) {
            (RoundDir::Up, true) => true,   // Positive, want upper -> round up
            (RoundDir::Down, true) => false, // Positive, want lower -> truncate
            (RoundDir::Up, false) => false,  // Negative, want upper -> less negative -> truncate
            (RoundDir::Down, false) => true, // Negative, want lower -> more negative -> round up
        };

    let final_quot = if should_round_up {
        quot + BigInt::one()
    } else {
        quot
    };

    // Apply sign
    let signed_mantissa = if is_positive_term {
        final_quot
    } else {
        -final_quot
    };

    Binary::new(signed_mantissa, -BigInt::from(precision_bits))
}

/// Computes error bound for arctan(1/k) Taylor series after n terms.
///
/// For alternating series with decreasing terms, the error is bounded by
/// the absolute value of the first omitted term:
/// |error| <= 1 / ((2n+1) * k^(2n+1))
///
/// We round UP to get a conservative (safe) error bound.
fn arctan_recip_error_bound(k: u64, num_terms: usize) -> Binary {
    let exponent = 2 * num_terms + 1;
    let coeff = BigInt::from(exponent); // 2n+1

    let k_big = BigInt::from(k);
    let mut k_power = BigInt::one();
    for _ in 0..exponent {
        k_power *= &k_big;
    }

    let denominator = &coeff * &k_power; // (2n+1) * k^(2n+1)

    // Compute 1/denominator, rounding UP for conservative error bound
    // TODO(correctness): Fixed 128-bit precision here has the same limitation as in
    // divide_one_by_bigint. For very high precision pi computations, this could underestimate
    // the error bound. Should use adaptive precision matching the requested output precision.
    let precision_bits: u64 = 128;
    let numerator = BigInt::one() << precision_bits as usize;
    let (quot, rem) = num_integer::Integer::div_rem(&numerator, &denominator);

    // Always round up for error bound
    let final_quot = if !rem.is_zero() {
        quot + BigInt::one()
    } else {
        quot
    };

    Binary::new(final_quot, -BigInt::from(precision_bits))
}

//=============================================================================
// Interval arithmetic helpers for use in sin.rs
//=============================================================================

// TODO: make this into FiniteBounds instead, using the same paradigm as the Bounds type

/// Represents an interval [lo, hi] for full interval propagation.
#[derive(Clone, Debug)]
pub struct Interval {
    pub lo: Binary,
    pub hi: Binary,
}

impl Interval {
    /// Creates a new interval [lo, hi].
    pub fn new(lo: Binary, hi: Binary) -> Self {
        // TODO: no debug assert! this should be just like in Bounds
        debug_assert!(lo <= hi, "Interval lower bound must be <= upper bound");
        Self { lo, hi }
    }

    /// Creates a point interval [x, x].
    pub fn point(x: Binary) -> Self {
        Self {
            lo: x.clone(),
            hi: x,
        }
    }

    /// Interval addition: [a,b] + [c,d] = [a+c, b+d]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.add(&other.lo),
            hi: self.hi.add(&other.hi),
        }
    }

    /// Interval subtraction: [a,b] - [c,d] = [a-d, b-c]
    /// Note the swap in the second operand!
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.sub(&other.hi), // a - d
            hi: self.hi.sub(&other.lo), // b - c
        }
    }

    /// Interval negation: -[a,b] = [-b, -a]
    pub fn neg(&self) -> Self {
        Self {
            lo: self.hi.neg(),
            hi: self.lo.neg(),
        }
    }

    /// Interval multiplication by a positive scalar k: k * [a,b] = [k*a, k*b]
    pub fn scale_positive(&self, k: &Binary) -> Self {
        debug_assert!(
            !k.mantissa().is_negative(),
            "scale_positive requires non-negative scalar"
        );
        Self {
            lo: self.lo.mul(k),
            hi: self.hi.mul(k),
        }
    }

    /// Interval multiplication by a BigInt (can be negative).
    pub fn scale_bigint(&self, k: &BigInt) -> Self {
        if k.is_negative() {
            // k * [a,b] = [k*b, k*a] when k < 0
            let k_binary = Binary::new(k.clone(), BigInt::zero());
            Self {
                lo: self.hi.mul(&k_binary),
                hi: self.lo.mul(&k_binary),
            }
        } else {
            let k_binary = Binary::new(k.clone(), BigInt::zero());
            Self {
                lo: self.lo.mul(&k_binary),
                hi: self.hi.mul(&k_binary),
            }
        }
    }

    /// Returns the width of the interval (hi - lo).
    pub fn width(&self) -> Binary {
        self.hi.sub(&self.lo)
    }

    /// Returns the midpoint of the interval.
    pub fn midpoint(&self) -> Binary {
        let sum = self.lo.add(&self.hi);
        // Divide by 2 by decrementing exponent
        Binary::new(sum.mantissa().clone(), sum.exponent() - BigInt::one())
    }

    /// Checks if this interval contains a point.
    pub fn contains(&self, point: &Binary) -> bool {
        &self.lo <= point && point <= &self.hi
    }

    /// Checks if this interval is entirely less than another.
    pub fn entirely_less_than(&self, other: &Self) -> bool {
        self.hi < other.lo
    }

    /// Checks if this interval is entirely greater than another.
    pub fn entirely_greater_than(&self, other: &Self) -> bool {
        self.lo > other.hi
    }

    /// Checks if this interval overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        !(self.entirely_less_than(other) || self.entirely_greater_than(other))
    }
}

/// Returns pi as an Interval with specified precision.
pub fn pi_interval_at_precision(precision_bits: u64) -> Interval {
    let (lo, hi) = pi_bounds_at_precision(precision_bits);
    Interval::new(lo, hi)
}

/// Returns 2*pi as an Interval with specified precision.
pub fn two_pi_interval_at_precision(precision_bits: u64) -> Interval {
    let (pi_lo, pi_hi) = pi_bounds_at_precision(precision_bits);
    // 2*pi: multiply by 2 (shift exponent by 1)
    let two_pi_lo = Binary::new(pi_lo.mantissa().clone(), pi_lo.exponent() + BigInt::one());
    let two_pi_hi = Binary::new(pi_hi.mantissa().clone(), pi_hi.exponent() + BigInt::one());
    Interval::new(two_pi_lo, two_pi_hi)
}

/// Returns pi/2 as an Interval with specified precision.
pub fn half_pi_interval_at_precision(precision_bits: u64) -> Interval {
    let (pi_lo, pi_hi) = pi_bounds_at_precision(precision_bits);
    // pi/2: divide by 2 (decrement exponent by 1)
    let half_pi_lo = Binary::new(pi_lo.mantissa().clone(), pi_lo.exponent() - BigInt::one());
    let half_pi_hi = Binary::new(pi_hi.mantissa().clone(), pi_hi.exponent() - BigInt::one());
    Interval::new(half_pi_lo, half_pi_hi)
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

    #[test]
    fn pi_bounds_contain_true_pi() {
        // pi = 3.14159265358979323846...
        // We check against a known approximation
        let (pi_lo, pi_hi) = compute_pi_bounds(20);

        // Convert to f64 for rough comparison
        // pi_lo and pi_hi should bracket 3.14159265...
        // TODO: fix this by converting the f64 approximation of pi to Binary and then comparing
        // We can't easily convert Binary to f64, but we can check the bounds are ordered
        assert!(pi_lo < pi_hi, "lower bound should be less than upper bound");

        // Check that the interval is reasonably tight (width < 1)
        let width = pi_hi.sub(&pi_lo);
        let one = bin(1, 0);
        assert!(width < one, "pi bounds should be tighter than width 1");

        // Check bounds bracket approximately 3.14
        // 3 < pi_lo should NOT hold (pi_lo should be > 3)
        // Actually, let's check: pi_lo > 3 and pi_hi < 4
        let three = bin(3, 0);
        let four = bin(4, 0);
        assert!(pi_lo > three, "pi lower bound should be > 3");
        assert!(pi_hi < four, "pi upper bound should be < 4");
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

         // TODO: improve this by converting the f64 approximation of pi to Binary and then comparing
        // Check basic sanity
        let three = bin(3, 0);
        let four = bin(4, 0);
        assert!(lower > three);
        assert!(upper < four);

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
        let a = Interval::new(bin(1, 0), bin(2, 0)); // [1, 2]
        let b = Interval::new(bin(3, 0), bin(5, 0)); // [3, 5]

        let result = a.sub(&b);
        // [1, 2] - [3, 5] = [1-5, 2-3] = [-4, -1]
        assert_eq!(result.lo, bin(-4, 0));
        assert_eq!(result.hi, bin(-1, 0));
    }

    // TODO: should this go with `neg` tests? is this actually needed or redundant?
    #[test]
    fn interval_negation() {
        let a = Interval::new(bin(1, 0), bin(3, 0)); // [1, 3]
        let neg_a = a.neg(); // [-3, -1]
        assert_eq!(neg_a.lo, bin(-3, 0));
        assert_eq!(neg_a.hi, bin(-1, 0));
    }
}
