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
    Binary, Bounds, FiniteBounds, ReciprocalRounding, UBinary, UXBinary, XBinary,
    reciprocal_of_biguint,
};
use crate::computable::Computable;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::Sane;

/// Initial number of Taylor series terms for pi computation.
pub(crate) const INITIAL_PI_TERMS: usize = 10;

/// Returns the number of bits in the binary representation of `x`.
///
/// Equivalent to floor(log2(x)) + 1 for x > 0, and 0 for x == 0.
///
/// Accepts and returns [`Sane`] so it composes naturally inside
/// [`sane_arithmetic!`] blocks.
fn bit_length(x: Sane) -> Sane {
    // leading_zeros() is always <= BITS, so this subtraction cannot underflow.
    // TODO: investigate whether the type system could prevent this case.
    let bits = usize::BITS
        .checked_sub(x.0.leading_zeros())
        .unwrap_or_else(|| unreachable!("leading_zeros() is always <= usize::BITS"));
    Sane(usize::try_from(bits).unwrap_or(0))
}

/// Computes the intermediate reciprocal precision needed for `num_terms` Taylor series terms.
///
/// Returns `(2n+1)*3 + bit_length(n+2) + bit_length(2n+1)` where `n = num_terms`.
///
/// ## Correctness proof
///
/// The Taylor series for arctan(1/5) has truncation error bounded by
/// `(1/5)^(2n+1) / (2n+1)`, which provides `(2n+1)*log2(5)` bits of accuracy.
/// Since `log2(5) < 3`, using `(2n+1)*3` conservatively exceeds the Taylor accuracy.
///
/// In the Machin formula `16*arctan(1/5) - 4*arctan(1/239)`, each of the `n` reciprocal
/// terms introduces at most 1 ULP of directed rounding error. The total rounding error
/// width is at most `20*(n+2) * 2^(-precision_bits)`. For this to not dominate the
/// Taylor truncation error, we need:
///
/// ```text
/// precision_bits >= (2n+1)*log2(5) + log2((n+2)*(2n+1)) - 0.68
/// ```
///
/// The margin `bit_length(n+2) + bit_length(2n+1)` covers `log2((n+2)*(2n+1))` because
/// `bit_length(x) = floor(log2(x)) + 1 >= log2(x)`.
fn precision_bits_for_num_terms(num_terms: usize) -> usize {
    crate::sane_arithmetic!(num_terms; {
        let n = num_terms;
        let two_n_plus_1 = 2 * n + 1;
        let taylor_bits = two_n_plus_1 * 3;
        let rounding_margin = bit_length(n + 2) + bit_length(two_n_plus_1);
        taylor_bits + rounding_margin
    })
}

/// Returns pi as a Computable that can be refined to arbitrary precision.
///
/// Uses Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239)
///
/// # Example
///
/// ```
/// use computable::{pi, XUsize};
///
/// let pi_val = pi();
/// let bounds = pi_val.refine_to_default(XUsize::Finite(50))?;
/// // bounds now contains pi to ~50 bits of precision
/// # Ok::<(), computable::ComputableError>(())
/// ```
pub fn pi() -> Computable {
    let node = Node::new(Arc::new(PiOp {
        num_terms: RwLock::new(INITIAL_PI_TERMS),
    }));
    Computable::from_node(node)
}

/// Returns bounds on pi with at least `precision_bits` bits of accuracy.
///
/// This is a helper function for use in sin.rs and other places that need
/// pi bounds without creating a full Computable.
///
/// The returned bounds (pi_lo, pi_hi) satisfy:
/// - pi_lo <= true_pi <= pi_hi
/// - (pi_hi - pi_lo) <= 2^(-precision_bits) approximately
pub fn pi_bounds_at_precision(precision_bits: usize) -> (Binary, Binary) {
    // Compute enough terms to achieve the desired precision.
    // For arctan(1/5), error after n terms is bounded by (1/5)^(2n+1)/(2n+1).
    // We need (2n+1)*log2(5) > precision_bits + 4, i.e. n > (precision_bits + 4) / (2*log2(5)) - 0.5.
    // Since log2(5) > 2, using (precision_bits + 10) / 4 is conservative (integer-only).
    let num_terms = crate::sane_arithmetic!(precision_bits; (precision_bits + 10) / 4).max(5);
    // Minimum precision to keep rounding error below Taylor truncation error
    let rounding_error_precision = crate::sane_arithmetic!(precision_bits, num_terms;
        precision_bits + bit_length(num_terms + 2) + bit_length(2 * num_terms + 1));
    let reciprocal_precision =
        precision_bits_for_num_terms(num_terms).max(rounding_error_precision);
    compute_pi_bounds(num_terms, reciprocal_precision)
}

/// Pi computation operation using Machin's formula.
pub struct PiOp {
    pub num_terms: RwLock<usize>,
}

impl NodeOp for PiOp {
    fn compute_bounds(&self) -> Result<Prefix, ComputableError> {
        let num_terms = *self.num_terms.read();
        let precision_bits = precision_bits_for_num_terms(num_terms);
        let (pi_lo, pi_hi) = compute_pi_bounds(num_terms, precision_bits);
        let bounds = Bounds::from_lower_and_width(
            XBinary::Finite(pi_lo.clone()),
            UXBinary::Finite(
                crate::binary::UBinary::try_from_binary(&pi_hi.sub(&pi_lo))
                    .unwrap_or_else(|_| crate::binary::UBinary::zero()),
            ),
        );
        Ok(Prefix::from(&bounds))
    }

    fn refine_step(&self, precision_bits: usize) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();

        // Leap to the needed term count based on precision_bits.
        // Same formula as pi_bounds_at_precision: n = (precision_bits + 10) / 4.
        if precision_bits <= crate::MAX_COMPUTATION_BITS {
            let needed = crate::sane_arithmetic!(precision_bits; (precision_bits + 10) / 4).max(1);
            if needed > *num_terms {
                *num_terms = needed;
                return Ok(true);
            }
        }

        // Fall through: double the number of terms (existing behavior)
        *num_terms = (*num_terms).saturating_mul(2).max(1_usize);
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }

    fn child_demand_budget(&self, _target_width: &UXBinary, _child_index: usize) -> UXBinary {
        unreachable!("PiOp has no children")
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
fn compute_pi_bounds(num_terms: usize, precision_bits: usize) -> (Binary, Binary) {
    // Compute arctan(1/5) bounds
    let (atan_5_lo, atan_5_hi) = arctan_recip_bounds(5, num_terms, precision_bits);

    // Compute arctan(1/239) bounds
    let (atan_239_lo, atan_239_hi) = arctan_recip_bounds(239, num_terms, precision_bits);

    // pi = 16*arctan(1/5) - 4*arctan(1/239)
    //
    // For interval arithmetic subtraction: [a,b] - [c,d] = [a-d, b-c]
    // So: pi_lo = 16*atan_5_lo - 4*atan_239_hi
    //     pi_hi = 16*atan_5_hi - 4*atan_239_lo

    let sixteen = Binary::new(BigInt::from(1_i32), BigInt::from(4_i32)); // 2^4 = 16
    let four = Binary::new(BigInt::from(1_i32), BigInt::from(2_i32)); // 2^2 = 4

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
fn arctan_recip_bounds(k: u64, num_terms: usize, precision_bits: usize) -> (Binary, Binary) {
    let k_big = BigInt::from(k);

    if num_terms == 0 {
        // No terms: error bound is 1 / (1 * k^1) = 1/k
        let error =
            reciprocal_of_biguint(k_big.magnitude(), precision_bits, ReciprocalRounding::Ceil);
        return (error.neg(), error);
    }

    let k_squared = &k_big * &k_big;
    let mut sum_lo = Binary::zero();
    let mut sum_hi = Binary::zero();

    // Start with k^1 in denominator
    let mut k_power = k_big; // k^(2i+1), starts at k^1

    for i in 0..num_terms {
        // Term i: (-1)^i / ((2i+1) * k^(2i+1))
        let coeff = BigInt::from(i) * 2_i64 + 1_i64; // 2i+1
        let denominator = &coeff * &k_power; // (2i+1) * k^(2i+1)

        let is_positive_term = i % 2 == 0;

        let term_lo = divide_one_by_bigint(
            &denominator,
            RoundDir::Down,
            is_positive_term,
            precision_bits,
        );
        let term_hi =
            divide_one_by_bigint(&denominator, RoundDir::Up, is_positive_term, precision_bits);
        sum_lo = sum_lo.add(&term_lo);
        sum_hi = sum_hi.add(&term_hi);

        // Prepare for next iteration: k^(2i+1) -> k^(2(i+1)+1) = k^(2i+3)
        k_power = &k_power * &k_squared;
    }

    // Derive error bound from the loop's final k_power state.
    // After the loop, k_power = k^(2*num_terms + 1) (advanced one past the last term).
    // Error = 1 / ((2n+1) * k^(2n+1))
    let error_coeff = BigInt::from(crate::sane_arithmetic!(num_terms; 2 * num_terms + 1));
    let error_denom = &error_coeff * &k_power;
    let error = reciprocal_of_biguint(
        error_denom.magnitude(),
        precision_bits,
        ReciprocalRounding::Ceil,
    );

    (sum_lo.sub(&error), sum_hi.add(&error))
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
fn divide_one_by_bigint(
    denominator: &BigInt,
    rounding: RoundDir,
    is_positive_term: bool,
    precision_bits: usize,
) -> Binary {
    let recip_rounding = match (rounding, is_positive_term) {
        (RoundDir::Down, true) => ReciprocalRounding::Floor,
        (RoundDir::Up, true) => ReciprocalRounding::Ceil,
        (RoundDir::Down, false) => ReciprocalRounding::Ceil,
        (RoundDir::Up, false) => ReciprocalRounding::Floor,
    };

    let unsigned_result =
        reciprocal_of_biguint(denominator.magnitude(), precision_bits, recip_rounding);

    if is_positive_term {
        unsigned_result
    } else {
        unsigned_result.neg()
    }
}

/// Returns pi as a FiniteBounds interval with specified precision.
pub fn pi_interval_at_precision(precision_bits: usize) -> FiniteBounds {
    let (lo, hi) = pi_bounds_at_precision(precision_bits);
    FiniteBounds::new(lo, hi)
}

/// Returns 2*pi as a FiniteBounds interval with specified precision.
pub fn two_pi_interval_at_precision(precision_bits: usize) -> FiniteBounds {
    use num_bigint::BigUint;

    let pi_interval = pi_interval_at_precision(precision_bits);
    // 2*pi: multiply by 2
    let two = UBinary::new(BigUint::from(1u32), BigInt::from(1_i32)); // 2^1 = 2
    pi_interval.scale_positive(&two)
}

/// Returns pi/2 as a FiniteBounds interval with specified precision.
pub fn half_pi_interval_at_precision(precision_bits: usize) -> FiniteBounds {
    let (pi_lo, pi_hi) = pi_bounds_at_precision(precision_bits);
    // pi/2: divide by 2 (decrement exponent by 1)
    let half_pi_lo = Binary::new(pi_lo.mantissa().clone(), pi_lo.exponent() - BigInt::one());
    let half_pi_hi = Binary::new(pi_hi.mantissa().clone(), pi_hi.exponent() - BigInt::one());
    FiniteBounds::new(half_pi_lo, half_pi_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::XBinary;
    use crate::refinement::XUsize;

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn epsilon_as_binary(n: usize) -> Binary {
        let n_i64 = i64::try_from(n).expect("precision fits in i64");
        Binary::new(
            BigInt::from(1_i32),
            BigInt::from(n_i64.checked_neg().expect("negation does not overflow")),
        )
    }

    fn unwrap_finite(x: &XBinary) -> Binary {
        match x {
            XBinary::Finite(b) => b.clone(),
            XBinary::NegInf | XBinary::PosInf => panic!("expected finite"),
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
        let (pi_lo, pi_hi) = compute_pi_bounds(20, precision_bits_for_num_terms(20));
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
        let (pi_lo_5, pi_hi_5) = compute_pi_bounds(5, precision_bits_for_num_terms(5));
        let (pi_lo_20, pi_hi_20) = compute_pi_bounds(20, precision_bits_for_num_terms(20));

        let width_5 = pi_hi_5.sub(&pi_lo_5);
        let width_20 = pi_hi_20.sub(&pi_lo_20);

        assert!(width_20 < width_5, "more terms should give tighter bounds");
    }

    #[test]
    fn pi_computable_refines() {
        let pi_comp = pi();
        let tolerance_exp = XUsize::Finite(20); // 2^-20 precision
        let prefix = pi_comp
            .refine_to_default(tolerance_exp)
            .expect("refine should succeed");

        let bounds = crate::binary::Bounds::from(&prefix);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());
        let pi_f64 = pi_f64_binary();

        assert!(
            upper >= pi_f64,
            "upper bound should be >= f64 pi approximation"
        );
        assert!(
            lower <= pi_f64,
            "lower bound should be <= f64 pi approximation (after simplification)"
        );

        let width = upper.sub(&lower);
        let eps_binary = epsilon_as_binary(20);
        assert!(width <= eps_binary, "width should be <= epsilon");
    }

    #[test]
    fn pi_bounds_at_precision_helper() {
        const PRECISION_BITS: usize = 50;
        let (lo, hi) = pi_bounds_at_precision(PRECISION_BITS);
        let width = hi.sub(&lo);

        let precision_i64 = i64::try_from(PRECISION_BITS).expect("precision fits in i64");
        let exp = precision_i64
            .checked_sub(1_i64)
            .expect("subtraction does not overflow")
            .checked_neg()
            .expect("negation does not overflow");
        let threshold = bin(1, exp);
        assert!(
            width < threshold,
            "{} bits of precision should give width < 2^-{}",
            PRECISION_BITS,
            PRECISION_BITS - 1
        );
    }

    #[test]
    fn pi_bounds_at_precision_256_bits() {
        let (lo, hi) = pi_bounds_at_precision(256);
        let width = hi.sub(&lo);

        let threshold = bin(1, -255);
        assert!(
            width < threshold,
            "256 bits of precision should give width < 2^-255, got width = {}",
            width
        );

        // Verify bounds still contain pi
        let pi_f64 = pi_f64_binary();
        assert!(hi >= pi_f64, "upper bound should be >= f64 pi");
    }

    #[test]
    fn pi_computable_refines_beyond_128_bits() {
        let pi_comp = pi();
        let tolerance_exp = XUsize::Finite(128); // 2^-128 precision
        let prefix = pi_comp
            .refine_to_default(tolerance_exp)
            .expect("refine to 2^-128 should succeed");

        let bounds = crate::binary::Bounds::from(&prefix);
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        let width = upper.sub(&lower);
        let eps_binary = epsilon_as_binary(128);
        assert!(
            width <= eps_binary,
            "width should be <= 2^-128, got width = {}",
            width
        );
    }
}
