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
use parking_lot::RwLock;

use crate::binary::{Binary, ReciprocalRounding, UXBinary, XBinary, reciprocal_of_biguint};
use crate::computable::Computable;
use crate::error::ComputableError;
use crate::node::{Node, NodeOp};
use crate::prefix::Prefix;
use crate::sane::Sane;

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
/// let prefix = pi_val.refine_to_default(XUsize::Finite(50))?;
/// // prefix now contains pi to ~50 bits of precision
/// # Ok::<(), computable::ComputableError>(())
/// ```
/// Creates a pi computation node for use in other ops (e.g., sin).
pub(crate) fn pi_node() -> Arc<Node> {
    Node::new(Arc::new(PiOp {
        state: RwLock::new(PiState {
            num_terms: 1,
            arctan_5: ArctanCache::empty(5),
            arctan_239: ArctanCache::empty(239),
        }),
    }))
}

pub fn pi() -> Computable {
    Computable::from_node(pi_node())
}

/// Returns a tight interval on pi with at least `precision_bits` bits of accuracy.
///
/// This is a helper function for use in sin.rs and other places that need
/// pi bounds without creating a full Computable. Returns `(lo, hi)` as Binary.
///
/// The returned bounds (pi_lo, pi_hi) satisfy:
/// - pi_lo <= true_pi <= pi_hi
/// - (pi_hi - pi_lo) <= 2^(-precision_bits) approximately
pub fn pi_prefix_at_precision(precision_bits: usize) -> (Binary, Binary) {
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
    compute_pi_interval(num_terms, reciprocal_precision)
}

/// Cached intermediate state for an arctan(1/k) Taylor series computation.
///
/// Stores partial sums and the power-of-k state so that additional terms
/// can be appended in O(delta) instead of recomputing from scratch.
struct ArctanCache {
    sum_lo: Binary,
    sum_hi: Binary,
    /// k^(2*num_terms+1), ready for the next term's denominator.
    k_power: BigInt,
    k_squared: BigInt,
    num_terms: usize,
    precision_bits: usize,
}

impl ArctanCache {
    /// Creates an empty cache for arctan(1/k). No terms are computed;
    /// `precision_bits` is set to 0 so the first `ensure_cache` call
    /// will recreate with the real precision and add terms.
    fn empty(k: u64) -> Self {
        let k_big = BigInt::from(k);
        let k_squared = &k_big * &k_big;
        Self {
            sum_lo: Binary::zero(),
            sum_hi: Binary::zero(),
            k_power: k_big,
            k_squared,
            num_terms: 0,
            precision_bits: 0,
        }
    }

    fn new(k: u64, precision_bits: usize) -> Self {
        let k_big = BigInt::from(k);
        let k_squared = &k_big * &k_big;
        Self {
            sum_lo: Binary::zero(),
            sum_hi: Binary::zero(),
            k_power: k_big,
            k_squared,
            num_terms: 0,
            precision_bits,
        }
    }

    fn add_terms(&mut self, count: usize) {
        let start = self.num_terms;
        let end = crate::sane_arithmetic!(start, count; start + count);
        for i in start..end {
            let coeff = BigInt::from(i) * 2_i64 + 1_i64;
            let denominator = &coeff * &self.k_power;
            let is_positive_term = i % 2 == 0;

            let term_lo = divide_one_by_bigint(
                &denominator,
                RoundDir::Down,
                is_positive_term,
                self.precision_bits,
            );
            let term_hi = divide_one_by_bigint(
                &denominator,
                RoundDir::Up,
                is_positive_term,
                self.precision_bits,
            );
            self.sum_lo = self.sum_lo.add(&term_lo);
            self.sum_hi = self.sum_hi.add(&term_hi);

            self.k_power = &self.k_power * &self.k_squared;
        }
        self.num_terms = end;
    }

    /// Returns (lower_bound, upper_bound) for arctan(1/k), including truncation error.
    fn interval(&self) -> (Binary, Binary) {
        if self.num_terms == 0 {
            let error = reciprocal_of_biguint(
                // k_power is k^1 at num_terms=0
                self.k_power.magnitude(),
                self.precision_bits,
                ReciprocalRounding::Ceil,
            );
            return (error.neg(), error);
        }
        let n = self.num_terms;
        let error_coeff = BigInt::from(crate::sane_arithmetic!(n; 2 * n + 1));
        let error_denom = &error_coeff * &self.k_power;
        let error = reciprocal_of_biguint(
            error_denom.magnitude(),
            self.precision_bits,
            ReciprocalRounding::Ceil,
        );
        (self.sum_lo.sub(&error), self.sum_hi.add(&error))
    }
}

/// Pi computation operation using Machin's formula.
pub struct PiOp {
    state: RwLock<PiState>,
}

struct PiState {
    num_terms: usize,
    arctan_5: ArctanCache,
    arctan_239: ArctanCache,
}

impl NodeOp for PiOp {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let mut state = self.state.write();
        let num_terms = state.num_terms;
        let needed_precision = precision_bits_for_num_terms(num_terms);

        // Extend or recreate arctan caches.
        Self::ensure_cache(&mut state.arctan_5, 5, num_terms, needed_precision);
        let (atan_5_lo, atan_5_hi) = state.arctan_5.interval();

        Self::ensure_cache(&mut state.arctan_239, 239, num_terms, needed_precision);
        let (atan_239_lo, atan_239_hi) = state.arctan_239.interval();

        // pi = 16*arctan(1/5) - 4*arctan(1/239)
        let sixteen = Binary::new(BigInt::from(1_i32), BigInt::from(4_i32));
        let four = Binary::new(BigInt::from(1_i32), BigInt::from(2_i32));

        let term1_lo = atan_5_lo.mul(&sixteen);
        let term1_hi = atan_5_hi.mul(&sixteen);
        let term2_lo = atan_239_lo.mul(&four);
        let term2_hi = atan_239_hi.mul(&four);

        let pi_lo = term1_lo.sub(&term2_hi);
        let pi_hi = term1_hi.sub(&term2_lo);

        Ok(Prefix::from_lower_upper(
            XBinary::Finite(pi_lo),
            XBinary::Finite(pi_hi),
        ))
    }

    fn refine_step(&self, precision_bits: usize) -> Result<bool, ComputableError> {
        let mut state = self.state.write();

        // Leap to the needed term count based on precision_bits.
        // Same formula as pi_prefix_at_precision: n = (precision_bits + 10) / 4.
        if precision_bits <= crate::MAX_COMPUTATION_BITS {
            let needed = crate::sane_arithmetic!(precision_bits; (precision_bits + 10) / 4).max(1);
            if needed > state.num_terms {
                state.num_terms = needed;
                return Ok(true);
            }
        }

        // With per-refiner budgets, the leap formula always produces needed > num_terms
        // when the coordinator dispatches (otherwise the refiner would have been skipped).
        unreachable!(
            "PiOp: leap did not advance; precision_bits={}, num_terms={}",
            precision_bits, state.num_terms
        )
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

impl PiOp {
    /// Extends the cache to `num_terms`, or recreates it if the precision requirement
    /// exceeds the cached precision.
    fn ensure_cache(cache: &mut ArctanCache, k: u64, num_terms: usize, needed_precision: usize) {
        if cache.precision_bits < needed_precision || cache.num_terms > num_terms {
            *cache = ArctanCache::new(k, needed_precision);
            cache.add_terms(num_terms);
        } else if cache.num_terms < num_terms {
            // Safe: guard ensures cache.num_terms < num_terms.
            #[allow(clippy::arithmetic_side_effects)]
            let delta = num_terms - cache.num_terms;
            cache.add_terms(delta);
        }
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
fn compute_pi_interval(num_terms: usize, precision_bits: usize) -> (Binary, Binary) {
    // Compute arctan(1/5) bounds
    let (atan_5_lo, atan_5_hi) = arctan_recip_interval(5, num_terms, precision_bits);

    // Compute arctan(1/239) bounds
    let (atan_239_lo, atan_239_hi) = arctan_recip_interval(239, num_terms, precision_bits);

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
fn arctan_recip_interval(k: u64, num_terms: usize, precision_bits: usize) -> (Binary, Binary) {
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

/// Returns pi as a FiniteInterval with specified precision.
#[cfg(test)]
pub(crate) fn pi_interval_at_precision(
    precision_bits: usize,
) -> crate::finite_interval::FiniteInterval {
    use crate::finite_interval::FiniteInterval;
    let (lo, hi) = pi_prefix_at_precision(precision_bits);
    FiniteInterval::new(lo, hi)
}

/// Returns 2*pi as a FiniteInterval with specified precision.
#[cfg(test)]
pub(crate) fn two_pi_interval_at_precision(
    precision_bits: usize,
) -> crate::finite_interval::FiniteInterval {
    use crate::binary::UBinary;
    use num_bigint::BigUint;

    let pi_interval = pi_interval_at_precision(precision_bits);
    // 2*pi: multiply by 2
    let two = UBinary::new(BigUint::from(1u32), BigInt::from(1_i32)); // 2^1 = 2
    pi_interval.scale_positive(&two)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refinement::XUsize;
    use crate::test_utils::{bin, epsilon_as_binary, to_bounds, unwrap_finite};

    /// Returns the f64 approximation of pi as a Binary.
    ///
    /// Note: f64 PI = 3.14159265358979311... which is slightly LESS than true pi
    /// (3.14159265358979323...) due to f64 rounding. The difference is ~1.2e-16.
    fn pi_f64_binary() -> Binary {
        Binary::from_f64(std::f64::consts::PI).expect("PI should convert to Binary")
    }

    #[test]
    fn pi_bounds_contain_true_pi() {
        let (pi_lo, pi_hi) = compute_pi_interval(20, precision_bits_for_num_terms(20));
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
        let (pi_lo_5, pi_hi_5) = compute_pi_interval(5, precision_bits_for_num_terms(5));
        let (pi_lo_20, pi_hi_20) = compute_pi_interval(20, precision_bits_for_num_terms(20));

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

        let bounds = to_bounds(&prefix);
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
    fn pi_prefix_at_precision_helper() {
        const PRECISION_BITS: usize = 50;
        let (lo, hi) = pi_prefix_at_precision(PRECISION_BITS);
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
    fn pi_prefix_at_precision_256_bits() {
        let (lo, hi) = pi_prefix_at_precision(256);
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
    fn arctan_cache_incremental_matches_from_scratch() {
        let precision_bits = precision_bits_for_num_terms(30);

        // Build incrementally: 10 terms, then extend to 30
        let mut cache = ArctanCache::new(5, precision_bits);
        cache.add_terms(10);
        cache.add_terms(20);
        let (inc_lo, inc_hi) = cache.interval();

        // Build from scratch: 30 terms
        let (scratch_lo, scratch_hi) = arctan_recip_interval(5, 30, precision_bits);

        assert_eq!(
            inc_lo, scratch_lo,
            "incremental lower bound should match from-scratch"
        );
        assert_eq!(
            inc_hi, scratch_hi,
            "incremental upper bound should match from-scratch"
        );
    }

    #[test]
    fn arctan_cache_incremental_matches_239() {
        let precision_bits = precision_bits_for_num_terms(25);

        let mut cache = ArctanCache::new(239, precision_bits);
        cache.add_terms(5);
        cache.add_terms(20);
        let (inc_lo, inc_hi) = cache.interval();

        let (scratch_lo, scratch_hi) = arctan_recip_interval(239, 25, precision_bits);

        assert_eq!(inc_lo, scratch_lo);
        assert_eq!(inc_hi, scratch_hi);
    }

    #[test]
    fn pi_computable_refines_beyond_128_bits() {
        let pi_comp = pi();
        let tolerance_exp = XUsize::Finite(128); // 2^-128 precision
        let prefix = pi_comp
            .refine_to_default(tolerance_exp)
            .expect("refine to 2^-128 should succeed");

        let bounds = to_bounds(&prefix);
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
