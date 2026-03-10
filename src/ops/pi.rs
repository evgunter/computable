//! Pi computation using Machin's formula with provably correct bounds.
//!
//! This module implements pi as a Computable number using:
//! - Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239)
//! - Taylor series for arctan with rigorous error bounds
//! - InvOp (Newton-Raphson) for incremental reciprocal refinement
//!
//! ## Key Design Decisions
//!
//! 1. **InvOp-backed reciprocals**: Each Taylor term `1/((2i+1)*k^(2i+1))` is an
//!    InvOp node. Existing approximations are refined incrementally (precision doubles
//!    per N-R step) rather than recomputed from scratch.
//! 2. **Full Interval Propagation**: Every intermediate computation tracks [lo, hi] bounds
//! 3. **Error Bound Tracking**: Taylor truncation error is computed conservatively
//! 4. **Dynamic Precision**: Pi bounds can be refined to arbitrary precision

use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use parking_lot::RwLock;

use crate::binary::{
    Binary, Bounds, FiniteBounds, ReciprocalRounding, UBinary, UXBinary, XBinary,
    reciprocal_of_biguint,
};
use crate::binary_utils::bisection::normalize_finite_to_bounds;
use crate::computable::Computable;
use crate::error::ComputableError;
use crate::node::{BaseNode, Node, NodeOp, TypedBaseNode};
use crate::sane::{Sane, U, XI};

use super::BaseOp;
use super::InvOp;

/// Initial number of Taylor series terms for pi computation.
pub(crate) const INITIAL_PI_TERMS: U = 10;

/// Returns the number of bits in the binary representation of `x`.
///
/// Equivalent to floor(log2(x)) + 1 for x > 0, and 0 for x == 0.
///
/// Accepts and returns [`Sane`] so it composes naturally inside
/// [`sane_arithmetic!`] blocks.
fn bit_length(x: Sane) -> Sane {
    // leading_zeros() is always <= BITS, so this subtraction cannot underflow.
    // TODO: investigate whether the type system could prevent this case.
    let bits = U::BITS
        .checked_sub(x.0.leading_zeros())
        .unwrap_or_else(|| unreachable!("leading_zeros() is always <= U::BITS"));
    Sane(bits)
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
fn precision_bits_for_num_terms(num_terms: U) -> U {
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
/// use computable::{pi, XI};
///
/// let pi_val = pi();
/// let bounds = pi_val.refine_to_default(XI::from_i32(-50))?;
/// // bounds now contain pi with width ≤ 2^(-50)
/// # Ok::<(), computable::ComputableError>(())
/// ```
pub fn pi() -> Computable {
    let node = Node::new(Arc::new(PiOp::new(INITIAL_PI_TERMS)));
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
pub fn pi_bounds_at_precision(precision_bits: U) -> (Binary, Binary) {
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

/// Cached inputs and result for `compute_bounds`, avoiding redundant Machin
/// formula recomputation when `compute_bounds` is called multiple times at the
/// same `num_terms` (which happens between `refine_step` calls).
pub struct PiBoundsCache {
    num_terms: U,
    result: Bounds,
}

/// Pi computation operation using Machin's formula.
///
/// Internally stores InvOp nodes for each Taylor series term. These are
/// driven manually by `compute_prefix` (not as graph children, since the
/// term count is dynamic).
pub struct PiOp {
    state: RwLock<PiState>,
    /// Cache of the last `compute_bounds` result, keyed on `num_terms`.
    /// Eliminates redundant Taylor series recomputation during bound propagation.
    bounds_cache: RwLock<Option<PiBoundsCache>>,
}

struct PiState {
    num_terms: U,
    atan5_terms: Vec<Arc<Node>>,
    atan239_terms: Vec<Arc<Node>>,
    /// 5^(2*num_terms+1), ready for the next term or error bound
    k5_power: BigInt,
    /// 239^(2*num_terms+1), ready for the next term or error bound
    k239_power: BigInt,
}

impl PiOp {
    pub fn new(num_terms: U) -> Self {
        PiOp {
            state: RwLock::new(PiState::new(num_terms)),
            bounds_cache: RwLock::new(None),
        }
    }
}

impl PiState {
    fn new(n: U) -> Self {
        let k5 = BigInt::from(5_i64);
        let k239 = BigInt::from(239_i64);
        let k5_sq = &k5 * &k5;
        let k239_sq = &k239 * &k239;

        let mut atan5_terms = Vec::with_capacity(crate::sane::u_as_usize(n));
        let mut atan239_terms = Vec::with_capacity(crate::sane::u_as_usize(n));

        let mut k5_power = k5; // 5^1
        let mut k239_power = k239; // 239^1

        for i in 0..n {
            let coeff = BigInt::from(i) * 2_i64 + 1_i64;

            let denom5 = &coeff * &k5_power;
            atan5_terms.push(make_inv_node(denom5));

            let denom239 = &coeff * &k239_power;
            atan239_terms.push(make_inv_node(denom239));

            k5_power = &k5_power * &k5_sq;
            k239_power = &k239_power * &k239_sq;
        }

        PiState {
            num_terms: n,
            atan5_terms,
            atan239_terms,
            k5_power,
            k239_power,
        }
    }

    fn extend_to(&mut self, n: U) {
        if n <= self.num_terms {
            return;
        }

        let k5_sq = BigInt::from(25_i64);
        let k239_sq = BigInt::from(239_i64 * 239_i64);

        for i in self.num_terms..n {
            let coeff = BigInt::from(i) * 2_i64 + 1_i64;

            let denom5 = &coeff * &self.k5_power;
            self.atan5_terms.push(make_inv_node(denom5));

            let denom239 = &coeff * &self.k239_power;
            self.atan239_terms.push(make_inv_node(denom239));

            self.k5_power = &self.k5_power * &k5_sq;
            self.k239_power = &self.k239_power * &k239_sq;
        }

        self.num_terms = n;
    }
}

/// Creates an InvOp node that computes `1/denominator` using Newton-Raphson.
fn make_inv_node(denominator: BigInt) -> Arc<Node> {
    let binary = Binary::new(denominator, 0);
    let base: Arc<dyn BaseNode> = Arc::new(TypedBaseNode::new(
        binary,
        |v: &Binary| {
            Ok(Bounds::new(
                XBinary::Finite(v.clone()),
                XBinary::Finite(v.clone()),
            ))
        },
        |v: Binary| Ok(v),
    ));
    let constant_node = Node::new(Arc::new(BaseOp { base }));
    Node::new(Arc::new(InvOp {
        inner: constant_node,
        division_state: RwLock::new(None),
    }))
}

/// Drives an InvOp node until its prefix width is at most `2^(-target_width_bits)`.
///
/// `refine_precision` is passed to `refine_step` as the seed precision hint.
/// It must be large enough that the InvOp's initial reciprocal seed is nonzero
/// (i.e., larger than `log2(denominator)`).
fn refine_node_to_precision(
    node: &Arc<Node>,
    target_width_bits: U,
    refine_precision: U,
) -> Result<(), ComputableError> {
    const MAX_ITERS: U = 64;
    let exp = i32::try_from(target_width_bits)
        .unwrap_or(i32::MAX)
        .checked_neg()
        .unwrap_or(i32::MIN);
    let tolerance = UBinary::new(BigUint::from(1_u32), exp);
    for _ in 0..MAX_ITERS {
        let bounds = node.get_bounds()?;
        match bounds.width() {
            UXBinary::Finite(w) if *w <= tolerance => return Ok(()),
            UXBinary::Finite(_) | UXBinary::Inf => {}
        }
        let target_exp = XI::from_i32(
            i32::try_from(refine_precision)
                .map(|p| p.checked_neg().unwrap_or(i32::MIN))
                .unwrap_or(i32::MIN),
        );
        node.refine_step(target_exp)?;
        let new_bounds = node.op.compute_bounds()?;
        node.set_bounds(new_bounds);
    }
    Err(ComputableError::MaxRefinementIterations { max: MAX_ITERS })
}

/// Sums arctan Taylor terms from InvOp nodes, applying sign alternation and error bounds.
///
/// Each `terms[i]` is an InvOp node computing `1/((2i+1)*k^(2i+1))`.
/// `k_power` is `k^(2*num_terms+1)` for the truncation error bound.
fn sum_arctan_terms(
    terms: &[Arc<Node>],
    num_terms: U,
    precision_bits: U,
    k_power: &BigInt,
) -> Result<(Binary, Binary), ComputableError> {
    // The InvOp seed precision must exceed the denominator's bit length,
    // otherwise floor(2^p / denom) = 0 and N-R is stuck at zero.
    // k_power = k^(2*num_terms+1) bounds all term denominators.
    let k_power_bits = crate::sane::bits_as_u(k_power.magnitude().bits());
    let refine_precision = precision_bits.max(k_power_bits.saturating_add(10));

    let mut sum_lo = Binary::zero();
    let mut sum_hi = Binary::zero();

    for (i, node) in terms
        .iter()
        .take(crate::sane::u_as_usize(num_terms))
        .enumerate()
    {
        refine_node_to_precision(node, precision_bits, refine_precision)?;

        let bounds = node.get_bounds()?;
        let lo = match bounds.small() {
            XBinary::Finite(b) => b.clone(),
            XBinary::NegInf | XBinary::PosInf => return Err(ComputableError::InfiniteBounds),
        };
        let hi = match bounds.large() {
            XBinary::Finite(b) => b.clone(),
            XBinary::NegInf | XBinary::PosInf => return Err(ComputableError::InfiniteBounds),
        };

        if i % 2 == 0 {
            // Positive term: sum_lo += lo, sum_hi += hi
            sum_lo = sum_lo.add(&lo);
            sum_hi = sum_hi.add(&hi);
        } else {
            // Negative term: sum_lo -= hi, sum_hi -= lo
            sum_lo = sum_lo.sub(&hi);
            sum_hi = sum_hi.sub(&lo);
        }
    }

    // Taylor truncation error: 1 / ((2n+1) * k^(2n+1))
    let error_coeff = BigInt::from(crate::sane_arithmetic!(num_terms; 2 * num_terms + 1));
    let error_denom = &error_coeff * k_power;
    let error = reciprocal_of_biguint(
        error_denom.magnitude(),
        precision_bits,
        ReciprocalRounding::Ceil,
    );

    Ok((sum_lo.sub(&error), sum_hi.add(&error)))
}

impl NodeOp for PiOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let state = self.state.read();
        let num_terms = state.num_terms;

        // Check cache: if num_terms hasn't changed, return the cached result.
        {
            let cache = self.bounds_cache.read();
            if let Some(cached) = cache.as_ref()
                && cached.num_terms == num_terms
            {
                return Ok(cached.result.clone());
            }
        }

        let precision_bits = precision_bits_for_num_terms(num_terms);

        // Compute arctan(1/5) bounds via InvOp nodes
        let (atan5_lo, atan5_hi) = sum_arctan_terms(
            &state.atan5_terms,
            num_terms,
            precision_bits,
            &state.k5_power,
        )?;

        // Compute arctan(1/239) bounds via InvOp nodes
        let (atan239_lo, atan239_hi) = sum_arctan_terms(
            &state.atan239_terms,
            num_terms,
            precision_bits,
            &state.k239_power,
        )?;

        // pi = 16*arctan(1/5) - 4*arctan(1/239)
        let sixteen = Binary::new(BigInt::from(1_i32), 4); // 2^4 = 16
        let four = Binary::new(BigInt::from(1_i32), 2); // 2^2 = 4

        let term1_lo = atan5_lo.mul(&sixteen);
        let term1_hi = atan5_hi.mul(&sixteen);

        let term2_lo = atan239_lo.mul(&four);
        let term2_hi = atan239_hi.mul(&four);

        // Interval subtraction: [term1_lo, term1_hi] - [term2_lo, term2_hi]
        let pi_lo = term1_lo.sub(&term2_hi);
        let pi_hi = term1_hi.sub(&term2_lo);

        // Normalize to prevent precision accumulation
        let finite = FiniteBounds::new(pi_lo, pi_hi);
        let result = normalize_finite_to_bounds(&finite)?;

        // Store in cache for future calls with the same num_terms.
        {
            let mut cache = self.bounds_cache.write();
            *cache = Some(PiBoundsCache {
                num_terms,
                result: result.clone(),
            });
        }

        Ok(result)
    }

    fn refine_step(&self, target_width_exp: XI) -> Result<bool, ComputableError> {
        let mut state = self.state.write();

        // Compute the needed term count from the target width exponent.
        // XI::Finite(e) means width ≤ 2^e. For precision we need |e| bits
        // when e < 0, and 0 bits when e >= 0 (coarse target).
        let needed = match target_width_exp {
            XI::NegInf => {
                // Exact target: need infinite precision — double terms.
                state.num_terms.checked_mul(2).unwrap_or(U::MAX)
            }
            XI::Finite {
                sign: crate::sane::Sign::Neg,
                magnitude,
            } => {
                // Fine target: precision_bits = magnitude.
                let precision_bits = magnitude;
                crate::sane_arithmetic!(precision_bits; (precision_bits + 10) / 4).max(1)
            }
            XI::Finite { .. } | XI::PosInf => {
                // Coarse target (e >= 0) or unbounded: no precision needed.
                // needed=1 ensures at least INITIAL_PI_TERMS terms suffice.
                1
            }
        };

        if needed > state.num_terms {
            state.extend_to(needed);
            Ok(true)
        } else {
            // Already have enough terms — no further refinement needed.
            Ok(false)
        }
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }

    fn child_demand_budget(&self, _target_width: &UXBinary, _child_idx: bool) -> UXBinary {
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
fn compute_pi_bounds(num_terms: U, precision_bits: U) -> (Binary, Binary) {
    // Compute arctan(1/5) bounds
    let (atan_5_lo, atan_5_hi) = arctan_recip_bounds(5, num_terms, precision_bits);

    // Compute arctan(1/239) bounds
    let (atan_239_lo, atan_239_hi) = arctan_recip_bounds(239, num_terms, precision_bits);

    // pi = 16*arctan(1/5) - 4*arctan(1/239)
    //
    // For interval arithmetic subtraction: [a,b] - [c,d] = [a-d, b-c]
    // So: pi_lo = 16*atan_5_lo - 4*atan_239_hi
    //     pi_hi = 16*atan_5_hi - 4*atan_239_lo

    let sixteen = Binary::new_normalized(BigInt::from(1_i32), 4); // 2^4 = 16
    let four = Binary::new_normalized(BigInt::from(1_i32), 2); // 2^2 = 4

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
fn arctan_recip_bounds(k: u64, num_terms: U, precision_bits: U) -> (Binary, Binary) {
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
        // coeff = 2i+1 always fits in i64 (guarded by sane_arithmetic!), avoiding
        // the BigInt::from(i) * 2 + 1 allocation chain in the original code.
        let coeff_usize = crate::sane_arithmetic!(i; 2 * i + 1);
        let coeff_i64 = i64::from(coeff_usize);
        let denominator = &k_power * coeff_i64; // (2i+1) * k^(2i+1)

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
    let error_coeff_usize = crate::sane_arithmetic!(num_terms; 2 * num_terms + 1);
    let error_coeff_i64 = i64::from(error_coeff_usize);
    let error_denom = &k_power * error_coeff_i64;
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
    precision_bits: U,
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
pub fn pi_interval_at_precision(precision_bits: U) -> FiniteBounds {
    let (lo, hi) = pi_bounds_at_precision(precision_bits);
    FiniteBounds::new(lo, hi)
}

/// Returns 2*pi as a FiniteBounds interval with specified precision.
pub fn two_pi_interval_at_precision(precision_bits: U) -> FiniteBounds {
    use num_bigint::BigUint;

    let pi_interval = pi_interval_at_precision(precision_bits);
    // 2*pi: multiply by 2
    let two = UBinary::new(BigUint::from(1u32), 1); // 2^1 = 2
    pi_interval.scale_positive(&two)
}

/// Returns pi/2 as a FiniteBounds interval with specified precision.
pub fn half_pi_interval_at_precision(precision_bits: U) -> FiniteBounds {
    let (pi_lo, pi_hi) = pi_bounds_at_precision(precision_bits);
    // pi/2: divide by 2 (decrement exponent by 1)
    let half_pi_lo = Binary::new_normalized(
        pi_lo.mantissa().clone(),
        pi_lo.exponent().checked_sub(1).unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in half_pi")
        }),
    );
    let half_pi_hi = Binary::new_normalized(
        pi_hi.mantissa().clone(),
        pi_hi.exponent().checked_sub(1).unwrap_or_else(|| {
            crate::detected_computable_would_exhaust_memory!("exponent overflow in half_pi")
        }),
    );
    FiniteBounds::new(half_pi_lo, half_pi_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::XBinary;
    use crate::sane::{I, XI};

    fn bin(mantissa: i64, exponent: I) -> Binary {
        Binary::new(BigInt::from(mantissa), exponent)
    }

    fn epsilon_as_binary(n: U) -> Binary {
        let n_i = I::try_from(n).expect("n fits in I");
        Binary::new(
            BigInt::from(1_i32),
            n_i.checked_neg().expect("negation does not overflow"),
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
        let tolerance_exp = XI::from_i32(-20); // 2^-20 precision
        let bounds = pi_comp
            .refine_to_default(tolerance_exp)
            .expect("refine should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(bounds.large());
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
        let eps_binary = epsilon_as_binary(20);
        assert!(width <= eps_binary, "width should be <= epsilon");
    }

    #[test]
    fn pi_bounds_at_precision_helper() {
        const PRECISION_BITS: U = 50;
        let (lo, hi) = pi_bounds_at_precision(PRECISION_BITS);
        let width = hi.sub(&lo);

        let precision_i = I::try_from(PRECISION_BITS).expect("PRECISION_BITS fits in I");
        let exp = precision_i
            .checked_sub(1)
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
        let tolerance_exp = XI::from_i32(-128); // 2^-128 precision
        let bounds = pi_comp
            .refine_to_default(tolerance_exp)
            .expect("refine to 2^-128 should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(bounds.large());

        let width = upper.sub(&lower);
        let eps_binary = epsilon_as_binary(128);
        assert!(
            width <= eps_binary,
            "width should be <= 2^-128, got width = {}",
            width
        );
    }
}
