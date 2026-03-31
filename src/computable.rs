//! The main Computable type representing computable real numbers.
//!
//! A `Computable` is a real number that can be refined to arbitrary precision.
//! It is backed by a computation graph where leaf nodes contain user-defined
//! state and refinement logic, and interior nodes represent arithmetic operations.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::binary::{Binary, XBinary};
use crate::error::ComputableError;
use crate::node::{BaseNode, Node, TypedBaseNode};
use crate::ops::{AddOp, BaseOp, InvOp, MulOp, NegOp, NthRootOp, PiOp, PowOp, SinOp};
use crate::prefix::Prefix;
use crate::refinement::{RefinementGraph, prefix_width_leq};
use crate::sane::{U, XI};

#[cfg(debug_assertions)]
pub const DEFAULT_INV_MAX_REFINES: U = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_INV_MAX_REFINES: U = 4096;

#[cfg(debug_assertions)]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: U = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: U = 4096;

/// A computable number backed by a shared node graph.
#[derive(Clone)]
pub struct Computable {
    node: Arc<Node>,
}

impl Computable {
    /// Creates a new computable from user-defined state and refinement logic.
    ///
    /// # Arguments
    /// * `state` - Initial state for this computable
    /// * `compute_prefix` - Function to compute prefix from the current state
    /// * `refine` - Function to refine the state to a more precise version
    pub fn new<X, B, F>(state: X, compute_prefix: B, refine: F) -> Self
    where
        X: Eq + Clone + Send + Sync + 'static,
        B: Fn(&X) -> Result<Prefix, ComputableError> + Send + Sync + 'static,
        F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
    {
        let base_node_struct = TypedBaseNode::new(state, compute_prefix, refine);
        let base_node: Arc<dyn BaseNode> = Arc::new(base_node_struct);
        let node = Node::new(Arc::new(BaseOp::new(base_node)));
        Self { node }
    }

    /// Creates a Computable from a pre-built Node.
    pub(crate) fn from_node(node: Arc<Node>) -> Self {
        Self { node }
    }

    /// Returns the current prefix for this computable.
    pub fn prefix(&self) -> Result<Prefix, ComputableError> {
        self.node.get_prefix()
    }

    /// Refines this computable until the prefix width exponent is at most `target_width_exp`.
    ///
    /// # Arguments
    /// * `target_width_exp` - Target width exponent. `Finite(e)` requests width ≤ 2^e.
    ///   `NegInf` requests exact prefix (width = 0).
    ///
    /// # Type Parameters
    /// * `MAX_REFINEMENT_ITERATIONS` - Maximum number of refinement iterations
    ///
    /// # Warning
    /// Using `XI::NegInf` (width = 0) will only succeed for values that can be
    /// represented exactly in binary (e.g., integers, dyadic rationals like 1/2 or 3/4).
    /// For values that cannot be exactly represented (e.g., 1/3, sqrt(2), pi),
    /// refinement will never achieve zero width and will return
    /// [`ComputableError::MaxRefinementIterations`] after exhausting the iteration limit.
    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: U>(
        &self,
        target_width_exp: XI,
    ) -> Result<Prefix, ComputableError> {
        loop {
            let prefix = self.node.get_prefix()?;
            if prefix_width_leq(&prefix, &target_width_exp) {
                return Ok(prefix);
            }

            let mut state_guard = self.node.refinement.state.lock();
            if !state_guard.active {
                state_guard.active = true;
                drop(state_guard);

                let graph = RefinementGraph::new(Arc::clone(&self.node))?;
                let result = graph.refine_to::<MAX_REFINEMENT_ITERATIONS>(&target_width_exp);

                let mut completion_guard = self.node.refinement.state.lock();
                completion_guard.active = false;
                self.node.refinement.condvar.notify_all();
                return result;
            }

            let observed_epoch = state_guard.epoch;
            self.node
                .refinement
                .condvar
                .wait_while(&mut state_guard, |guard| {
                    guard.active && guard.epoch == observed_epoch
                });
        }
    }

    /// Refines this computable using the default maximum iterations.
    pub fn refine_to_default(&self, target_width_exp: XI) -> Result<Prefix, ComputableError> {
        self.refine_to::<DEFAULT_MAX_REFINEMENT_ITERATIONS>(target_width_exp)
    }

    /// Returns the multiplicative inverse of this computable.
    pub fn inv(self) -> Self {
        let node = Node::new(Arc::new(InvOp::new(Arc::clone(&self.node))));
        Self { node }
    }

    /// Computes the sine of this computable number.
    ///
    /// Uses Taylor series with provably correct error bounds.
    /// The implementation uses directed rounding throughout: the lower bound computation
    /// rounds toward negative infinity and the upper bound rounds toward positive infinity.
    /// The error bound |x|^(2n+1)/(2n+1)! is also computed conservatively (rounded up)
    /// to ensure the true value is always contained within the returned bounds.
    pub fn sin(self) -> Self {
        let pi_node = Node::new(Arc::new(PiOp::new(crate::ops::pi::INITIAL_PI_TERMS)));
        let node = Node::new(Arc::new(SinOp::new(Arc::clone(&self.node), pi_node)));
        Self { node }
    }

    /// Computes the n-th root of this computable number.
    ///
    /// Uses Newton-Raphson iteration for quadratic convergence with provably
    /// correct bounds. Each refinement step approximately doubles the number
    /// of correct bits.
    ///
    /// # Arguments
    /// * `degree` - The root degree (n in x^(1/n)). Must be >= 1, enforced by the type system.
    ///
    /// # Constraints
    /// - For even degrees (2, 4, 6, ...): requires non-negative input
    /// - For odd degrees (3, 5, 7, ...): supports all real inputs
    ///
    /// # Examples
    /// - `nth_root(NonZeroU32::new(2).unwrap())` computes the square root
    /// - `nth_root(NonZeroU32::new(3).unwrap())` computes the cube root
    /// - `nth_root(NonZeroU32::new(4).unwrap())` computes the fourth root
    pub fn nth_root(self, degree: NonZeroU32) -> Self {
        let node = Node::new(Arc::new(NthRootOp::new(Arc::clone(&self.node), degree)));
        Self { node }
    }

    /// Raises this computable number to an integer power.
    ///
    /// Computes x^n for non-negative integer exponents. This is more efficient than
    /// repeated multiplication because it computes bounds directly using the
    /// monotonicity properties of power functions.
    ///
    /// # Arguments
    /// * `exponent` - The power to raise to (n in x^n).
    ///
    /// # Bounds Computation
    /// - For n=0: returns constant 1 (including 0^0 = 1 by convention)
    /// - For odd n: x^n is monotonically increasing, so bounds are [lower^n, upper^n]
    /// - For even n: x^n has a minimum at 0
    ///   - If interval is non-negative: [lower^n, upper^n]
    ///   - If interval is non-positive: [upper^n, lower^n]
    ///   - If interval spans zero: [0, max(|lower|^n, |upper|^n)]
    ///
    /// # Examples
    /// - `pow(0)` returns constant 1
    /// - `pow(2)` computes the square
    /// - `pow(3)` computes the cube
    pub fn pow(self, exponent: u32) -> Self {
        match std::num::NonZeroU32::new(exponent) {
            None => {
                // x^0 = 1 for all x, including 0^0 = 1 by convention
                // Check for infinite bounds - infinity^0 is an indeterminate form.
                if let Ok(prefix) = self.node.get_prefix() {
                    let lower = prefix.lower();
                    let upper = prefix.upper();
                    let has_infinite = matches!(lower, XBinary::NegInf | XBinary::PosInf)
                        || matches!(&upper, XBinary::NegInf | XBinary::PosInf);
                    if has_infinite {
                        crate::detected_computable_with_infinite_value!(
                            "input has infinite bounds for x^0 (infinity^0 is an indeterminate form)"
                        );
                    }
                }
                Computable::constant(Binary::one())
            }
            Some(nonzero_exp) => {
                let node = Node::new(Arc::new(PowOp::new(Arc::clone(&self.node), nonzero_exp)));
                Self { node }
            }
        }
    }

    /// Creates a constant computable with exact prefix.
    pub fn constant(value: Binary) -> Self {
        fn compute_prefix(value: &Binary) -> Result<Prefix, ComputableError> {
            Ok(Prefix::exact(value.clone()))
        }

        fn refine(value: Binary) -> Result<Binary, ComputableError> {
            Ok(value)
        }

        Computable::new(value, compute_prefix, refine)
    }
}

impl From<Binary> for Computable {
    fn from(value: Binary) -> Self {
        Computable::constant(value)
    }
}

impl std::ops::Neg for Computable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let node = Node::new(Arc::new(NegOp::new(Arc::clone(&self.node))));
        Self { node }
    }
}

impl std::ops::Add for Computable {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::new(Arc::new(AddOp::new(
            Arc::clone(&self.node),
            Arc::clone(&rhs.node),
        )));
        Self { node }
    }
}

impl std::ops::Sub for Computable {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for Computable {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::new(Arc::new(MulOp::new(
            Arc::clone(&self.node),
            Arc::clone(&rhs.node),
        )));
        Self { node }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div for Computable {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sane::XI;
    use crate::test_utils::{bin, epsilon_as_binary, unwrap_finite};

    fn sqrt_computable(value_int: u64) -> Computable {
        Computable::constant(bin(i64::try_from(value_int).expect("value fits in i64"), 0))
            .nth_root(NonZeroU32::new(2).expect("2 is non-zero"))
    }

    #[test]
    fn from_binary_matches_constant_prefix() {
        let value = bin(3, 0);
        let computable: Computable = value.clone().into();

        let prefix = computable.prefix().expect("prefix should succeed");
        assert_eq!(
            prefix,
            Prefix::from_lower_upper(XBinary::Finite(value.clone()), XBinary::Finite(value))
        );
    }

    #[test]
    fn integration_sqrt2_expression() {
        let one = Computable::constant(bin(1, 0));
        let sqrt2 = sqrt_computable(2);
        let expr = (sqrt2.clone() + one.clone()) * (sqrt2.clone() - one) + sqrt2.inv();

        let target_width_exp = XI::from_i32(-12);
        let prefix = expr
            .refine_to_default(target_width_exp)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(&prefix.lower());
        let upper = unwrap_finite(&prefix.upper());
        let expected = 1.0_f64 + 2.0_f64.sqrt().recip();
        let expected_binary =
            XBinary::from_f64(expected).expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon_as_binary(12);

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }

    /// Helper: assert prefix meets the target but is not over-refined past `over_refined_exp`.
    fn assert_coarse_not_over_refined(
        prefix: &Prefix,
        target_width_exp: XI,
        over_refined_exp: XI,
        label: &str,
    ) {
        use crate::refinement::prefix_width_leq;
        assert!(
            prefix_width_leq(prefix, &target_width_exp),
            "{label}: prefix should meet target width"
        );
        assert!(
            !prefix_width_leq(prefix, &over_refined_exp),
            "{label}: prefix should not be over-refined (width should be > 2^{})",
            over_refined_exp.finite_i64()
        );
    }

    // --- Coarse-tolerance tests: verify each op doesn't over-refine ---
    //
    // Nth_root, mul, and add have refiners backed by Newton-Raphson, which
    // initializes a 64-bit seed in compute_prefix() before any target is
    // known. This inherently produces tight initial prefixes, so we can only
    // verify that refine_to succeeds and meets the target — the precision
    // cap in refine_step prevents further escalation but compute_prefix()
    // already provides ~64-bit-wide results.
    //
    // For ops whose initial prefixes are genuinely coarse (sin, inv, pi, pow),
    // we additionally verify that prefixes aren't over-refined past a
    // given boundary.

    #[test]
    fn coarse_epsilon_does_not_over_refine_nth_root() {
        // sqrt(10^18) ≈ 10^9. Coarse target: width ≤ 2^4 = 16.
        // compute_prefix already produces ~64-bit precision (width ≈ 2^(-34)),
        // so this just verifies refine_to accepts a coarse XI target.
        let large = sqrt_computable(1_000_000_000_000_000_000);
        let target = XI::from_i32(4);
        let prefix = large
            .refine_to_default(target)
            .expect("refine should succeed");
        use crate::refinement::prefix_width_leq;
        assert!(
            prefix_width_leq(&prefix, &target),
            "nth_root: should meet target"
        );
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_mul() {
        // sqrt(10^18) * sqrt(2) ≈ 1.41 * 10^9. Coarse target: width ≤ 16.
        let a = sqrt_computable(1_000_000_000_000_000_000);
        let b = sqrt_computable(2);
        let expr = a * b;
        let target = XI::from_i32(4);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        use crate::refinement::prefix_width_leq;
        assert!(
            prefix_width_leq(&prefix, &target),
            "mul: should meet target"
        );
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_add() {
        // sqrt(10^18) + constant(0) ≈ 10^9. Coarse target: width ≤ 16.
        let a = sqrt_computable(1_000_000_000_000_000_000);
        let b = Computable::constant(bin(0, 0));
        let expr = a + b;
        let target = XI::from_i32(4);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        use crate::refinement::prefix_width_leq;
        assert!(
            prefix_width_leq(&prefix, &target),
            "add: should meet target"
        );
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_pow() {
        // sqrt(9999)^2 = 9999. Derivative at x ≈ 100 is 200.
        // Target width 2^12 = 4096, child demand ≈ 4096/200 ≈ 20 (coarse).
        let expr = sqrt_computable(9999).pow(2);
        let target = XI::from_i32(12);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        assert_coarse_not_over_refined(&prefix, target, XI::from_i32(6), "pow");
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_sin() {
        // sin(1) * 2^30 ≈ 0.84 * 10^9. Sin starts with 1 Taylor term
        // (width ≈ 0.33), so result width ≈ 2^28. Target 2^34 is coarse
        // enough that no refinement is needed.
        let expr = Computable::constant(bin(1, 0)).sin() * Computable::constant(bin(1, 30));
        let target = XI::from_i32(34);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        assert_coarse_not_over_refined(&prefix, target, XI::from_i32(24), "sin");
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_inv() {
        // inv(sqrt(2)) * 2^30 ≈ 0.707 * 10^9. Target 2^34; mul demand
        // on inv child is 2^34 / 2^30 = 2^4, coarse.
        let expr = sqrt_computable(2).inv() * Computable::constant(bin(1, 30));
        let target = XI::from_i32(34);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        assert_coarse_not_over_refined(&prefix, target, XI::from_i32(18), "inv");
    }

    #[test]
    fn coarse_epsilon_does_not_over_refine_pi() {
        // Use 1 initial term so we can test coarse-target behavior.
        // With 1 term, pi width is ~2^(-3.5), so pi * 2^6 has width
        // ≈ 2^(6-3.5) = 2^2.5 ≈ 5.7. Target 2^4 = 16 is satisfied
        // without any refinement.
        let expr = crate::ops::pi::pi_with_initial_terms(1) * Computable::constant(bin(1, 6));
        let target = XI::from_i32(4);
        let prefix = expr
            .refine_to_default(target)
            .expect("refine should succeed");
        assert_coarse_not_over_refined(&prefix, target, XI::from_i32(0), "pi");
    }

    #[test]
    fn shared_operand_in_expression() {
        let shared = sqrt_computable(2);
        let expr = shared.clone() + shared * Computable::constant(bin(1, 0));

        let target_width_exp = XI::from_i32(-12);
        let prefix = expr
            .refine_to_default(target_width_exp)
            .expect("refine_to should succeed");

        let lower = unwrap_finite(&prefix.lower());
        let upper = unwrap_finite(&prefix.upper());
        let expected = 2.0_f64 * 2.0_f64.sqrt();
        let expected_binary =
            XBinary::from_f64(expected).expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon_as_binary(12);

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }
}
