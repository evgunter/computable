//! The main Computable type representing computable real numbers.
//!
//! A `Computable` is a real number that can be refined to arbitrary precision.
//! It is backed by a computation graph where leaf nodes contain user-defined
//! state and refinement logic, and interior nodes represent arithmetic operations.

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::binary::{Binary, UBinary, XBinary};
use crate::error::ComputableError;
use crate::node::{BaseNode, Node, TypedBaseNode};
use crate::ops::{AddOp, BaseOp, InvOp, MulOp, NegOp, NthRootOp, PowOp, SinOp};
use crate::binary::Bounds;
use crate::refinement::{bounds_width_leq, RefinementGraph};

use parking_lot::RwLock;

#[cfg(debug_assertions)]
pub const DEFAULT_INV_MAX_REFINES: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_INV_MAX_REFINES: usize = 4096;

#[cfg(debug_assertions)]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 4096;

/// A computable number backed by a shared node graph.
#[derive(Clone)]
pub struct Computable {
    pub(crate) node: Arc<Node>,
}

impl Computable {
    /// Creates a new computable from user-defined state and refinement logic.
    ///
    /// # Arguments
    /// * `state` - Initial state for this computable
    /// * `bounds` - Function to compute bounds from the current state
    /// * `refine` - Function to refine the state to a more precise version
    pub fn new<X, B, F>(state: X, bounds: B, refine: F) -> Self
    where
        X: Eq + Clone + Send + Sync + 'static,
        B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
        F: Fn(X) -> X + Send + Sync + 'static,
    {
        let base_node_struct = TypedBaseNode::new(state, bounds, refine);
        let base_node: Arc<dyn BaseNode> = Arc::new(base_node_struct);
        let node = Node::new(Arc::new(BaseOp { base: base_node }));
        Self { node }
    }

    /// Returns the current bounds for this computable.
    pub fn bounds(&self) -> Result<Bounds, ComputableError> {
        self.node.get_bounds()
    }

    /// Refines this computable until the bounds width is at most epsilon.
    ///
    /// # Arguments
    /// * `epsilon` - Maximum width for the returned bounds
    ///
    /// # Type Parameters
    /// * `MAX_REFINEMENT_ITERATIONS` - Maximum number of refinement iterations
    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        &self,
        epsilon: UBinary,
    ) -> Result<Bounds, ComputableError> {
        // TODO: it may be desirable to allow epsilon = 0, but probably only after we implement automatic checking of short-prefix bounds
        // (e.g. as-is, sqrt(4) may never refine to a width of 0 because we just use binary search)
        if epsilon.mantissa().is_zero() {
            return Err(ComputableError::NonpositiveEpsilon);
        }

        loop {
            let bounds = self.node.get_bounds()?;
            if bounds_width_leq(&bounds, &epsilon) {
                return Ok(bounds);
            }

            let mut state_guard = self.node.refinement.state.lock();
            if !state_guard.active {
                state_guard.active = true;
                drop(state_guard);

                let graph = RefinementGraph::new(Arc::clone(&self.node))?;
                let result = graph.refine_to::<MAX_REFINEMENT_ITERATIONS>(&epsilon);

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
    pub fn refine_to_default(&self, epsilon: UBinary) -> Result<Bounds, ComputableError> {
        self.refine_to::<DEFAULT_MAX_REFINEMENT_ITERATIONS>(epsilon)
    }

    /// Returns the multiplicative inverse of this computable.
    pub fn inv(self) -> Self {
        let node = Node::new(Arc::new(InvOp {
            inner: Arc::clone(&self.node),
            precision_bits: RwLock::new(BigInt::zero()),
        }));
        Self { node }
    }

    /// Computes the sine of this computable number.
    ///
    /// Uses Taylor series with provably correct error bounds.
    /// The error bound |x|^(2n+1)/(2n+1)! is computed conservatively (rounded up)
    /// to ensure the true value is always contained within the returned bounds.
    ///
    /// TODO: Ideally, we'd round differently for lower vs upper bounds in the sum
    /// computation (round down for lower, round up for upper). Currently we round up
    /// for the error bound which covers intermediate rounding, but directed rounding
    /// throughout would be more precise.
    pub fn sin(self) -> Self {
        let node = Node::new(Arc::new(SinOp {
            inner: Arc::clone(&self.node),
            num_terms: RwLock::new(BigInt::one()),
        }));
        Self { node }
    }

    /// Computes the n-th root of this computable number.
    ///
    /// Uses binary search (bisection) for guaranteed convergence with provably
    /// correct bounds. For each refinement step, the interval is halved.
    ///
    /// # Arguments
    /// * `degree` - The root degree (n in x^(1/n)). Must be >= 1.
    ///
    /// # Constraints
    /// - For even degrees (2, 4, 6, ...): requires non-negative input
    /// - For odd degrees (3, 5, 7, ...): supports all real inputs
    ///
    /// # Examples
    /// - `nth_root(2)` computes the square root
    /// - `nth_root(3)` computes the cube root
    /// - `nth_root(4)` computes the fourth root
    // TODO: Refactor to remove this assert. Options include returning self for degree=0
    // (since x^(1/0) is undefined, but we could define it as identity or error gracefully).
    pub fn nth_root(self, degree: u32) -> Self {
        assert!(degree >= 1, "Root degree must be at least 1");
        let node = Node::new(Arc::new(NthRootOp {
            inner: Arc::clone(&self.node),
            degree,
            bisection_state: RwLock::new(None),
        }));
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
        if exponent == 0 {
            // x^0 = 1 for all x, including 0^0 = 1 by convention
            return Computable::constant(Binary::new(
                num_bigint::BigInt::from(1),
                num_bigint::BigInt::from(0),
            ));
        }
        let node = Node::new(Arc::new(PowOp {
            inner: Arc::clone(&self.node),
            exponent,
        }));
        Self { node }
    }

    /// Creates a constant computable with exact bounds.
    #[allow(clippy::type_complexity)]
    pub fn constant(value: Binary) -> Self {
        fn bounds(value: &Binary) -> Result<Bounds, ComputableError> {
            Ok(Bounds::new(
                XBinary::Finite(value.clone()),
                XBinary::Finite(value.clone()),
            ))
        }

        fn refine(value: Binary) -> Binary {
            value
        }

        Computable::new(value, bounds, refine)
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
        let node = Node::new(Arc::new(NegOp {
            inner: Arc::clone(&self.node),
        }));
        Self { node }
    }
}

impl std::ops::Add for Computable {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::new(Arc::new(AddOp {
            left: Arc::clone(&self.node),
            right: Arc::clone(&rhs.node),
        }));
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
        let node = Node::new(Arc::new(MulOp {
            left: Arc::clone(&self.node),
            right: Arc::clone(&rhs.node),
        }));
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
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::test_utils::{bin, ubin, unwrap_finite};

    fn sqrt_computable(value_int: u64) -> Computable {
        Computable::constant(bin(value_int as i64, 0)).nth_root(2)
    }

    #[test]
    fn from_binary_matches_constant_bounds() {
        let value = bin(3, 0);
        let computable: Computable = value.clone().into();

        let bounds = computable.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            Bounds::new(
                XBinary::Finite(value.clone()),
                XBinary::Finite(value)
            )
        );
    }

    #[test]
    fn integration_sqrt2_expression() {
        let one = Computable::constant(bin(1, 0));
        let sqrt2 = sqrt_computable(2);
        let expr = (sqrt2.clone() + one.clone()) * (sqrt2.clone() - one) + sqrt2.inv();

        let epsilon = ubin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = bounds.large();
        let upper = unwrap_finite(&upper);
        let expected = 1.0_f64 + 2.0_f64.sqrt().recip();
        let expected_binary = XBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon.to_binary();

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }

    #[test]
    fn shared_operand_in_expression() {
        let shared = sqrt_computable(2);
        let expr = shared.clone() + shared * Computable::constant(bin(1, 0));

        let epsilon = ubin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = bounds.large();
        let upper = unwrap_finite(&upper);
        let expected = 2.0_f64 * 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon.to_binary();

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }
}
