#![warn(
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used
)]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crossbeam_channel::{Receiver, Sender, unbounded};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use parking_lot::{Condvar, Mutex, RwLock};

mod binary;
mod concurrency;
mod ordered_pair;

pub use binary::{Binary, BinaryError, UBinary, UXBinary, XBinary, XBinaryError};
use binary::{reciprocal_rounded_abs_extended, ReciprocalRounding};
use concurrency::StopFlag;
pub use ordered_pair::{Bounds, Interval, IntervalError};

/// Shared API for retrieving bounds with lazy computation.
trait BoundsAccess {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputableError {
    NonpositiveEpsilon,
    InvalidBoundsOrder,
    BoundsWorsened,
    StateUnchanged,
    ExcludedValueUnreachable,
    RefinementChannelClosed,
    MaxRefinementIterations { max: usize },
    Binary(BinaryError),
}

impl fmt::Display for ComputableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonpositiveEpsilon => write!(f, "epsilon must be positive"),
            Self::InvalidBoundsOrder => write!(f, "computed bounds are not ordered"),
            Self::BoundsWorsened => write!(f, "refinement produced worse bounds"),
            Self::StateUnchanged => write!(f, "refinement did not change state"),
            Self::ExcludedValueUnreachable => write!(f, "cannot refine bounds to exclude value"),
            Self::RefinementChannelClosed => {
                write!(f, "refinement coordination channel closed")
            }
            Self::MaxRefinementIterations { max } => {
                write!(f, "maximum refinement iterations ({max}) reached")
            }
            Self::Binary(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ComputableError {}

impl From<BinaryError> for ComputableError {
    fn from(error: BinaryError) -> Self {
        Self::Binary(error)
    }
}

/// Type-erased base node so we can store heterogeneous leaf states in a single graph.
/// This is also the hook for future user-defined base nodes.
trait BaseNode: Send + Sync {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
    fn refine(&self) -> Result<(), ComputableError>;
}

/// Concrete base node that owns the user-provided state and refinement logic.
struct TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> X + Send + Sync + 'static,
{
    /// Snapshot ties a particular state with its computed bounds to avoid recomputation.
    snapshot: RwLock<BaseSnapshot<X>>,
    bounds: B,
    refine: F,
}

/// Cached base state plus bounds derived from that state.
#[derive(Clone)]
struct BaseSnapshot<X> {
    state: X,
    bounds: Option<Bounds>,
}

impl<X, B, F> TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> X + Send + Sync + 'static,
{
    fn new(state: X, bounds: B, refine: F) -> Self {
        Self {
            snapshot: RwLock::new(BaseSnapshot {
                state,
                bounds: None,
            }),
            bounds,
            refine,
        }
    }

    fn snapshot_bounds(&self, snapshot: &mut BaseSnapshot<X>) -> Result<Bounds, ComputableError> {
        if let Some(bounds) = &snapshot.bounds {
            return Ok(bounds.clone());
        }
        let bounds = (self.bounds)(&snapshot.state)?;
        snapshot.bounds = Some(bounds.clone());
        Ok(bounds)
    }
}

impl<X, B, F> BaseNode for TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> X + Send + Sync + 'static,
{
    /// Returns cached bounds for the current base state, computing and caching if needed.
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        let mut snapshot = self.snapshot.write();
        let bounds = self.snapshot_bounds(&mut snapshot)?;
        Ok(bounds)
    }

    /// Refines the base state and computes the new bounds for that refined state.
    fn refine(&self) -> Result<(), ComputableError> {
        let mut snapshot = self.snapshot.write();
        let previous_bounds = self.snapshot_bounds(&mut snapshot)?;
        let previous_state = snapshot.state.clone();
        let next_state = (self.refine)(previous_state.clone());
        if next_state == previous_state {
            if previous_bounds.small() == &previous_bounds.large() {
                return Ok(());
            }
            return Err(ComputableError::StateUnchanged);
        }

        let next_bounds = (self.bounds)(&next_state)?;
        let lower_worsened = next_bounds.small() < previous_bounds.small();
        let upper_worsened = next_bounds.large() > previous_bounds.large();
        if lower_worsened || upper_worsened {
            return Err(ComputableError::BoundsWorsened);
        }

        snapshot.state = next_state;
        snapshot.bounds = Some(next_bounds);

        Ok(())
    }
}

impl<T: BaseNode + ?Sized> BoundsAccess for T {
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        BaseNode::get_bounds(self)
    }
}

/// A computable number backed by a shared node graph.
#[derive(Clone)]
pub struct Computable {
    node: Arc<Node>,
}

impl Computable {
    pub fn new<X, B, F>(state: X, bounds: B, refine: F) -> Self
    where
        X: Eq + Clone + Send + Sync + 'static,
        B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
        F: Fn(X) -> X + Send + Sync + 'static,
    {
        let base_node_struct = TypedBaseNode::new(state, bounds, refine);
        let base_node = Arc::new(base_node_struct);
        let node = Node::new(Arc::new(BaseOp { base: base_node }));
        Self { node }
    }

    pub fn bounds(&self) -> Result<Bounds, ComputableError> {
        self.node.get_bounds()
    }

    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        &self,
        epsilon: UBinary,
    ) -> Result<Bounds, ComputableError> {
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

    pub fn refine_to_default(&self, epsilon: UBinary) -> Result<Bounds, ComputableError> {
        self.refine_to::<DEFAULT_MAX_REFINEMENT_ITERATIONS>(epsilon)
    }

    pub fn inv(self) -> Self {
        let node = Node::new(Arc::new(InvOp {
            inner: Arc::clone(&self.node),
            precision_bits: RwLock::new(BigInt::zero()),
        }));
        Self { node }
    }

    /// Computes the sine of this computable number.
    ///
    /// Uses argument reduction to bring the input into [-pi/4, pi/4],
    /// then applies a polynomial approximation for high precision.
    pub fn sin(self) -> Self {
        let node = Node::new(Arc::new(SinOp::new(Arc::clone(&self.node))));
        Self { node }
    }

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

/// Node operator for composed computables.
/// TODO: enable user-defined composed nodes (e.g. sine) via this trait.
trait NodeOp: Send + Sync {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError>;
    fn refine_step(&self) -> Result<bool, ComputableError>;
    fn children(&self) -> Vec<Arc<Node>>;
    fn is_refiner(&self) -> bool;
}

struct BaseOp {
    base: Arc<dyn BaseNode>,
}

impl NodeOp for BaseOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        BoundsAccess::get_bounds(self.base.as_ref())
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        self.base.refine()?;
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        Vec::new()
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

struct NegOp {
    inner: Arc<Node>,
}

impl NodeOp for NegOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        let lower = existing.small().neg();
        let upper = existing.large().neg();
        Bounds::new_checked(upper, lower).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        false
    }
}

struct AddOp {
    left: Arc<Node>,
    right: Arc<Node>,
}

impl NodeOp for AddOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let left_bounds = self.left.get_bounds()?;
        let right_bounds = self.right.get_bounds()?;
        let lower = left_bounds.small().add_lower(right_bounds.small());
        let upper = left_bounds.large().add_upper(&right_bounds.large());
        Bounds::new_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }
}

struct MulOp {
    left: Arc<Node>,
    right: Arc<Node>,
}

impl NodeOp for MulOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let left_bounds = self.left.get_bounds()?;
        let right_bounds = self.right.get_bounds()?;
        let left_lower = left_bounds.small();
        let left_upper = left_bounds.large();
        let right_lower = right_bounds.small();
        let right_upper = right_bounds.large();

        let candidates = [
            left_lower.mul(right_lower),
            left_lower.mul(&right_upper),
            left_upper.mul(right_lower),
            left_upper.mul(&right_upper),
        ];

        let mut min = candidates[0].clone();
        let mut max = candidates[0].clone();
        for candidate in candidates.iter().skip(1) {
            if candidate < &min {
                min = candidate.clone();
            }
            if candidate > &max {
                max = candidate.clone();
            }
        }

        Bounds::new_checked(min, max).map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        Ok(false)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }

    fn is_refiner(&self) -> bool {
        false
    }
}

// TODO: Improve inv() precision strategy. Currently precision_bits starts at 0 and
// increments by 1 on each refine_step. This is simple but potentially inefficient:
// - For a given epsilon, we don't know how many bits are needed upfront
// - Each step recomputes the reciprocal from scratch at the new precision
// Consider: adaptive precision based on current bounds width, or Newton-Raphson iteration.
struct InvOp {
    inner: Arc<Node>,
    precision_bits: RwLock<BigInt>,
}

impl NodeOp for InvOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        reciprocal_bounds(&existing, &self.precision_bits.read())
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut precision = self.precision_bits.write();
        *precision += BigInt::one();
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Computes PI to the specified number of bits using the Machin formula:
/// pi/4 = 4*arctan(1/5) - arctan(1/239)
/// arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
fn compute_pi(precision_bits: &BigInt) -> Binary {
    use num_traits::ToPrimitive;

    // We need extra bits for intermediate computations to avoid accumulated error
    let extra_bits = 32i64;
    let total_bits = precision_bits.to_i64().unwrap_or(64) + extra_bits;

    // Compute arctan(1/5) * 4 and arctan(1/239) using scaled integers
    let arctan_1_5 = scaled_arctan(&BigInt::from(5), total_bits);
    let arctan_1_239 = scaled_arctan(&BigInt::from(239), total_bits);

    // pi/4 = 4*arctan(1/5) - arctan(1/239)
    // pi = 16*arctan(1/5) - 4*arctan(1/239)
    let pi_scaled = BigInt::from(16) * arctan_1_5 - BigInt::from(4) * arctan_1_239;

    Binary::new(pi_scaled, BigInt::from(-total_bits))
}

/// Computes arctan(1/x) scaled by 2^bits using the Taylor series.
fn scaled_arctan(x: &BigInt, bits: i64) -> BigInt {
    let scale = BigInt::from(1) << (bits as usize);
    let x_sq = x * x;
    let mut result = BigInt::zero();
    let mut term = &scale / x; // First term: (1/x) * scale
    let mut k = 1i64;

    loop {
        if k % 2 == 1 {
            result += &term;
        } else {
            result -= &term;
        }

        // Next term: term * (-1)^(k+1) / (2k+1) / x^2
        // Since we're computing arctan(1/x), each term is divided by x^2 and (2k+1)
        term = &term / &x_sq;
        term = &term / BigInt::from(2 * k + 1);
        term = &term * BigInt::from(2 * k - 1);

        k += 1;

        // Stop when term contribution is negligible
        if term.magnitude().bits() == 0 || k > bits {
            break;
        }
    }

    result
}

/// Computes sin(x) using polynomial approximation for x in [-pi/4, pi/4].
/// Uses the Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
/// Returns bounds [lower, upper] for the result.
fn sin_polynomial(x_lower: &Binary, x_upper: &Binary, precision_bits: &BigInt) -> (Binary, Binary) {
    use num_traits::ToPrimitive;

    let bits = precision_bits.to_i64().unwrap_or(64).max(16);

    // For the polynomial approximation, we compute using the midpoint
    // and then add error bounds based on the input interval width and truncation error

    // Compute sin at both endpoints to get bounds (sin is monotonic on [-pi/4, pi/4])
    let sin_lower = sin_taylor(x_lower, bits);
    let sin_upper = sin_taylor(x_upper, bits);

    // sin is increasing on [-pi/4, pi/4], so:
    // - if x_lower <= x_upper, sin(x_lower) <= sin(x_upper)
    let (result_lower, result_upper) = if sin_lower <= sin_upper {
        (sin_lower, sin_upper)
    } else {
        (sin_upper, sin_lower)
    };

    // Add truncation error margin: for Taylor series truncated at term n,
    // error is bounded by |x|^(n+2)/(n+2)! which is very small for |x| <= pi/4
    let error_margin = Binary::new(BigInt::from(1), BigInt::from(-bits + 4));
    let lower_with_error = result_lower.sub(&error_margin);
    let upper_with_error = result_upper.add(&error_margin);

    (lower_with_error, upper_with_error)
}

/// Computes sin(x) using Taylor series to the specified precision.
fn sin_taylor(x: &Binary, bits: i64) -> Binary {
    // sin(x) = x - x^3/6 + x^5/120 - x^7/5040 + ...
    // = sum_{k=0}^{inf} (-1)^k * x^(2k+1) / (2k+1)!

    let mut result = x.clone();
    let mut term = x.clone();
    let x_sq = x.mul(x);

    for k in 1i64..=(bits / 2 + 8) {
        // term = term * x^2 / ((2k) * (2k+1))
        term = term.mul(&x_sq);
        let divisor = BigInt::from(2 * k) * BigInt::from(2 * k + 1);
        // Divide by divisor: multiply mantissa, adjust exponent
        let term_mantissa = term.mantissa().clone();
        let term_exponent = term.exponent().clone();

        // To divide, we shift the mantissa left and then divide
        let shift = (bits + 10) as usize;
        let shifted_mantissa = &term_mantissa << shift;
        let new_mantissa = shifted_mantissa / &divisor;
        term = Binary::new(new_mantissa, term_exponent - BigInt::from(shift as i64));

        // Alternate signs
        if k % 2 == 1 {
            result = result.sub(&term);
        } else {
            result = result.add(&term);
        }

        // Stop when term is negligible
        if term.mantissa().magnitude().bits() < 4 {
            break;
        }
    }

    result
}

/// Computes cos(x) using Taylor series for x in [-pi/4, pi/4].
fn cos_polynomial(x_lower: &Binary, x_upper: &Binary, precision_bits: &BigInt) -> (Binary, Binary) {
    use num_traits::ToPrimitive;

    let bits = precision_bits.to_i64().unwrap_or(64).max(16);

    // cos is decreasing on [0, pi/4] and increasing on [-pi/4, 0]
    // So we need to check if 0 is in the interval
    let zero = Binary::zero();

    let cos_lower_pt = cos_taylor(x_lower, bits);
    let cos_upper_pt = cos_taylor(x_upper, bits);

    let (result_lower, result_upper) = if x_lower <= &zero && x_upper >= &zero {
        // 0 is in the interval, maximum is at x=0 which gives cos(0)=1
        let cos_at_zero = Binary::new(BigInt::from(1), BigInt::from(0));
        let min_cos = if cos_lower_pt <= cos_upper_pt {
            cos_lower_pt
        } else {
            cos_upper_pt
        };
        (min_cos, cos_at_zero)
    } else if x_lower > &zero {
        // Entirely positive, cos is decreasing
        (cos_upper_pt, cos_lower_pt)
    } else {
        // Entirely negative, cos is increasing
        (cos_lower_pt, cos_upper_pt)
    };

    let error_margin = Binary::new(BigInt::from(1), BigInt::from(-bits + 4));
    let lower_with_error = result_lower.sub(&error_margin);
    let upper_with_error = result_upper.add(&error_margin);

    (lower_with_error, upper_with_error)
}

/// Computes cos(x) using Taylor series.
fn cos_taylor(x: &Binary, bits: i64) -> Binary {
    // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720 + ...

    let one = Binary::new(BigInt::from(1), BigInt::from(0));
    let mut result = one;
    let x_sq = x.mul(x);
    let mut term = Binary::new(BigInt::from(1), BigInt::from(0));

    for k in 1i64..=(bits / 2 + 8) {
        // term = term * x^2 / ((2k-1) * (2k))
        term = term.mul(&x_sq);
        let divisor = BigInt::from(2 * k - 1) * BigInt::from(2 * k);
        let term_mantissa = term.mantissa().clone();
        let term_exponent = term.exponent().clone();

        let shift = (bits + 10) as usize;
        let shifted_mantissa = &term_mantissa << shift;
        let new_mantissa = shifted_mantissa / &divisor;
        term = Binary::new(new_mantissa, term_exponent - BigInt::from(shift as i64));

        if k % 2 == 1 {
            result = result.sub(&term);
        } else {
            result = result.add(&term);
        }

        if term.mantissa().magnitude().bits() < 4 {
            break;
        }
    }

    result
}

/// State for computing sin bounds with argument reduction.
struct SinState {
    precision_bits: BigInt,
}

struct SinOp {
    inner: Arc<Node>,
    state: RwLock<SinState>,
}

impl SinOp {
    fn new(inner: Arc<Node>) -> Self {
        Self {
            inner,
            state: RwLock::new(SinState {
                precision_bits: BigInt::from(16),
            }),
        }
    }

    /// Computes bounds on sin(x) given bounds on x.
    fn compute_sin_bounds(&self, input_bounds: &Bounds) -> Result<Bounds, ComputableError> {
        let state = self.state.read();
        let precision = &state.precision_bits;

        // Extract finite bounds or return [-1, 1] for infinite inputs
        let (x_lower, x_upper) = match (input_bounds.small(), input_bounds.large()) {
            (XBinary::Finite(l), XBinary::Finite(u)) => (l.clone(), u),
            _ => {
                // Infinite bounds: sin is bounded by [-1, 1]
                return Ok(Bounds::new(
                    XBinary::Finite(Binary::new(BigInt::from(-1), BigInt::from(0))),
                    XBinary::Finite(Binary::new(BigInt::from(1), BigInt::from(0))),
                ));
            }
        };

        // Compute pi to required precision
        let pi = compute_pi(precision);
        let two_pi = pi.mul(&Binary::new(BigInt::from(2), BigInt::from(0)));
        let half_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::from(1));
        let quarter_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::from(2));

        // Check if interval spans more than 2*pi - if so, sin takes all values in [-1, 1]
        let interval_width = x_upper.sub(&x_lower);
        if interval_width >= two_pi {
            return Ok(Bounds::new(
                XBinary::Finite(Binary::new(BigInt::from(-1), BigInt::from(0))),
                XBinary::Finite(Binary::new(BigInt::from(1), BigInt::from(0))),
            ));
        }

        // Reduce arguments to [0, 2*pi) and compute sin bounds
        let (sin_lower, sin_upper) =
            self.compute_sin_with_reduction(&x_lower, &x_upper, &pi, &half_pi, &quarter_pi, precision)?;

        // Clamp to [-1, 1] for safety
        let one = Binary::new(BigInt::from(1), BigInt::from(0));
        let neg_one = Binary::new(BigInt::from(-1), BigInt::from(0));

        let clamped_lower = if sin_lower < neg_one {
            neg_one.clone()
        } else {
            sin_lower
        };
        let clamped_upper = if sin_upper > one { one } else { sin_upper };

        Bounds::new_checked(
            XBinary::Finite(clamped_lower),
            XBinary::Finite(clamped_upper),
        )
        .map_err(|_| ComputableError::InvalidBoundsOrder)
    }

    /// Computes sin bounds with argument reduction.
    fn compute_sin_with_reduction(
        &self,
        x_lower: &Binary,
        x_upper: &Binary,
        pi: &Binary,
        half_pi: &Binary,
        quarter_pi: &Binary,
        precision: &BigInt,
    ) -> Result<(Binary, Binary), ComputableError> {
        let two_pi = pi.mul(&Binary::new(BigInt::from(2), BigInt::from(0)));

        // Check if the interval contains a maximum (pi/2 + 2*k*pi) or minimum (3*pi/2 + 2*k*pi)
        let contains_max = self.interval_contains_critical_point(x_lower, x_upper, half_pi, &two_pi);
        let three_half_pi = half_pi.add(pi);
        let contains_min =
            self.interval_contains_critical_point(x_lower, x_upper, &three_half_pi, &two_pi);

        if contains_max && contains_min {
            // Contains both max and min
            return Ok((
                Binary::new(BigInt::from(-1), BigInt::from(0)),
                Binary::new(BigInt::from(1), BigInt::from(0)),
            ));
        }

        // Compute sin at the endpoints
        let sin_at_lower = self.sin_reduced(x_lower, pi, half_pi, quarter_pi, precision);
        let sin_at_upper = self.sin_reduced(x_upper, pi, half_pi, quarter_pi, precision);

        let mut result_lower = sin_at_lower.0.clone();
        let mut result_upper = sin_at_lower.1.clone();

        // Expand bounds to include sin at upper endpoint
        if sin_at_upper.0 < result_lower {
            result_lower = sin_at_upper.0;
        }
        if sin_at_upper.1 > result_upper {
            result_upper = sin_at_upper.1;
        }

        // If contains max, upper bound is 1
        if contains_max {
            result_upper = Binary::new(BigInt::from(1), BigInt::from(0));
        }

        // If contains min, lower bound is -1
        if contains_min {
            result_lower = Binary::new(BigInt::from(-1), BigInt::from(0));
        }

        Ok((result_lower, result_upper))
    }

    /// Checks if the interval [x_lower, x_upper] contains a point of the form
    /// critical + k * period for some integer k.
    fn interval_contains_critical_point(
        &self,
        x_lower: &Binary,
        x_upper: &Binary,
        critical: &Binary,
        period: &Binary,
    ) -> bool {
        // Find k such that critical + k * period is in [x_lower, x_upper]
        // k_min = ceil((x_lower - critical) / period)
        // k_max = floor((x_upper - critical) / period)
        // Contains critical point iff k_min <= k_max

        let diff_lower = x_lower.sub(critical);
        let diff_upper = x_upper.sub(critical);

        // Compute k_min and k_max approximately
        // k_min = ceil(diff_lower / period)
        // k_max = floor(diff_upper / period)

        let k_lower = self.floor_div(&diff_lower, period);
        let k_upper = self.floor_div(&diff_upper, period);

        // If k_lower < k_upper, there's definitely an integer between them
        if k_lower < k_upper {
            return true;
        }

        // If k_lower == k_upper, check if critical + k*period is actually in the interval
        if k_lower == k_upper {
            let k_val = k_lower;
            let point = critical.add(&period.mul(&Binary::new(k_val, BigInt::from(0))));
            return &point >= x_lower && &point <= x_upper;
        }

        false
    }

    /// Computes floor(a / b) as a BigInt.
    fn floor_div(&self, a: &Binary, b: &Binary) -> BigInt {
        // a / b = (a_m * 2^a_e) / (b_m * 2^b_e) = (a_m / b_m) * 2^(a_e - b_e)
        // For floor division, we need to be careful with signs

        let a_m = a.mantissa();
        let b_m = b.mantissa();
        let exp_diff = a.exponent() - b.exponent();

        // Scale to common precision for division
        let shift = 64i64;

        if exp_diff >= BigInt::from(0) {
            // a has larger exponent
            let exp_diff_usize: usize = exp_diff.to_string().parse().unwrap_or(64);
            let scaled_a = a_m << exp_diff_usize;
            &scaled_a / b_m
        } else {
            // b has larger exponent
            let exp_diff_usize: usize = (-exp_diff.clone()).to_string().parse().unwrap_or(64);
            let scaled_b = b_m << exp_diff_usize;
            let scaled_a = a_m << (shift as usize);
            let result = &scaled_a / &scaled_b;
            result >> (shift as usize)
        }
    }

    /// Computes sin(x) with argument reduction, returning bounds.
    fn sin_reduced(
        &self,
        x: &Binary,
        pi: &Binary,
        half_pi: &Binary,
        quarter_pi: &Binary,
        precision: &BigInt,
    ) -> (Binary, Binary) {
        let two_pi = pi.mul(&Binary::new(BigInt::from(2), BigInt::from(0)));

        // Reduce x to [0, 2*pi)
        let reduced = self.reduce_to_period(x, &two_pi);

        // Determine which octant we're in and apply appropriate identity
        let (reduced_arg, negate, use_cos) = self.determine_reduction(&reduced, pi, half_pi, quarter_pi);

        // Compute sin or cos in [-pi/4, pi/4]
        let (mut lower, mut upper) = if use_cos {
            cos_polynomial(&reduced_arg, &reduced_arg, precision)
        } else {
            sin_polynomial(&reduced_arg, &reduced_arg, precision)
        };

        // Apply negation if needed
        if negate {
            let temp = lower.neg();
            lower = upper.neg();
            upper = temp;
        }

        (lower, upper)
    }

    /// Reduces x to [0, period) using modular arithmetic.
    fn reduce_to_period(&self, x: &Binary, period: &Binary) -> Binary {
        use num_traits::Signed;

        // k = floor(x / period)
        let k = self.floor_div(x, period);

        // reduced = x - k * period
        let k_binary = Binary::new(k, BigInt::from(0));
        let offset = period.mul(&k_binary);
        let reduced = x.sub(&offset);

        // Ensure result is in [0, period)
        if reduced.mantissa().is_negative() {
            reduced.add(period)
        } else if &reduced >= period {
            reduced.sub(period)
        } else {
            reduced
        }
    }

    /// Determines how to reduce the argument for sin computation.
    /// Returns (reduced_arg, should_negate, use_cos).
    fn determine_reduction(
        &self,
        x: &Binary,
        pi: &Binary,
        half_pi: &Binary,
        quarter_pi: &Binary,
    ) -> (Binary, bool, bool) {
        let three_quarter_pi = quarter_pi.add(half_pi);
        let five_quarter_pi = pi.add(quarter_pi);
        let three_half_pi = half_pi.add(pi);
        let seven_quarter_pi = three_half_pi.add(quarter_pi);

        // Octant 0: [0, pi/4] -> sin(x)
        if x <= quarter_pi {
            return (x.clone(), false, false);
        }

        // Octant 1: [pi/4, pi/2] -> cos(pi/2 - x)
        if x <= half_pi {
            let reduced = half_pi.sub(x);
            return (reduced, false, true);
        }

        // Octant 2: [pi/2, 3pi/4] -> cos(x - pi/2)
        if x <= &three_quarter_pi {
            let reduced = x.sub(half_pi);
            return (reduced, false, true);
        }

        // Octant 3: [3pi/4, pi] -> sin(pi - x)
        if x <= pi {
            let reduced = pi.sub(x);
            return (reduced, false, false);
        }

        // Octant 4: [pi, 5pi/4] -> -sin(x - pi)
        if x <= &five_quarter_pi {
            let reduced = x.sub(pi);
            return (reduced, true, false);
        }

        // Octant 5: [5pi/4, 3pi/2] -> -cos(3pi/2 - x)
        if x <= &three_half_pi {
            let reduced = three_half_pi.sub(x);
            return (reduced, true, true);
        }

        // Octant 6: [3pi/2, 7pi/4] -> -cos(x - 3pi/2)
        if x <= &seven_quarter_pi {
            let reduced = x.sub(&three_half_pi);
            return (reduced, true, true);
        }

        // Octant 7: [7pi/4, 2pi] -> sin(x - 2pi) = -sin(2pi - x)
        let two_pi = pi.mul(&Binary::new(BigInt::from(2), BigInt::from(0)));
        let reduced = two_pi.sub(x);
        (reduced, true, false)
    }
}

impl NodeOp for SinOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        self.compute_sin_bounds(&input_bounds)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut state = self.state.write();
        state.precision_bits += BigInt::from(8);
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

/// Node in the computation graph. The op stores structure/state; the cache stores
/// the last bounds computed for this node.
///
/// NOTE: The bounds_cache is not automatically invalidated when children are refined.
/// Updates are explicitly propagated via apply_update during refinement. If get_bounds()
/// is called between refinement steps (outside of refine_to), it may return stale cached
/// values. Consider whether this is acceptable for your use case.
struct Node {
    id: usize,
    op: Arc<dyn NodeOp>,
    bounds_cache: RwLock<Option<Bounds>>,
    refinement: RefinementSync,
}

impl Node {
    fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            bounds_cache: RwLock::new(None),
            refinement: RefinementSync::new(),
        })
    }

    /// Returns cached bounds if already computed.
    fn cached_bounds(&self) -> Option<Bounds> {
        self.bounds_cache.read().clone()
    }

    /// Returns cached bounds, computing and caching if needed.
    /// Combinators are infallible, so bounds are lazily computed on demand.
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        if let Some(bounds) = self.cached_bounds() {
            return Ok(bounds);
        }
        let bounds = self.compute_bounds()?;
        self.set_bounds(bounds.clone());
        Ok(bounds)
    }

    fn set_bounds(&self, bounds: Bounds) {
        let mut cache = self.bounds_cache.write();
        *cache = Some(bounds);
        self.refinement.notify_bounds_updated();
    }

    /// Computes bounds for this node from current children/base state.
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        self.op.compute_bounds()
    }

    /// Performs one refinement step. Returns whether refinement was applied.
    fn refine_step(&self) -> Result<bool, ComputableError> {
        self.op.refine_step()
    }

    fn children(&self) -> Vec<Arc<Node>> {
        self.op.children()
    }

    fn is_refiner(&self) -> bool {
        self.op.is_refiner()
    }
}

struct RefinementSync {
    state: Mutex<RefinementState>,
    condvar: Condvar,
}

struct RefinementState {
    active: bool,
    epoch: u64,
}

impl RefinementSync {
    fn new() -> Self {
        Self {
            state: Mutex::new(RefinementState {
                active: false,
                epoch: 0,
            }),
            condvar: Condvar::new(),
        }
    }

    fn notify_bounds_updated(&self) {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        self.condvar.notify_all();
    }
}

impl BoundsAccess for Node {
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        Node::get_bounds(self)
    }
}

#[derive(Clone, Copy)]
enum RefineCommand {
    Step,
    Stop,
}

/// Handle for a background refiner task.
struct RefinerHandle {
    sender: Sender<RefineCommand>,
}

/// Snapshot of the node graph used to coordinate parallel refinement.
struct RefinementGraph {
    root: Arc<Node>,
    nodes: HashMap<usize, Arc<Node>>,    // node id -> node
    parents: HashMap<usize, Vec<usize>>, // child id -> parent ids
    refiners: Vec<Arc<Node>>,
}

impl RefinementGraph {
    fn new(root: Arc<Node>) -> Result<Self, ComputableError> {
        let mut nodes = HashMap::new();
        let mut parents: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut refiners = Vec::new();

        let mut stack = vec![Arc::clone(&root)];
        while let Some(node) = stack.pop() {
            if nodes.contains_key(&node.id) {
                continue;
            }
            let node_id = node.id;
            nodes.insert(node_id, Arc::clone(&node));

            if node.is_refiner() {
                refiners.push(Arc::clone(&node));
            }
            for child in node.children() {
                parents.entry(child.id).or_default().push(node_id);
                stack.push(child);
            }
        }

        let graph = Self {
            root,
            nodes,
            parents,
            refiners,
        };

        Ok(graph)
    }

    // TODO: The README describes an async/event-driven refinement model where:
    // - Branches refine continuously and publish updates
    // - Other nodes "subscribe" to updates and recompute live
    // - Multiplication "receives refinements of its left branch and is listening for refinements of b"
    //
    // However, this implementation uses SYNCHRONOUS LOCK-STEP refinement:
    // 1. Send Step command to ALL refiners
    // 2. Wait for ALL refiners to complete one step
    // 3. Collect and propagate ALL updates
    // 4. Check if precision met
    // 5. Repeat
    //
    // There is no subscription/listener pattern - all refiners do exactly one step in lockstep.
    // The README's detailed example (sqrt(a + ab)) describes behavior that doesn't match this
    // implementation. Either the README should be updated to reflect the actual synchronous
    // model, or the implementation should be changed to the async model described.
    fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        &self,
        epsilon: &UBinary,
    ) -> Result<Bounds, ComputableError> {
        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());
            let mut refiners = Vec::new();
            let mut iterations = 0usize;
            let (update_tx, update_rx) = unbounded();

            for node in &self.refiners {
                refiners.push(spawn_refiner(
                    scope,
                    Arc::clone(node),
                    Arc::clone(&stop_flag),
                    update_tx.clone(),
                ));
            }
            drop(update_tx);

            let shutdown_refiners = |handles: Vec<RefinerHandle>, stop_signal: &Arc<StopFlag>| {
                stop_signal.stop();
                for refiner in &handles {
                    let _ = refiner.sender.send(RefineCommand::Stop);
                }
            };

            let result = (|| {
                let mut root_bounds;
                loop {
                    root_bounds = self.root.get_bounds()?;
                    if bounds_width_leq(&root_bounds, epsilon) {
                        break;
                    }
                    if iterations >= MAX_REFINEMENT_ITERATIONS {
                        // TODO: allow individual refiners to stop at the max without
                        // failing the whole refinement, only erroring if all refiners
                        // are exhausted before meeting the target precision.
                        return Err(ComputableError::MaxRefinementIterations {
                            max: MAX_REFINEMENT_ITERATIONS,
                        });
                    }
                    iterations += 1;

                    for refiner in &refiners {
                        refiner
                            .sender
                            .send(RefineCommand::Step)
                            .map_err(|_| ComputableError::RefinementChannelClosed)?;
                    }

                    let mut expected_updates = refiners.len();
                    while expected_updates > 0 {
                        let update_result = match update_rx.recv() {
                            Ok(update_result) => update_result,
                            Err(_) => {
                                return Err(ComputableError::RefinementChannelClosed);
                            }
                        };
                        expected_updates -= 1;
                        match update_result {
                            Ok(update) => self.apply_update(update)?,
                            Err(error) => {
                                return Err(error);
                            }
                        }
                    }
                }

                self.root.get_bounds()
            })();

            shutdown_refiners(refiners, &stop_flag);
            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }

    fn apply_update(&self, update: NodeUpdate) -> Result<(), ComputableError> {
        let mut queue = VecDeque::new();
        if let Some(node) = self.nodes.get(&update.node_id) {
            node.set_bounds(update.bounds);
            queue.push_back(node.id);
        }

        while let Some(changed_id) = queue.pop_front() {
            let Some(parents) = self.parents.get(&changed_id) else {
                continue;
            };
            for parent_id in parents {
                let parent = self
                    .nodes
                    .get(parent_id)
                    .ok_or(ComputableError::RefinementChannelClosed)?;
                let next_bounds = parent.compute_bounds()?;
                if parent.cached_bounds().as_ref() != Some(&next_bounds) {
                    parent.set_bounds(next_bounds);
                    queue.push_back(*parent_id);
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
struct NodeUpdate {
    node_id: usize,
    bounds: Bounds,
}

fn spawn_refiner<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> RefinerHandle {
    let (command_tx, command_rx) = unbounded();
    scope.spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refiner_loop(node, stop, command_rx, updates)
        }));
    });

    RefinerHandle {
        sender: command_tx,
    }
}

fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    commands: Receiver<RefineCommand>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> Result<(), ComputableError> {
    while !stop.is_stopped() {
        match commands.recv() {
            Ok(RefineCommand::Step) => match node.refine_step() {
                Ok(true) => {
                    let bounds = node.compute_bounds()?;
                    node.set_bounds(bounds.clone());
                    if updates
                        .send(Ok(NodeUpdate {
                            node_id: node.id,
                            bounds,
                        }))
                        .is_err()
                    {
                        break;
                    }
                }
                Ok(false) => {
                    let bounds = node
                        .cached_bounds()
                        .ok_or(ComputableError::RefinementChannelClosed)?;
                    if updates
                        .send(Ok(NodeUpdate {
                            node_id: node.id,
                            bounds,
                        }))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    let _ = updates.send(Err(error));
                    break;
                }
            },
            Ok(RefineCommand::Stop) | Err(_) => break,
        }
    }
    Ok(())
}

#[cfg(debug_assertions)]
pub const DEFAULT_INV_MAX_REFINES: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_INV_MAX_REFINES: usize = 4096;

#[cfg(debug_assertions)]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 64;
#[cfg(not(debug_assertions))]
pub const DEFAULT_MAX_REFINEMENT_ITERATIONS: usize = 4096;

fn reciprocal_bounds(bounds: &Bounds, precision_bits: &BigInt) -> Result<Bounds, ComputableError> {
    let lower = bounds.small();
    let upper = bounds.large();
    let zero = XBinary::zero();
    if lower <= &zero && upper >= zero {
        return Ok(Bounds::new(
            XBinary::NegInf,
            XBinary::PosInf,
        ));
    }

    let (lower_bound, upper_bound) = if upper < zero {
        let lower_bound = reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            ReciprocalRounding::Ceil,
        )?
        .neg();
        let upper_bound = reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            ReciprocalRounding::Floor,
        )?
        .neg();
        (lower_bound, upper_bound)
    } else {
        let lower_bound = reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            ReciprocalRounding::Floor,
        )?;
        let upper_bound = reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            ReciprocalRounding::Ceil,
        )?;
        (lower_bound, upper_bound)
    };

    Bounds::new_checked(lower_bound, upper_bound)
        .map_err(|_| ComputableError::InvalidBoundsOrder)
}

/// Compares bounds width (UXBinary) against epsilon (UBinary).
/// Returns true if width <= epsilon.
fn bounds_width_leq(bounds: &Bounds, epsilon: &UBinary) -> bool {
    match bounds.width() {
        UXBinary::PosInf => false,
        UXBinary::Finite(uwidth) => *uwidth <= *epsilon,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use num_bigint::BigUint;
    use std::sync::{Arc, Barrier};
    use std::thread;

    type IntervalState = Bounds;

    // --- test utilities ---

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn ubin(mantissa: u64, exponent: i64) -> UBinary {
        UBinary::new(BigUint::from(mantissa), BigInt::from(exponent))
    }

    fn xbin(mantissa: i64, exponent: i64) -> XBinary {
        XBinary::Finite(bin(mantissa, exponent))
    }

    fn unwrap_finite(input: &XBinary) -> Binary {
        match input {
            XBinary::Finite(value) => value.clone(),
            XBinary::NegInf | XBinary::PosInf => {
                panic!("expected finite extended binary")
            }
        }
    }

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn midpoint_between(lower: &XBinary, upper: &XBinary) -> Binary {
        let mid_sum = unwrap_finite(lower).add(&unwrap_finite(upper));
        let exponent = mid_sum.exponent() - BigInt::one();
        Binary::new(mid_sum.mantissa().clone(), exponent)
    }

    fn interval_refine(state: IntervalState) -> IntervalState {
        let midpoint = midpoint_between(state.small(), &state.large());
        Bounds::new(
            XBinary::Finite(midpoint.clone()),
            XBinary::Finite(midpoint),
        )
    }

    fn interval_refine_strict(state: IntervalState) -> IntervalState {
        let midpoint = midpoint_between(state.small(), &state.large());
        Bounds::new(state.small().clone(), XBinary::Finite(midpoint))
    }

    fn interval_midpoint_computable(lower: i64, upper: i64) -> Computable {
        let interval_state = Bounds::new(xbin(lower, 0), xbin(upper, 0));
        Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine,
        )
    }

    fn sqrt_computable(value_int: u64) -> Computable {
        let interval_state = Bounds::new(xbin(1, 0), xbin(value_int as i64, 0));
        let bounds = |inner_state: &IntervalState| Ok(inner_state.clone());
        let refine = move |inner_state: IntervalState| {
            let mid = midpoint_between(inner_state.small(), &inner_state.large());
            let mid_sq = mid.mul(&mid);
            let value = bin(value_int as i64, 0);

            if mid_sq <= value {
                Bounds::new(XBinary::Finite(mid), inner_state.large().clone())
            } else {
                Bounds::new(inner_state.small().clone(), XBinary::Finite(mid))
            }
        };

        Computable::new(interval_state, bounds, refine)
    }

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_ubinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    fn unwrap_finite_ubinary(input: &UXBinary) -> UBinary {
        match input {
            UXBinary::Finite(value) => value.clone(),
            UXBinary::PosInf => {
                panic!("expected finite unsigned extended binary")
            }
        }
    }

    fn assert_width_nonnegative(bounds: &Bounds) {
        assert!(*bounds.width() >= UXBinary::zero());
    }

    // --- tests for different results of refinement (mostly errors) ---

    // Note: negative epsilon is now impossible at the type level since we use UBinary

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(0, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::NonpositiveEpsilon)));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected = xbin(1, 0);
        let upper = bounds.large();
        let width = unwrap_finite_ubinary(bounds.width());

        assert!(bounds.small() <= &expected && &expected <= &upper);
        assert!(width < epsilon);
        let refined_bounds = computable.bounds().expect("bounds should succeed");
        let refined_upper = refined_bounds.large();
        assert!(
            refined_bounds.small() <= &expected
                && &expected <= &refined_upper
        );
    }

    #[test]
    fn computable_refine_to_rejects_unchanged_state() {
        let interval_state = Bounds::new(xbin(0, 0), xbin(2, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            |inner_state| inner_state,
        );
        let epsilon = ubin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::StateUnchanged)));
    }

    #[test]
    fn computable_refine_to_enforces_max_iterations() {
        let computable = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let epsilon = ubin(1, -1);
        let result = computable.refine_to::<5>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 5 })
        ));
    }

    // test the "normal case" where the bounds shrink but never meet
    #[test]
    fn computable_refine_to_handles_non_meeting_bounds() {
        let interval_state = Bounds::new(xbin(0, 0), xbin(4, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine_strict,
        );
        let epsilon = ubin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let upper = bounds.large();
        assert!(bounds.small() < &upper);
        assert!(bounds_width_leq(&bounds, &epsilon));
        assert_eq!(computable.bounds().expect("bounds should succeed"), bounds);
    }

    #[test]
    fn computable_refine_to_rejects_worsened_bounds() {
        let interval_state = Bounds::new(xbin(0, 0), xbin(1, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            |inner_state: IntervalState| {
                let upper = inner_state.large();
                let worse_upper = unwrap_finite(&upper).add(&bin(1, 0));
                Bounds::new(
                    inner_state.small().clone(),
                    XBinary::Finite(worse_upper),
                )
            },
        );
        let epsilon = ubin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    // --- tests for bounds of arithmetic operations ---

    #[test]
    fn computable_add_combines_bounds() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left + right;
        let sum_bounds = sum.bounds().expect("bounds should succeed");
        assert_eq!(sum_bounds, Bounds::new(xbin(1, 0), xbin(5, 0)));
    }

    #[test]
    fn computable_sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(diff_bounds, Bounds::new(xbin(2, 0), xbin(5, 0)));
    }

    #[test]
    fn computable_neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-3, 0), xbin(-1, 0)));
    }

    #[test]
    fn computable_inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            Bounds::new(XBinary::NegInf, XBinary::PosInf)
        );
    }

    #[test]
    fn computable_inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = ubin(1, -8);
        let bounds = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected_binary = XBinary::from_f64(1.0 / 3.0)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected_value, &epsilon);
    }

    #[test]
    fn computable_mul_combines_bounds_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(2, 0), xbin(12, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-12, 0), xbin(-2, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-10, 0), xbin(15, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-8, 0), xbin(12, 0)));
    }

    #[test]
    fn computable_from_binary_matches_constant_bounds() {
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

    // --- test more complex expressions ---

    #[test]
    fn computable_integration_sqrt2_expression() {
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
    fn computable_shared_operand_in_expression() {
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

    // --- concurrency tests ---

    #[test]
    fn computable_refine_shared_clone_updates_original() {
        let original = sqrt_computable(2);
        let cloned = original.clone();
        let epsilon = ubin(1, -12);

        let _ = cloned
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let bounds = original.bounds().expect("bounds should succeed");
        assert!(bounds_width_leq(&bounds, &epsilon));
    }

    #[test]
    fn computable_refine_to_channel_closure() {
        let computable = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |_| panic!("refiner panic"),
        );

        let epsilon = ubin(1, -4);
        let result = computable.refine_to::<2>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::RefinementChannelClosed)
        ));
    }

    #[test]
    fn computable_refine_to_max_iterations_multiple_refiners() {
        let left = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let right = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let expr = left + right;
        let epsilon = ubin(1, -4);
        let result = expr.refine_to::<2>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 2 })
        ));
    }

    #[test]
    fn computable_refine_to_error_path_stops_refiners() {
        let stable = interval_midpoint_computable(0, 2);
        let faulty = Computable::new(
            Bounds::new(xbin(0, 0), xbin(1, 0)),
            |state| Ok(state.clone()),
            |state| Bounds::new(state.small().clone(), xbin(2, 0)),
        );
        let expr = stable + faulty;
        let epsilon = ubin(1, -4);
        let result = expr.refine_to::<3>(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    #[test]
    fn concurrent_bounds_reads_during_failed_refinement() {
        let computable = Arc::new(Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        ));
        let epsilon = ubin(1, -6);
        let reader = Arc::clone(&computable);
        let handle = thread::spawn(move || {
            for _ in 0..8 {
                let bounds = reader.bounds().expect("bounds should succeed");
                assert_width_nonnegative(&bounds);
            }
        });

        let result = computable.refine_to::<3>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 3 })
        ));
        handle.join().expect("reader thread should join");
    }

    // NOTE: this test could be fallible, since it uses timing to measure success. perhaps it should be an integration test rather than a unit test
    #[test]
    fn refinement_parallelizes_multiple_refiners() {
        use std::time::{Duration, Instant};

        const SLEEP_MS: u64 = 10;

        let slow_refiner = || {
            Computable::new(
                0usize,
                |_| {
                    Ok(Bounds::new(
                        XBinary::NegInf,
                        XBinary::PosInf,
                    ))
                },
                |state| {
                    thread::sleep(Duration::from_millis(SLEEP_MS));
                    state + 1
                },
            )
        };

        let expr = slow_refiner() + slow_refiner() + slow_refiner() + slow_refiner();
        let epsilon = ubin(1, -6);

        let start = Instant::now();
        let result = expr.refine_to::<1>(epsilon);
        let elapsed = start.elapsed();

        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 1 })
        ));
        assert!(
            elapsed.as_millis() as u64 > SLEEP_MS,
            "refinement must not have actually run"
        );
        assert!(
            (elapsed.as_millis() as u64) < 2 * SLEEP_MS,
            "expected parallel refinement under {}ms, elapsed {elapsed:?}",
            2 * SLEEP_MS
        );
    }

    #[test]
    fn concurrent_refine_to_shared_expression() {
        let sqrt2 = sqrt_computable(2);
        let base_expression =
            (sqrt2.clone() + sqrt2.clone()) * (Computable::constant(bin(1, 0)) + sqrt2.clone());
        let expression = Arc::new(base_expression);
        let epsilon = ubin(1, -10);
        // Coordinate multiple threads calling refine_to on the same computable.
        let barrier = Arc::new(Barrier::new(4));

        let mut handles = Vec::new();
        for _ in 0..3 {
            let shared_expression = Arc::clone(&expression);
            let shared_barrier = Arc::clone(&barrier);
            let thread_epsilon = epsilon.clone();
            handles.push(thread::spawn(move || {
                shared_barrier.wait();
                shared_expression.refine_to_default(thread_epsilon)
            }));
        }

        barrier.wait();
        let main_bounds = expression
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let main_upper = main_bounds.large();
        assert!(bounds_width_leq(&main_bounds, &epsilon));

        for handle in handles {
            let bounds = handle
                .join()
                .expect("thread should join")
                .expect("refine_to should succeed");
            let bounds_upper = bounds.large();
            assert_width_nonnegative(&bounds);
            assert!(bounds_width_leq(&bounds, &epsilon));
            assert!(bounds.small() <= &main_upper);
            assert!(main_bounds.small() <= &bounds_upper);
        }
    }

    #[test]
    fn concurrent_refine_to_uses_single_refiner() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::Duration;

        let active_refines = Arc::new(AtomicUsize::new(0));
        let saw_overlap = Arc::new(AtomicBool::new(false));

        let shared_active = Arc::clone(&active_refines);
        let shared_overlap = Arc::clone(&saw_overlap);
        let computable = Computable::new(
            Bounds::new(xbin(0, 0), xbin(4, 0)),
            |state| Ok(state.clone()),
            move |state: IntervalState| {
                let prior = shared_active.fetch_add(1, Ordering::SeqCst);
                if prior > 0 {
                    shared_overlap.store(true, Ordering::SeqCst);
                }
                thread::sleep(Duration::from_millis(10));
                let next = interval_refine(state);
                shared_active.fetch_sub(1, Ordering::SeqCst);
                next
            },
        );

        let shared = Arc::new(computable);
        let epsilon = ubin(1, -6);
        let barrier = Arc::new(Barrier::new(3));

        let mut handles = Vec::new();
        for _ in 0..2 {
            let shared_value = Arc::clone(&shared);
            let shared_barrier = Arc::clone(&barrier);
            let thread_epsilon = epsilon.clone();
            handles.push(thread::spawn(move || {
                shared_barrier.wait();
                shared_value
                    .refine_to_default(thread_epsilon)
                    .expect("refine_to should succeed")
            }));
        }

        barrier.wait();
        let main_bounds = shared
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        for handle in handles {
            let bounds = handle.join().expect("thread should join");
            assert_width_nonnegative(&bounds);
        }

        assert!(!saw_overlap.load(Ordering::SeqCst));
        assert!(bounds_width_leq(&main_bounds, &epsilon));
    }

    #[test]
    fn concurrent_bounds_reads_during_refinement() {
        let base_value = interval_midpoint_computable(0, 4);
        let shared_value = Arc::new(base_value);
        let epsilon = ubin(1, -8);
        // Reader thread repeatedly calls bounds while refinement is running.
        let barrier = Arc::new(Barrier::new(2));

        let reader = {
            let reader_value = Arc::clone(&shared_value);
            let reader_barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                reader_barrier.wait();
                for _ in 0..32 {
                    let bounds = reader_value.bounds().expect("bounds should succeed");
                    assert_width_nonnegative(&bounds);
                }
            })
        };

        barrier.wait();
        let refined = shared_value
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        reader.join().expect("reader should join");
        assert_width_nonnegative(&refined);
    }

    // --- sin tests ---

    #[test]
    fn computable_sin_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_zero
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0) = 0
        let expected = bin(0, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn computable_sin_of_pi_over_2() {
        // pi/2 ~= 1.5707963...
        // We approximate it as 3217/2048 ~= 1.5708...
        let pi_over_2 = Computable::constant(bin(3217, -11));
        let sin_pi_2 = pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi/2) = 1
        let expected_f64 = (std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi/2) should be very close to 1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn computable_sin_of_pi() {
        // pi ~= 3.14159...
        // We approximate it as 6434/2048 ~= 3.1416...
        let pi_approx = Computable::constant(bin(6434, -11));
        let sin_pi = pi_approx.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi) ~= 0 (should be close to 0)
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi) should be very close to 0
        let small_bound = bin(1, -4);
        let neg_small_bound = bin(-1, -4);
        assert!(lower >= neg_small_bound);
        assert!(upper <= small_bound);
    }

    #[test]
    fn computable_sin_of_negative_pi_over_2() {
        // -pi/2 ~= -1.5707963...
        let neg_pi_over_2 = Computable::constant(bin(-3217, -11));
        let sin_neg_pi_2 = neg_pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_neg_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(-pi/2) = -1
        let expected_f64 = (-std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(-pi/2) should be very close to -1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn computable_sin_bounds_always_in_minus_one_to_one() {
        // Test with a large value that exercises argument reduction
        let large_value = Computable::constant(bin(100, 0));
        let sin_large = large_value.sin();
        let bounds = sin_large.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        assert!(lower >= neg_one);
        assert!(upper <= one);
    }

    #[test]
    fn computable_sin_of_small_value() {
        // For small x, sin(x) ~= x
        let small = Computable::constant(bin(1, -4)); // 1/16 = 0.0625
        let sin_small = small.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_small
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0.0625) ~= 0.0624593...
        let expected = XBinary::from_f64(0.0625_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn computable_sin_interval_spanning_maximum() {
        // An interval that spans pi/2 (where sin has maximum)
        let interval_state = Bounds::new(xbin(1, 0), xbin(2, 0)); // [1, 2] includes pi/2 ~= 1.57
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine,
        );
        let sin_interval = computable.sin();
        let bounds = sin_interval.bounds().expect("bounds should succeed");

        let upper = unwrap_finite(&bounds.large());

        // The upper bound should be close to 1 since the interval contains pi/2
        assert!(upper >= bin(1, -1)); // Upper bound should be at least 0.5
    }

    #[test]
    fn computable_sin_with_infinite_input_bounds() {
        let unbounded = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let sin_unbounded = unbounded.sin();
        let bounds = sin_unbounded.bounds().expect("bounds should succeed");

        // sin of unbounded input should be [-1, 1]
        assert_eq!(bounds.small(), &xbin(-1, 0));
        assert_eq!(&bounds.large(), &xbin(1, 0));
    }

    #[test]
    fn computable_sin_expression_with_arithmetic() {
        // Test sin(x) + cos-like expression: sin(x)^2 + sin(x + pi/2)^2 should be close to 1
        // Here we just test that sin works in expressions
        let x = Computable::constant(bin(1, 0)); // x = 1
        let sin_x = x.clone().sin();
        let two = Computable::constant(bin(2, 0));
        let expr = sin_x.clone() * two; // 2 * sin(1)

        let epsilon = ubin(1, -8);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // 2 * sin(1) ~= 2 * 0.8414... ~= 1.6829...
        let expected = XBinary::from_f64(2.0 * 1.0_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }
}
