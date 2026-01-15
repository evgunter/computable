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
use num_traits::{One, Signed, Zero};
use parking_lot::{Condvar, Mutex, RwLock};

mod binary;
mod concurrency;
mod ordered_pair;

pub use binary::{Binary, BinaryError, ExtendedBinary, ExtendedBinaryError};
use concurrency::StopFlag;
pub use ordered_pair::{OrderedPair, OrderedPairError};

pub type Bounds = OrderedPair<ExtendedBinary>;

pub fn bounds_lower(bounds: &Bounds) -> &ExtendedBinary {
    bounds.small()
}

pub fn bounds_upper(bounds: &Bounds) -> ExtendedBinary {
    bounds.large()
}

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
            if bounds_lower(&previous_bounds) == &bounds_upper(&previous_bounds) {
                return Ok(());
            }
            return Err(ComputableError::StateUnchanged);
        }

        let next_bounds = (self.bounds)(&next_state)?;
        let lower_worsened = bounds_lower(&next_bounds) < bounds_lower(&previous_bounds);
        let upper_worsened =  bounds_upper(&next_bounds) > bounds_upper(&previous_bounds);
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
        epsilon: Binary,
    ) -> Result<Bounds, ComputableError> {
        if !epsilon.mantissa().is_positive() {
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

    pub fn refine_to_default(&self, epsilon: Binary) -> Result<Bounds, ComputableError> {
        self.refine_to::<DEFAULT_MAX_REFINEMENT_ITERATIONS>(epsilon)
    }

    pub fn inv(self) -> Self {
        let node = Node::new(Arc::new(InvOp {
            inner: Arc::clone(&self.node),
            precision_bits: RwLock::new(BigInt::zero()),
        }));
        Self { node }
    }

    #[allow(clippy::type_complexity)]
    pub fn constant(value: Binary) -> Self {
        fn bounds(value: &Binary) -> Result<Bounds, ComputableError> {
            Ok(OrderedPair::new(
                ExtendedBinary::Finite(value.clone()),
                ExtendedBinary::Finite(value.clone()),
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
        let lower = bounds_lower(&existing).neg();
        let upper = bounds_upper(&existing).neg();
        OrderedPair::new_checked(upper, lower).map_err(|_| ComputableError::InvalidBoundsOrder)
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
        let lower = bounds_lower(&left_bounds).add_lower(bounds_lower(&right_bounds));
        let upper = bounds_upper(&left_bounds).add_upper(&bounds_upper(&right_bounds));
        OrderedPair::new_checked(lower, upper).map_err(|_| ComputableError::InvalidBoundsOrder)
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
        let left_lower = bounds_lower(&left_bounds);
        let left_upper = bounds_upper(&left_bounds);
        let right_lower = bounds_lower(&right_bounds);
        let right_upper = bounds_upper(&right_bounds);

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

        OrderedPair::new_checked(min, max).map_err(|_| ComputableError::InvalidBoundsOrder)
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

/// Node in the computation graph. The op stores structure/state; the cache stores
/// the last bounds computed for this node.
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

/// Handle for a background refiner thread.
struct RefinerHandle {
    sender: Sender<RefineCommand>,
    join: thread::JoinHandle<Result<(), ComputableError>>,
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

    fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        &self,
        epsilon: &Binary,
    ) -> Result<Bounds, ComputableError> {
        let stop_flag = Arc::new(StopFlag::new());
        let mut refiners = Vec::new();
        let mut iterations = 0usize;
        let (update_tx, update_rx) = unbounded();

        for node in &self.refiners {
            refiners.push(spawn_refiner(
                Arc::clone(node),
                Arc::clone(&stop_flag),
                update_tx.clone(),
            ));
        }
        drop(update_tx);

        let shutdown_refiners = |handles: Vec<RefinerHandle>| {
            stop_flag.stop();
            for refiner in &handles {
                let _ = refiner.sender.send(RefineCommand::Stop);
            }
            for refiner in handles {
                let _ = refiner.join.join();
            }
        };

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
                shutdown_refiners(refiners);
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
                        shutdown_refiners(refiners);
                        return Err(ComputableError::RefinementChannelClosed);
                    }
                };
                expected_updates -= 1;
                match update_result {
                    Ok(update) => self.apply_update(update)?,
                    Err(error) => {
                        shutdown_refiners(refiners);
                        return Err(error);
                    }
                }
            }
        }

        shutdown_refiners(refiners);

        self.root.get_bounds()
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

fn spawn_refiner(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> RefinerHandle {
    let (command_tx, command_rx) = unbounded();
    let join = thread::spawn(move || refiner_loop(node, stop, command_rx, updates));

    RefinerHandle {
        sender: command_tx,
        join,
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
    let lower = bounds_lower(bounds);
    let upper = bounds_upper(bounds);
    let zero = ExtendedBinary::zero();
    if lower <= &zero && upper >= zero {
        return Ok(OrderedPair::new(
            ExtendedBinary::NegInf,
            ExtendedBinary::PosInf,
        ));
    }

    let (lower_bound, upper_bound) = if upper < zero {
        let lower_bound = binary::reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            binary::ReciprocalRounding::Ceil,
        )?
        .neg();
        let upper_bound = binary::reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            binary::ReciprocalRounding::Floor,
        )?
        .neg();
        (lower_bound, upper_bound)
    } else {
        let lower_bound = binary::reciprocal_rounded_abs_extended(
            &upper,
            precision_bits,
            binary::ReciprocalRounding::Floor,
        )?;
        let upper_bound = binary::reciprocal_rounded_abs_extended(
            lower,
            precision_bits,
            binary::ReciprocalRounding::Ceil,
        )?;
        (lower_bound, upper_bound)
    };

    OrderedPair::new_checked(lower_bound, upper_bound)
        .map_err(|_| ComputableError::InvalidBoundsOrder)
}

fn bounds_width_leq(bounds: &Bounds, epsilon: &Binary) -> bool {
    let width_value = bounds.width();
    let ExtendedBinary::Finite(width) = width_value else {
        return false;
    };
    width <= epsilon
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    type IntervalState = Bounds;

    // --- test utilities ---

    fn bin(mantissa: i64, exponent: i64) -> Binary {
        Binary::new(BigInt::from(mantissa), BigInt::from(exponent))
    }

    fn ext(mantissa: i64, exponent: i64) -> ExtendedBinary {
        ExtendedBinary::Finite(bin(mantissa, exponent))
    }

    fn unwrap_finite(input: &ExtendedBinary) -> Binary {
        match input {
            ExtendedBinary::Finite(value) => value.clone(),
            ExtendedBinary::NegInf | ExtendedBinary::PosInf => {
                panic!("expected finite extended binary")
            }
        }
    }

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn midpoint_between(lower: &ExtendedBinary, upper: &ExtendedBinary) -> Binary {
        let mid_sum = unwrap_finite(lower).add(&unwrap_finite(upper));
        let exponent = mid_sum.exponent() - BigInt::one();
        Binary::new(mid_sum.mantissa().clone(), exponent)
    }

    fn interval_refine(state: IntervalState) -> IntervalState {
        let midpoint = midpoint_between(state.small(), &state.large());
        OrderedPair::new(
            ExtendedBinary::Finite(midpoint.clone()),
            ExtendedBinary::Finite(midpoint),
        )
    }

    fn interval_refine_strict(state: IntervalState) -> IntervalState {
        let midpoint = midpoint_between(state.small(), &state.large());
        OrderedPair::new(state.small().clone(), ExtendedBinary::Finite(midpoint))
    }

    fn interval_midpoint_computable(lower: i64, upper: i64) -> Computable {
        let interval_state = OrderedPair::new(ext(lower, 0), ext(upper, 0));
        Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine,
        )
    }

    fn sqrt_computable(value_int: u64) -> Computable {
        let interval_state = OrderedPair::new(ext(1, 0), ext(value_int as i64, 0));
        let bounds = |inner_state: &IntervalState| Ok(inner_state.clone());
        let refine = move |inner_state: IntervalState| {
            let mid = midpoint_between(inner_state.small(), &inner_state.large());
            let mid_sq = mid.mul(&mid);
            let value = bin(value_int as i64, 0);

            if mid_sq <= value {
                OrderedPair::new(ExtendedBinary::Finite(mid), inner_state.large().clone())
            } else {
                OrderedPair::new(inner_state.small().clone(), ExtendedBinary::Finite(mid))
            }
        };

        Computable::new(interval_state, bounds, refine)
    }

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &Binary,
    ) {
        let lower = unwrap_finite(bounds_lower(bounds));
        let upper_xb = bounds_upper(bounds);
        let width = unwrap_finite(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(&width <= epsilon);
    }

    fn assert_bounds_ordered(bounds: &Bounds) {
        assert!(bounds_lower(bounds) <= &bounds_upper(bounds));
    }

    // --- tests for different results of refinement (mostly errors) ---

    #[test]
    fn computable_refine_to_rejects_negative_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(-1, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::NonpositiveEpsilon)));
    }

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(0, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::NonpositiveEpsilon)));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = bin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected = ext(1, 0);
        let upper = bounds_upper(&bounds);
        let width = unwrap_finite(bounds.width());

        assert!(bounds_lower(&bounds) <= &expected && &expected <= &upper);
        assert!(width < epsilon);
        let refined_bounds = computable.bounds().expect("bounds should succeed");
        let refined_upper = bounds_upper(&refined_bounds);
        assert!(
            bounds_lower(&refined_bounds) <= &expected
                && &expected <= &refined_upper
        );
    }

    #[test]
    fn computable_refine_to_rejects_unchanged_state() {
        let interval_state = OrderedPair::new(ext(0, 0), ext(2, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            |inner_state| inner_state,
        );
        let epsilon = bin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::StateUnchanged)));
    }

    #[test]
    fn computable_refine_to_enforces_max_iterations() {
        let computable = Computable::new(
            0usize,
            |_| {
                Ok(OrderedPair::new(
                    ExtendedBinary::NegInf,
                    ExtendedBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let epsilon = bin(1, -1);
        let result = computable.refine_to::<5>(epsilon);
        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 5 })
        ));
    }

    // test the "normal case" where the bounds shrink but never meet
    #[test]
    fn computable_refine_to_handles_non_meeting_bounds() {
        let interval_state = OrderedPair::new(ext(0, 0), ext(4, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine_strict,
        );
        let epsilon = bin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let upper = bounds_upper(&bounds);
        assert!(bounds_lower(&bounds) < &upper);
        assert!(bounds_width_leq(&bounds, &epsilon));
        assert_eq!(computable.bounds().expect("bounds should succeed"), bounds);
    }

    #[test]
    fn computable_refine_to_rejects_worsened_bounds() {
        let interval_state = OrderedPair::new(ext(0, 0), ext(1, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            |inner_state: IntervalState| {
                let upper = inner_state.large();
                let worse_upper = unwrap_finite(&upper).add(&bin(1, 0));
                OrderedPair::new(
                    inner_state.small().clone(),
                    ExtendedBinary::Finite(worse_upper),
                )
            },
        );
        let epsilon = bin(1, -2);
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
        assert_eq!(sum_bounds, OrderedPair::new(ext(1, 0), ext(5, 0)));
    }

    #[test]
    fn computable_sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(diff_bounds, OrderedPair::new(ext(2, 0), ext(5, 0)));
    }

    #[test]
    fn computable_neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(bounds, OrderedPair::new(ext(-3, 0), ext(-1, 0)));
    }

    #[test]
    fn computable_inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(ExtendedBinary::NegInf, ExtendedBinary::PosInf)
        );
    }

    #[test]
    fn computable_inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = bin(1, -8);
        let bounds = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected_binary = ExtendedBinary::from_f64(1.0 / 3.0)
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
        assert_eq!(bounds, OrderedPair::new(ext(2, 0), ext(12, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, OrderedPair::new(ext(-12, 0), ext(-2, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, OrderedPair::new(ext(-10, 0), ext(15, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, OrderedPair::new(ext(-8, 0), ext(12, 0)));
    }

    #[test]
    fn computable_from_binary_matches_constant_bounds() {
        let value = bin(3, 0);
        let computable: Computable = value.clone().into();

        let bounds = computable.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            OrderedPair::new(
                ExtendedBinary::Finite(value.clone()),
                ExtendedBinary::Finite(value)
            )
        );
    }

    // --- test more complex expressions ---

    #[test]
    fn computable_integration_sqrt2_expression() {
        let one = Computable::constant(bin(1, 0));
        let sqrt2 = sqrt_computable(2);
        let expr = (sqrt2.clone() + one.clone()) * (sqrt2.clone() - one) + sqrt2.inv();

        let epsilon = bin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds_lower(&bounds));
        let upper = bounds_upper(&bounds);
        let upper = unwrap_finite(&upper);
        let expected = 1.0_f64 + 2.0_f64.sqrt().recip();
        let expected_binary = ExtendedBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon;

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }

    #[test]
    fn computable_shared_operand_in_expression() {
        let shared = sqrt_computable(2);
        let expr = shared.clone() + shared * Computable::constant(bin(1, 0));

        let epsilon = bin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds_lower(&bounds));
        let upper = bounds_upper(&bounds);
        let upper = unwrap_finite(&upper);
        let expected = 2.0_f64 * 2.0_f64.sqrt();
        let expected_binary = ExtendedBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon;

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
        let epsilon = bin(1, -12);

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
                Ok(OrderedPair::new(
                    ExtendedBinary::NegInf,
                    ExtendedBinary::PosInf,
                ))
            },
            |_| panic!("refiner panic"),
        );

        let epsilon = bin(1, -4);
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
                Ok(OrderedPair::new(
                    ExtendedBinary::NegInf,
                    ExtendedBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let right = Computable::new(
            0usize,
            |_| {
                Ok(OrderedPair::new(
                    ExtendedBinary::NegInf,
                    ExtendedBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let expr = left + right;
        let epsilon = bin(1, -4);
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
            OrderedPair::new(ext(0, 0), ext(1, 0)),
            |state| Ok(state.clone()),
            |state| OrderedPair::new(state.small().clone(), ext(2, 0)),
        );
        let expr = stable + faulty;
        let epsilon = bin(1, -4);
        let result = expr.refine_to::<3>(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    #[test]
    fn concurrent_bounds_reads_during_failed_refinement() {
        let computable = Arc::new(Computable::new(
            0usize,
            |_| {
                Ok(OrderedPair::new(
                    ExtendedBinary::NegInf,
                    ExtendedBinary::PosInf,
                ))
            },
            |state| state + 1,
        ));
        let epsilon = bin(1, -6);
        let reader = Arc::clone(&computable);
        let handle = thread::spawn(move || {
            for _ in 0..8 {
                let bounds = reader.bounds().expect("bounds should succeed");
                assert_bounds_ordered(&bounds);
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
                    Ok(OrderedPair::new(
                        ExtendedBinary::NegInf,
                        ExtendedBinary::PosInf,
                    ))
                },
                |state| {
                    thread::sleep(Duration::from_millis(SLEEP_MS));
                    state + 1
                },
            )
        };

        let expr = slow_refiner() + slow_refiner() + slow_refiner() + slow_refiner();
        let epsilon = bin(1, -6);

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
        let epsilon = bin(1, -10);
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
        let main_upper = bounds_upper(&main_bounds);
        assert!(bounds_width_leq(&main_bounds, &epsilon));

        for handle in handles {
            let bounds = handle
                .join()
                .expect("thread should join")
                .expect("refine_to should succeed");
            let bounds_upper = bounds_upper(&bounds);
            assert_bounds_ordered(&bounds);
            assert!(bounds_width_leq(&bounds, &epsilon));
            assert!(bounds_lower(&bounds) <= &main_upper);
            assert!(bounds_lower(&main_bounds) <= &bounds_upper);
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
            OrderedPair::new(ext(0, 0), ext(4, 0)),
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
        let epsilon = bin(1, -6);
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
            assert_bounds_ordered(&bounds);
        }

        assert!(!saw_overlap.load(Ordering::SeqCst));
        assert!(bounds_width_leq(&main_bounds, &epsilon));
    }

    #[test]
    fn concurrent_bounds_reads_during_refinement() {
        let base_value = interval_midpoint_computable(0, 4);
        let shared_value = Arc::new(base_value);
        let epsilon = bin(1, -8);
        // Reader thread repeatedly calls bounds while refinement is running.
        let barrier = Arc::new(Barrier::new(2));

        let reader = {
            let reader_value = Arc::clone(&shared_value);
            let reader_barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                reader_barrier.wait();
                for _ in 0..32 {
                    let bounds = reader_value.bounds().expect("bounds should succeed");
                    assert_bounds_ordered(&bounds);
                }
            })
        };

        barrier.wait();
        let refined = shared_value
            .refine_to_default(epsilon)
            .expect("refine_to should succeed");

        reader.join().expect("reader should join");
        assert_bounds_ordered(&refined);
    }
}
