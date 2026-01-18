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

pub use binary::{
    shortest_binary_in_bounds, Binary, BinaryError, UBinary, UXBinary, XBinary, XBinaryError,
};
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
// doubles on each refine_step. This is simple but potentially inefficient:
// - For a given epsilon, we don't know how many bits are needed upfront
// - Each step recomputes the reciprocal from scratch at the new precision
// Consider: adaptive precision based on current bounds width, or Newton-Raphson iteration.
struct InvOp {
    inner: Arc<Node>,
    precision_bits: RwLock<BigInt>,
}

/// Initial precision bits to start with for inv refinement.
/// Starting at a reasonable value avoids unnecessary early iterations.
const INV_INITIAL_PRECISION_BITS: i64 = 4;

impl NodeOp for InvOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let existing = self.inner.get_bounds()?;
        reciprocal_bounds(&existing, &self.precision_bits.read())
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut precision = self.precision_bits.write();
        // Double precision each step for O(log n) convergence.
        // If precision is 0, start with initial value to bootstrap.
        // Once the TODO above is implemented (reusing precision calculation state),
        // this should be changed back to linear increment to avoid unnecessary
        // computation to higher precision than requested.
        if precision.is_zero() {
            *precision = BigInt::from(INV_INITIAL_PRECISION_BITS);
        } else {
            *precision *= 2;
        }
        Ok(true)
    }

    fn children(&self) -> Vec<Arc<Node>> {
        vec![Arc::clone(&self.inner)]
    }

    fn is_refiner(&self) -> bool {
        true
    }
}

struct SinOp {
    inner: Arc<Node>,
    num_terms: RwLock<BigInt>,
}

impl NodeOp for SinOp {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        let input_bounds = self.inner.get_bounds()?;
        let num_terms = self.num_terms.read().clone();
        sin_bounds(&input_bounds, &num_terms)
    }

    fn refine_step(&self) -> Result<bool, ComputableError> {
        let mut num_terms = self.num_terms.write();
        *num_terms += BigInt::one();
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

// TODO: make this a computable number so that the results remain provably correct (right now they're logically incorrect because of the approximation used for pi!)
/// Computes bounds for sin(x) using Taylor series with rigorous error bounds.
///
/// Returns a high-precision representation of π as a Binary number.
/// Uses ~64 bits of precision (~19 decimal digits).
fn pi_binary() -> Binary {
    // π * 2^61 ≈ 7244019458077122842.70...
    // 7244019458077122843 is odd (ends in 3)
    let mantissa = BigInt::parse_bytes(b"7244019458077122843", 10)
        .unwrap_or_else(|| BigInt::from(3));
    Binary::new(mantissa, BigInt::from(-61))
}

/// Returns 2π as a Binary number (for range reduction).
fn two_pi_binary() -> Binary {
    let pi = pi_binary();
    Binary::new(pi.mantissa().clone(), pi.exponent() + BigInt::one())
}

/// Reduces x to the range [-π, π] by subtracting multiples of 2π.
fn reduce_to_pi_range(x: &Binary) -> Binary {
    use num_traits::Signed;

    let two_pi = two_pi_binary();
    let pi = pi_binary();

    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };

    if abs_x <= pi {
        return x.clone();
    }

    let k = compute_reduction_factor(x, &two_pi);
    let k_times_two_pi = multiply_by_integer(&two_pi, &k);
    x.sub(&k_times_two_pi)
}

/// Reduces x to the range [-π/2, π/2] and returns (reduced_x, sign_flip).
/// sign_flip indicates whether the final sin value needs to be negated.
fn reduce_to_half_pi_range(x: &Binary) -> (Binary, bool) {
    let pi = pi_binary();
    let half_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::one());
    let neg_half_pi = half_pi.neg();

    let reduced = reduce_to_pi_range(x);

    if reduced > half_pi {
        // x in (π/2, π]: use sin(x) = sin(π - x)
        (pi.sub(&reduced), false)
    } else if reduced < neg_half_pi {
        // x in [-π, -π/2): use sin(x) = -sin(π + x)
        (pi.add(&reduced), true)
    } else {
        (reduced, false)
    }
}

/// Computes k = round(x / period).
fn compute_reduction_factor(x: &Binary, period: &Binary) -> BigInt {
    use num_traits::Signed;

    let precision_bits = 64i64;
    let mx = x.mantissa();
    let ex = x.exponent();
    let mp = period.mantissa();
    let ep = period.exponent();

    let shifted_mx = mx << precision_bits as usize;
    let quotient = &shifted_mx / mp;
    let result_exp = ex - ep - BigInt::from(precision_bits);

    if result_exp >= BigInt::zero() {
        use num_traits::ToPrimitive;
        let shift = result_exp.to_usize().unwrap_or(0);
        &quotient << shift
    } else {
        use num_traits::ToPrimitive;
        let shift = (-&result_exp).to_usize().unwrap_or(0);
        if shift == 0 {
            quotient.clone()
        } else {
            let half = BigInt::one() << (shift - 1);
            let rounded = if quotient.is_negative() {
                &quotient - &half
            } else {
                &quotient + &half
            };
            rounded >> shift
        }
    }
}

/// Multiplies a Binary by a BigInt integer.
fn multiply_by_integer(b: &Binary, k: &BigInt) -> Binary {
    Binary::new(b.mantissa() * k, b.exponent().clone())
}

/// Truncates a Binary to at most `precision_bits` of mantissa.
fn truncate_precision(x: &Binary, precision_bits: usize) -> Binary {
    let mantissa = x.mantissa();
    let exponent = x.exponent();
    let bit_length = mantissa.magnitude().bits() as usize;

    if bit_length <= precision_bits {
        return x.clone();
    }

    let shift = bit_length - precision_bits;
    let truncated_mantissa = mantissa >> shift;
    let new_exponent = exponent + BigInt::from(shift);
    Binary::new(truncated_mantissa, new_exponent)
}

/// Checks if an interval [a, b] contains critical points of sin (where sin = ±1).
/// Returns (contains_max, contains_min).
fn interval_contains_critical_points(lower: &Binary, upper: &Binary) -> (bool, bool) {
    let pi = pi_binary();
    let two_pi = two_pi_binary();
    let half_pi = Binary::new(pi.mantissa().clone(), pi.exponent() - BigInt::one());
    let neg_half_pi = half_pi.neg();

    let width = upper.sub(lower);
    if width >= two_pi {
        return (true, true);
    }

    let reduced_lower = reduce_to_pi_range(lower);
    let reduced_upper = reduced_lower.add(&width);

    let contains_max = interval_contains_point(&reduced_lower, &reduced_upper, &half_pi, &two_pi);
    let contains_min = interval_contains_point(&reduced_lower, &reduced_upper, &neg_half_pi, &two_pi);

    (contains_max, contains_min)
}

/// Checks if an interval [a, b] contains a point p (or p + k*period for any integer k).
fn interval_contains_point(lower: &Binary, upper: &Binary, point: &Binary, period: &Binary) -> bool {
    let mut p = point.clone();

    while p < lower.sub(period) {
        p = p.add(period);
    }
    while p > upper.add(period) {
        p = p.sub(period);
    }

    if &p >= lower && &p <= upper {
        return true;
    }
    let p_plus = p.add(period);
    if &p_plus >= lower && &p_plus <= upper {
        return true;
    }
    let p_minus = p.sub(period);
    &p_minus >= lower && &p_minus <= upper
}

/// The Taylor series is: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
/// After n terms, the error is bounded by |x|^(2n+1)/(2n+1)!
///
/// This implementation uses:
/// - Range reduction to [-π/2, π/2] for efficient Taylor series convergence
/// - Critical point detection for tight bounds on intervals containing extrema
/// - Directed rounding for provably correct interval arithmetic
fn sin_bounds(input_bounds: &Bounds, num_terms: &BigInt) -> Result<Bounds, ComputableError> {
    use num_traits::ToPrimitive;

    let neg_one = Binary::new(BigInt::from(-1), BigInt::zero());
    let pos_one = Binary::new(BigInt::from(1), BigInt::zero());

    // Extract finite bounds, or return [-1, 1] for any infinite bounds
    let lower = input_bounds.small();
    let upper = input_bounds.large();
    let (lower_bin, upper_bin) = match (lower, &upper) {
        (XBinary::Finite(l), XBinary::Finite(u)) => (l, u),
        _ => {
            return Ok(Bounds::new(
                XBinary::Finite(neg_one),
                XBinary::Finite(pos_one),
            ));
        }
    };

    // Check for critical points (where sin reaches ±1)
    let (contains_max, contains_min) = interval_contains_critical_points(lower_bin, upper_bin);

    // Convert num_terms to usize for computation (capped at reasonable limit)
    let n = num_terms.to_usize().unwrap_or(1).max(1);

    // Apply range reduction to both endpoints for efficient Taylor series
    let (reduced_lower, lower_sign_flip) = reduce_to_half_pi_range(lower_bin);
    let (reduced_upper, upper_sign_flip) = reduce_to_half_pi_range(upper_bin);

    // TODO: This truncation loses precision and isn't accounted for in error bounds.
    // For provable correctness, either remove truncation or add the truncation error
    // (at most 2^(-64) * |reduced_value|) to the final error bound (while increasing the
    // precision bits so that the answer still converges instead of remaining stuck at 64 bits of precision).
    // Truncate to 64 bits to keep mantissas manageable
    let reduced_lower = truncate_precision(&reduced_lower, 64);
    let reduced_upper = truncate_precision(&reduced_upper, 64);

    // Compute Taylor series bounds on reduced values
    let sin_lower_raw = taylor_sin_bounds(&reduced_lower, n);
    let sin_upper_raw = taylor_sin_bounds(&reduced_upper, n);

    // Apply sign flips if needed
    let (sin_lower_lo, sin_lower_hi) = if lower_sign_flip {
        (sin_lower_raw.1.neg(), sin_lower_raw.0.neg())
    } else {
        sin_lower_raw
    };
    let (sin_upper_lo, sin_upper_hi) = if upper_sign_flip {
        (sin_upper_raw.1.neg(), sin_upper_raw.0.neg())
    } else {
        sin_upper_raw
    };

    // Combine endpoint bounds
    let mut result_lower = if sin_lower_lo <= sin_upper_lo {
        sin_lower_lo
    } else {
        sin_upper_lo
    };
    let mut result_upper = if sin_lower_hi >= sin_upper_hi {
        sin_lower_hi
    } else {
        sin_upper_hi
    };

    // If interval contains critical points, extend bounds accordingly
    if contains_max {
        result_upper = pos_one.clone();
    }
    if contains_min {
        result_lower = neg_one.clone();
    }

    // Final clamp to [-1, 1]
    if result_lower < neg_one {
        result_lower = neg_one.clone();
    }
    if result_upper > pos_one {
        result_upper = pos_one;
    }

    Bounds::new_checked(XBinary::Finite(result_lower), XBinary::Finite(result_upper))
        .map_err(|_| ComputableError::InvalidBoundsOrder)
}

/// Rounding direction for directed rounding in interval arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundingDirection {
    /// Round toward negative infinity (floor)
    Down,
    /// Round toward positive infinity (ceiling)
    Up,
}

/// Computes Taylor series bounds for sin(x) with n terms.
/// Returns (lower_bound, upper_bound) accounting for truncation error.
///
/// Taylor series: sin(x) = sum_{k=0}^{n-1} (-1)^k * x^(2k+1) / (2k+1)!
/// Error after n terms: |R_n| <= |x|^(2n+1) / (2n+1)!
///
/// Uses directed rounding to compute provably correct bounds:
/// - Lower bound: all intermediate operations round DOWN (toward -inf)
/// - Upper bound: all intermediate operations round UP (toward +inf)
fn taylor_sin_bounds(x: &Binary, n: usize) -> (Binary, Binary) {
    if n == 0 {
        // No terms: just use error bound (always round UP for conservative bounds)
        let error = taylor_error_bound(x, 0);
        return (error.neg(), error);
    }

    // Compute lower and upper partial sums with directed rounding
    let sum_lower = taylor_sin_partial_sum(x, n, RoundingDirection::Down);
    let sum_upper = taylor_sin_partial_sum(x, n, RoundingDirection::Up);

    // Compute error bound (always round UP for conservative bounds)
    let error = taylor_error_bound(x, n);

    // Return bounds: lower_sum - error, upper_sum + error
    (sum_lower.sub(&error), sum_upper.add(&error))
}

/// Computes Taylor series partial sum for sin(x) with directed rounding.
///
/// For RoundingDirection::Down: rounds all division operations toward -infinity
/// For RoundingDirection::Up: rounds all division operations toward +infinity
fn taylor_sin_partial_sum(x: &Binary, n: usize, rounding: RoundingDirection) -> Binary {
    let mut sum = Binary::zero();
    let mut power = x.clone(); // x^1
    let mut factorial = BigInt::one(); // 1!

    for k in 0..n {
        // Term k: (-1)^k * x^(2k+1) / (2k+1)!
        let term_num = if k % 2 == 0 {
            power.clone()
        } else {
            power.neg()
        };

        // Divide by factorial with directed rounding
        let term = divide_by_factorial_directed(&term_num, &factorial, rounding);
        sum = sum.add(&term);

        // Prepare for next term: multiply power by x^2
        if k + 1 < n {
            power = power.mul(x).mul(x);
            // factorial *= (2k+2) * (2k+3)
            let next_k = k + 1;
            factorial *= BigInt::from(2 * next_k) * BigInt::from(2 * next_k + 1);
        }
    }

    sum
}

/// Computes |x|^(2n+1) / (2n+1)! as an upper bound on Taylor series truncation error.
/// Always rounds UP to be conservative.
fn taylor_error_bound(x: &Binary, n: usize) -> Binary {
    use num_traits::Signed;

    // Compute |x|^(2n+1)
    let abs_x = if x.mantissa().is_negative() {
        x.neg()
    } else {
        x.clone()
    };

    let exp = 2 * n + 1;
    let mut power = Binary::new(BigInt::one(), BigInt::zero()); // 1
    for _ in 0..exp {
        power = power.mul(&abs_x);
    }

    // Compute (2n+1)!
    let mut factorial = BigInt::one();
    for i in 1..=exp {
        factorial *= BigInt::from(i);
    }

    // error = power / factorial (round UP for conservative error bound)
    divide_by_factorial_directed(&power, &factorial, RoundingDirection::Up)
}

/// Divides a Binary by a BigInt factorial with directed rounding.
///
/// Rounding semantics:
/// - `RoundingDirection::Up`: rounds toward +infinity (ceiling)
/// - `RoundingDirection::Down`: rounds toward -infinity (floor)
///
/// This is essential for interval arithmetic: when computing a lower bound,
/// round DOWN; when computing an upper bound, round UP.
fn divide_by_factorial_directed(
    value: &Binary,
    factorial: &BigInt,
    rounding: RoundingDirection,
) -> Binary {
    use num_integer::Integer;
    use num_traits::Signed;

    if factorial.is_zero() {
        return value.clone();
    }

    let mantissa = value.mantissa();
    let exponent = value.exponent();

    // We need to compute mantissa / factorial with the result as a Binary.
    // To get a good approximation, we shift the mantissa up by some bits before dividing.
    // The number of bits we shift determines our precision.
    let precision_bits = 64_u64; // Extra precision for intermediate computation

    // shifted_mantissa = |mantissa| * 2^precision_bits
    let abs_mantissa = mantissa.magnitude().clone();
    let shifted_mantissa = &abs_mantissa << precision_bits as usize;

    // Compute |mantissa| / factorial
    let (quot, rem) = shifted_mantissa.div_rem(factorial.magnitude());

    // Determine how to round based on direction and sign
    // For directed rounding toward +/- infinity:
    // - Round UP (+inf): positive values round away from zero, negative round toward zero
    // - Round DOWN (-inf): positive values round toward zero, negative round away from zero
    let is_negative = mantissa.is_negative();
    let has_remainder = !rem.is_zero();

    let result_magnitude = if has_remainder {
        match (rounding, is_negative) {
            // Rounding UP (toward +infinity):
            // - Positive: round away from zero (add 1)
            // - Negative: round toward zero (truncate)
            (RoundingDirection::Up, false) => quot + BigInt::one().magnitude(),
            (RoundingDirection::Up, true) => quot,
            // Rounding DOWN (toward -infinity):
            // - Positive: round toward zero (truncate)
            // - Negative: round away from zero (add 1)
            (RoundingDirection::Down, false) => quot,
            (RoundingDirection::Down, true) => quot + BigInt::one().magnitude(),
        }
    } else {
        // Exact division, no rounding needed
        quot
    };

    // Adjust sign
    let signed_mantissa = if is_negative {
        -BigInt::from(result_magnitude)
    } else {
        BigInt::from(result_magnitude)
    };

    // New exponent = original_exponent - precision_bits
    let new_exponent = exponent - BigInt::from(precision_bits);

    Binary::new(signed_mantissa, new_exponent)
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
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    fn unwrap_finite_uxbinary(input: &UXBinary) -> UBinary {
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
        let width = unwrap_finite_uxbinary(bounds.width());

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

    #[test]
    fn directed_rounding_produces_valid_bounds() {
        // Test that directed rounding produces well-ordered bounds that contain the true value.
        //
        // Key invariants:
        // 1. lower <= upper (bounds are ordered)
        // 2. lower_sum <= upper_sum (directed rounding produces correct ordering)
        // 3. The bounds interval width decreases with more terms
        // 4. Bounds remain within [-1, 1] (sin range)

        let test_cases = [
            bin(1, -2),   // 0.25
            bin(1, 0),    // 1.0
            bin(3, 0),    // 3.0
            bin(-1, 0),   // -1.0
            bin(5, -1),   // 2.5
            bin(-3, -1),  // -1.5
        ];

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        for x in &test_cases {
            // Compute Taylor bounds with directed rounding
            let (lower, upper) = taylor_sin_bounds(x, 10);

            // Verify bounds are ordered correctly
            assert!(
                lower <= upper,
                "Lower bound {} should be <= upper bound {} for x = {}",
                lower, upper, x
            );

            // Verify bounds are within sin's range [-1, 1]
            assert!(
                lower >= neg_one,
                "Lower bound {} should be >= -1 for x = {}",
                lower, x
            );
            assert!(
                upper <= one,
                "Upper bound {} should be <= 1 for x = {}",
                upper, x
            );
        }
    }

    #[test]
    fn directed_rounding_bounds_converge() {
        // Verify that bounds get tighter as we add more terms
        let x = bin(1, 0); // 1.0

        let (lower5, upper5) = taylor_sin_bounds(&x, 5);
        let (lower10, upper10) = taylor_sin_bounds(&x, 10);

        let width5 = upper5.sub(&lower5);
        let width10 = upper10.sub(&lower10);

        // More terms should give tighter bounds
        assert!(
            width10 < width5,
            "Bounds with 10 terms (width {}) should be tighter than 5 terms (width {})",
            width10, width5
        );
    }

    #[test]
    fn directed_rounding_symmetry() {
        // Test that sin(-x) bounds are the negation of sin(x) bounds
        // This verifies that the directed rounding handles negative inputs correctly

        let x = bin(1, -2); // 0.25
        let neg_x = bin(-1, -2); // -0.25

        let (lower_x, upper_x) = taylor_sin_bounds(&x, 10);
        let (lower_neg_x, upper_neg_x) = taylor_sin_bounds(&neg_x, 10);

        // sin(-x) = -sin(x), so bounds should be negated and swapped
        // lower(-x) should equal -upper(x)
        // upper(-x) should equal -lower(x)

        // Allow small differences due to rounding
        let neg_upper_x = upper_x.neg();
        let neg_lower_x = lower_x.neg();

        // The bounds should be approximately symmetric
        // We just verify they're in the right ballpark
        assert!(
            lower_neg_x <= neg_upper_x.add(&bin(1, -50)),
            "lower(sin(-x)) should be approximately -upper(sin(x))"
        );
        assert!(
            neg_lower_x <= upper_neg_x.add(&bin(1, -50)),
            "-lower(sin(x)) should be approximately upper(sin(-x))"
        );
    }

    #[test]
    fn directed_rounding_lower_bound_is_lower() {
        // Verify that rounding down produces smaller values than rounding up
        let x = bin(1, 0); // 1.0
        let n = 5;

        let sum_down = taylor_sin_partial_sum(&x, n, RoundingDirection::Down);
        let sum_up = taylor_sin_partial_sum(&x, n, RoundingDirection::Up);

        // The down-rounded sum should be <= up-rounded sum
        assert!(
            sum_down <= sum_up,
            "Rounding down {} should produce <= rounding up {}",
            sum_down, sum_up
        );
    }
}
