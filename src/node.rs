//! Computation graph nodes with lazy bounds computation and caching.
//!
//! This module provides the core abstractions for the computation graph:
//! - `BaseNode` trait for user-defined leaf nodes with custom refinement logic
//! - `TypedBaseNode` for type-erased storage of heterogeneous leaf states
//! - `Node` for the computation graph with bounds caching
//! - `NodeOp` trait for composable operations (arithmetic, transcendental, etc.)

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::{Condvar, Mutex, RwLock};

use crate::binary::Bounds;
use crate::error::ComputableError;

/// Shared API for retrieving bounds with lazy computation.
pub trait BoundsAccess {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
}

/// Type-erased base node so we can store heterogeneous leaf states in a single graph.
/// This is also the hook for future user-defined base nodes.
pub trait BaseNode: Send + Sync {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
    /// Refines the base state. Returns:
    /// - Ok(true) if state changed (progress made)
    /// - Ok(false) if state unchanged and bounds are exact (constant, no progress needed)
    /// - Err(StateUnchanged) if state unchanged but bounds are not exact
    /// - Err(...) for other errors
    fn refine(&self) -> Result<bool, ComputableError>;
}

/// Cached base state plus bounds derived from that state.
#[derive(Clone)]
struct BaseSnapshot<X> {
    state: X,
    bounds: Option<Bounds>,
}

/// Concrete base node that owns the user-provided state and refinement logic.
pub struct TypedBaseNode<X, B, F>
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

impl<X, B, F> TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Bounds, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> X + Send + Sync + 'static,
{
    pub fn new(state: X, bounds: B, refine: F) -> Self {
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
    /// Returns Ok(true) if state changed, Ok(false) if already converged (constant).
    fn refine(&self) -> Result<bool, ComputableError> {
        let mut snapshot = self.snapshot.write();
        let previous_bounds = self.snapshot_bounds(&mut snapshot)?;
        let previous_state = snapshot.state.clone();
        let next_state = (self.refine)(previous_state.clone());
        if next_state == previous_state {
            if previous_bounds.small() == &previous_bounds.large() {
                // Constant case - no progress needed
                return Ok(false);
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

        Ok(true)
    }
}

impl<T: BaseNode + ?Sized> BoundsAccess for T {
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        BaseNode::get_bounds(self)
    }
}

/// Node operator for composed computables.
///
/// This trait is currently internal. Operations like `SinOp`, `InvOp`, and `NthRootOp`
/// implement this trait to provide custom refinement logic beyond simple arithmetic.
// TODO: ensure it is possible to create user-defined composed nodes.
pub trait NodeOp: Send + Sync {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError>;
    fn refine_step(&self) -> Result<bool, ComputableError>;
    fn children(&self) -> Vec<Arc<Node>>;
    fn is_refiner(&self) -> bool;
}

/// Synchronization state for coordinating refinement across threads.
pub struct RefinementSync {
    pub state: Mutex<RefinementState>,
    pub condvar: Condvar,
}

/// State tracking for active refinement.
pub struct RefinementState {
    pub active: bool,
    pub epoch: u64,
}

impl RefinementSync {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(RefinementState {
                active: false,
                epoch: 0,
            }),
            condvar: Condvar::new(),
        }
    }

    pub fn notify_bounds_updated(&self) {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        self.condvar.notify_all();
    }
}

impl Default for RefinementSync {
    fn default() -> Self {
        Self::new()
    }
}

/// Node in the computation graph. The op stores structure/state; the cache stores
/// the last bounds computed for this node.
///
/// NOTE: The bounds_cache is not automatically invalidated when children are refined.
/// Updates are explicitly propagated via apply_update during refinement. If get_bounds()
/// is called between refinement steps (outside of refine_to), it may return stale cached
/// values. Consider whether this is acceptable for your use case.
pub struct Node {
    pub id: usize,
    pub op: Arc<dyn NodeOp>,
    pub bounds_cache: RwLock<Option<Bounds>>,
    pub refinement: RefinementSync,
}

impl Node {
    pub fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            bounds_cache: RwLock::new(None),
            refinement: RefinementSync::new(),
        })
    }

    /// Returns cached bounds if already computed.
    pub fn cached_bounds(&self) -> Option<Bounds> {
        self.bounds_cache.read().clone()
    }

    /// Returns cached bounds, computing and caching if needed.
    /// Combinators are infallible, so bounds are lazily computed on demand.
    pub fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        if let Some(bounds) = self.cached_bounds() {
            return Ok(bounds);
        }
        let bounds = self.compute_bounds()?;
        self.set_bounds(bounds.clone());
        Ok(bounds)
    }

    pub fn set_bounds(&self, bounds: Bounds) {
        let mut cache = self.bounds_cache.write();
        *cache = Some(bounds);
        self.refinement.notify_bounds_updated();
    }

    /// Computes bounds for this node from current children/base state.
    pub fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
        self.op.compute_bounds()
    }

    /// Performs one refinement step. Returns whether refinement was applied.
    pub fn refine_step(&self) -> Result<bool, ComputableError> {
        self.op.refine_step()
    }

    pub fn children(&self) -> Vec<Arc<Node>> {
        self.op.children()
    }

    pub fn is_refiner(&self) -> bool {
        self.op.is_refiner()
    }
}

impl BoundsAccess for Node {
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        Node::get_bounds(self)
    }
}
