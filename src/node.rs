//! Computation graph nodes with lazy prefix computation and caching.
//!
//! This module provides the core abstractions for the computation graph:
//! - `BaseNode` trait for user-defined leaf nodes with custom refinement logic
//! - `TypedBaseNode` for type-erased storage of heterogeneous leaf states
//! - `Node` for the computation graph with prefix caching
//! - `NodeOp` trait for composable operations (arithmetic, transcendental, etc.)

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::{Condvar, Mutex, RwLock};

use crate::binary::{Bounds, UXBinary};
use crate::error::ComputableError;
use crate::prefix::Prefix;
use crate::sane::XIsize;

/// Shared API for retrieving bounds with lazy computation.
pub trait BoundsAccess {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
}

/// Type-erased base node so we can store heterogeneous leaf states in a single graph.
/// This is also the hook for future user-defined base nodes.
pub trait BaseNode: Send + Sync {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
    fn refine(&self) -> Result<(), ComputableError>;
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
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
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
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
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
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
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
        let next_state = (self.refine)(previous_state.clone())?;
        if next_state == previous_state {
            if previous_bounds.small() == previous_bounds.large() {
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

/// Node operator for composed computables.
///
/// This trait is currently internal. Operations like `SinOp`, `InvOp`, and `NthRootOp`
/// implement this trait to provide custom refinement logic beyond simple arithmetic.
// TODO: ensure it is possible to create user-defined composed nodes.
pub trait NodeOp: Send + Sync {
    fn compute_bounds(&self) -> Result<Bounds, ComputableError>;

    /// Computes prefix for this node. Default converts compute_bounds() result.
    fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        let bounds = self.compute_bounds()?;
        Ok(Prefix::from_lower_upper(
            bounds.small().clone(),
            bounds.large().clone(),
        ))
    }

    /// Performs one refinement step targeting the given width exponent.
    ///
    /// Contract: after returning `Ok(true)`, `compute_prefix()` should return
    /// a Prefix with `width_exponent < current` (at least 1 bit improvement).
    /// Ideally `width_exponent ≤ target_width_exp`.
    fn refine_step(&self, target_width_exp: XIsize) -> Result<bool, ComputableError>;

    fn children(&self) -> Vec<Arc<Node>>;
    fn is_refiner(&self) -> bool;

    /// Computes the demand budget for a child node given a target width for
    /// this node's output.
    ///
    /// Returns the maximum width the child at `child_index` can have while
    /// still allowing this node to meet the target. Every non-leaf NodeOp
    /// must implement this — there is no default, so forgetting to implement
    /// it for a new operation is a compile error.
    fn child_demand_budget(&self, target_width: &UXBinary, child_index: usize) -> UXBinary;

    /// Whether this op's `child_demand_budget` depends on cached bounds.
    ///
    /// Returns `true` for ops like MulOp (budget depends on sibling's
    /// max_abs) and `false` for ops like AddOp (budget is just target/2).
    /// The coordinator uses this to skip budget recomputation for subtrees
    /// where budgets can't change as bounds tighten.
    fn budget_depends_on_bounds(&self) -> bool {
        false
    }
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
/// the last prefix computed for this node.
///
/// NOTE: The prefix_cache is not automatically invalidated when children are refined.
/// Updates are explicitly propagated via apply_update during refinement. If get_prefix()
/// is called between refinement steps (outside of refine_to), it may return stale cached
/// values. Consider whether this is acceptable for your use case.
pub struct Node {
    pub id: usize,
    pub op: Arc<dyn NodeOp>,
    /// Prefix cache for the refinement system (power-of-2 width).
    pub prefix_cache: RwLock<Option<Prefix>>,
    /// Exact bounds cache for arithmetic operations.
    bounds_cache: RwLock<Option<Bounds>>,
    pub refinement: RefinementSync,
}

impl Node {
    pub fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            prefix_cache: RwLock::new(None),
            bounds_cache: RwLock::new(None),
            refinement: RefinementSync::new(),
        })
    }

    /// Returns cached prefix if already computed.
    pub fn cached_prefix(&self) -> Option<Prefix> {
        self.prefix_cache.read().clone()
    }

    /// Returns cached bounds if already computed.
    pub fn cached_bounds(&self) -> Option<Bounds> {
        self.bounds_cache.read().clone()
    }

    /// Returns exact bounds, computing and caching if needed.
    ///
    /// Only caches bounds (not prefix) to avoid the overhead of
    /// `Prefix::from_lower_upper` on every intermediate node during
    /// cascading evaluation. The prefix is derived lazily by
    /// `get_prefix()` when needed.
    pub fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        if let Some(bounds) = self.cached_bounds() {
            return Ok(bounds);
        }
        let bounds = self.op.compute_bounds()?;
        {
            let mut cache = self.bounds_cache.write();
            *cache = Some(bounds.clone());
        }
        Ok(bounds)
    }

    /// Returns cached prefix, computing and caching if needed.
    ///
    /// If bounds are already cached (from `get_bounds()`), derives the
    /// prefix from those bounds without recomputing them. This avoids
    /// redundant `compute_bounds()` calls during cascading evaluation
    /// where `get_bounds()` has already been called on children.
    ///
    /// Uses direct cache writes (no condvar notification) since this is
    /// initial lazy evaluation, not a refinement update. Notifications
    /// are only needed when bounds *change* during refinement (via
    /// `set_prefix` / `set_prefix_and_bounds`).
    pub fn get_prefix(&self) -> Result<Prefix, ComputableError> {
        if let Some(prefix) = self.cached_prefix() {
            return Ok(prefix);
        }
        // If bounds are cached, derive prefix from them. This is
        // cheaper than compute_prefix() which would re-call
        // compute_bounds() on children.
        let prefix = if let Some(bounds) = self.cached_bounds() {
            Prefix::from_lower_upper(bounds.small().clone(), bounds.large().clone())
        } else {
            self.compute_prefix()?
        };
        {
            let mut cache = self.prefix_cache.write();
            *cache = Some(prefix.clone());
        }
        Ok(prefix)
    }

    #[allow(dead_code)]
    pub fn set_prefix(&self, prefix: Prefix) {
        {
            let mut cache = self.prefix_cache.write();
            *cache = Some(prefix);
        }
        // Invalidate bounds cache — bounds derived from children may be stale.
        {
            let mut cache = self.bounds_cache.write();
            *cache = None;
        }
        self.refinement.notify_bounds_updated();
    }

    /// Sets the exact bounds cache and derives prefix from it.
    #[allow(dead_code)]
    pub fn set_bounds(&self, bounds: Bounds) {
        let prefix = Prefix::from_lower_upper(bounds.small().clone(), bounds.large().clone());
        self.set_prefix_and_bounds(prefix, bounds);
    }

    /// Sets both prefix and exact bounds caches without re-deriving the prefix.
    ///
    /// Use this when the prefix has already been computed from the bounds
    /// (e.g. during propagation where the prefix is needed for comparison
    /// before deciding whether to update). Avoids the redundant
    /// `Prefix::from_lower_upper` that `set_bounds` would perform.
    pub fn set_prefix_and_bounds(&self, prefix: Prefix, bounds: Bounds) {
        {
            let mut cache = self.prefix_cache.write();
            *cache = Some(prefix);
        }
        {
            let mut cache = self.bounds_cache.write();
            *cache = Some(bounds);
        }
        self.refinement.notify_bounds_updated();
    }

    /// Computes prefix for this node from current children/base state.
    pub fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        self.op.compute_prefix()
    }

    /// Performs one refinement step targeting the given width exponent.
    pub fn refine_step(&self, target_width_exp: XIsize) -> Result<bool, ComputableError> {
        self.op.refine_step(target_width_exp)
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
