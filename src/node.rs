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

use crate::binary::UXBinary;
use crate::error::ComputableError;
use crate::prefix::Prefix;

/// Shared API for retrieving prefix with lazy computation.
pub trait PrefixAccess {
    fn get_prefix(&self) -> Result<Prefix, ComputableError>;
}

/// Type-erased base node so we can store heterogeneous leaf states in a single graph.
/// This is also the hook for future user-defined base nodes.
pub trait BaseNode: Send + Sync {
    fn get_prefix(&self) -> Result<Prefix, ComputableError>;
    fn refine(&self) -> Result<(), ComputableError>;
}

/// Cached base state plus prefix derived from that state.
#[derive(Clone)]
struct BaseSnapshot<X> {
    state: X,
    prefix: Option<Prefix>,
}

/// Concrete base node that owns the user-provided state and refinement logic.
pub struct TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Prefix, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
{
    /// Snapshot ties a particular state with its computed prefix to avoid recomputation.
    snapshot: RwLock<BaseSnapshot<X>>,
    prefix_fn: B,
    refine: F,
}

impl<X, B, F> TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Prefix, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
{
    pub fn new(state: X, prefix_fn: B, refine: F) -> Self {
        Self {
            snapshot: RwLock::new(BaseSnapshot {
                state,
                prefix: None,
            }),
            prefix_fn,
            refine,
        }
    }

    fn snapshot_prefix(&self, snapshot: &mut BaseSnapshot<X>) -> Result<Prefix, ComputableError> {
        if let Some(prefix) = &snapshot.prefix {
            return Ok(prefix.clone());
        }
        let prefix = (self.prefix_fn)(&snapshot.state)?;
        snapshot.prefix = Some(prefix.clone());
        Ok(prefix)
    }
}

impl<X, B, F> BaseNode for TypedBaseNode<X, B, F>
where
    X: Eq + Clone + Send + Sync + 'static,
    B: Fn(&X) -> Result<Prefix, ComputableError> + Send + Sync + 'static,
    F: Fn(X) -> Result<X, ComputableError> + Send + Sync + 'static,
{
    /// Returns the prefix for the current base state, computing and caching if needed.
    fn get_prefix(&self) -> Result<Prefix, ComputableError> {
        let mut snapshot = self.snapshot.write();
        self.snapshot_prefix(&mut snapshot)
    }

    /// Refines the base state and computes the new prefix for that refined state.
    fn refine(&self) -> Result<(), ComputableError> {
        let mut snapshot = self.snapshot.write();
        let previous_prefix = self.snapshot_prefix(&mut snapshot)?;
        let previous_state = snapshot.state.clone();
        let next_state = (self.refine)(previous_state.clone())?;
        if next_state == previous_state {
            if previous_prefix.lower() == previous_prefix.upper() {
                return Ok(());
            }
            return Err(ComputableError::StateUnchanged);
        }

        let next_prefix = (self.prefix_fn)(&next_state)?;
        let lower_worsened = next_prefix.lower() < previous_prefix.lower();
        let upper_worsened = next_prefix.upper() > previous_prefix.upper();
        if lower_worsened || upper_worsened {
            return Err(ComputableError::BoundsWorsened);
        }

        snapshot.state = next_state;
        snapshot.prefix = Some(next_prefix);

        Ok(())
    }
}

impl<T: BaseNode + ?Sized> PrefixAccess for T {
    fn get_prefix(&self) -> Result<Prefix, ComputableError> {
        BaseNode::get_prefix(self)
    }
}

/// Node operator for composed computables.
///
/// This trait is currently internal. Operations like `SinOp`, `InvOp`, and `NthRootOp`
/// implement this trait to provide custom refinement logic beyond simple arithmetic.
// TODO: ensure it is possible to create user-defined composed nodes.
pub trait NodeOp: Send + Sync {
    fn compute_prefix(&self) -> Result<Prefix, ComputableError>;
    fn refine_step(&self, precision_bits: usize) -> Result<bool, ComputableError>;
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

    /// Whether this op's `child_demand_budget` depends on cached prefix.
    ///
    /// Returns `true` for ops like MulOp (budget depends on sibling's
    /// max_abs) and `false` for ops like AddOp (budget is just target/2).
    /// The coordinator uses this to skip budget recomputation for subtrees
    /// where budgets can't change as prefix tightens.
    fn budget_depends_on_prefix(&self) -> bool {
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

    pub fn notify_prefix_updated(&self) {
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
    pub prefix_cache: RwLock<Option<Prefix>>,
    pub refinement: RefinementSync,
}

impl Node {
    pub fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            prefix_cache: RwLock::new(None),
            refinement: RefinementSync::new(),
        })
    }

    /// Returns cached prefix if already computed.
    pub fn cached_prefix(&self) -> Option<Prefix> {
        self.prefix_cache.read().clone()
    }

    /// Returns the prefix, computing and caching if needed.
    /// Combinators are infallible, so prefix is lazily computed on demand.
    pub fn get_prefix(&self) -> Result<Prefix, ComputableError> {
        if let Some(prefix) = self.cached_prefix() {
            return Ok(prefix);
        }
        let prefix = self.compute_prefix()?;
        self.set_prefix(prefix.clone());
        Ok(prefix)
    }

    pub fn set_prefix(&self, prefix: Prefix) {
        let mut cache = self.prefix_cache.write();
        *cache = Some(prefix);
        self.refinement.notify_prefix_updated();
    }

    /// Computes prefix for this node from current children/base state.
    pub fn compute_prefix(&self) -> Result<Prefix, ComputableError> {
        self.op.compute_prefix()
    }

    /// Performs one refinement step. Returns whether refinement was applied.
    pub fn refine_step(&self, precision_bits: usize) -> Result<bool, ComputableError> {
        self.op.refine_step(precision_bits)
    }

    pub fn children(&self) -> Vec<Arc<Node>> {
        self.op.children()
    }

    pub fn is_refiner(&self) -> bool {
        self.op.is_refiner()
    }
}

impl PrefixAccess for Node {
    fn get_prefix(&self) -> Result<Prefix, ComputableError> {
        Node::get_prefix(self)
    }
}
