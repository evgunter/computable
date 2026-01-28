//! Computation graph nodes with lazy bounds computation and caching.
//!
//! This module provides the core abstractions for the computation graph:
//! - `BaseNode` trait for user-defined leaf nodes with custom refinement logic
//! - `TypedBaseNode` for type-erased storage of heterogeneous leaf states
//! - `Node` for the computation graph with bounds caching
//! - `NodeOp` trait for composable operations (arithmetic, transcendental, etc.)
//!
//! ## Blackholing for Concurrent Lazy Evaluation
//!
//! Inspired by GHC's runtime system, this module implements "blackholing" for lazy
//! bounds computation. When a thread starts computing bounds for a node:
//!
//! 1. The node is "blackholed" - marked as being evaluated
//! 2. Other threads that try to get bounds will wait rather than duplicate work
//! 3. When computation completes, all waiters are notified and receive the result
//! 4. If computation fails, the error is propagated to all waiters
//!
//! This prevents redundant computation in concurrent evaluation scenarios while
//! maintaining correct semantics for lazy evaluation.
//!
//! ## Blackholing for Refinement
//!
//! The blackholing pattern is also extended to refinement operations. When a thread
//! starts refining a node:
//!
//! 1. The node's refinement is "blackholed" - marked as being refined
//! 2. Other threads that try to refine wait rather than duplicate work
//! 3. When refinement completes, waiters are notified with the new precision level
//! 4. Precision is tracked monotonically - it can only increase, never decrease
//!
//! This creates a "thunk-like" refinement model where:
//! - Bounds are revealed progressively (like digits in a stream)
//! - Previous precision is never lost
//! - Concurrent refinement is coordinated efficiently

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::{Condvar, Mutex, RwLock};

use crate::binary::Bounds;
use crate::error::ComputableError;
use crate::normalized::{BoundsBlackhole, ClaimResult, PrecisionLevel};

/// Shared API for retrieving bounds with lazy computation.
pub trait BoundsAccess {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
}

// ============================================================================
// Blackhole - now imported from normalized.rs
// ============================================================================
//
// The unified BoundsBlackhole in normalized.rs replaces the previous separate
// Blackhole (for bounds caching) and RefinementBlackhole (for precision tracking).
// See normalized.rs for the implementation.

/// Re-export for backward compatibility with existing code.
pub type Blackhole = BoundsBlackhole;

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
///
/// This tracks whether a top-level refinement is active and provides
/// epoch-based change detection. The actual bounds/precision tracking
/// is now unified in the Node's BoundsBlackhole.
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

/// Node in the computation graph. The op stores structure/state; the blackhole
/// provides lazy bounds computation with proper concurrent synchronization.
///
/// ## Unified Blackholing
///
/// The `blackhole` field implements GHC-style blackholing for BOTH:
/// - **Initial computation**: When bounds are first requested, one thread computes
/// - **Refinement**: When higher precision is requested, one thread refines
///
/// The blackhole tracks both the cached bounds AND the precision level, providing
/// a unified API for lazy evaluation and incremental refinement.
///
/// ## Usage
///
/// - `get_bounds()`: Returns cached bounds, computing if needed (blocks)
/// - `get_bounds_with_precision(target)`: Refines to target precision (blocks)
/// - `cached_bounds()`: Peek at cached bounds without blocking
/// - `set_bounds(bounds)`: Update bounds (used during refinement propagation)
///
/// NOTE: The blackhole is not automatically invalidated when children are refined.
/// Updates are explicitly propagated via apply_update during refinement.
pub struct Node {
    pub id: usize,
    pub op: Arc<dyn NodeOp>,
    /// Unified blackhole for bounds computation AND precision tracking.
    pub blackhole: BoundsBlackhole,
    pub refinement: RefinementSync,
}

impl Node {
    pub fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            blackhole: BoundsBlackhole::new(),
            refinement: RefinementSync::new(),
        })
    }

    /// Returns cached bounds if already computed, without blocking.
    ///
    /// This is a non-blocking peek at the blackhole state. Returns `None` if:
    /// - Bounds have never been computed (NotEvaluated)
    /// - Another thread is currently computing (BeingEvaluated)
    /// - A previous computation failed (Failed)
    ///
    /// Use `get_bounds()` for blocking semantics that will compute if needed.
    pub fn cached_bounds(&self) -> Option<Bounds> {
        self.blackhole.peek()
    }

    /// Returns the current precision level of cached bounds.
    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        self.blackhole.current_precision()
    }

    /// Returns cached bounds, computing and caching if needed.
    ///
    /// This is the main entry point for lazy bounds computation with blackholing:
    ///
    /// 1. If bounds are already computed, returns them immediately
    /// 2. If another thread is computing, blocks until they finish
    /// 3. If no one is computing, claims the blackhole and computes
    /// 4. If computation fails, the error is cached and propagated to waiters
    ///
    /// This implements the core blackholing pattern from GHC's runtime system,
    /// ensuring that:
    /// - Only one thread computes bounds at a time
    /// - Other threads wait rather than duplicating work
    /// - Errors are properly propagated to all waiters
    pub fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        // Use zero precision to just get any cached value or compute
        match self.blackhole.try_claim(&PrecisionLevel::zero())? {
            ClaimResult::AlreadyMeets(bounds) => Ok(bounds),
            ClaimResult::Claimed { .. } => {
                // We claimed the blackhole - compute bounds
                match self.compute_bounds() {
                    Ok(bounds) => {
                        self.blackhole.complete(bounds.clone());
                        self.refinement.notify_bounds_updated();
                        Ok(bounds)
                    }
                    Err(err) => {
                        // Computation failed - store error for other waiters
                        self.blackhole.fail(err.clone());
                        Err(err)
                    }
                }
            }
        }
    }

    /// Updates the cached bounds directly.
    ///
    /// This is used during refinement to propagate new bounds through the graph.
    /// Unlike GHC's immutable thunks, our bounds can be refined to higher precision.
    pub fn set_bounds(&self, bounds: Bounds) {
        self.blackhole.update(bounds);
        self.refinement.notify_bounds_updated();
    }

    /// Computes bounds for this node from current children/base state.
    ///
    /// This directly calls the operation's compute_bounds without going through
    /// the blackhole. Used internally when we've already claimed the blackhole.
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

    /// Refines this node until the target precision is reached.
    ///
    /// This uses the unified blackhole to coordinate concurrent access:
    /// - If already at target precision, returns immediately
    /// - If another thread is refining, waits for it
    /// - If this thread claims, loops until target precision is met
    ///
    /// # Arguments
    /// * `target` - The target precision level
    /// * `max_iterations` - Maximum refinement steps to prevent infinite loops
    ///
    /// Returns the bounds after refinement reaches target precision.
    pub fn refine_to_precision(
        &self,
        target: &PrecisionLevel,
        max_iterations: usize,
    ) -> Result<Bounds, ComputableError> {
        for _ in 0..max_iterations {
            match self.blackhole.try_claim(target)? {
                ClaimResult::AlreadyMeets(bounds) => {
                    return Ok(bounds);
                }
                ClaimResult::Claimed { .. } => {
                    // We claimed - perform one refinement step
                    match self.refine_step() {
                        Ok(_refined) => {
                            let bounds = self.compute_bounds()?;
                            self.blackhole.complete(bounds.clone());
                            self.refinement.notify_bounds_updated();
                            // Loop to check if we've reached target precision
                        }
                        Err(e) => {
                            self.blackhole.fail(e.clone());
                            return Err(e);
                        }
                    }
                }
            }
        }
        // Max iterations reached - return current bounds
        self.get_bounds()
    }
}

impl BoundsAccess for Node {
    fn get_bounds(&self) -> Result<Bounds, ComputableError> {
        Node::get_bounds(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::xbin;
    use num_bigint::BigInt;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    fn test_bounds() -> Bounds {
        Bounds::new(xbin(1, 0), xbin(2, 0))
    }

    fn different_bounds() -> Bounds {
        Bounds::new(xbin(10, 0), xbin(20, 0))
    }

    // =========================================================================
    // BoundsBlackhole Unit Tests (using Blackhole alias)
    // =========================================================================
    // Note: The full BoundsBlackhole tests are in normalized.rs
    // These tests verify the integration with Node.

    #[test]
    fn blackhole_new_is_not_evaluated() {
        let bh = BoundsBlackhole::new();
        assert!(bh.peek().is_none());
    }

    #[test]
    fn blackhole_update_and_peek() {
        let bh = BoundsBlackhole::new();
        let bounds = test_bounds();

        bh.update(bounds.clone());
        assert_eq!(bh.peek(), Some(bounds));
    }

    #[test]
    fn blackhole_reset_clears_value() {
        let bh = BoundsBlackhole::new();
        let bounds = test_bounds();

        bh.update(bounds);
        assert!(bh.peek().is_some());

        bh.reset();
        assert!(bh.peek().is_none());
    }

    // =========================================================================
    // Node Blackholing Tests
    // =========================================================================

    /// Test helper: creates a NodeOp that counts invocations
    struct CountingOp {
        compute_count: Arc<AtomicUsize>,
        compute_delay: Duration,
        bounds: Bounds,
    }

    impl NodeOp for CountingOp {
        fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
            self.compute_count.fetch_add(1, AtomicOrdering::SeqCst);
            if !self.compute_delay.is_zero() {
                thread::sleep(self.compute_delay);
            }
            Ok(self.bounds.clone())
        }

        fn refine_step(&self) -> Result<bool, ComputableError> {
            Ok(false)
        }

        fn children(&self) -> Vec<Arc<Node>> {
            vec![]
        }

        fn is_refiner(&self) -> bool {
            false
        }
    }

    /// Test helper: creates a NodeOp that always fails
    struct FailingOp {
        fail_count: Arc<AtomicUsize>,
    }

    impl NodeOp for FailingOp {
        fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
            self.fail_count.fetch_add(1, AtomicOrdering::SeqCst);
            Err(ComputableError::DomainError)
        }

        fn refine_step(&self) -> Result<bool, ComputableError> {
            Ok(false)
        }

        fn children(&self) -> Vec<Arc<Node>> {
            vec![]
        }

        fn is_refiner(&self) -> bool {
            false
        }
    }

    #[test]
    fn node_get_bounds_computes_once() {
        let compute_count = Arc::new(AtomicUsize::new(0));
        let node = Node::new(Arc::new(CountingOp {
            compute_count: Arc::clone(&compute_count),
            compute_delay: Duration::ZERO,
            bounds: test_bounds(),
        }));

        // First call computes
        let bounds1 = node.get_bounds().expect("should succeed");
        assert_eq!(bounds1, test_bounds());
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);

        // Second call returns cached
        let bounds2 = node.get_bounds().expect("should succeed");
        assert_eq!(bounds2, test_bounds());
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn node_concurrent_get_bounds_computes_once() {
        let compute_count = Arc::new(AtomicUsize::new(0));
        let node = Arc::new(Node::new(Arc::new(CountingOp {
            compute_count: Arc::clone(&compute_count),
            compute_delay: Duration::from_millis(20),
            bounds: test_bounds(),
        })));
        // Unwrap the Arc<Arc<Node>> - Node::new returns Arc<Node>
        let node = Arc::into_inner(node).expect("single owner");

        let node = Arc::new(node);
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let node = Arc::clone(&node);
                let bar = Arc::clone(&barrier);

                thread::spawn(move || {
                    bar.wait();
                    node.get_bounds()
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();

        // All results should succeed with same bounds
        for result in results {
            assert_eq!(result.expect("should succeed"), test_bounds());
        }

        // Computation should happen exactly once
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn node_get_bounds_propagates_error() {
        let fail_count = Arc::new(AtomicUsize::new(0));
        let node = Node::new(Arc::new(FailingOp {
            fail_count: Arc::clone(&fail_count),
        }));

        // First call fails
        let result1 = node.get_bounds();
        assert!(matches!(result1, Err(ComputableError::DomainError)));
        assert_eq!(fail_count.load(AtomicOrdering::SeqCst), 1);

        // Second call returns cached error (doesn't recompute)
        let result2 = node.get_bounds();
        assert!(matches!(result2, Err(ComputableError::DomainError)));
        assert_eq!(fail_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn node_set_bounds_updates_cache() {
        let compute_count = Arc::new(AtomicUsize::new(0));
        let node = Node::new(Arc::new(CountingOp {
            compute_count: Arc::clone(&compute_count),
            compute_delay: Duration::ZERO,
            bounds: test_bounds(),
        }));

        // First get computes
        let _ = node.get_bounds().expect("should succeed");
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);

        // Set new bounds
        let new_bounds = different_bounds();
        node.set_bounds(new_bounds.clone());

        // Get returns new bounds without recomputing
        let bounds = node.get_bounds().expect("should succeed");
        assert_eq!(bounds, new_bounds);
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn node_cached_bounds_is_nonblocking() {
        let compute_count = Arc::new(AtomicUsize::new(0));
        let node = Node::new(Arc::new(CountingOp {
            compute_count: Arc::clone(&compute_count),
            compute_delay: Duration::ZERO,
            bounds: test_bounds(),
        }));

        // Before computation, cached_bounds returns None
        assert!(node.cached_bounds().is_none());
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 0);

        // After computation, cached_bounds returns the value
        let _ = node.get_bounds().expect("should succeed");
        assert_eq!(node.cached_bounds(), Some(test_bounds()));
    }

    #[test]
    fn node_concurrent_error_propagation() {
        let fail_count = Arc::new(AtomicUsize::new(0));
        let node = Arc::new(Node::new(Arc::new(FailingOp {
            fail_count: Arc::clone(&fail_count),
        })));
        // Unwrap the Arc<Arc<Node>>
        let node = Arc::into_inner(node).expect("single owner");

        let node = Arc::new(node);
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let node = Arc::clone(&node);
                let bar = Arc::clone(&barrier);

                thread::spawn(move || {
                    bar.wait();
                    node.get_bounds()
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();

        // All results should have the same error
        for result in results {
            assert!(matches!(result, Err(ComputableError::DomainError)));
        }

        // Error computation should happen exactly once
        assert_eq!(fail_count.load(AtomicOrdering::SeqCst), 1);
    }

    // =========================================================================
    // Precision-Based Refinement Tests
    // =========================================================================

    #[test]
    fn node_refine_to_precision_basic() {
        use crate::normalized::PrecisionLevel;

        // Create a simple refining node
        let refinements = Arc::new(AtomicUsize::new(0));
        let node = {
            let refinements = Arc::clone(&refinements);
            Node::new(Arc::new(RefiningOp {
                refinement_count: refinements,
                // Each refinement halves the bounds width by decreasing exponent
                // Width progression: 2^0, 2^-1, 2^-2, 2^-3, ...
                bounds_fn: Arc::new(|count| {
                    let exp = -(count as i64);
                    Bounds::new(xbin(0, 0), xbin(1, exp))
                }),
            }))
        };

        // Initial precision should be None (no refinement yet)
        assert!(node.current_precision().is_none());

        // Refine to zero precision - should just compute initial bounds
        let target_zero = PrecisionLevel::zero();
        let bounds0 = node.refine_to_precision(&target_zero, 100).expect("should refine");
        assert!(node.current_precision().is_some());

        // Now refine to higher precision (8 bits) - should require multiple steps
        let target_8 = PrecisionLevel::from_bits(BigInt::from(8));
        let bounds1 = node.refine_to_precision(&target_8, 100).expect("should refine");

        // The bounds should have narrowed significantly
        assert!(bounds1.width() < bounds0.width());

        // Check that we actually did multiple refinements
        assert!(refinements.load(AtomicOrdering::SeqCst) >= 8);
    }

    #[test]
    fn node_refine_to_precision_multi_step() {
        use crate::normalized::PrecisionLevel;
        use num_bigint::BigInt;

        // Create a refining node that requires many steps
        let refinements = Arc::new(AtomicUsize::new(0));
        let node = {
            let refinements = Arc::clone(&refinements);
            Node::new(Arc::new(RefiningOp {
                refinement_count: refinements,
                bounds_fn: Arc::new(|count| {
                    let exp = -(count as i64);
                    Bounds::new(xbin(0, 0), xbin(1, exp))
                }),
            }))
        };

        // Request precision of 16 bits - requires at least 16 refinement steps
        let target = PrecisionLevel::from_bits(BigInt::from(16));
        let bounds = node.refine_to_precision(&target, 100).expect("should refine");

        // Should have done at least 16 refinements
        assert!(refinements.load(AtomicOrdering::SeqCst) >= 16);

        // Bounds width should be <= 2^-16
        let precision = node.current_precision().expect("should have precision");
        assert!(precision.meets(&target));

        // Verify the bounds are tight
        match bounds.width() {
            crate::binary::UXBinary::Finite(w) => {
                // Width should be very small
                assert!(w.exponent() <= &BigInt::from(-16));
            }
            _ => panic!("Expected finite width"),
        }
    }

    #[test]
    fn node_refine_to_precision_already_met() {
        use crate::normalized::PrecisionLevel;
        use num_bigint::BigInt;

        // Create a node with exact bounds (zero width)
        let node = Node::new(Arc::new(ConstantOp {
            bounds: Bounds::new(xbin(42, 0), xbin(42, 0)),
        }));

        // Get initial bounds (triggers computation)
        let _ = node.get_bounds().expect("should succeed");

        // Any precision target should be immediately met
        let target = PrecisionLevel::from_bits(BigInt::from(100));
        let bounds = node.refine_to_precision(&target, 100).expect("should succeed");

        assert_eq!(*bounds.small(), bounds.large()); // Zero width
    }

    /// A simple op that returns constant bounds.
    struct ConstantOp {
        bounds: Bounds,
    }

    impl NodeOp for ConstantOp {
        fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
            Ok(self.bounds.clone())
        }
        fn refine_step(&self) -> Result<bool, ComputableError> {
            Ok(false) // No refinement available
        }
        fn children(&self) -> Vec<Arc<Node>> {
            vec![]
        }
        fn is_refiner(&self) -> bool {
            false
        }
    }

    /// A node op that supports refinement with configurable behavior.
    struct RefiningOp {
        refinement_count: Arc<AtomicUsize>,
        bounds_fn: Arc<dyn Fn(usize) -> Bounds + Send + Sync>,
    }

    impl NodeOp for RefiningOp {
        fn compute_bounds(&self) -> Result<Bounds, ComputableError> {
            let count = self.refinement_count.load(AtomicOrdering::SeqCst);
            Ok((self.bounds_fn)(count))
        }
        fn refine_step(&self) -> Result<bool, ComputableError> {
            self.refinement_count.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(true)
        }
        fn children(&self) -> Vec<Arc<Node>> {
            vec![]
        }
        fn is_refiner(&self) -> bool {
            true
        }
    }
}
