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
use crate::normalized::{PrecisionLevel, RefinementBlackhole, RefinementClaimResult};

/// Shared API for retrieving bounds with lazy computation.
pub trait BoundsAccess {
    fn get_bounds(&self) -> Result<Bounds, ComputableError>;
}

// ============================================================================
// Blackhole Implementation for Lazy Bounds Computation
// ============================================================================
//
// This implements a blackholing mechanism similar to GHC's runtime:
// - NotEvaluated: The thunk has never been forced
// - BeingEvaluated: A thread is currently computing this value (the "blackhole")
// - Evaluated: The value has been computed and cached
// - Failed: Computation failed with an error
//
// When a thread encounters a BeingEvaluated state, it waits on a condvar
// rather than re-computing the value. This is equivalent to GHC's blocking
// queue mechanism but using Rust's synchronization primitives.

/// State of a lazy bounds computation, implementing the blackhole pattern.
///
/// This enum represents the four possible states of a lazy computation:
/// - `NotEvaluated`: Initial state, computation hasn't started
/// - `BeingEvaluated`: A thread is actively computing (the "blackhole")
/// - `Evaluated`: Computation completed successfully with a cached result
/// - `Failed`: Computation failed with an error
///
/// The key insight from GHC's blackholing is that when multiple threads
/// try to evaluate the same thunk, only one should do the work while
/// others wait for the result.
#[derive(Clone)]
pub enum BlackholeState {
    /// The bounds have never been computed.
    NotEvaluated,
    /// A thread is currently computing the bounds (the "blackhole").
    /// Other threads should wait rather than duplicate work.
    BeingEvaluated,
    /// The bounds have been successfully computed and cached.
    Evaluated(Bounds),
    /// The computation failed. Stores the error for waiters.
    /// Unlike GHC's <<loop>> detection, we propagate actual computation errors.
    Failed(ComputableError),
}

/// Synchronization wrapper for blackhole state.
///
/// Combines the state with a condition variable for efficient waiting.
/// This is analogous to GHC's BLACKHOLE_BQ (blackhole with blocking queue)
/// but uses Rust's parking_lot primitives for better performance.
pub struct Blackhole {
    state: Mutex<BlackholeState>,
    condvar: Condvar,
}

impl Blackhole {
    /// Creates a new blackhole in the NotEvaluated state.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(BlackholeState::NotEvaluated),
            condvar: Condvar::new(),
        }
    }

    /// Creates a new blackhole with an already-evaluated value.
    /// Useful for testing or when the value is known at construction time.
    #[cfg(test)]
    pub fn with_value(bounds: Bounds) -> Self {
        Self {
            state: Mutex::new(BlackholeState::Evaluated(bounds)),
            condvar: Condvar::new(),
        }
    }

    /// Attempts to claim the blackhole for evaluation.
    ///
    /// Returns `Ok(Some(bounds))` if already evaluated,
    /// `Ok(None)` if this thread should compute the value,
    /// or blocks if another thread is computing.
    ///
    /// This is the core of the blackholing mechanism:
    /// 1. If NotEvaluated, transition to BeingEvaluated and return None (caller computes)
    /// 2. If BeingEvaluated, wait on condvar until state changes
    /// 3. If Evaluated, return the cached bounds
    /// 4. If Failed, return the error
    fn try_claim_or_wait(&self) -> Result<Option<Bounds>, ComputableError> {
        let mut state = self.state.lock();
        loop {
            match &*state {
                BlackholeState::NotEvaluated => {
                    // We're the first thread - claim the blackhole
                    *state = BlackholeState::BeingEvaluated;
                    return Ok(None);
                }
                BlackholeState::BeingEvaluated => {
                    // Another thread is computing - wait for it
                    // This is equivalent to blocking on GHC's BLACKHOLE_BQ
                    self.condvar.wait(&mut state);
                    // Loop to re-check state after waking
                }
                BlackholeState::Evaluated(bounds) => {
                    // Already computed - return cached value
                    return Ok(Some(bounds.clone()));
                }
                BlackholeState::Failed(err) => {
                    // Previous computation failed - propagate error
                    return Err(err.clone());
                }
            }
        }
    }

    /// Completes the evaluation with a successful result.
    ///
    /// Transitions from BeingEvaluated to Evaluated and wakes all waiters.
    fn complete(&self, bounds: Bounds) {
        let mut state = self.state.lock();
        *state = BlackholeState::Evaluated(bounds);
        self.condvar.notify_all();
    }

    /// Completes the evaluation with an error.
    ///
    /// Transitions from BeingEvaluated to Failed and wakes all waiters.
    /// This is similar to how GHC handles exceptions during thunk evaluation,
    /// but we explicitly propagate the error rather than re-throwing.
    fn fail(&self, err: ComputableError) {
        let mut state = self.state.lock();
        *state = BlackholeState::Failed(err);
        self.condvar.notify_all();
    }

    /// Resets the blackhole to allow re-evaluation.
    ///
    /// This is used when bounds need to be recomputed, such as when
    /// child nodes have been refined and cached bounds are stale.
    /// Unlike GHC's immutable thunks, our bounds can be refined.
    #[allow(dead_code)] // API for future use in cache invalidation
    pub fn reset(&self) {
        let mut state = self.state.lock();
        *state = BlackholeState::NotEvaluated;
        // No need to notify - waiters will re-check when they wake
    }

    /// Updates the cached value directly without going through evaluation.
    ///
    /// Used during refinement to propagate new bounds.
    pub fn update(&self, bounds: Bounds) {
        let mut state = self.state.lock();
        *state = BlackholeState::Evaluated(bounds);
        self.condvar.notify_all();
    }

    /// Returns the current cached bounds if evaluated, without blocking.
    pub fn peek(&self) -> Option<Bounds> {
        let state = self.state.lock();
        match &*state {
            BlackholeState::Evaluated(bounds) => Some(bounds.clone()),
            _ => None,
        }
    }
}

impl Default for Blackhole {
    fn default() -> Self {
        Self::new()
    }
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
/// This integrates the blackholing pattern for refinement with the existing
/// coordination mechanism. It tracks:
/// - Whether a top-level refinement is active
/// - The current epoch for detecting updates
/// - The refinement blackhole for precision tracking
pub struct RefinementSync {
    pub state: Mutex<RefinementState>,
    pub condvar: Condvar,
    /// Blackhole for tracking refinement precision and coordinating concurrent access.
    pub blackhole: RefinementBlackhole,
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
            blackhole: RefinementBlackhole::new(),
        }
    }

    pub fn notify_bounds_updated(&self) {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        self.condvar.notify_all();
    }

    /// Returns the current precision level from the refinement blackhole.
    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        self.blackhole.current_precision()
    }

    /// Attempts to claim the refinement for reaching a target precision.
    ///
    /// This integrates with the blackhole to ensure:
    /// - Only one thread refines at a time
    /// - Precision is tracked monotonically
    /// - Threads can wait for specific precision levels
    pub fn try_claim_for_precision(
        &self,
        target: &PrecisionLevel,
    ) -> Result<RefinementClaimResult, ComputableError> {
        self.blackhole.try_claim(target)
    }

    /// Completes a refinement step and updates the blackhole.
    pub fn complete_refinement(&self, bounds: Bounds) -> Result<(), ComputableError> {
        self.blackhole.complete(bounds)?;
        self.notify_bounds_updated();
        Ok(())
    }

    /// Fails the refinement with an error.
    pub fn fail_refinement(&self, err: ComputableError) {
        self.blackhole.fail(err);
        self.notify_bounds_updated();
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
/// ## Blackholing Semantics
///
/// The `blackhole` field implements GHC-style blackholing for lazy evaluation:
/// - When `get_bounds()` is called and bounds haven't been computed, the node
///   is "blackholed" (marked as being evaluated)
/// - Other threads calling `get_bounds()` will block until evaluation completes
/// - This prevents duplicate work when multiple threads access the same node
///
/// ## Refinement and Updates
///
/// Unlike GHC's immutable thunks, computable bounds can be refined to higher
/// precision. The `set_bounds()` method updates the blackhole state and notifies
/// any waiters. The `refinement` field coordinates higher-level refinement
/// across the computation graph.
///
/// NOTE: The blackhole is not automatically invalidated when children are refined.
/// Updates are explicitly propagated via apply_update during refinement. If get_bounds()
/// is called between refinement steps (outside of refine_to), it may return stale cached
/// values. Consider whether this is acceptable for your use case.
pub struct Node {
    pub id: usize,
    pub op: Arc<dyn NodeOp>,
    /// Blackhole for lazy bounds computation with concurrent synchronization.
    /// This replaces the simple RwLock<Option<Bounds>> with proper blackholing.
    pub blackhole: Blackhole,
    pub refinement: RefinementSync,
}

impl Node {
    pub fn new(op: Arc<dyn NodeOp>) -> Arc<Self> {
        static NODE_IDS: AtomicUsize = AtomicUsize::new(0);
        Arc::new(Self {
            id: NODE_IDS.fetch_add(1, Ordering::Relaxed),
            op,
            blackhole: Blackhole::new(),
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
        // Try to get cached bounds or claim the blackhole
        match self.blackhole.try_claim_or_wait()? {
            Some(bounds) => {
                // Already computed - return cached value
                Ok(bounds)
            }
            None => {
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

    /// Returns the current precision level of this node's refinement.
    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        self.refinement.current_precision()
    }

    /// Attempts to refine this node to reach the target precision.
    ///
    /// This uses the refinement blackhole to coordinate concurrent access:
    /// - If already at target precision, returns immediately
    /// - If another thread is refining, waits for it
    /// - If this thread claims the refinement, performs one step
    ///
    /// Returns the bounds after refinement (or the existing bounds if
    /// target precision was already met).
    pub fn refine_to_precision(&self, target: &PrecisionLevel) -> Result<Bounds, ComputableError> {
        match self.refinement.try_claim_for_precision(target)? {
            RefinementClaimResult::AlreadyMeets(bounds) => {
                // Already at target precision - return the cached bounds
                Ok(bounds)
            }
            RefinementClaimResult::Claimed { .. } => {
                // We claimed the refinement - perform a step
                match self.refine_step() {
                    Ok(_refined) => {
                        // Compute new bounds and complete
                        let bounds = self.compute_bounds()?;
                        self.refinement.complete_refinement(bounds.clone())?;
                        self.blackhole.update(bounds.clone());
                        Ok(bounds)
                    }
                    Err(e) => {
                        self.refinement.fail_refinement(e.clone());
                        Err(e)
                    }
                }
            }
        }
    }

    /// Updates both the bounds cache and refinement tracking atomically.
    ///
    /// This is used during refinement propagation to ensure consistency
    /// between the bounds blackhole and refinement blackhole states.
    pub fn update_bounds_with_precision(&self, bounds: Bounds) -> Result<(), ComputableError> {
        // Update the refinement blackhole (which tracks precision)
        self.refinement.blackhole.update(bounds.clone())?;
        // Update the bounds blackhole (for caching)
        self.blackhole.update(bounds);
        self.refinement.notify_bounds_updated();
        Ok(())
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
    // Blackhole Unit Tests
    // =========================================================================

    #[test]
    fn blackhole_new_is_not_evaluated() {
        let bh = Blackhole::new();
        assert!(bh.peek().is_none());
    }

    #[test]
    fn blackhole_with_value_is_evaluated() {
        let bounds = test_bounds();
        let bh = Blackhole::with_value(bounds.clone());
        assert_eq!(bh.peek(), Some(bounds));
    }

    #[test]
    fn blackhole_try_claim_returns_none_for_new() {
        let bh = Blackhole::new();
        let result = bh.try_claim_or_wait();
        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn blackhole_try_claim_returns_bounds_after_complete() {
        let bh = Blackhole::new();
        let bounds = test_bounds();

        // First claim
        assert!(matches!(bh.try_claim_or_wait(), Ok(None)));

        // Complete with value
        bh.complete(bounds.clone());

        // Second claim should return the value
        assert_eq!(bh.try_claim_or_wait(), Ok(Some(bounds)));
    }

    #[test]
    fn blackhole_try_claim_returns_error_after_fail() {
        let bh = Blackhole::new();

        // Claim
        assert!(matches!(bh.try_claim_or_wait(), Ok(None)));

        // Fail
        bh.fail(ComputableError::DomainError);

        // Next attempt should get error
        assert!(matches!(
            bh.try_claim_or_wait(),
            Err(ComputableError::DomainError)
        ));
    }

    #[test]
    fn blackhole_update_sets_value() {
        let bh = Blackhole::new();
        let bounds = test_bounds();

        bh.update(bounds.clone());

        assert_eq!(bh.peek(), Some(bounds.clone()));
        assert_eq!(bh.try_claim_or_wait(), Ok(Some(bounds)));
    }

    #[test]
    fn blackhole_reset_clears_value() {
        let bh = Blackhole::new();
        let bounds = test_bounds();

        bh.update(bounds);
        assert!(bh.peek().is_some());

        bh.reset();
        assert!(bh.peek().is_none());
    }

    // =========================================================================
    // Concurrent Blackhole Tests
    // =========================================================================

    #[test]
    fn blackhole_concurrent_claim_computes_once() {
        let bh = Arc::new(Blackhole::new());
        let compute_count = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let bh = Arc::clone(&bh);
                let count = Arc::clone(&compute_count);
                let bar = Arc::clone(&barrier);

                thread::spawn(move || {
                    bar.wait();

                    match bh.try_claim_or_wait() {
                        Ok(Some(bounds)) => bounds,
                        Ok(None) => {
                            // We claimed it - compute
                            count.fetch_add(1, AtomicOrdering::SeqCst);
                            thread::sleep(Duration::from_millis(10));
                            let bounds = test_bounds();
                            bh.complete(bounds.clone());
                            bounds
                        }
                        Err(_) => panic!("unexpected error"),
                    }
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();

        // All results should be the same
        let expected = test_bounds();
        for result in results {
            assert_eq!(result, expected);
        }

        // Computation should happen exactly once
        assert_eq!(compute_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn blackhole_waiters_get_error_on_fail() {
        let bh = Arc::new(Blackhole::new());
        let barrier = Arc::new(Barrier::new(3));

        // Claim the blackhole first
        assert!(matches!(bh.try_claim_or_wait(), Ok(None)));

        // Spawn waiters
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let bh = Arc::clone(&bh);
                let bar = Arc::clone(&barrier);

                thread::spawn(move || {
                    bar.wait();
                    bh.try_claim_or_wait()
                })
            })
            .collect();

        // Give waiters time to start waiting
        barrier.wait();
        thread::sleep(Duration::from_millis(10));

        // Fail the computation
        bh.fail(ComputableError::DomainError);

        // All waiters should get the error
        for handle in handles {
            let result = handle.join().expect("join");
            assert!(matches!(result, Err(ComputableError::DomainError)));
        }
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
    // Note: PrecisionLevel and RefinementBlackhole tests have been moved to
    // src/normalized.rs as part of the normalized bounds refactor.
    // =========================================================================
}
