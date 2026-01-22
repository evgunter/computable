//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - `NodeUpdate`: Update message from a refiner to the coordinator
//!
//! The refinement model is a true pub/sub system:
//! - Each refiner node runs autonomously in its own thread
//! - Refiners continuously refine and publish bounds updates
//! - Updates propagate upward through the graph automatically
//! - A coordinator monitors the root node and signals completion when target precision is met
//! - No lock-step coordination: refiners run at their own pace with backpressure via bounded channel

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Sender, bounded};

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;

/// Update message from a refiner thread.
#[derive(Clone)]
pub struct NodeUpdate {
    pub node_id: usize,
    /// The bounds computed by the refiner. These are sent for debugging/logging
    /// purposes but the coordinator recomputes fresh bounds when applying.
    #[allow(dead_code)]
    pub bounds: Bounds,
}

/// Snapshot of the node graph used to coordinate parallel refinement.
pub struct RefinementGraph {
    pub root: Arc<Node>,
    pub nodes: HashMap<usize, Arc<Node>>,    // node id -> node
    pub parents: HashMap<usize, Vec<usize>>, // child id -> parent ids
    pub refiners: Vec<Arc<Node>>,
}

impl RefinementGraph {
    pub fn new(root: Arc<Node>) -> Result<Self, ComputableError> {
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

    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: usize>(
        &self,
        epsilon: &UBinary,
    ) -> Result<Bounds, ComputableError> {
        // Check if we already meet the precision requirement
        let initial_bounds = self.root.get_bounds()?;
        if bounds_width_leq(&initial_bounds, epsilon) {
            return Ok(initial_bounds);
        }

        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());

            // Initialize cached bounds for all nodes before starting refiners
            // This ensures get_bounds() returns cached values rather than computing
            // and caching, which would bypass the coordinator's single-threaded updates
            for node in self.nodes.values() {
                if node.cached_bounds().is_none()
                    && let Ok(bounds) = node.compute_bounds()
                {
                    node.set_bounds(bounds);
                }
            }

            // Use bounded channel for backpressure - refiners block when buffer is full
            // A small buffer (1) allows some pipelining while still providing backpressure
            let (update_tx, update_rx) = bounded(16);

            // Spawn autonomous refiner threads
            for node in &self.refiners {
                spawn_refiner(
                    scope,
                    Arc::clone(node),
                    Arc::clone(&stop_flag),
                    update_tx.clone(),
                );
            }
            drop(update_tx);

            let result = (|| {
                let mut iterations = 0usize;
                let mut received_first_update = false;

                loop {
                    // Wait for at least one update before checking bounds
                    // This ensures refiners have had a chance to initialize their state
                    if received_first_update {
                        // Check current root bounds
                        let root_bounds = self.root.get_bounds()?;
                        if bounds_width_leq(&root_bounds, epsilon) {
                            // Return the bounds that passed the check
                            return Ok(root_bounds);
                        }
                    }

                    // Receive updates as they arrive (event-driven, not lock-step)
                    // Use recv_timeout to allow periodic bounds checks
                    match update_rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(update_result) => {
                            received_first_update = true;
                            match update_result {
                                Ok(update) => {
                                    self.apply_update(update)?;
                                    // Count updates toward max iterations
                                    // Scale by refiner count to match lock-step semantics
                                    // where one "iteration" = all refiners stepping once
                                    iterations += 1;
                                    let effective_iterations =
                                        iterations / self.refiners.len().max(1);
                                    if effective_iterations >= MAX_REFINEMENT_ITERATIONS {
                                        return Err(ComputableError::MaxRefinementIterations {
                                            max: MAX_REFINEMENT_ITERATIONS,
                                        });
                                    }
                                }
                                Err(error) => {
                                    return Err(error);
                                }
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            // Timeout allows us to check bounds periodically
                            // and prevents deadlock if all refiners stopped
                            continue;
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            // All refiners have exited - check if we reached target
                            let final_bounds = self.root.get_bounds()?;
                            if bounds_width_leq(&final_bounds, epsilon) {
                                return Ok(final_bounds);
                            }
                            return Err(ComputableError::RefinementChannelClosed);
                        }
                    }
                }
            })();

            // Signal all refiners to stop
            stop_flag.stop();

            // Drain pending updates until channel is disconnected
            // This ensures refiners blocked on send() can unblock and exit
            // Use a timeout to prevent infinite waits if refiners are stuck
            let drain_start = std::time::Instant::now();
            let drain_timeout = Duration::from_secs(5);
            loop {
                if drain_start.elapsed() > drain_timeout {
                    // Timeout - refiners may be stuck in long computations
                    // They'll eventually exit when their current refine_step completes
                    break;
                }
                match update_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(update_result) => {
                        if let Ok(update) = update_result {
                            let _ = self.apply_update(update);
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Keep waiting - refiners may still be processing
                        continue;
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        // All refiners have exited
                        break;
                    }
                }
            }

            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }

    /// Apply an update from a refiner and propagate bounds changes upward.
    /// Returns true if any bounds actually changed, false if the update was redundant.
    ///
    /// All bound updates happen through this function (single-threaded) to ensure
    /// consistency. We recompute bounds rather than using the bounds from the update,
    /// because the update may have been computed with stale child bounds.
    fn apply_update(&self, update: NodeUpdate) -> Result<bool, ComputableError> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut any_changed = false;

        // Recompute bounds for the refiner node (don't use update.bounds directly,
        // as they may have been computed with stale child bounds)
        if let Some(node) = self.nodes.get(&update.node_id) {
            let fresh_bounds = node.compute_bounds()?;
            let current = node.cached_bounds();
            if current.as_ref() != Some(&fresh_bounds) {
                node.set_bounds(fresh_bounds);
                any_changed = true;
            }
            queue.push_back(node.id);
            visited.insert(node.id);
        }

        // Propagate to parents
        while let Some(changed_id) = queue.pop_front() {
            let Some(parents) = self.parents.get(&changed_id) else {
                continue;
            };
            for parent_id in parents {
                if visited.contains(parent_id) {
                    continue;
                }
                visited.insert(*parent_id);

                let parent = self
                    .nodes
                    .get(parent_id)
                    .ok_or(ComputableError::RefinementChannelClosed)?;

                // Compute parent bounds from children's current cached bounds
                let next_bounds = parent.compute_bounds()?;
                let current_bounds = parent.cached_bounds();

                // Only update and continue propagation if bounds changed
                if current_bounds.as_ref() != Some(&next_bounds) {
                    parent.set_bounds(next_bounds);
                    any_changed = true;
                    queue.push_back(*parent_id);
                }
            }
        }

        Ok(any_changed)
    }
}

fn spawn_refiner<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) {
    scope.spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refiner_loop(node, stop, updates)
        }));
    });
}

fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> Result<(), ComputableError> {
    // Track last sent bounds to detect when no progress is being made
    let mut last_bounds: Option<Bounds> = None;
    let mut no_progress_count = 0usize;

    // Run autonomously - refine continuously until stopped
    while !stop.is_stopped() {
        match node.refine_step() {
            Ok(true) => {
                // Made progress - compute new bounds
                // NOTE: We do NOT call set_bounds here. The coordinator will set bounds
                // based on updates to ensure consistent, single-threaded updates.
                let bounds = node.compute_bounds()?;

                // Track whether bounds actually changed
                let bounds_changed = last_bounds.as_ref() != Some(&bounds);
                last_bounds = Some(bounds.clone());

                if bounds_changed {
                    no_progress_count = 0;
                } else {
                    no_progress_count += 1;
                }

                // Send update to coordinator using blocking send
                // This naturally synchronizes with the coordinator
                let update = Ok(NodeUpdate {
                    node_id: node.id,
                    bounds,
                });
                loop {
                    if stop.is_stopped() {
                        return Ok(());
                    }
                    match updates.send_timeout(update.clone(), Duration::from_millis(10)) {
                        Ok(()) => break,
                        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                            // Channel full, retry after checking stop flag
                            continue;
                        }
                        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                            // Coordinator has shut down
                            return Ok(());
                        }
                    }
                }

                // Small delay after sending to give coordinator time to process
                // This reduces contention and improves consistency
                if no_progress_count > 10 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            Ok(false) => {
                // No progress made - small backoff to avoid busy-spinning
                thread::sleep(Duration::from_micros(100));
            }
            Err(error) => {
                // Send error with retries to ensure delivery
                let error_update = Err(error);
                loop {
                    match updates.send_timeout(error_update.clone(), Duration::from_millis(50)) {
                        Ok(()) => break,
                        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                            // Keep trying to send error
                            continue;
                        }
                        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                            // Coordinator already shut down
                            break;
                        }
                    }
                }
                break;
            }
        }
    }
    Ok(())
}

/// Compares bounds width (UXBinary) against epsilon (UBinary).
/// Returns true if width <= epsilon.
pub fn bounds_width_leq(bounds: &Bounds, epsilon: &UBinary) -> bool {
    match bounds.width() {
        UXBinary::Inf => false,
        UXBinary::Finite(uwidth) => *uwidth <= *epsilon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::XBinary;
    use crate::computable::Computable;
    use crate::error::ComputableError;
    use crate::test_utils::{
        bin, interval_midpoint_computable, interval_noop_computable, interval_refine,
        midpoint_between, ubin, unwrap_finite, xbin,
    };
    use num_traits::Zero;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    type IntervalState = Bounds;

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn interval_refine_strict(state: IntervalState) -> IntervalState {
        let midpoint = midpoint_between(state.small(), &state.large());
        Bounds::new(state.small().clone(), XBinary::Finite(midpoint))
    }

    fn sqrt_computable(value_int: u64) -> Computable {
        Computable::constant(bin(value_int as i64, 0))
            .nth_root(std::num::NonZeroU32::new(2).expect("2 is non-zero"))
    }

    fn assert_width_nonnegative(bounds: &Bounds) {
        assert!(*bounds.width() >= UXBinary::zero());
    }

    // --- tests for different results of refinement (mostly errors) ---

    #[test]
    fn refine_to_accepts_zero_epsilon_for_exact_values() {
        // A computable that collapses to a single point (width = 0) after refinement
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(0, 0);
        let bounds = computable
            .refine_to_default(epsilon)
            .expect("refine_to with epsilon=0 should succeed when bounds converge exactly");

        // After refinement, bounds should be exactly [1, 1]
        assert_eq!(bounds.small(), &xbin(1, 0));
        assert_eq!(bounds.large(), xbin(1, 0));

        // Width should be exactly zero
        assert!(matches!(bounds.width(), UXBinary::Finite(w) if w.mantissa().is_zero()));
    }

    #[test]
    fn refine_to_with_zero_epsilon_on_constant_succeeds_immediately() {
        // A constant computable already has exact bounds (width = 0)
        let computable = Computable::constant(bin(42, 0));
        let epsilon = ubin(0, 0);
        let bounds = computable
            .refine_to_default(epsilon)
            .expect("refine_to with epsilon=0 should succeed for constants");

        // Bounds should be exactly [42, 42]
        assert_eq!(bounds.small(), &xbin(42, 0));
        assert_eq!(bounds.large(), xbin(42, 0));
    }

    #[test]
    fn refine_to_with_zero_epsilon_on_non_exact_value_returns_max_iterations() {
        // 1/3 cannot be represented exactly in binary, so epsilon=0 should
        // eventually hit max iterations rather than hanging forever.
        let one = Computable::constant(bin(1, 0));
        let three = Computable::constant(bin(3, 0));
        let one_third = one / three;

        let epsilon = ubin(0, 0);
        // Use a small max iterations count to keep the test fast
        let result = one_third.refine_to::<10>(epsilon);

        assert!(
            matches!(
                result,
                Err(ComputableError::MaxRefinementIterations { max: 10 })
            ),
            "expected MaxRefinementIterations error for non-exact value with epsilon=0, got {:?}",
            result
        );
    }

    #[test]
    fn refine_to_returns_refined_state() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected = xbin(1, 0);
        let upper = bounds.large();
        let width = match bounds.width() {
            UXBinary::Finite(w) => w.clone(),
            UXBinary::Inf => panic!("expected finite width"),
        };

        assert!(bounds.small() <= &expected && &expected <= &upper);
        assert!(width < epsilon);
        let refined_bounds = computable.bounds().expect("bounds should succeed");
        let refined_upper = refined_bounds.large();
        assert!(refined_bounds.small() <= &expected && &expected <= &refined_upper);
    }

    #[test]
    fn refine_to_rejects_unchanged_state() {
        let computable = interval_noop_computable(0, 2);
        let epsilon = ubin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::StateUnchanged)));
    }

    #[test]
    fn refine_to_enforces_max_iterations() {
        let computable = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
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
    fn refine_to_handles_non_meeting_bounds() {
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

        // In the async refinement model, refiners may continue making progress
        // after we decide to return. The final bounds should be at least as tight
        // as the returned bounds (since refinement never makes bounds looser).
        let final_bounds = computable.bounds().expect("bounds should succeed");
        assert!(
            *final_bounds.width() <= *bounds.width(),
            "final bounds should be at least as tight as returned bounds"
        );
    }

    #[test]
    fn refine_to_rejects_worsened_bounds() {
        let interval_state = Bounds::new(xbin(0, 0), xbin(1, 0));
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            |inner_state: IntervalState| {
                let upper = inner_state.large();
                let worse_upper = unwrap_finite(&upper).add(&bin(1, 0));
                Bounds::new(inner_state.small().clone(), XBinary::Finite(worse_upper))
            },
        );
        let epsilon = ubin(1, -2);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    // --- concurrency tests ---

    #[test]
    fn refine_shared_clone_updates_original() {
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
    fn refine_to_channel_closure() {
        let computable = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
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
    fn refine_to_max_iterations_multiple_refiners() {
        let left = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| state + 1,
        );
        let right = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
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
    fn refine_to_error_path_stops_refiners() {
        let stable = interval_midpoint_computable(0, 2);
        let faulty = Computable::new(
            Bounds::new(xbin(0, 0), xbin(1, 0)),
            |state| Ok(state.clone()),
            |state| Bounds::new(state.small().clone(), xbin(2, 0)),
        );
        let expr = stable + faulty;
        let epsilon = ubin(1, -4);
        let result = expr.refine_to::<10>(epsilon);
        // In async refinement, we may get BoundsWorsened or MaxRefinementIterations
        // depending on timing. Both indicate the refinement failed, which is correct.
        // BoundsWorsened is preferred but timing-dependent.
        assert!(
            matches!(
                result,
                Err(ComputableError::BoundsWorsened)
                    | Err(ComputableError::MaxRefinementIterations { .. })
            ),
            "expected error but got {:?}",
            result
        );
    }

    #[test]
    fn concurrent_bounds_reads_during_failed_refinement() {
        let computable = Arc::new(Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
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

    // NOTE: this test verifies that refinement runs in parallel (4 refiners sleeping
    // in parallel should take ~SLEEP_MS, not 4*SLEEP_MS). The test is inherently
    // timing-sensitive and may be affected by thread scheduling under heavy load.
    #[test]
    fn refinement_parallelizes_multiple_refiners() {
        use std::time::Instant;

        // Using 20ms gives more margin for thread scheduling delays
        const SLEEP_MS: u64 = 20;

        let slow_refiner = || {
            Computable::new(
                0usize,
                |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
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
        // With 4 refiners sleeping 20ms each in parallel, total should be ~20-40ms
        // (not 80ms if sequential). Allow margin for thread scheduling and overhead.
        // In async model, there's additional overhead from channel communication and
        // bounds propagation, so we use 75ms as the upper bound (< 80ms sequential).
        let elapsed_ms = elapsed.as_millis() as u64;
        assert!(
            elapsed_ms >= SLEEP_MS / 2,
            "elapsed time {} ms too short, refinement may not have run (expected >= {} ms)",
            elapsed_ms,
            SLEEP_MS / 2
        );
        let parallel_threshold = 75; // Less than 80ms (sequential) but allows for overhead
        assert!(
            elapsed_ms < parallel_threshold,
            "expected parallel refinement under {} ms, but took {} ms (would be {} ms if sequential)",
            parallel_threshold,
            elapsed_ms,
            4 * SLEEP_MS
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
}
