//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - `NodeUpdate`: Update message from a refiner to the coordinator
//!
//! The refinement model follows a pub/sub pattern as described in the README:
//! 1. Each refiner runs autonomously in its own thread, continuously refining
//! 2. Refiners publish updates via bounded(0) channels for natural backpressure
//! 3. Coordinator receives updates as they arrive and propagates through the graph
//! 4. Precision is checked after each update propagation
//! 5. When precision is met or max iterations reached, coordinator signals stop via StopFlag

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;

/// Update message from a refiner thread.
#[derive(Clone)]
pub struct NodeUpdate {
    pub node_id: usize,
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
        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());
            // Use bounded(1) channel - allows one queued message per refiner
            // This provides backpressure while allowing some pipelining
            let (update_tx, update_rx) = bounded(1);

            // Spawn all refiners - they start running autonomously immediately
            for node in &self.refiners {
                spawn_refiner(
                    scope,
                    Arc::clone(node),
                    Arc::clone(&stop_flag),
                    update_tx.clone(),
                );
            }
            drop(update_tx);

            // In event-driven mode, we count individual updates.
            // Multiply MAX_ITERATIONS by number of refiners to get equivalent work budget.
            let max_updates = MAX_REFINEMENT_ITERATIONS.saturating_mul(self.refiners.len().max(1));

            let result = self.event_driven_loop(&update_rx, &stop_flag, epsilon, max_updates);

            // Shutdown all refiners
            stop_flag.stop();
            // Refiners will exit when they see stop flag or channel closes
            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }

    fn event_driven_loop(
        &self,
        update_rx: &Receiver<Result<NodeUpdate, ComputableError>>,
        stop_flag: &Arc<StopFlag>,
        epsilon: &UBinary,
        max_updates: usize,
    ) -> Result<Bounds, ComputableError> {
        let mut updates_processed = 0usize;

        // Check initial precision
        let initial_bounds = self.root.get_bounds()?;
        if bounds_width_leq(&initial_bounds, epsilon) {
            stop_flag.stop();
            return Ok(initial_bounds);
        }

        // Event loop: receive updates as they arrive from autonomous refiners
        // Refiners are already running and will send updates as they refine
        // With bounded(0) channel, refiners block on send until we receive,
        // providing natural backpressure similar to the original's continue signals
        loop {
            let result = match update_rx.recv() {
                Ok(r) => r,
                Err(_) => {
                    // All senders dropped - refiners finished before we met precision.
                    // This typically happens when refiners panic or exit early.
                    stop_flag.stop();
                    return Err(ComputableError::RefinementChannelClosed);
                }
            };

            match result {
                Ok(update) => {
                    self.apply_update(update)?;
                    updates_processed += 1;

                    // Check if precision met
                    let root_bounds = self.root.get_bounds()?;
                    if bounds_width_leq(&root_bounds, epsilon) {
                        stop_flag.stop();
                        return Ok(root_bounds);
                    }

                    // Check iteration limit
                    if updates_processed >= max_updates {
                        stop_flag.stop();
                        return Err(ComputableError::MaxRefinementIterations {
                            max: max_updates / self.refiners.len().max(1),
                        });
                    }

                    // No continue signal needed - refiners run autonomously
                    // Backpressure is provided by bounded(0) channel
                }
                Err(error) => {
                    stop_flag.stop();
                    return Err(error);
                }
            }
        }
    }

    fn apply_update(&self, update: NodeUpdate) -> Result<(), ComputableError> {
        let mut queue = VecDeque::new();
        if let Some(node) = self.nodes.get(&update.node_id) {
            // With autonomous refiners, we might receive stale updates.
            // Only apply if the update bounds are at least as good as current.
            // This can happen when refiners get ahead of the coordinator.
            let should_apply = match node.cached_bounds() {
                None => true,
                Some(current) => {
                    // Update is valid if it doesn't worsen either bound
                    update.bounds.small() >= current.small()
                        && update.bounds.large() <= current.large()
                }
            };
            if should_apply {
                node.set_bounds(update.bounds);
                queue.push_back(node.id);
            }
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

/// Backoff duration when refine_step makes no progress (100 microseconds)
const REFINER_BACKOFF: Duration = Duration::from_micros(100);

fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> Result<(), ComputableError> {
    // Autonomous refiner loop: runs continuously without waiting for coordinator
    while !stop.is_stopped() {
        match node.refine_step() {
            Ok(made_progress) => {
                // Compute bounds (either new or cached)
                let bounds = if made_progress {
                    let b = node.compute_bounds()?;
                    node.set_bounds(b.clone());
                    b
                } else {
                    // No progress - add small backoff to avoid busy-spinning
                    thread::sleep(REFINER_BACKOFF);
                    match node.cached_bounds() {
                        Some(b) => b,
                        None => node.compute_bounds()?,
                    }
                };

                // With bounded(1) channel, send blocks if one message is already queued
                // This provides natural backpressure - refiners can't get too far ahead
                if updates
                    .send(Ok(NodeUpdate {
                        node_id: node.id,
                        bounds,
                    }))
                    .is_err()
                {
                    // Channel closed - coordinator is done
                    break;
                }
            }
            Err(error) => {
                let _ = updates.send(Err(error));
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

// TODO: make a macro for basically
// mod tests {
//     #![allow(clippy::expect_used, clippy::panic)]
// and then put ratchet tests for clippy::expect_used and clippy:panic
// (they should NEVER be allowed in the code except in tests--we ONLY use unreachable! and debug_assert! in non-test code)

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::XBinary;
    use crate::computable::Computable;
    use crate::error::ComputableError;
    use crate::test_utils::{
        bin, interval_midpoint_computable, interval_refine, midpoint_between, ubin, unwrap_finite,
        xbin,
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
        // With autonomous refiners, bounds may continue to improve after refine_to returns.
        // Verify that the current bounds are at least as good as what was returned.
        let current_bounds = computable.bounds().expect("bounds should succeed");
        assert!(
            current_bounds.small() >= bounds.small(),
            "lower bound should not get worse"
        );
        assert!(
            &current_bounds.large() <= &upper,
            "upper bound should not get worse"
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
        let result = expr.refine_to::<3>(epsilon);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
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

    // NOTE: this test could be fallible, since it uses timing to measure success. perhaps it should be an integration test rather than a unit test
    #[test]
    fn refinement_parallelizes_multiple_refiners() {
        use std::time::Instant;

        const SLEEP_MS: u64 = 10;

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
        assert!(
            elapsed.as_millis() as u64 > SLEEP_MS,
            "refinement must not have actually run"
        );
        // Allow some margin for thread spawning overhead and context switching
        assert!(
            (elapsed.as_millis() as u64) < 3 * SLEEP_MS,
            "expected parallel refinement under {}ms, elapsed {elapsed:?}",
            3 * SLEEP_MS
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
