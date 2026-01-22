//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - Refiner threads for leaf nodes that continuously refine and publish updates
//! - Propagator threads for intermediate nodes that subscribe to children and propagate
//!
//! The refinement model is true pub/sub:
//! 1. Refiners run continuously, publishing updates to a shared update bus
//! 2. Intermediate nodes (propagators) subscribe to the bus, filter for child updates,
//!    recompute their own bounds, and publish back to the bus
//! 3. Coordinator monitors root node updates and stops when precision is met
//! 4. Stop flag terminates all threads

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded, select};

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;

/// Backoff duration when a refiner makes no progress.
const NO_PROGRESS_BACKOFF_US: u64 = 100;

/// Timeout for propagator receive to check stop flag.
const PROPAGATOR_TIMEOUT_MS: u64 = 1;


/// Notification sent when a node's bounds have been updated.
#[derive(Clone)]
pub struct BoundsNotification {
    pub node_id: usize,
}

/// Notification sent when a refiner completes an iteration.
#[derive(Clone, Copy)]
pub struct IterationNotification;

/// Error notification from a thread.
#[derive(Clone)]
pub struct ErrorNotification {
    pub error: ComputableError,
}

/// Snapshot of the node graph used to coordinate parallel refinement.
pub struct RefinementGraph {
    pub root: Arc<Node>,
    pub nodes: HashMap<usize, Arc<Node>>,    // node id -> node
    pub parents: HashMap<usize, Vec<usize>>, // child id -> parent ids
    pub children: HashMap<usize, Vec<usize>>, // parent id -> child ids
    pub refiners: Vec<Arc<Node>>,
    pub propagators: Vec<Arc<Node>>,          // intermediate nodes (non-refiners with children)
}

impl RefinementGraph {
    pub fn new(root: Arc<Node>) -> Result<Self, ComputableError> {
        let mut nodes = HashMap::new();
        let mut parents: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut refiners = Vec::new();
        let mut propagators = Vec::new();

        let mut stack = vec![Arc::clone(&root)];
        while let Some(node) = stack.pop() {
            if nodes.contains_key(&node.id) {
                continue;
            }
            let node_id = node.id;
            nodes.insert(node_id, Arc::clone(&node));

            let node_children = node.children();
            if node.is_refiner() {
                refiners.push(Arc::clone(&node));
            } else if !node_children.is_empty() {
                // Non-refiner with children = intermediate node needing propagator
                propagators.push(Arc::clone(&node));
            }

            for child in &node_children {
                parents.entry(child.id).or_default().push(node_id);
                children.entry(node_id).or_default().push(child.id);
                stack.push(Arc::clone(child));
            }
        }

        let graph = Self {
            root,
            nodes,
            parents,
            children,
            refiners,
            propagators,
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

            // Check if precision already met before spawning threads
            let root_bounds = match self.root.get_bounds() {
                Ok(b) => b,
                Err(e) => {
                    outcome = Some(Err(e));
                    return;
                }
            };
            if bounds_width_leq(&root_bounds, epsilon) {
                outcome = Some(Ok(root_bounds));
                return;
            }

            // Build subscription infrastructure:
            // - Each parent (propagator + coordinator) has an inbox receiver
            // - Children have senders to their parents' inboxes
            // - Refiners with children also need inboxes (e.g., InvOp)
            let mut inbox_senders: HashMap<usize, Vec<Sender<BoundsNotification>>> = HashMap::new();
            let mut inbox_receivers: HashMap<usize, Receiver<BoundsNotification>> = HashMap::new();

            // Create inbox for each propagator
            for propagator in &self.propagators {
                let (tx, rx) = bounded(4); // Small buffer for backpressure
                inbox_receivers.insert(propagator.id, rx);
                // Register this inbox with all children
                if let Some(child_ids) = self.children.get(&propagator.id) {
                    for child_id in child_ids {
                        inbox_senders.entry(*child_id).or_default().push(tx.clone());
                    }
                }
            }

            // Create inbox for each refiner that has children (like InvOp)
            // These refiners need to be notified when their children update
            let refiners_with_children: Vec<Arc<Node>> = self
                .refiners
                .iter()
                .filter(|n| self.children.get(&n.id).map(|c| !c.is_empty()).unwrap_or(false))
                .cloned()
                .collect();

            for refiner in &refiners_with_children {
                let (tx, rx) = bounded(4);
                inbox_receivers.insert(refiner.id, rx);
                if let Some(child_ids) = self.children.get(&refiner.id) {
                    for child_id in child_ids {
                        inbox_senders.entry(*child_id).or_default().push(tx.clone());
                    }
                }
            }

            // Create inbox for coordinator (receives from root)
            let (coord_tx, coord_rx) = bounded(4);
            inbox_senders.entry(self.root.id).or_default().push(coord_tx);

            // Iteration channel - refiners report each step for max_iterations tracking
            let (iter_tx, iter_rx) = bounded::<IterationNotification>(16);

            // Error channel for threads to report failures
            let (error_tx, error_rx) = bounded::<ErrorNotification>(1);

            // Spawn refiner threads
            // Refiners with children run a hybrid loop: refine + listen for child updates
            // Refiners without children just refine continuously
            for node in &self.refiners {
                let senders = inbox_senders.get(&node.id).cloned().unwrap_or_default();
                let inbox_rx = inbox_receivers.remove(&node.id);
                spawn_refiner(
                    scope,
                    Arc::clone(node),
                    Arc::clone(&stop_flag),
                    senders,
                    inbox_rx,
                    iter_tx.clone(),
                    error_tx.clone(),
                );
            }

            // Spawn propagator threads - they receive from their inbox, send to parents' inboxes
            for node in &self.propagators {
                let inbox_rx = inbox_receivers.remove(&node.id).unwrap();
                let senders = inbox_senders.get(&node.id).cloned().unwrap_or_default();
                spawn_propagator(
                    scope,
                    Arc::clone(node),
                    Arc::clone(&stop_flag),
                    inbox_rx,
                    senders,
                    error_tx.clone(),
                );
            }

            drop(iter_tx);
            drop(error_tx);

            // Coordinator: receive root updates and check precision
            // Also receive iteration notifications for max_iterations tracking
            // Adjust max iterations by number of refiners since each sends independently
            let max_iterations_adjusted = MAX_REFINEMENT_ITERATIONS * self.refiners.len().max(1);
            let mut iterations = 0usize;

            let result = (|| {
                loop {
                    select! {
                        recv(coord_rx) -> msg => {
                            match msg {
                                Ok(_notification) => {
                                    // Root updated - check precision
                                    let root_bounds = self.root.get_bounds()?;
                                    if bounds_width_leq(&root_bounds, epsilon) {
                                        stop_flag.stop();
                                        return Ok(root_bounds);
                                    }
                                }
                                Err(_) => {
                                    // Channel closed - check if precision met
                                    let root_bounds = self.root.get_bounds()?;
                                    if bounds_width_leq(&root_bounds, epsilon) {
                                        return Ok(root_bounds);
                                    }
                                    return Err(ComputableError::RefinementChannelClosed);
                                }
                            }
                        }
                        recv(iter_rx) -> msg => {
                            match msg {
                                Ok(_) => {
                                    iterations += 1;

                                    // Check max iterations (adjusted for number of refiners)
                                    if iterations >= max_iterations_adjusted {
                                        stop_flag.stop();
                                        return Err(ComputableError::MaxRefinementIterations {
                                            max: MAX_REFINEMENT_ITERATIONS,
                                        });
                                    }
                                }
                                Err(_) => {
                                    // All refiners done - check if precision met
                                    let root_bounds = self.root.get_bounds()?;
                                    if bounds_width_leq(&root_bounds, epsilon) {
                                        return Ok(root_bounds);
                                    }
                                    return Err(ComputableError::RefinementChannelClosed);
                                }
                            }
                        }
                        recv(error_rx) -> msg => {
                            if let Ok(err_notification) = msg {
                                stop_flag.stop();
                                return Err(err_notification.error);
                            }
                        }
                    }
                }
            })();

            stop_flag.stop();
            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }
}

fn spawn_refiner<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    parent_senders: Vec<Sender<BoundsNotification>>,
    child_inbox: Option<Receiver<BoundsNotification>>,
    iter_tx: Sender<IterationNotification>,
    error_tx: Sender<ErrorNotification>,
) {
    scope.spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refiner_loop(node, stop, parent_senders, child_inbox, iter_tx, error_tx)
        }));
    });
}

fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    parent_senders: Vec<Sender<BoundsNotification>>,
    child_inbox: Option<Receiver<BoundsNotification>>,
    iter_tx: Sender<IterationNotification>,
    error_tx: Sender<ErrorNotification>,
) {
    // Track cached bounds to detect changes from child updates
    let mut cached_bounds: Option<Bounds> = None;

    while !stop.is_stopped() {
        // If we have a child inbox, check for child updates first
        if let Some(ref inbox) = child_inbox {
            // Non-blocking check for child updates
            while let Ok(_) = inbox.try_recv() {
                // Child updated - recompute our bounds
                let new_bounds = match node.compute_bounds() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = error_tx.send(ErrorNotification { error: e });
                        return;
                    }
                };

                if cached_bounds.as_ref() != Some(&new_bounds) {
                    node.set_bounds(new_bounds.clone());
                    cached_bounds = Some(new_bounds);

                    // Notify parents with blocking send
                    let notification = BoundsNotification { node_id: node.id };
                    for sender in &parent_senders {
                        if sender.send(notification.clone()).is_err() {
                            return;
                        }
                    }
                }
            }
        }

        match node.refine_step() {
            Ok(true) => {
                // Report iteration for max_iterations tracking
                if iter_tx.send(IterationNotification).is_err() {
                    break;
                }

                // Check stop flag before computing bounds (early exit)
                if stop.is_stopped() {
                    break;
                }
                // Made progress - compute and update bounds
                let bounds = match node.compute_bounds() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = error_tx.send(ErrorNotification { error: e });
                        break;
                    }
                };
                node.set_bounds(bounds.clone());
                cached_bounds = Some(bounds);

                // Publish to all parent inboxes
                let notification = BoundsNotification { node_id: node.id };
                for sender in &parent_senders {
                    // Use blocking send to ensure updates are seen
                    if sender.send(notification.clone()).is_err() {
                        return;
                    }
                }
            }
            Ok(false) => {
                // No progress - back off
                thread::sleep(Duration::from_micros(NO_PROGRESS_BACKOFF_US));
            }
            Err(error) => {
                let _ = error_tx.send(ErrorNotification { error });
                break;
            }
        }
    }
}

fn spawn_propagator<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    inbox_rx: Receiver<BoundsNotification>,
    parent_senders: Vec<Sender<BoundsNotification>>,
    error_tx: Sender<ErrorNotification>,
) {
    scope.spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            propagator_loop(node, stop, inbox_rx, parent_senders, error_tx)
        }));
    });
}

fn propagator_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    inbox_rx: Receiver<BoundsNotification>,
    parent_senders: Vec<Sender<BoundsNotification>>,
    error_tx: Sender<ErrorNotification>,
) {
    // Initial bounds computation
    let mut cached_bounds = match node.compute_bounds() {
        Ok(b) => {
            node.set_bounds(b.clone());
            b
        }
        Err(e) => {
            let _ = error_tx.send(ErrorNotification { error: e });
            return;
        }
    };

    // Publish initial bounds to all parents
    let notification = BoundsNotification { node_id: node.id };
    for sender in &parent_senders {
        if sender.send(notification.clone()).is_err() {
            return;
        }
    }

    let timeout = Duration::from_millis(PROPAGATOR_TIMEOUT_MS);

    while !stop.is_stopped() {
        // Wait for any child update in our inbox
        match inbox_rx.recv_timeout(timeout) {
            Ok(_notification) => {
                // A child updated - recompute our bounds
                let new_bounds = match node.compute_bounds() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = error_tx.send(ErrorNotification { error: e });
                        return;
                    }
                };

                // Only propagate if bounds actually changed
                if new_bounds != cached_bounds {
                    node.set_bounds(new_bounds.clone());
                    cached_bounds = new_bounds;

                    // Publish to parent inboxes with blocking send
                    let notification = BoundsNotification { node_id: node.id };
                    for sender in &parent_senders {
                        if sender.send(notification.clone()).is_err() {
                            return;
                        }
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Just check stop flag and continue
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Channel closed, exit
                return;
            }
        }
    }
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
        assert_eq!(computable.bounds().expect("bounds should succeed"), bounds);
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
        // In continuous mode, refiners may start a second iteration before stop flag is set.
        // Allow up to 3 * SLEEP_MS to account for this, while still verifying parallelism
        // (sequential execution would take 4 * SLEEP_MS = 40ms).
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
