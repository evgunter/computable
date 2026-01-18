//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - `RefinerHandle`: Handle for controlling a background refiner thread
//! - `NodeUpdate`: Update message from a refiner to the coordinator
//!
//! The refinement model is synchronous lock-step:
//! 1. Send Step command to ALL refiners
//! 2. Wait for ALL refiners to complete one step
//! 3. Collect and propagate ALL updates
//! 4. Check if precision met
//! 5. Repeat
//!
//! TODO: The README describes an async/event-driven refinement model where:
//! - Branches refine continuously and publish updates
//! - Other nodes "subscribe" to updates and recompute live
//! Either the README should be updated to reflect the actual synchronous
//! model, or the implementation should be changed to the async model described.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;
use crate::binary::Bounds;

/// Command sent to a refiner thread.
#[derive(Clone, Copy)]
pub enum RefineCommand {
    Step,
    Stop,
}

/// Handle for a background refiner task.
pub struct RefinerHandle {
    pub sender: Sender<RefineCommand>,
}

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

/// Compares bounds width (UXBinary) against epsilon (UBinary).
/// Returns true if width <= epsilon.
pub fn bounds_width_leq(bounds: &Bounds, epsilon: &UBinary) -> bool {
    match bounds.width() {
        UXBinary::PosInf => false,
        UXBinary::Finite(uwidth) => *uwidth <= *epsilon,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::binary::{Binary, XBinary};
    use crate::computable::Computable;
    use crate::error::ComputableError;
    use num_bigint::{BigInt, BigUint};
    use num_traits::One;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    type IntervalState = Bounds;

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

    fn assert_width_nonnegative(bounds: &Bounds) {
        assert!(*bounds.width() >= UXBinary::zero());
    }

    // --- tests for different results of refinement (mostly errors) ---

    #[test]
    fn refine_to_rejects_zero_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(0, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::NonpositiveEpsilon)));
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
            UXBinary::PosInf => panic!("expected finite width"),
        };

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
    fn refine_to_max_iterations_multiple_refiners() {
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
        use std::time::Instant;

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
