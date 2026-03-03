//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - `RefinerHandle`: Handle for controlling a background refiner thread
//! - `NodeUpdate`: Update message from a refiner to the coordinator
//!
//! The refinement model is round-based with early exit, per-refiner exhaustion,
//! and demand-based skipping:
//! 1. Check if root precision meets target — if so, return immediately
//! 2. Count active refiners — if none remain, return exhaustion error
//! 3. Check iteration cap
//! 4. Compute demand budget (ε / 2^⌈log₂(N)⌉) and skip refiners already below it
//! 5. Send Step command to remaining active refiners (safety valve: if all were
//!    skipped but root precision isn't met, step the least-precise refiners,
//!    skipping extreme outliers whose width is negligible vs the widest)
//! 6. After EACH response, apply the update and check precision (early exit)
//! 7. If a refiner reports Exhausted, mark it inactive and track the reason
//! 8. If a refiner reports Error, return the error immediately
//! 9. Repeat
//!
//! This preserves demand pacing (refiners only advance when the coordinator
//! permits) while adding three improvements over plain lock-step:
//! - Early exit: precision is checked after each individual update, not after
//!   waiting for all refiners in a round
//! - Per-refiner exhaustion: converged or state-unchanged refiners are removed
//!   from future rounds instead of causing the entire refinement to fail
//! - Demand-based skipping: refiners whose bounds are already narrow enough
//!   (width ≤ ε/N) are skipped, avoiding wasted work on fast-converging operands

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, unbounded};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary, XBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;

/// A `usize` extended with positive infinity, analogous to `UXBinary`.
///
/// When used as a tolerance exponent: `Finite(n)` means epsilon = 2^(-n),
/// `Inf` means epsilon = 0 (exact convergence required).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XUsize {
    Finite(usize),
    Inf,
}

/// Command sent to a refiner thread.
#[derive(Clone, Copy)]
pub enum RefineCommand {
    Step { precision_bits: usize },
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

/// Reason a refiner has stopped producing further updates.
enum ExhaustionReason {
    /// refine_step() returned Ok(false) — bounds have converged to a point.
    Converged,
    /// refine_step() returned Err(StateUnchanged) — state did not change.
    StateUnchanged,
}

/// Message sent from a refiner thread to the coordinator.
enum RefinerMessage {
    /// Normal update: refine_step succeeded and bounds may have narrowed.
    Update(NodeUpdate),
    /// Refiner has exhausted itself and will not produce further updates.
    Exhausted {
        update: NodeUpdate,
        reason: ExhaustionReason,
    },
    /// Refiner encountered a fatal error.
    Error(ComputableError),
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
        tolerance_exp: &XUsize,
    ) -> Result<Bounds, ComputableError> {
        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());
            let mut refiners = Vec::new();
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
                    // Safe to discard: the refiner may have already exited
                    // (e.g. due to an error), and we're shutting down regardless.
                    let _shutdown = refiner.sender.send(RefineCommand::Stop);
                }
            };

            let result = (|| {
                let num_refiners = refiners.len();
                let mut active = vec![true; num_refiners];
                let mut all_state_unchanged = true;

                let precision_met =
                    |root: &Arc<Node>, tol: &XUsize| -> Result<bool, ComputableError> {
                        let bounds = root.get_bounds()?;
                        Ok(bounds_width_leq(&bounds, tol))
                    };

                for _iteration in 0..MAX_REFINEMENT_ITERATIONS {
                    // Check if precision is already met.
                    if precision_met(&self.root, tolerance_exp)? {
                        return self.root.get_bounds();
                    }

                    // Count active refiners; if none remain, select exhaustion error.
                    let active_count = active.iter().filter(|&&a| a).count();
                    if active_count == 0 {
                        return if all_state_unchanged {
                            Err(ComputableError::StateUnchanged)
                        } else {
                            Err(ComputableError::MaxRefinementIterations {
                                max: MAX_REFINEMENT_ITERATIONS,
                            })
                        };
                    }

                    // Compute demand budget for this round.
                    let demand_budget = compute_demand_budget(tolerance_exp, active_count);
                    let precision_bits = match demand_budget {
                        XUsize::Finite(n) => n,
                        XUsize::Inf => usize::MAX,
                    };

                    // Send Step to active refiners ABOVE demand budget only.
                    let mut expected = 0_usize;
                    for (i, refiner) in refiners.iter().enumerate() {
                        if !active[i] {
                            continue;
                        }
                        let skip = self.refiners[i]
                            .cached_bounds()
                            .is_some_and(|b| bounds_width_leq(&b, &demand_budget));
                        if !skip {
                            refiner
                                .sender
                                .send(RefineCommand::Step { precision_bits })
                                .map_err(|_send_err| ComputableError::RefinementChannelClosed)?;
                            expected = expected.checked_add(1).unwrap_or_else(|| {
                                unreachable!("expected <= refiners.len(), cannot overflow usize")
                            });
                        }
                    }

                    // Safety valve: all active refiners were below demand but root
                    // precision isn't met. Step the least-precise refiners, skipping
                    // extreme outliers whose width is negligible vs the widest.
                    if expected == 0 {
                        // Find widest active refiner width.
                        let max_width = (0..refiners.len())
                            .filter(|&i| active[i])
                            .map(|i| {
                                self.refiners[i]
                                    .cached_bounds()
                                    .map_or(UXBinary::Inf, |b| b.width().clone())
                            })
                            .max();

                        // Step all active refiners except extreme outliers
                        // (those whose width is negligibly small vs the widest).
                        for (i, refiner) in refiners.iter().enumerate() {
                            if !active[i] {
                                continue;
                            }
                            let dominated = max_width.as_ref().is_some_and(|max_w| {
                                self.refiners[i]
                                    .cached_bounds()
                                    .is_some_and(|b| is_width_dominated(b.width(), max_w))
                            });
                            if !dominated {
                                refiner
                                    .sender
                                    .send(RefineCommand::Step { precision_bits })
                                    .map_err(|_send_err| {
                                        ComputableError::RefinementChannelClosed
                                    })?;
                                expected = expected.checked_add(1).unwrap_or_else(|| {
                                    unreachable!(
                                        "expected <= refiners.len(), cannot overflow usize"
                                    )
                                });
                            }
                        }
                    }

                    // Collect responses, checking precision after each update.
                    for _ in 0..expected {
                        let message = match update_rx.recv() {
                            Ok(msg) => msg,
                            Err(_) => return Err(ComputableError::RefinementChannelClosed),
                        };

                        match message {
                            RefinerMessage::Update(update) => {
                                self.apply_update(update)?;
                                if precision_met(&self.root, tolerance_exp)? {
                                    return self.root.get_bounds();
                                }
                            }
                            RefinerMessage::Exhausted { update, reason } => {
                                let exhausted_node_id = update.node_id;
                                self.apply_update(update)?;
                                // Find and mark this refiner inactive.
                                for (i, refiner_node) in self.refiners.iter().enumerate() {
                                    if refiner_node.id == exhausted_node_id {
                                        active[i] = false;
                                        break;
                                    }
                                }
                                if !matches!(reason, ExhaustionReason::StateUnchanged) {
                                    all_state_unchanged = false;
                                }
                                if precision_met(&self.root, tolerance_exp)? {
                                    return self.root.get_bounds();
                                }
                            }
                            RefinerMessage::Error(error) => {
                                return Err(error);
                            }
                        }
                    }
                }

                Err(ComputableError::MaxRefinementIterations {
                    max: MAX_REFINEMENT_ITERATIONS,
                })
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
    updates: Sender<RefinerMessage>,
) -> RefinerHandle {
    let (command_tx, command_rx) = unbounded();
    scope.spawn(move || {
        refiner_loop(node, stop, command_rx, updates);
    });

    RefinerHandle { sender: command_tx }
}

fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    commands: Receiver<RefineCommand>,
    updates: Sender<RefinerMessage>,
) {
    while !stop.is_stopped() {
        match commands.recv() {
            Ok(RefineCommand::Step { precision_bits }) => match node.refine_step(precision_bits) {
                Ok(true) => {
                    let bounds = match node.compute_bounds() {
                        Ok(b) => b,
                        Err(e) => {
                            // Safe to discard: send fails only when the coordinator
                            // has already dropped update_rx (i.e. it already has a result).
                            let _send = updates.send(RefinerMessage::Error(e));
                            break;
                        }
                    };
                    node.set_bounds(bounds.clone());
                    if updates
                        .send(RefinerMessage::Update(NodeUpdate {
                            node_id: node.id,
                            bounds,
                        }))
                        .is_err()
                    {
                        break;
                    }
                }
                Ok(false) => {
                    let bounds = match node.compute_bounds() {
                        Ok(b) => b,
                        Err(e) => {
                            // Safe to discard: send fails only when the coordinator
                            // has already dropped update_rx (i.e. it already has a result).
                            let _send = updates.send(RefinerMessage::Error(e));
                            break;
                        }
                    };
                    node.set_bounds(bounds.clone());
                    // Safe to discard: send fails only when the coordinator
                    // has already dropped update_rx (i.e. it already has a result).
                    let _send = updates.send(RefinerMessage::Exhausted {
                        update: NodeUpdate {
                            node_id: node.id,
                            bounds,
                        },
                        reason: ExhaustionReason::Converged,
                    });
                    break;
                }
                Err(ComputableError::StateUnchanged) => {
                    let bounds = node
                        .cached_bounds()
                        .unwrap_or_else(|| Bounds::new(XBinary::NegInf, XBinary::PosInf));
                    // Safe to discard: send fails only when the coordinator
                    // has already dropped update_rx (i.e. it already has a result).
                    let _send = updates.send(RefinerMessage::Exhausted {
                        update: NodeUpdate {
                            node_id: node.id,
                            bounds,
                        },
                        reason: ExhaustionReason::StateUnchanged,
                    });
                    break;
                }
                Err(error) => {
                    // Safe to discard: send fails only when the coordinator
                    // has already dropped update_rx (i.e. it already has a result).
                    let _send = updates.send(RefinerMessage::Error(error));
                    break;
                }
            },
            Ok(RefineCommand::Stop) | Err(_) => break,
        }
    }
}

/// Demand budget = epsilon / 2^ceil(log2(N)).
/// A refiner with width <= budget is "precise enough" to skip this round.
///
/// Precondition: `num_active >= 1` (the coordinator returns an exhaustion error
/// before dispatching when no active refiners remain).
fn compute_demand_budget(tolerance_exp: &XUsize, num_active: usize) -> XUsize {
    debug_assert!(
        num_active >= 1,
        "compute_demand_budget called with 0 active refiners"
    );
    match tolerance_exp {
        XUsize::Inf => XUsize::Inf,
        XUsize::Finite(exp) => {
            // ceil(log2(N)) = BITS - leading_zeros(N), which for N>=1 gives the
            // number of bits needed to represent N, i.e. the shift amount that
            // makes 2^shift >= N.
            let shift_u32 = usize::BITS
                .checked_sub(num_active.leading_zeros())
                .unwrap_or_else(|| unreachable!("leading_zeros() is always <= usize::BITS"));
            #[allow(clippy::as_conversions)] // u32 always fits in usize
            let shift = shift_u32 as usize;
            let tolerance = *exp;
            let budget = crate::sane_arithmetic!(tolerance, shift; tolerance + shift);
            XUsize::Finite(budget)
        }
    }
}

/// Refiners whose width is <= max_width >> SAFETY_VALVE_SKIP_SHIFT are
/// skipped in the safety valve. This prevents exponentially expensive steps
/// on fast-converging refiners (e.g. PiOp) that are already far more precise
/// than the bottleneck. The exact value barely matters — any reasonable
/// threshold prevents the pathological case.
const SAFETY_VALVE_SKIP_SHIFT: i64 = 4; // skip if width << 4 <= max_width, i.e. width * 16 <= max_width

/// Returns true when `width` is negligibly narrow compared to `max_width`,
/// i.e. `width << SAFETY_VALVE_SKIP_SHIFT <= max_width`.
fn is_width_dominated(width: &UXBinary, max_width: &UXBinary) -> bool {
    match width {
        UXBinary::Inf => false,
        UXBinary::Finite(w) => {
            let shifted = UBinary::new(
                w.mantissa().clone(),
                w.exponent() + BigInt::from(SAFETY_VALVE_SKIP_SHIFT),
            );
            UXBinary::Finite(shifted) <= *max_width
        }
    }
}

/// Compares bounds width against a tolerance exponent.
///
/// `Finite(n)` means epsilon = 2^(-n); `Inf` means epsilon = 0.
/// Returns true if width <= epsilon.
pub fn bounds_width_leq(bounds: &Bounds, tolerance_exp: &XUsize) -> bool {
    match bounds.width() {
        UXBinary::Inf => false,
        UXBinary::Finite(width) => match tolerance_exp {
            XUsize::Inf => width.mantissa().is_zero(),
            XUsize::Finite(exp) => *width <= UBinary::new(BigUint::from(1u32), -BigInt::from(*exp)),
        },
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
        midpoint_between, unwrap_finite, xbin,
    };
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    type IntervalState = Bounds;

    fn interval_bounds(state: &IntervalState) -> Bounds {
        state.clone()
    }

    fn interval_refine_strict(state: IntervalState) -> Result<IntervalState, ComputableError> {
        let midpoint = midpoint_between(state.small(), &state.large());
        Ok(Bounds::new(
            state.small().clone(),
            XBinary::Finite(midpoint),
        ))
    }

    fn sqrt_computable(value_int: u64) -> Computable {
        Computable::constant(bin(i64::try_from(value_int).expect("value fits in i64"), 0))
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
        let tolerance_exp = XUsize::Inf;
        let bounds = computable
            .refine_to_default(tolerance_exp)
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
        let tolerance_exp = XUsize::Inf;
        let bounds = computable
            .refine_to_default(tolerance_exp)
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

        let tolerance_exp = XUsize::Inf;
        // Use a small max iterations count to keep the test fast
        let result = one_third.refine_to::<10>(tolerance_exp);

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
        let tolerance_exp = XUsize::Finite(1);
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let expected = xbin(1, 0);
        let upper = bounds.large();

        assert!(bounds.small() <= &expected && &expected <= &upper);
        assert!(bounds_width_leq(&bounds, &tolerance_exp));
        let refined_bounds = computable.bounds().expect("bounds should succeed");
        let refined_upper = refined_bounds.large();
        assert!(refined_bounds.small() <= &expected && &expected <= &refined_upper);
    }

    #[test]
    fn refine_to_rejects_unchanged_state() {
        let computable = interval_noop_computable(0, 2);
        let tolerance_exp = XUsize::Finite(2);
        let result = computable.refine_to_default(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::StateUnchanged)));
    }

    #[test]
    fn refine_to_enforces_max_iterations() {
        let computable = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let tolerance_exp = XUsize::Finite(1);
        let result = computable.refine_to::<5>(tolerance_exp);
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
        let tolerance_exp = XUsize::Finite(1);
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let upper = bounds.large();
        assert!(bounds.small() < &upper);
        assert!(bounds_width_leq(&bounds, &tolerance_exp));
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
                Ok(Bounds::new(
                    inner_state.small().clone(),
                    XBinary::Finite(worse_upper),
                ))
            },
        );
        let tolerance_exp = XUsize::Finite(2);
        let result = computable.refine_to_default(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    // --- concurrency tests ---

    #[test]
    fn refine_shared_clone_updates_original() {
        let original = sqrt_computable(2);
        let cloned = original.clone();
        let tolerance_exp = XUsize::Finite(12);

        let _bounds = cloned
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let bounds = original.bounds().expect("bounds should succeed");
        assert!(bounds_width_leq(&bounds, &tolerance_exp));
    }

    #[test]
    fn refine_to_propagates_refiner_error() {
        let computable = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |_| Err(ComputableError::DomainError),
        );

        let tolerance_exp = XUsize::Finite(4);
        let result = computable.refine_to::<2>(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::DomainError)));
    }

    #[test]
    fn refine_to_max_iterations_multiple_refiners() {
        let left = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let right = Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let expr = left + right;
        let tolerance_exp = XUsize::Finite(4);
        let result = expr.refine_to::<2>(tolerance_exp);
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
            |state| Ok(Bounds::new(state.small().clone(), xbin(2, 0))),
        );
        let expr = stable + faulty;
        let tolerance_exp = XUsize::Finite(4);
        let result = expr.refine_to::<3>(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    #[test]
    fn concurrent_bounds_reads_during_failed_refinement() {
        let computable = Arc::new(Computable::new(
            0usize,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        ));
        let tolerance_exp = XUsize::Finite(6);
        let reader = Arc::clone(&computable);
        let handle = thread::spawn(move || {
            for _ in 0_i32..8_i32 {
                let bounds = reader.bounds().expect("bounds should succeed");
                assert_width_nonnegative(&bounds);
            }
        });

        let result = computable.refine_to::<3>(tolerance_exp);
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
        let sleep_duration = Duration::from_millis(SLEEP_MS);

        let slow_refiner = || {
            Computable::new(
                0usize,
                |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
                |state| {
                    thread::sleep(Duration::from_millis(SLEEP_MS));
                    Ok(state + 1)
                },
            )
        };

        let expr = slow_refiner() + slow_refiner() + slow_refiner() + slow_refiner();
        let tolerance_exp = XUsize::Finite(6);

        let start = Instant::now();
        let result = expr.refine_to::<1>(tolerance_exp);
        let elapsed = start.elapsed();

        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 1 })
        ));
        // Use Duration comparison instead of as_millis() truncation
        // to avoid off-by-one issues when elapsed is e.g. 10.5ms
        assert!(
            elapsed >= sleep_duration,
            "refinement must not have actually run, elapsed {elapsed:?}"
        );
        assert!(
            elapsed < 2 * sleep_duration,
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
        let tolerance_exp = XUsize::Finite(10);
        // Coordinate multiple threads calling refine_to on the same computable.
        let barrier = Arc::new(Barrier::new(4));

        let mut handles = Vec::new();
        for _ in 0_i32..3_i32 {
            let shared_expression = Arc::clone(&expression);
            let shared_barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                shared_barrier.wait();
                shared_expression.refine_to_default(tolerance_exp)
            }));
        }

        barrier.wait();
        let main_bounds = expression
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let main_upper = main_bounds.large();
        assert!(bounds_width_leq(&main_bounds, &tolerance_exp));

        for handle in handles {
            let bounds = handle
                .join()
                .expect("thread should join")
                .expect("refine_to should succeed");
            let bounds_upper = bounds.large();
            assert_width_nonnegative(&bounds);
            assert!(bounds_width_leq(&bounds, &tolerance_exp));
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
        let tolerance_exp = XUsize::Finite(6);
        let barrier = Arc::new(Barrier::new(3));

        let mut handles = Vec::new();
        for _ in 0_i32..2_i32 {
            let shared_value = Arc::clone(&shared);
            let shared_barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                shared_barrier.wait();
                shared_value
                    .refine_to_default(tolerance_exp)
                    .expect("refine_to should succeed")
            }));
        }

        barrier.wait();
        let main_bounds = shared
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        for handle in handles {
            let bounds = handle.join().expect("thread should join");
            assert_width_nonnegative(&bounds);
        }

        assert!(!saw_overlap.load(Ordering::SeqCst));
        assert!(bounds_width_leq(&main_bounds, &tolerance_exp));
    }

    #[test]
    fn concurrent_bounds_reads_during_refinement() {
        let base_value = interval_midpoint_computable(0, 4);
        let shared_value = Arc::new(base_value);
        let tolerance_exp = XUsize::Finite(8);
        // Reader thread repeatedly calls bounds while refinement is running.
        let barrier = Arc::new(Barrier::new(2));

        let reader = {
            let reader_value = Arc::clone(&shared_value);
            let reader_barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                reader_barrier.wait();
                for _ in 0_i32..32_i32 {
                    let bounds = reader_value.bounds().expect("bounds should succeed");
                    assert_width_nonnegative(&bounds);
                }
            })
        };

        barrier.wait();
        let refined = shared_value
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        reader.join().expect("reader should join");
        assert_width_nonnegative(&refined);
    }

    /// Demonstrates a flaw in demand-based skipping: y starts with width 3/8,
    /// which is already below the target precision of 1/2, so ideally y would
    /// be skipped entirely. But the demand budget (2^(-3) = 1/8 for tolerance=1
    /// with 2 refiners) is tighter than necessary, so y gets stepped anyway,
    /// causing refinement to wait for y's expensive sleep.
    #[test]
    fn demand_skipping_unnecessarily_steps_already_precise_refiner() {
        use std::time::Instant;

        const SLOW_STEP_MS: u64 = 1000;

        // x: starts at [0, 1024] (width = 1024), converges by halving the
        // upper bound each step (width halves each round). No sleeps, so
        // convergence is fast but gradual — takes ~12 steps to reach width < 0.5.
        let x = Computable::new(
            Bounds::new(xbin(0, 0), xbin(1024, 0)),
            |state| Ok(state.clone()),
            interval_refine_strict,
        );

        // y: starts at [0, 3/8] (width = 3/8, below the target precision of
        // 1/2 but ABOVE the demand budget of 1/8). Each step sleeps 1s, so
        // if refinement steps y unnecessarily it'll be slow.
        let y = Computable::new(
            Bounds::new(xbin(0, 0), xbin(3, -3)),
            |state| Ok(state.clone()),
            move |state: Bounds| {
                thread::sleep(Duration::from_millis(SLOW_STEP_MS));
                interval_refine(state)
            },
        );

        let sum = x + y;
        let tolerance_exp = XUsize::Finite(1); // target width ≤ 0.5

        let start = Instant::now();
        let bounds = sum
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let elapsed = start.elapsed();

        assert!(
            bounds_width_leq(&bounds, &tolerance_exp),
            "bounds should meet target precision"
        );
        // Ideally this would finish without stepping y (< 1s), but the demand
        // budget is too tight, so y gets stepped unnecessarily.
        assert!(
            elapsed >= Duration::from_millis(SLOW_STEP_MS),
            "expected y to be stepped (demand budget flaw), but finished in {elapsed:?}"
        );
    }
}
