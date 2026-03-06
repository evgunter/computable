//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - `RefinerHandle`: Handle for controlling a background refiner thread
//! - `NodeUpdate`: Update message from a refiner to the coordinator
//!
//! The refinement model is an event loop with per-refiner step tracking,
//! demand-based skipping, and outstanding-step management:
//! 1. Check if root precision meets target — if so, return immediately
//! 2. Check eligibility (active and under per-refiner step limit) — if no
//!    eligible refiners and none outstanding, return exhaustion/iteration error
//! 3. Compute per-refiner demand budgets by walking the graph top-down from
//!    root, using each operation's sensitivity to map parent targets to child
//!    budgets (e.g. AddOp splits ε/2 per child, MulOp uses ε/(2·|sibling|)).
//!    Dispatch eligible, non-outstanding refiners above their budget. Refiners
//!    not reachable through passive combinators are always stepped.
//! 4. Collect responses: block for one, drain any immediately available via
//!    try_recv, then check precision (early exit). Outstanding refiners from
//!    the same batch may still be in flight — the next iteration re-dispatches
//!    any eligible non-outstanding refiners while slow ones continue computing.
//! 5. If a refiner reports Exhausted, mark it inactive and track the reason
//! 6. If a refiner reports Error, return the error immediately
//! 7. Loop back to step 1
//!
//! This event-loop design preserves demand pacing while allowing fast refiners
//! to advance when a slow refiner is still computing:
//! - Non-blocking: each recv processes one response (plus any try_recv'd extras),
//!   checks precision for early exit, then re-dispatches eligible refiners
//! - Per-refiner step limit: each refiner gets at most MAX steps, matching the
//!   old round-based iteration cap per refiner
//! - Per-refiner exhaustion: converged or state-unchanged refiners are removed
//!   from future dispatch instead of causing the entire refinement to fail
//! - Demand-based skipping: per-refiner budgets propagated through the graph
//!   skip refiners whose bounds are already narrow enough for their position
//!   in the computation, avoiding wasted work on fast-converging operands

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, unbounded};
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;
use crate::prefix::Prefix;
use crate::sane::{XIsize, XUsize};

/// Command sent to a refiner thread.
#[derive(Clone, Copy)]
pub enum RefineCommand {
    Step { target_width_exp: XIsize },
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
    pub prefix: Prefix,
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
        // Eagerly populate all prefix caches so we can identify exact-bounds refiners.
        let root_prefix = self.root.get_prefix()?;
        if prefix_width_leq(&root_prefix, tolerance_exp) {
            return Ok(root_prefix.to_bounds());
        }

        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());
            let (update_tx, update_rx) = unbounded();

            // Only spawn threads for refiners whose bounds aren't already exact.
            let mut refiner_nodes = Vec::new();
            let mut refiner_handles = Vec::new();
            for node in &self.refiners {
                let exact = node
                    .cached_prefix()
                    .is_some_and(|p| p.width_exponent() == XIsize::NegInf);
                if !exact {
                    refiner_handles.push(spawn_refiner(
                        scope,
                        Arc::clone(node),
                        Arc::clone(&stop_flag),
                        update_tx.clone(),
                    ));
                    refiner_nodes.push(Arc::clone(node));
                }
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
                let num_refiners = refiner_handles.len();
                let mut active = vec![true; num_refiners];
                let mut all_state_unchanged = true;
                let mut steps = vec![0usize; num_refiners];
                let mut outstanding = vec![false; num_refiners];
                let mut eligible_count = num_refiners;

                // Build node_id → refiner index mapping for routing responses.
                let mut refiner_index: HashMap<usize, usize> = HashMap::new();
                for (i, node) in refiner_nodes.iter().enumerate() {
                    refiner_index.insert(node.id, i);
                }

                // Determine two properties per refiner:
                //
                // is_leaf_refiner: subtree contains no other refiners.
                //   Leaf refiners are self-improving and can always be
                //   re-dispatched. Non-leaf refiners should only be
                //   re-dispatched when another refiner responds (inputs
                //   may have changed via apply_update propagation).
                //
                // budget_is_static: the path from root to this refiner
                //   passes only through ops with static budgets (AddOp,
                //   NegOp). Static budgets don't change as bounds tighten,
                //   so they never need refreshing at wave boundaries.
                let refiner_id_set: HashSet<usize> = refiner_nodes.iter().map(|n| n.id).collect();
                // For each refiner, collect the indices of sub-refiners in
                // its subtree. Leaf refiners have no sub-refiners.
                let sub_refiner_indices: Vec<Vec<usize>> = refiner_nodes
                    .iter()
                    .map(|node| {
                        let mut subs = Vec::new();
                        let mut seen = HashSet::new();
                        let mut stack: Vec<Arc<Node>> = node.children();
                        while let Some(child) = stack.pop() {
                            if refiner_id_set.contains(&child.id)
                                && let Some(&idx) = refiner_index.get(&child.id)
                                && seen.insert(idx)
                            {
                                subs.push(idx);
                            }
                            stack.extend(child.children());
                        }
                        subs
                    })
                    .collect();
                let is_leaf_refiner: Vec<bool> = sub_refiner_indices
                    .iter()
                    .map(|subs| subs.is_empty())
                    .collect();
                // Walk root-to-refiner paths to check if any op along the
                // way has bounds-dependent budgets. We do this by walking
                // the BFS used for budget propagation and tracking whether
                // any ancestor had budget_depends_on_bounds.
                let budget_is_static: Vec<bool> = {
                    let mut node_is_static: HashMap<usize, bool> = HashMap::new();
                    node_is_static.insert(self.root.id, true);
                    let mut queue = VecDeque::new();
                    queue.push_back(self.root.id);
                    while let Some(node_id) = queue.pop_front() {
                        let Some(node) = self.nodes.get(&node_id) else {
                            continue;
                        };
                        let parent_static = node_is_static.get(&node_id).copied().unwrap_or(true);
                        let child_static = parent_static && !node.op.budget_depends_on_bounds();
                        for child in node.children() {
                            let entry = node_is_static.entry(child.id).or_insert(true);
                            // If ANY path is non-static, mark non-static.
                            *entry = *entry && child_static;
                            queue.push_back(child.id);
                        }
                    }
                    refiner_nodes
                        .iter()
                        .map(|node| node_is_static.get(&node.id).copied().unwrap_or(true))
                        .collect()
                };
                let _any_budget_dynamic = budget_is_static.iter().any(|&s| !s);

                // Track whether each refiner should be re-dispatched.
                // True initially (first dispatch) and when another refiner
                // responds (inputs may have changed via propagation). Leaf
                // refiners are always re-dispatched since they don't depend
                // on other refiners' bounds.
                let mut needs_redispatch = vec![true; num_refiners];
                // For non-leaf refiners: track how many sub-refiners have
                // responded since the last dispatch. Only mark for redispatch
                // when ALL sub-refiners have responded (so compute_bounds
                // reads fully-updated inputs, not partially-stale ones).
                let mut sub_responded_count: Vec<usize> = sub_refiner_indices
                    .iter()
                    .map(|subs| subs.len()) // start "all responded" so first dispatch proceeds
                    .collect();
                let mut sub_responded: Vec<Vec<bool>> = sub_refiner_indices
                    .iter()
                    .map(|subs| vec![true; subs.len()])
                    .collect();
                // Queue of refiners ready for dispatch consideration.
                // Avoids scanning all N refiners each iteration — only
                // check refiners whose needs_redispatch was recently set.
                let mut dispatch_queue: VecDeque<usize> = (0..num_refiners).collect();
                let mut in_queue = vec![true; num_refiners];

                let precision_met =
                    |root: &Arc<Node>, tol: &XUsize| -> Result<bool, ComputableError> {
                        let prefix = root.get_prefix()?;
                        Ok(prefix_width_leq(&prefix, tol))
                    };

                let mut refiner_budgets: Vec<Option<UXBinary>> = {
                    let map = self.compute_propagated_budgets(tolerance_exp);
                    refiner_nodes
                        .iter()
                        .map(|node| map.get(&node.id).cloned())
                        .collect()
                };

                loop {
                    // 1. Check if there's any remaining work (eligible or outstanding).
                    //    (Precision is checked after each response in the collection
                    //    phase, and before the loop via the pre-spawn check at the
                    //    top of refine_to. No need to re-check here.)
                    let any_eligible = eligible_count > 0;
                    let any_outstanding = outstanding.iter().any(|&o| o);

                    if !any_eligible && !any_outstanding {
                        let all_exhausted = active.iter().all(|&a| !a);
                        return if all_exhausted && all_state_unchanged {
                            Err(ComputableError::StateUnchanged)
                        } else {
                            Err(ComputableError::MaxRefinementIterations {
                                max: MAX_REFINEMENT_ITERATIONS,
                            })
                        };
                    }

                    // 3. Dispatch eligible, non-outstanding refiners above demand budget.
                    //
                    //    Propagated budgets: walk the graph top-down from root, using
                    //    each op's sensitivity to compute per-refiner budgets.
                    //
                    //    The propagated budgets are provably sufficient: if every
                    //    refiner meets its budget, the root meets the target. This
                    //    follows from the sensitivity bounds at each combinator:
                    //    - AddOp: w_out = w_a + w_b ≤ ε/2 + ε/2 = ε
                    //    - MulOp: w_out ≤ |a|·w_b + |b|·w_a ≤ ε/2 + ε/2 = ε
                    //      (no cross-term: |a| uses the endpoint, not the center)
                    //    - PowOp: w_out ≤ n·max_abs^(n-1)·w_in ≤ ε (MVT)
                    //
                    //    Budgets are refreshed at wave boundaries (when nothing
                    //    is outstanding) so that sensitivity-based budgets
                    //    like MulOp's child_budget = ε/(2·|sibling|) loosen
                    //    as sibling bounds tighten, allowing refiners to stop
                    //    earlier. The initial budgets are the most conservative
                    //    (widest bounds → tightest budgets) and remain provably
                    //    sufficient throughout, so the refresh only helps —
                    //    it never makes budgets tighter than initial.
                    //
                    //    Only refresh if the graph has bounds-dependent ops
                    //    (MulOp, PowOp, etc.). For pure AddOp/NegOp trees,
                    //    budgets are static functions of the tolerance.
                    // Only refresh if there are active non-leaf refiners
                    // with dynamic budgets. Leaf refiners always
                    // self-redispatch, so budget loosening doesn't change
                    // their dispatch behavior.
                    let needs_refresh = !any_outstanding
                        && (0..num_refiners)
                            .any(|i| active[i] && !budget_is_static[i] && !is_leaf_refiner[i]);
                    if needs_refresh {
                        let map = self.compute_propagated_budgets(tolerance_exp);
                        for (i, node) in refiner_nodes.iter().enumerate() {
                            if !budget_is_static[i] {
                                refiner_budgets[i] = map.get(&node.id).cloned();
                            }
                        }
                    }
                    let mut dispatched = 0usize;

                    while let Some(i) = dispatch_queue.pop_front() {
                        in_queue[i] = false;
                        if !active[i]
                            || outstanding[i]
                            || steps[i] >= MAX_REFINEMENT_ITERATIONS
                            || !needs_redispatch[i]
                        {
                            continue;
                        }
                        let skip = refiner_budgets[i].as_ref().is_some_and(|budget| {
                            refiner_nodes[i]
                                .cached_prefix()
                                .is_some_and(|p| p.width() <= *budget)
                        });
                        if !skip {
                            // Convert budget to target_width_exp for the refiner.
                            let target_width_exp = refiner_budgets[i]
                                .as_ref()
                                .map(uxbinary_to_exp)
                                .unwrap_or_else(|| tolerance_to_exp(tolerance_exp));
                            refiner_handles[i]
                                .sender
                                .send(RefineCommand::Step { target_width_exp })
                                .map_err(|_send_err| ComputableError::RefinementChannelClosed)?;
                            outstanding[i] = true;
                            needs_redispatch[i] = false;
                            // Reset sub-refiner tracking for non-leaf refiners.
                            // Keep exhausted sub-refiners marked as responded
                            // (they won't respond again, so clearing them
                            // would make the gate unreachable).
                            if !is_leaf_refiner[i] {
                                sub_responded_count[i] = 0;
                                for (k, flag) in sub_responded[i].iter_mut().enumerate() {
                                    let sub_idx = sub_refiner_indices[i][k];
                                    if active[sub_idx] {
                                        *flag = false;
                                    } else {
                                        // Exhausted sub-refiner: keep as responded.
                                        sub_responded_count[i] =
                                            sub_responded_count[i].saturating_add(1);
                                    }
                                }
                            }
                            dispatched = dispatched.checked_add(1).unwrap_or_else(|| {
                                unreachable!(
                                    "dispatched <= refiner_handles.len(), cannot overflow usize"
                                )
                            });
                        }
                    }

                    // Stall recovery: if nothing was dispatched and nothing
                    // is outstanding, check if any active above-budget refiner
                    // is blocked only by needs_redispatch. If so, force-enable
                    // it (its dependencies have all responded or been skipped,
                    // so there's nothing to wait for). If no such refiner
                    // exists, it's a true stall.
                    if dispatched == 0 && !outstanding.iter().any(|&o| o) {
                        let mut any_forced = false;
                        for i in 0..num_refiners {
                            if active[i]
                                && !needs_redispatch[i]
                                && steps[i] < MAX_REFINEMENT_ITERATIONS
                            {
                                let above_budget =
                                    !refiner_budgets[i].as_ref().is_some_and(|budget| {
                                        refiner_nodes[i]
                                            .cached_prefix()
                                            .is_some_and(|p| p.width() <= *budget)
                                    });
                                if above_budget {
                                    needs_redispatch[i] = true;
                                    if !in_queue[i] {
                                        dispatch_queue.push_back(i);
                                        in_queue[i] = true;
                                    }
                                    any_forced = true;
                                }
                            }
                        }
                        if any_forced {
                            continue; // retry dispatch with forced refiners
                        }
                        return Err(ComputableError::MaxRefinementIterations {
                            max: MAX_REFINEMENT_ITERATIONS,
                        });
                    }

                    // 4. Receive responses and update state.
                    //
                    //    Block for one response, drain any immediately available
                    //    via try_recv, then loop back to dispatch. Checking
                    //    precision after each response allows early exit when
                    //    apply_update propagation meets the target mid-batch.
                    //
                    //    After each response, mark OTHER refiners as needing
                    //    re-dispatch (their inputs may have changed). The
                    //    responding refiner is only marked for re-dispatch if
                    //    its bounds actually improved — this prevents wasteful
                    //    re-dispatches when compute_bounds reads stale inputs
                    //    from slow refiners that haven't responded yet.

                    // Process a response: apply the update and return the refiner
                    // index + exhaustion info so the caller can update bookkeeping
                    // arrays without borrow conflicts with the `outstanding` loop.
                    let apply_response = |message: RefinerMessage| -> Result<
                        (usize, Option<ExhaustionReason>, Vec<usize>),
                        ComputableError,
                    > {
                        match message {
                            RefinerMessage::Update(update) => {
                                let idx = refiner_index[&update.node_id];
                                let changed = self.apply_update(update)?;
                                Ok((idx, None, changed))
                            }
                            RefinerMessage::Exhausted { update, reason } => {
                                let idx = refiner_index[&update.node_id];
                                let changed = self.apply_update(update)?;
                                Ok((idx, Some(reason), changed))
                            }
                            RefinerMessage::Error(error) => Err(error),
                        }
                    };

                    // Inline helper: update bookkeeping after a response.
                    // Not a closure to avoid holding a mutable borrow on
                    // `outstanding` across the straggler collection loop.
                    macro_rules! record_completion {
                        ($idx:expr, $exhaustion:expr, $changed:expr) => {{
                            let idx = $idx;
                            let changed_nodes: &[usize] = &$changed;
                            outstanding[idx] = false;
                            steps[idx] = steps[idx].checked_add(1).unwrap_or_else(|| {
                                unreachable!("steps <= MAX_REFINEMENT_ITERATIONS, cannot overflow")
                            });
                            // Track eligibility: refiner becomes ineligible
                            // when it exhausts or hits the step limit.
                            if let Some(reason) = $exhaustion {
                                active[idx] = false;
                                eligible_count = eligible_count.saturating_sub(1);
                                if !matches!(reason, ExhaustionReason::StateUnchanged) {
                                    all_state_unchanged = false;
                                }
                            } else if steps[idx] >= MAX_REFINEMENT_ITERATIONS {
                                eligible_count = eligible_count.saturating_sub(1);
                            }
                            // Leaf responding refiner: always re-dispatch
                            // (self-improving, independent of other refiners).
                            if is_leaf_refiner[idx] {
                                needs_redispatch[idx] = true;
                                if !in_queue[idx] {
                                    dispatch_queue.push_back(idx);
                                    in_queue[idx] = true;
                                }
                            } else if needs_redispatch[idx] && !in_queue[idx] {
                                // Non-leaf refiner already marked (all subs
                                // responded) but was outstanding when the
                                // queue was drained. Re-enqueue now.
                                dispatch_queue.push_back(idx);
                                in_queue[idx] = true;
                            }
                            // Use the apply_update propagation path to
                            // determine which non-leaf refiners had their
                            // inputs changed. If a non-leaf refiner's node
                            // was recomputed during propagation, its
                            // compute_bounds read updated child bounds →
                            // it should count toward that refiner's
                            // sub-responded gate.
                            for &changed_id in changed_nodes {
                                if let Some(&j) = refiner_index.get(&changed_id) {
                                    if j != idx && !is_leaf_refiner[j] {
                                        if let Some(sub_pos) =
                                            sub_refiner_indices[j].iter().position(|&s| s == idx)
                                        {
                                            if !sub_responded[j][sub_pos] {
                                                sub_responded[j][sub_pos] = true;
                                                sub_responded_count[j] = sub_responded_count[j]
                                                    .checked_add(1)
                                                    .unwrap_or_else(|| unreachable!());
                                            }
                                            if sub_responded_count[j]
                                                >= sub_refiner_indices[j].len()
                                            {
                                                needs_redispatch[j] = true;
                                                if !in_queue[j] {
                                                    dispatch_queue.push_back(j);
                                                    in_queue[j] = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }};
                    }

                    // Block for the first response.
                    {
                        let first = match update_rx.recv() {
                            Ok(msg) => msg,
                            Err(_) => return Err(ComputableError::RefinementChannelClosed),
                        };
                        let (idx, exhaustion, changed) = apply_response(first)?;
                        record_completion!(idx, exhaustion, changed);
                        if precision_met(&self.root, tolerance_exp)? {
                            return self.root.get_bounds();
                        }
                    }

                    // Drain any immediately available responses.
                    while let Ok(msg) = update_rx.try_recv() {
                        let (idx, exhaustion, changed) = apply_response(msg)?;
                        record_completion!(idx, exhaustion, changed);
                        if precision_met(&self.root, tolerance_exp)? {
                            return self.root.get_bounds();
                        }
                    }
                }
            })();

            shutdown_refiners(refiner_handles, &stop_flag);
            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }

    /// Applies a node update and propagates prefix changes upward.
    /// Returns the set of node IDs that were changed by propagation
    /// (excluding the directly-updated node).
    fn apply_update(&self, update: NodeUpdate) -> Result<Vec<usize>, ComputableError> {
        let mut changed_by_propagation = Vec::new();
        let mut queue = VecDeque::new();
        if let Some(node) = self.nodes.get(&update.node_id) {
            node.set_prefix(update.prefix);
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
                let next_prefix = parent.compute_prefix()?;
                if parent.cached_prefix().as_ref() != Some(&next_prefix) {
                    parent.set_prefix(next_prefix);
                    changed_by_propagation.push(*parent_id);
                    queue.push_back(*parent_id);
                }
            }
        }

        Ok(changed_by_propagation)
    }

    /// Computes per-refiner demand budgets by walking the graph top-down.
    ///
    /// Starting from the root with the overall tolerance, propagates budgets
    /// through passive combinators using their sensitivity to child widths.
    /// For DAG nodes (shared subexpressions under multiple parents), takes the
    /// tightest (minimum) budget.
    ///
    /// Refiners that are not reachable through passive combinators (e.g. children
    /// of other refiners) will not appear in the returned map.
    fn compute_propagated_budgets(&self, tolerance_exp: &XUsize) -> HashMap<usize, UXBinary> {
        let target_width = tolerance_to_uxbinary(tolerance_exp);
        let mut budgets: HashMap<usize, UXBinary> = HashMap::new();
        budgets.insert(self.root.id, target_width);

        let mut queue = VecDeque::new();
        queue.push_back(self.root.id);

        while let Some(node_id) = queue.pop_front() {
            let Some(node) = self.nodes.get(&node_id) else {
                continue;
            };

            let Some(budget) = budgets.get(&node_id).cloned() else {
                continue;
            };

            let children = node.children();
            for (child_idx, child) in children.iter().enumerate() {
                let child_budget = node.op.child_demand_budget(&budget, child_idx);
                let entry = budgets.entry(child.id).or_insert(UXBinary::Inf);
                if child_budget < *entry {
                    *entry = child_budget;
                    queue.push_back(child.id);
                }
            }
        }

        budgets
    }
}

/// Converts a tolerance exponent to a UXBinary width value.
///
/// `Finite(n)` → width 2^(-n), `Inf` → width 0 (exact convergence).
fn tolerance_to_uxbinary(tolerance_exp: &XUsize) -> UXBinary {
    match tolerance_exp {
        XUsize::Inf => UXBinary::zero(),
        XUsize::Finite(exp) => {
            UXBinary::Finite(UBinary::new(BigUint::from(1u32), -BigInt::from(*exp)))
        }
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
            Ok(RefineCommand::Step { target_width_exp }) => {
                // Keep refining until the prefix visibly changes or we exhaust.
                // This prevents infinite loops where the underlying bounds improve
                // but the Prefix (power-of-2 rounded width) stays the same.
                let old_prefix = node.cached_prefix();
                let mut extra_steps = 0usize;

                loop {
                    match node.refine_step(target_width_exp) {
                        Ok(true) => {
                            let prefix = match node.compute_prefix() {
                                Ok(p) => p,
                                Err(e) => {
                                    let _send = updates.send(RefinerMessage::Error(e));
                                    return;
                                }
                            };

                            // Check if prefix visibly changed
                            let changed = old_prefix.as_ref() != Some(&prefix);
                            node.set_prefix(prefix.clone());

                            if changed || extra_steps >= 16 {
                                if updates
                                    .send(RefinerMessage::Update(NodeUpdate {
                                        node_id: node.id,
                                        prefix,
                                    }))
                                    .is_err()
                                {
                                    return;
                                }
                                break;
                            }
                            // Prefix didn't change — do another refine step
                            extra_steps = extra_steps.saturating_add(1);
                        }
                        Ok(false) => {
                            let prefix = match node.compute_prefix() {
                                Ok(p) => p,
                                Err(e) => {
                                    let _send = updates.send(RefinerMessage::Error(e));
                                    return;
                                }
                            };
                            node.set_prefix(prefix.clone());
                            let _send = updates.send(RefinerMessage::Exhausted {
                                update: NodeUpdate {
                                    node_id: node.id,
                                    prefix,
                                },
                                reason: ExhaustionReason::Converged,
                            });
                            return;
                        }
                        Err(ComputableError::StateUnchanged) => {
                            let prefix = node.cached_prefix().unwrap_or_else(Prefix::unbounded);
                            let _send = updates.send(RefinerMessage::Exhausted {
                                update: NodeUpdate {
                                    node_id: node.id,
                                    prefix,
                                },
                                reason: ExhaustionReason::StateUnchanged,
                            });
                            return;
                        }
                        Err(error) => {
                            let _send = updates.send(RefinerMessage::Error(error));
                            return;
                        }
                    }
                }
            }
            Ok(RefineCommand::Stop) | Err(_) => break,
        }
    }
}

/// Compares bounds width against a tolerance exponent.
///
/// `Finite(n)` means epsilon = 2^(-n); `Inf` means epsilon = 0.
/// Returns true if width <= epsilon.
#[cfg(test)]
pub fn bounds_width_leq(bounds: &Bounds, tolerance_exp: &XUsize) -> bool {
    match bounds.width() {
        UXBinary::Inf => false,
        UXBinary::Finite(width) => match tolerance_exp {
            XUsize::Inf => width.mantissa().is_zero(),
            XUsize::Finite(exp) => *width <= UBinary::new(BigUint::from(1u32), -BigInt::from(*exp)),
        },
    }
}

/// Compares prefix width against a tolerance exponent.
///
/// `Finite(n)` means epsilon = 2^(-n); `Inf` means epsilon = 0.
/// Returns true if width_exponent <= -tolerance_exp.
pub fn prefix_width_leq(prefix: &Prefix, tolerance_exp: &XUsize) -> bool {
    let we = prefix.width_exponent();
    #[allow(clippy::arithmetic_side_effects)] // XUsize::neg() is total (no panics)
    let target = -(*tolerance_exp);
    we <= target
}

/// Converts a UXBinary width to an `XIsize` exponent.
///
/// Returns the smallest e such that 2^e >= width, or `PosInf` for infinity.
fn uxbinary_to_exp(ux: &UXBinary) -> XIsize {
    match ux {
        UXBinary::Inf => XIsize::PosInf,
        UXBinary::Finite(ub) => {
            if ub.mantissa().is_zero() {
                return XIsize::NegInf;
            }
            let bits = ub.mantissa().bits();
            let bits_isize = isize::try_from(bits).unwrap_or(isize::MAX);
            let exp_isize = ub.exponent().to_isize().unwrap_or(isize::MAX);
            // width = mantissa * 2^exponent, mantissa has `bits` bits
            // so width < 2^(bits + exponent), ceil = bits + exponent if mantissa != power of 2
            // For a simple upper bound: bits + exponent (may overcount by 1, conservative)
            XIsize::Finite(bits_isize.saturating_add(exp_isize))
        }
    }
}

/// Converts a tolerance exponent to an `XIsize` target for refiners.
fn tolerance_to_exp(tolerance_exp: &XUsize) -> XIsize {
    #[allow(clippy::arithmetic_side_effects)] // XUsize::neg() is total (no panics)
    {
        -(*tolerance_exp)
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

    #[test]
    fn refinement_parallelizes_multiple_refiners() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let concurrent_count = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let slow_refiner = |concurrent: Arc<AtomicUsize>, max_conc: Arc<AtomicUsize>| {
            Computable::new(
                0usize,
                |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
                move |state| {
                    let prev = concurrent.fetch_add(1, Ordering::SeqCst);
                    max_conc.fetch_max(prev + 1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10));
                    concurrent.fetch_sub(1, Ordering::SeqCst);
                    Ok(state + 1)
                },
            )
        };

        let expr = slow_refiner(Arc::clone(&concurrent_count), Arc::clone(&max_concurrent))
            + slow_refiner(Arc::clone(&concurrent_count), Arc::clone(&max_concurrent))
            + slow_refiner(Arc::clone(&concurrent_count), Arc::clone(&max_concurrent))
            + slow_refiner(Arc::clone(&concurrent_count), Arc::clone(&max_concurrent));
        let tolerance_exp = XUsize::Finite(6);

        let result = expr.refine_to::<1>(tolerance_exp);

        assert!(matches!(
            result,
            Err(ComputableError::MaxRefinementIterations { max: 1 })
        ));
        assert!(
            max_concurrent.load(Ordering::SeqCst) >= 4,
            "expected all 4 refiners to run concurrently, max observed: {}",
            max_concurrent.load(Ordering::SeqCst)
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

    /// Tests that propagated demand budgets correctly skip a precise refiner.
    ///
    /// y starts at width 3/16 = 0.1875, which is below the AddOp propagated
    /// budget of ε/2 = 0.25 (tolerance ε = 0.5 with 2 refiners). With the
    /// old flat budget (ε/4 = 0.125), y would have been stepped unnecessarily
    /// since 0.1875 > 0.125. The propagated budget correctly identifies that
    /// y is precise enough and skips it.
    #[test]
    fn propagated_budget_skips_precise_add_operand() {
        use std::time::Instant;

        const SLOW_STEP_MS: u64 = 1000;

        // x: starts at [0, 1024] (width = 1024), converges by halving the
        // upper bound each step (width halves each round). No sleeps, so
        // convergence is fast but gradual — takes ~12 steps to reach width < 0.25.
        let x = Computable::new(
            Bounds::new(xbin(0, 0), xbin(1024, 0)),
            |state| Ok(state.clone()),
            interval_refine_strict,
        );

        // y: starts at [0, 3/16] (width = 3/16 = 0.1875). Below the
        // propagated budget of ε/2 = 0.25, but above the old flat budget
        // of ε/4 = 0.125. Each step sleeps 1s — if stepped, the test is slow.
        let y = Computable::new(
            Bounds::new(xbin(0, 0), xbin(3, -4)),
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
        // With propagated budgets, y is correctly skipped (width 0.1875 ≤
        // budget 0.25), so refinement finishes fast — only x needs to converge.
        assert!(
            elapsed < Duration::from_millis(SLOW_STEP_MS),
            "expected y to be skipped (propagated budget), but took {elapsed:?}"
        );
    }

    /// Demonstrates the event loop's benefit: fast refiners are re-dispatched
    /// immediately while slow refiners are still computing, reducing total
    /// wall-clock time compared to a round-based model.
    ///
    /// x: fast, [0, 1024], halves each step (microseconds per step)
    /// y: slow, [0, 4], halves each step (200ms per step)
    /// tolerance: width ≤ 1 (tolerance_exp = 0)
    ///
    /// Round model: sum width = 1028/2^k after k rounds, needs k ≥ 11 → 2200ms
    /// Event loop: x finishes ~20 steps in microseconds while y does 3 steps
    ///             (4→2→1→0.5), sum width ≈ 0.5 + tiny ≤ 1 after ~600ms
    #[test]
    fn event_loop_does_not_block_fast_refiner_on_slow_refiner() {
        use std::time::Instant;

        const SLOW_STEP_MS: u64 = 200;

        // x: fast refiner with wide initial bounds. Halves each step with no
        // sleep, so all ~20 needed steps complete in microseconds.
        let x = Computable::new(
            Bounds::new(xbin(0, 0), xbin(1024, 0)),
            |state| Ok(state.clone()),
            interval_refine_strict,
        );

        // y: slow refiner with narrower initial bounds. Halves each step but
        // sleeps 200ms per step, so each step is expensive.
        let y = Computable::new(
            Bounds::new(xbin(0, 0), xbin(4, 0)),
            |state| Ok(state.clone()),
            move |state: Bounds| {
                thread::sleep(Duration::from_millis(SLOW_STEP_MS));
                interval_refine_strict(state)
            },
        );

        let sum = x + y;
        let tolerance_exp = XUsize::Finite(0); // target width ≤ 1

        let start = Instant::now();
        let bounds = sum
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let elapsed = start.elapsed();

        assert!(
            bounds_width_leq(&bounds, &tolerance_exp),
            "bounds should meet target precision"
        );
        // Event loop: ~600ms (3 y-steps of 200ms each, x runs concurrently).
        // A round-based model would need ~11 rounds × 200ms = ~2200ms.
        assert!(
            elapsed < Duration::from_millis(800),
            "expected event loop to finish fast (~600ms), but took {elapsed:?} \
             (round-based model would take ~2200ms)"
        );
    }
}
