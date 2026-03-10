//! Parallel refinement infrastructure for the computation graph.
//!
//! This module provides the machinery for refining computable numbers to a desired precision:
//! - `RefinementGraph`: Snapshot of the computation graph for coordinating refinement
//! - Worker pool: threads are reused across refiners via a shared command channel
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

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, unbounded};
use num_bigint::BigUint;
use num_traits::Zero;

use crate::binary::Bounds;
use crate::binary::{UBinary, UXBinary};
use crate::concurrency::StopFlag;
use crate::error::ComputableError;
use crate::node::Node;
use crate::prefix::Prefix;
use crate::sane::{self, U, XI, XU};

/// Converts a node ID (`U` = u32) to `usize` for Vec indexing.
/// Guaranteed lossless because `usize` is at least 32 bits on all supported platforms.
#[inline]
fn id(n: U) -> usize {
    sane::u_as_usize(n)
}

/// Command sent to the shared worker pool, tagged with a refiner index.
#[derive(Clone, Copy)]
enum WorkerCommand {
    Step {
        refiner_idx: usize,
        target_width_exp: XI,
    },
    Stop,
}

/// Update message from a refiner thread.
#[derive(Clone)]
pub struct NodeUpdate {
    pub node_id: U,
    pub prefix: Prefix,
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
    pub nodes: Vec<Option<Arc<Node>>>, // node id -> node (indexed by node_id)
    pub parents: Vec<Vec<usize>>,      // child id -> parent ids (indexed by node_id)
    pub refiners: Vec<Arc<Node>>,
    /// Size for Vec-based budget maps (max node ID + 1).
    budget_vec_len: usize,
}

impl RefinementGraph {
    pub fn new(root: Arc<Node>) -> Result<Self, ComputableError> {
        // First pass: collect all node IDs to determine the Vec size.
        let mut all_ids = Vec::new();
        {
            let mut seen = HashSet::new();
            let mut stack = vec![Arc::clone(&root)];
            while let Some(node) = stack.pop() {
                if !seen.insert(node.id) {
                    continue;
                }
                all_ids.push(node.id);
                for child in node.children() {
                    stack.push(child);
                }
            }
        }

        // Pre-compute Vec size for all indexed Vecs: max_node_id + 1.
        // Safe: all_ids is non-empty (contains at least root), and node IDs
        // are sequential from a global AtomicUsize counter.
        let budget_vec_len = id(
            all_ids
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .unwrap_or_else(|| unreachable!("node ID at U::MAX implies memory exhaustion")),
        );

        let mut nodes: Vec<Option<Arc<Node>>> = vec![None; budget_vec_len];
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); budget_vec_len];
        let mut refiners = Vec::new();

        let mut stack = vec![Arc::clone(&root)];
        while let Some(node) = stack.pop() {
            if nodes[id(node.id)].is_some() {
                continue;
            }
            let node_id = id(node.id);
            nodes[node_id] = Some(Arc::clone(&node));

            if node.is_refiner() {
                refiners.push(Arc::clone(&node));
            }
            for child in node.children() {
                parents[id(child.id)].push(node_id);
                stack.push(child);
            }
        }

        let graph = Self {
            root,
            nodes,
            parents,
            refiners,
            budget_vec_len,
        };

        Ok(graph)
    }

    pub fn refine_to<const MAX_REFINEMENT_ITERATIONS: U>(
        &self,
        tolerance_exp: &XU,
    ) -> Result<Bounds, ComputableError> {
        // Eagerly populate all prefix caches so we can identify exact-bounds refiners.
        let root_prefix = self.root.get_prefix()?;
        if prefix_width_leq(&root_prefix, tolerance_exp) {
            return Ok(root_prefix.to_bounds());
        }

        // Only refine nodes whose bounds aren't already exact.
        let refiner_nodes: Vec<Arc<Node>> = self
            .refiners
            .iter()
            .filter(|node| !node.cached_bounds().is_some_and(|b| b.width().is_zero()))
            .map(Arc::clone)
            .collect();

        // Fast path: single leaf refiner — skip thread spawn and channel overhead.
        // When there's exactly one non-exact refiner, it's trivially a leaf (no
        // other active refiners can appear in its subtree). We run the refinement
        // loop synchronously, propagating updates via apply_update.
        if refiner_nodes.len() == 1 {
            return self
                .refine_single::<MAX_REFINEMENT_ITERATIONS>(&refiner_nodes[0], tolerance_exp);
        }

        let mut outcome = None;
        thread::scope(|scope| {
            let stop_flag = Arc::new(StopFlag::new());
            let (update_tx, update_rx) = unbounded();

            // Spawn a capped worker pool with a shared command channel.
            // All workers pull from the same queue, so no single worker
            // becomes a bottleneck when many refiners are assigned.
            let num_refiners_spawned = refiner_nodes.len();
            let num_workers = worker_count(num_refiners_spawned);
            let (work_tx, work_rx) = unbounded::<WorkerCommand>();
            let refiner_nodes_shared = Arc::new(refiner_nodes.clone());
            for _ in 0..num_workers {
                let nodes = Arc::clone(&refiner_nodes_shared);
                let stop = Arc::clone(&stop_flag);
                let rx = work_rx.clone();
                let tx = update_tx.clone();
                scope.spawn(move || {
                    worker_loop(nodes, stop, rx, tx);
                });
            }
            drop(update_tx);

            let shutdown_workers = |work_sender: Sender<WorkerCommand>,
                                    stop_signal: &Arc<StopFlag>,
                                    n_workers: usize| {
                stop_signal.stop();
                for _ in 0..n_workers {
                    // Safe to discard: workers may have already exited.
                    let _shutdown = work_sender.send(WorkerCommand::Stop);
                }
            };

            let result = (|| {
                let num_refiners = num_refiners_spawned;
                let mut active = vec![true; num_refiners];
                let mut all_state_unchanged = true;
                let mut steps = vec![0_u32; num_refiners];
                let mut outstanding = vec![false; num_refiners];
                let mut outstanding_count = 0usize;
                let mut eligible_count = num_refiners;

                // Build node_id → refiner index mapping for routing responses.
                let mut refiner_index: Vec<Option<usize>> = vec![None; self.budget_vec_len];
                for (i, node) in refiner_nodes.iter().enumerate() {
                    refiner_index[id(node.id)] = Some(i);
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
                let mut refiner_id_set: Vec<bool> = vec![false; self.budget_vec_len];
                for node in &refiner_nodes {
                    refiner_id_set[id(node.id)] = true;
                }
                // For each refiner, collect the indices of sub-refiners in
                // its subtree. Leaf refiners have no sub-refiners.
                let sub_refiner_indices: Vec<Vec<usize>> = refiner_nodes
                    .iter()
                    .map(|node| {
                        let mut subs = Vec::new();
                        let mut seen = HashSet::new();
                        let mut stack: Vec<Arc<Node>> = node.children();
                        while let Some(child) = stack.pop() {
                            if refiner_id_set[id(child.id)]
                                && let Some(&idx) = refiner_index[id(child.id)].as_ref()
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
                    let mut node_is_static: Vec<Option<bool>> = vec![None; self.budget_vec_len];
                    node_is_static[id(self.root.id)] = Some(true);
                    let mut queue = VecDeque::new();
                    queue.push_back(id(self.root.id));
                    while let Some(node_id) = queue.pop_front() {
                        let Some(node) = self.nodes[node_id].as_ref() else {
                            continue;
                        };
                        let parent_static = node_is_static[node_id].unwrap_or(true);
                        let child_static = parent_static && !node.op.budget_depends_on_bounds();
                        for child in node.children() {
                            let entry = node_is_static[id(child.id)].get_or_insert(true);
                            // If ANY path is non-static, mark non-static.
                            *entry = *entry && child_static;
                            queue.push_back(id(child.id));
                        }
                    }
                    refiner_nodes
                        .iter()
                        .map(|node| node_is_static[id(node.id)].unwrap_or(true))
                        .collect()
                };
                // Count of active non-leaf refiners with dynamic budgets.
                // Used to quickly check if budget refresh is needed without
                // scanning all N refiners each iteration.
                let mut dynamic_nonleaf_count = (0..num_refiners)
                    .filter(|&i| !budget_is_static[i] && !is_leaf_refiner[i])
                    .count();

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

                let precision_met = |root: &Arc<Node>, tol: &XU| -> Result<bool, ComputableError> {
                    let prefix = root.get_prefix()?;
                    Ok(prefix_width_leq(&prefix, tol))
                };

                let target_width = tolerance_to_uxbinary(tolerance_exp);
                let mut cached_budget_map = self.compute_propagated_budgets(&target_width, None);
                let mut refiner_budgets: Vec<Option<UXBinary>> = {
                    refiner_nodes
                        .iter()
                        .map(|node| cached_budget_map[id(node.id)].clone())
                        .collect()
                };
                let mut last_refresh_root_we = self.root.get_prefix()?.width_exponent();

                loop {
                    // 1. Check if there's any remaining work (eligible or outstanding).
                    //    (Precision is checked after each response in the collection
                    //    phase, and before the loop via the pre-spawn check at the
                    //    top of refine_to. No need to re-check here.)
                    let any_eligible = eligible_count > 0;
                    let any_outstanding = outstanding_count > 0;

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
                    let refresh_candidate = !any_outstanding && dynamic_nonleaf_count > 0;
                    let needs_refresh = refresh_candidate && {
                        let current_we = self
                            .root
                            .cached_prefix()
                            .unwrap_or_else(Prefix::unbounded)
                            .width_exponent();
                        current_we < last_refresh_root_we
                    };
                    if needs_refresh {
                        let map = self
                            .compute_propagated_budgets(&target_width, Some(&cached_budget_map));
                        for (i, node) in refiner_nodes.iter().enumerate() {
                            if !budget_is_static[i] {
                                refiner_budgets[i] = map[id(node.id)].clone();
                            }
                        }
                        cached_budget_map = map;
                        last_refresh_root_we = self
                            .root
                            .cached_prefix()
                            .unwrap_or_else(Prefix::unbounded)
                            .width_exponent();
                    }
                    let mut dispatched = 0u32;

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
                                .cached_bounds()
                                .is_some_and(|b| *b.width() <= *budget)
                        });
                        if !skip {
                            // Convert budget to target_width_exp for the refiner.
                            let target_width_exp = refiner_budgets[i]
                                .as_ref()
                                .map(uxbinary_to_exp)
                                .unwrap_or_else(|| tolerance_to_exp(tolerance_exp));
                            work_tx
                                .send(WorkerCommand::Step {
                                    refiner_idx: i,
                                    target_width_exp,
                                })
                                .map_err(|_send_err| ComputableError::RefinementChannelClosed)?;
                            outstanding[i] = true;
                            outstanding_count =
                                outstanding_count.checked_add(1).unwrap_or_else(|| {
                                    unreachable!(
                                        "outstanding_count <= num_refiners, cannot overflow"
                                    )
                                });
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
                                            sub_responded_count[i].checked_add(1).unwrap_or_else(|| {
                                                unreachable!("sub_responded_count bounded by sub_refiner_indices[i].len()")
                                            });
                                    }
                                }
                            }
                            dispatched = dispatched.checked_add(1).unwrap_or_else(|| {
                                unreachable!("dispatched <= num_refiners, cannot overflow usize")
                            });
                        }
                    }

                    // Stall recovery: if nothing was dispatched and nothing
                    // is outstanding, check if any active above-budget refiner
                    // is blocked only by needs_redispatch. If so, force-enable
                    // it (its dependencies have all responded or been skipped,
                    // so there's nothing to wait for). If no such refiner
                    // exists, it's a true stall.
                    if dispatched == 0 && outstanding_count == 0 {
                        let mut any_forced = false;
                        for i in 0..num_refiners {
                            if active[i]
                                && !needs_redispatch[i]
                                && steps[i] < MAX_REFINEMENT_ITERATIONS
                            {
                                let above_budget =
                                    !refiner_budgets[i].as_ref().is_some_and(|budget| {
                                        refiner_nodes[i]
                                            .cached_bounds()
                                            .is_some_and(|b| *b.width() <= *budget)
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
                                let idx = refiner_index[id(update.node_id)].unwrap_or_else(|| {
                                    unreachable!("refiner_index populated for all refiner node IDs")
                                });
                                let changed = self.apply_update(update)?;
                                Ok((idx, None, changed))
                            }
                            RefinerMessage::Exhausted { update, reason } => {
                                let idx = refiner_index[id(update.node_id)].unwrap_or_else(|| {
                                    unreachable!("refiner_index populated for all refiner node IDs")
                                });
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
                            outstanding_count = outstanding_count.checked_sub(1).unwrap_or_else(|| {
                                unreachable!("outstanding_count > 0: each completion decrements at most once")
                            });
                            steps[idx] = steps[idx].checked_add(1).unwrap_or_else(|| {
                                unreachable!("steps <= MAX_REFINEMENT_ITERATIONS, cannot overflow")
                            });
                            // Track eligibility: refiner becomes ineligible
                            // when it exhausts or hits the step limit.
                            if let Some(reason) = $exhaustion {
                                active[idx] = false;
                                eligible_count = eligible_count.checked_sub(1).unwrap_or_else(|| {
                                    unreachable!("eligible_count > 0: each refiner decrements at most once")
                                });
                                if !budget_is_static[idx] && !is_leaf_refiner[idx] {
                                    dynamic_nonleaf_count = dynamic_nonleaf_count.saturating_sub(1);
                                }
                                if !matches!(reason, ExhaustionReason::StateUnchanged) {
                                    all_state_unchanged = false;
                                }
                            } else if steps[idx] >= MAX_REFINEMENT_ITERATIONS {
                                eligible_count = eligible_count.checked_sub(1).unwrap_or_else(|| {
                                    unreachable!("eligible_count > 0: each refiner decrements at most once")
                                });
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
                                if let Some(&j) = refiner_index[changed_id].as_ref() {
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

                    // Block for the first response. Only check precision
                    // when the root's bounds actually changed (the root was
                    // either the directly-updated refiner or was reached by
                    // apply_update propagation), skipping the check otherwise.
                    let root_id = self.root.id;
                    let root_id_usize = id(root_id);
                    {
                        let first = match update_rx.recv() {
                            Ok(msg) => msg,
                            Err(_) => return Err(ComputableError::RefinementChannelClosed),
                        };
                        let (idx, exhaustion, changed) = apply_response(first)?;
                        let root_changed =
                            refiner_nodes[idx].id == root_id || changed.contains(&root_id_usize);
                        record_completion!(idx, exhaustion, changed);
                        if root_changed && precision_met(&self.root, tolerance_exp)? {
                            return self.root.get_bounds();
                        }
                    }

                    // Drain any immediately available responses.
                    while let Ok(msg) = update_rx.try_recv() {
                        let (idx, exhaustion, changed) = apply_response(msg)?;
                        let root_changed =
                            refiner_nodes[idx].id == root_id || changed.contains(&root_id_usize);
                        record_completion!(idx, exhaustion, changed);
                        if root_changed && precision_met(&self.root, tolerance_exp)? {
                            return self.root.get_bounds();
                        }
                    }
                }
            })();

            shutdown_workers(work_tx, &stop_flag, num_workers);
            outcome = Some(result);
        });

        match outcome {
            Some(result) => result,
            None => Err(ComputableError::RefinementChannelClosed),
        }
    }

    /// Synchronous single-refiner refinement loop.
    ///
    /// Executes the refinement loop directly without spawning threads or
    /// creating channels. For single-refiner graphs (e.g. `pi()`, `sqrt()`,
    /// `inv()`), this avoids significant per-call overhead from thread spawn,
    /// channel creation, and worker pool management.
    ///
    /// The refiner may or may not be the root node. When it's not the root
    /// (e.g. a BaseOp child under a NthRootOp), `apply_update` propagates
    /// bounds changes upward through the graph.
    fn refine_single<const MAX_REFINEMENT_ITERATIONS: U>(
        &self,
        node: &Arc<Node>,
        tolerance_exp: &XU,
    ) -> Result<Bounds, ComputableError> {
        let target_width_exp = tolerance_to_exp(tolerance_exp);

        for _step in 0..MAX_REFINEMENT_ITERATIONS {
            let old_prefix = node.cached_prefix();
            let mut extra_steps = 0usize;

            // Inner loop: keep refining until the prefix visibly changes or
            // we exhaust, mirroring the logic in execute_refine_step.
            loop {
                match node.refine_step(target_width_exp) {
                    Ok(true) => {
                        let bounds = node.op.compute_bounds()?;
                        let prefix = Prefix::from_lower_upper(
                            bounds.small().clone(),
                            bounds.large().clone(),
                        );
                        let changed = old_prefix.as_ref() != Some(&prefix);
                        node.set_prefix_and_bounds(prefix.clone(), bounds.clone());

                        if changed || extra_steps >= 16 {
                            // Propagate upward to parents (handles root != refiner).
                            self.apply_update(NodeUpdate {
                                node_id: node.id,
                                prefix,
                                bounds,
                            })?;
                            break;
                        }
                        extra_steps = extra_steps.checked_add(1).unwrap_or_else(|| {
                            unreachable!("extra_steps bounded by loop break at 16")
                        });
                    }
                    Ok(false) => {
                        // Converged: update, propagate, and return.
                        let bounds = node.op.compute_bounds()?;
                        let prefix = Prefix::from_lower_upper(
                            bounds.small().clone(),
                            bounds.large().clone(),
                        );
                        node.set_prefix_and_bounds(prefix.clone(), bounds.clone());
                        self.apply_update(NodeUpdate {
                            node_id: node.id,
                            prefix,
                            bounds,
                        })?;
                        return self.root.get_bounds();
                    }
                    Err(ComputableError::StateUnchanged) => {
                        // Precision wasn't met (checked at loop top and at
                        // refine_to entry), and the refiner can't improve.
                        return Err(ComputableError::StateUnchanged);
                    }
                    Err(e) => return Err(e),
                }
            }

            // Check if root precision now meets the target.
            let root_prefix = self.root.get_prefix()?;
            if prefix_width_leq(&root_prefix, tolerance_exp) {
                return Ok(root_prefix.to_bounds());
            }
        }

        Err(ComputableError::MaxRefinementIterations {
            max: MAX_REFINEMENT_ITERATIONS,
        })
    }

    /// Applies a node update and propagates prefix changes upward.
    /// Returns the set of node IDs that were changed by propagation
    /// (excluding the directly-updated node).
    ///
    /// **Bounds-cache preservation**: uses `set_prefix_and_bounds` instead
    /// of `set_prefix` so that each recomputed parent's exact bounds stay
    /// cached. When the next ancestor calls `child.get_bounds()` during
    /// its own `compute_bounds()`, it hits the cache instead of
    /// recomputing from scratch. (`set_prefix` invalidates bounds_cache,
    /// forcing every level to redundantly re-derive bounds.)
    fn apply_update(&self, update: NodeUpdate) -> Result<Vec<usize>, ComputableError> {
        let mut changed_by_propagation = Vec::new();
        let mut queue = VecDeque::new();
        if let Some(node) = self.nodes[id(update.node_id)].as_ref() {
            node.set_prefix_and_bounds(update.prefix, update.bounds);
            queue.push_back(id(node.id));
        }

        while let Some(changed_id) = queue.pop_front() {
            let parents = &self.parents[changed_id];
            if parents.is_empty() {
                continue;
            }
            for parent_id in parents {
                let parent = self.nodes[*parent_id]
                    .as_ref()
                    .ok_or(ComputableError::RefinementChannelClosed)?;
                let next_bounds = parent.op.compute_bounds()?;
                let next_prefix = Prefix::from_lower_upper(
                    next_bounds.small().clone(),
                    next_bounds.large().clone(),
                );
                if parent.cached_prefix().as_ref() != Some(&next_prefix) {
                    // Cache both prefix AND exact bounds so ancestors'
                    // compute_bounds() hits the bounds cache.
                    parent.set_prefix_and_bounds(next_prefix, next_bounds);
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
    /// of other refiners) will not appear in the returned vec (their slot will
    /// be `None`).
    fn compute_propagated_budgets(
        &self,
        target_width: &UXBinary,
        cached_budgets: Option<&[Option<UXBinary>]>,
    ) -> Vec<Option<UXBinary>> {
        let mut budgets: Vec<Option<UXBinary>> = vec![None; self.budget_vec_len];
        budgets[id(self.root.id)] = Some(target_width.clone());

        let mut queue = VecDeque::new();
        queue.push_back(id(self.root.id));

        while let Some(node_id) = queue.pop_front() {
            let Some(node) = self.nodes[node_id].as_ref() else {
                continue;
            };

            let Some(budget) = budgets[node_id].clone() else {
                continue;
            };

            // If this node has a static budget (doesn't depend on bounds)
            // and the cached budget matches, skip recomputing children
            // and copy their cached budgets instead.
            if let Some(cached) = cached_budgets
                && !node.op.budget_depends_on_bounds()
                && let Some(cached_budget) = cached.get(node_id).and_then(|b| b.as_ref())
                && *cached_budget == budget
            {
                let children = node.children();
                let mut all_cached = true;
                for child in &children {
                    if let Some(cached_child) = cached.get(id(child.id)).and_then(|b| b.as_ref()) {
                        let entry = budgets[id(child.id)].get_or_insert(UXBinary::Inf);
                        if *cached_child < *entry {
                            *entry = cached_child.clone();
                            queue.push_back(id(child.id));
                        }
                    } else {
                        all_cached = false;
                        break;
                    }
                }
                if all_cached {
                    continue;
                }
            }

            let children = node.children();
            for (child_idx, child) in children.iter().enumerate() {
                // Operations have at most 2 children, so child_idx always fits in U.
                #[allow(clippy::as_conversions)]
                let child_budget = node.op.child_demand_budget(&budget, child_idx as U);
                let entry = budgets[id(child.id)].get_or_insert(UXBinary::Inf);
                if child_budget < *entry {
                    *entry = child_budget;
                    queue.push_back(id(child.id));
                }
            }
        }

        budgets
    }
}

/// Converts a tolerance exponent to a UXBinary width value.
///
/// `Finite(n)` → width 2^(-n), `Inf` → width 0 (exact convergence).
fn tolerance_to_uxbinary(tolerance_exp: &XU) -> UXBinary {
    match tolerance_exp {
        XU::Inf => UXBinary::zero(),
        XU::Finite(exp) => UXBinary::Finite(UBinary::new(
            BigUint::from(1u32),
            i64::from(*exp)
                .checked_neg()
                .unwrap_or_else(|| {
                    crate::detected_computable_would_exhaust_memory!("tolerance negation overflow")
                }),
        )),
    }
}

/// Returns the number of worker threads to spawn for the given refiner count.
///
/// Caps at available CPU parallelism to avoid spawning excessive OS threads
/// when the graph contains many refiners (e.g. 1000 NthRoot nodes).
fn worker_count(num_refiners: usize) -> usize {
    if num_refiners == 0 {
        return 0;
    }
    let cpus = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4_usize);
    num_refiners.min(cpus)
}

/// Worker loop that processes commands for multiple refiners from a shared queue.
///
/// Each worker receives commands tagged with a refiner index, looks up the
/// corresponding node, and executes the refinement step. Multiple workers
/// pull from the same channel, providing work-stealing behavior. This allows
/// N refiners to be multiplexed onto fewer OS threads.
fn worker_loop(
    refiner_nodes: Arc<Vec<Arc<Node>>>,
    stop: Arc<StopFlag>,
    commands: Receiver<WorkerCommand>,
    updates: Sender<RefinerMessage>,
) {
    while !stop.is_stopped() {
        match commands.recv() {
            Ok(WorkerCommand::Step {
                refiner_idx,
                target_width_exp,
            }) => {
                let Some(node) = refiner_nodes.get(refiner_idx) else {
                    continue;
                };
                execute_refine_step(node, target_width_exp, &updates);
            }
            Ok(WorkerCommand::Stop) | Err(_) => break,
        }
    }
}

/// Executes a single refine step for a node and sends the result.
///
/// Keeps refining until the prefix visibly changes or we exhaust.
/// This prevents infinite loops where the underlying bounds improve
/// but the Prefix (power-of-2 rounded width) stays the same.
fn execute_refine_step(
    node: &Arc<Node>,
    target_width_exp: XI,
    updates: &Sender<RefinerMessage>,
) {
    let old_prefix = node.cached_prefix();
    let mut extra_steps = 0usize;

    loop {
        match node.refine_step(target_width_exp) {
            Ok(true) => {
                let bounds = match node.op.compute_bounds() {
                    Ok(b) => b,
                    Err(e) => {
                        let _send = updates.send(RefinerMessage::Error(e));
                        return;
                    }
                };
                let prefix =
                    Prefix::from_lower_upper(bounds.small().clone(), bounds.large().clone());

                // Check if prefix visibly changed
                let changed = old_prefix.as_ref() != Some(&prefix);
                node.set_prefix_and_bounds(prefix.clone(), bounds.clone());

                if changed || extra_steps >= 16 {
                    let _send = updates.send(RefinerMessage::Update(NodeUpdate {
                        node_id: node.id,
                        prefix,
                        bounds,
                    }));
                    return;
                }
                // Prefix didn't change — do another refine step
                extra_steps = extra_steps
                    .checked_add(1)
                    .unwrap_or_else(|| unreachable!("extra_steps bounded by loop break at 16"));
            }
            Ok(false) => {
                let bounds = match node.op.compute_bounds() {
                    Ok(b) => b,
                    Err(e) => {
                        let _send = updates.send(RefinerMessage::Error(e));
                        return;
                    }
                };
                let prefix =
                    Prefix::from_lower_upper(bounds.small().clone(), bounds.large().clone());
                node.set_prefix_and_bounds(prefix.clone(), bounds.clone());
                let _send = updates.send(RefinerMessage::Exhausted {
                    update: NodeUpdate {
                        node_id: node.id,
                        prefix,
                        bounds,
                    },
                    reason: ExhaustionReason::Converged,
                });
                return;
            }
            Err(ComputableError::StateUnchanged) => {
                let prefix = node.cached_prefix().unwrap_or_else(Prefix::unbounded);
                let bounds = node.cached_bounds().unwrap_or_else(|| prefix.to_bounds());
                let _send = updates.send(RefinerMessage::Exhausted {
                    update: NodeUpdate {
                        node_id: node.id,
                        prefix,
                        bounds,
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

/// Compares bounds width against a tolerance exponent.
///
/// `Finite(n)` means epsilon = 2^(-n); `Inf` means epsilon = 0.
/// Returns true if width <= epsilon.
#[cfg(test)]
pub fn bounds_width_leq(bounds: &Bounds, tolerance_exp: &XU) -> bool {
    match bounds.width() {
        UXBinary::Inf => false,
        UXBinary::Finite(width) => match tolerance_exp {
            XU::Inf => width.mantissa().is_zero(),
            XU::Finite(exp) => {
                *width
                    <= UBinary::new(
                        BigUint::from(1u32),
                        i64::try_from(*exp)
                            .unwrap_or_else(|_err| {
                                crate::detected_computable_would_exhaust_memory!(
                                    "tolerance exceeds i64"
                                )
                            })
                            .checked_neg()
                            .unwrap_or_else(|| {
                                crate::detected_computable_would_exhaust_memory!(
                                    "tolerance negation overflow"
                                )
                            }),
                    )
            }
        },
    }
}

/// Compares prefix width against a tolerance exponent.
///
/// `Finite(n)` means epsilon = 2^(-n); `Inf` means epsilon = 0.
/// Returns true if width_exponent <= -tolerance_exp.
pub fn prefix_width_leq(prefix: &Prefix, tolerance_exp: &XU) -> bool {
    let we = prefix.width_exponent();
    #[allow(clippy::arithmetic_side_effects)] // XU::neg() is total (no panics)
    let target = -(*tolerance_exp);
    we <= target
}

/// Converts a UXBinary width to an `XI` exponent.
///
/// Returns the smallest e such that 2^e >= width, or `PosInf` for infinity.
fn uxbinary_to_exp(ux: &UXBinary) -> XI {
    match ux {
        UXBinary::Inf => XI::PosInf,
        UXBinary::Finite(ub) => {
            if ub.mantissa().is_zero() {
                return XI::NegInf;
            }
            let bits = ub.mantissa().bits();
            let Ok(bits_i32) = i32::try_from(bits) else {
                // mantissa has more bits than i32 can represent: width is astronomically large
                return XI::PosInf;
            };
            let Ok(exp_i32) = i32::try_from(ub.exponent()) else {
                // exponent doesn't fit in i32: width is astronomically large (or small)
                // Positive exponent → PosInf; negative exponent with huge mantissa → still large
                return if ub.exponent() < 0 {
                    // Huge negative exponent: width ≈ 0, exponent is extremely negative
                    XI::NegInf
                } else {
                    XI::PosInf
                };
            };
            // width = mantissa * 2^exponent, mantissa has `bits` bits
            // so width < 2^(bits + exponent), ceil = bits + exponent if mantissa != power of 2
            // For a simple upper bound: bits + exponent (may overcount by 1, conservative)
            match bits_i32.checked_add(exp_i32) {
                Some(v) => XI::Finite(v),
                // Overflow in positive direction: astronomically large width
                None if exp_i32 > 0 => XI::PosInf,
                // Overflow in negative direction: astronomically small width
                None => XI::NegInf,
            }
        }
    }
}

/// Converts a tolerance exponent to an `XI` target for refiners.
fn tolerance_to_exp(tolerance_exp: &XU) -> XI {
    #[allow(clippy::arithmetic_side_effects)] // XU::neg() is total (no panics)
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
        let midpoint = midpoint_between(state.small(), state.large());
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
        let tolerance_exp = XU::Inf;
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to with epsilon=0 should succeed when bounds converge exactly");

        // After refinement, bounds should be exactly [1, 1]
        assert_eq!(bounds.small(), &xbin(1, 0));
        assert_eq!(bounds.large(), &xbin(1, 0));

        // Width should be exactly zero
        assert!(matches!(bounds.width(), UXBinary::Finite(w) if w.mantissa().is_zero()));
    }

    #[test]
    fn refine_to_with_zero_epsilon_on_constant_succeeds_immediately() {
        // A constant computable already has exact bounds (width = 0)
        let computable = Computable::constant(bin(42, 0));
        let tolerance_exp = XU::Inf;
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to with epsilon=0 should succeed for constants");

        // Bounds should be exactly [42, 42]
        assert_eq!(bounds.small(), &xbin(42, 0));
        assert_eq!(bounds.large(), &xbin(42, 0));
    }

    #[test]
    fn refine_to_with_zero_epsilon_on_non_exact_value_returns_max_iterations() {
        // 1/3 cannot be represented exactly in binary, so epsilon=0 should
        // eventually hit max iterations rather than hanging forever.
        let one = Computable::constant(bin(1, 0));
        let three = Computable::constant(bin(3, 0));
        let one_third = one / three;

        let tolerance_exp = XU::Inf;
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
        let tolerance_exp = XU::Finite(1);
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        let expected = xbin(1, 0);
        let upper = bounds.large();

        assert!(bounds.small() <= &expected && expected <= *upper);
        assert!(bounds_width_leq(&bounds, &tolerance_exp));
        let refined_bounds = computable.bounds().expect("bounds should succeed");
        let refined_upper = refined_bounds.large();
        assert!(refined_bounds.small() <= &expected && expected <= *refined_upper);
    }

    #[test]
    fn refine_to_rejects_unchanged_state() {
        let computable = interval_noop_computable(0, 2);
        let tolerance_exp = XU::Finite(2);
        let result = computable.refine_to_default(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::StateUnchanged)));
    }

    #[test]
    fn refine_to_enforces_max_iterations() {
        let computable = Computable::new(
            0u32,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let tolerance_exp = XU::Finite(1);
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
        let tolerance_exp = XU::Finite(1);
        let bounds = computable
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");
        assert!(bounds.small() < bounds.large());
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
                let worse_upper = unwrap_finite(inner_state.large()).add(&bin(1, 0));
                Ok(Bounds::new(
                    inner_state.small().clone(),
                    XBinary::Finite(worse_upper),
                ))
            },
        );
        let tolerance_exp = XU::Finite(2);
        let result = computable.refine_to_default(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    // --- concurrency tests ---

    #[test]
    fn refine_shared_clone_updates_original() {
        let original = sqrt_computable(2);
        let cloned = original.clone();
        let tolerance_exp = XU::Finite(12);

        let _bounds = cloned
            .refine_to_default(tolerance_exp)
            .expect("refine_to should succeed");

        let bounds = original.bounds().expect("bounds should succeed");
        assert!(bounds_width_leq(&bounds, &tolerance_exp));
    }

    #[test]
    fn refine_to_propagates_refiner_error() {
        let computable = Computable::new(
            0u32,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |_| Err(ComputableError::DomainError),
        );

        let tolerance_exp = XU::Finite(4);
        let result = computable.refine_to::<2>(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::DomainError)));
    }

    #[test]
    fn refine_to_max_iterations_multiple_refiners() {
        let left = Computable::new(
            0u32,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let right = Computable::new(
            0u32,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        );
        let expr = left + right;
        let tolerance_exp = XU::Finite(4);
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
        let tolerance_exp = XU::Finite(4);
        let result = expr.refine_to::<3>(tolerance_exp);
        assert!(matches!(result, Err(ComputableError::BoundsWorsened)));
    }

    #[test]
    fn concurrent_bounds_reads_during_failed_refinement() {
        let computable = Arc::new(Computable::new(
            0u32,
            |_| Ok(Bounds::new(XBinary::NegInf, XBinary::PosInf)),
            |state| Ok(state + 1),
        ));
        let tolerance_exp = XU::Finite(6);
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
                0u32,
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
        let tolerance_exp = XU::Finite(6);

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
        let tolerance_exp = XU::Finite(10);
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
            assert_width_nonnegative(&bounds);
            assert!(bounds_width_leq(&bounds, &tolerance_exp));
            assert!(bounds.small() <= main_upper);
            assert!(main_bounds.small() <= bounds.large());
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
        let tolerance_exp = XU::Finite(6);
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
        let tolerance_exp = XU::Finite(8);
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
        let tolerance_exp = XU::Finite(1); // target width ≤ 0.5

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
        let tolerance_exp = XU::Finite(0); // target width ≤ 1

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
