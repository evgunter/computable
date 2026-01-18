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
use crate::ordered_pair::Bounds;

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
