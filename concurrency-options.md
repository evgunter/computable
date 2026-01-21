# Concurrency Model Options for Pub/Sub Refinement

This document describes three implementation options for switching from lock-step refinement to the event-driven pub/sub model described in the README.

All options assume we keep **scoped threads** for execution.

---

## Option 1: Crossbeam `select!` with Event-Driven Coordinator

### Summary
Minimal change from current implementation. Keep the central coordinator but switch from lock-step (wait for all refiners) to event-driven (react to each update as it arrives).

### How it works
- Refiners push updates to a shared channel (current approach)
- Coordinator uses `select!` to receive from any refiner *without waiting for all*
- Propagates each update immediately through the graph
- Checks precision after each propagation; stops early when met

### Key changes to `refinement.rs`

Current lock-step approach:
```rust
// Wait for ALL refiners each iteration
for refiner in &refiners {
    refiner.sender.send(RefineCommand::Step)?;
}
for _ in 0..refiners.len() {
    let update = update_rx.recv()?;  // blocks until all arrive
    apply_update(update)?;
}
```

New event-driven approach:
```rust
// React to each update as it arrives
loop {
    crossbeam_channel::select! {
        recv(update_rx) -> update => {
            apply_update(update?)?;
            if precision_met() {
                stop_flag.stop();
                return Ok(root_bounds);
            }
        }
    }
}
```

### Refiners still need modification
Refiners currently wait for `RefineCommand::Step`. They would need to either:
- Run continuously (see Option 2), or
- Be sent steps more aggressively by the coordinator

### Pros
- Minimal change from current implementation
- Already using crossbeam
- Achieves early stopping

### Cons
- Centralized coordinator (not true decentralized pub/sub)
- Refiners still need to be told when to refine (or run continuously)
- Doesn't fully match README's "subscribe to updates" language

---

## Option 2: Continuous Refiners + Central Coordinator

### Summary
Refiners run in a continuous loop, refining and pushing updates until stopped. Coordinator receives updates as they arrive, propagates immediately, and stops when precision is met.

### How it works
- Refiners run in a loop, refining continuously until `StopFlag` is set
- Each refinement pushes an update to the coordinator channel
- Coordinator receives updates as they arrive, propagates, checks precision
- When precision met, sets `StopFlag` and all refiners exit their loops

### Refiner thread implementation
```rust
fn refiner_loop(
    node: Arc<Node>,
    stop: Arc<StopFlag>,
    updates: Sender<Result<NodeUpdate, ComputableError>>,
) -> Result<(), ComputableError> {
    while !stop.is_stopped() {
        match node.refine_step() {
            Ok(changed) => {
                let bounds = if changed {
                    let b = node.compute_bounds()?;
                    node.set_bounds(b.clone());
                    b
                } else {
                    // No change; may want to add backoff or skip sending
                    node.cached_bounds().ok_or(ComputableError::RefinementChannelClosed)?
                };
                
                if updates.send(Ok(NodeUpdate { node_id: node.id, bounds })).is_err() {
                    break;
                }
            }
            Err(e) => {
                let _ = updates.send(Err(e));
                break;
            }
        }
    }
    Ok(())
}
```

### Coordinator implementation
```rust
fn coordinate_refinement(
    graph: &RefinementGraph,
    epsilon: &UBinary,
    update_rx: Receiver<Result<NodeUpdate, ComputableError>>,
    stop_flag: Arc<StopFlag>,
) -> Result<Bounds, ComputableError> {
    loop {
        // Check if we're already done
        let root_bounds = graph.root.get_bounds()?;
        if bounds_width_leq(&root_bounds, epsilon) {
            stop_flag.stop();
            return Ok(root_bounds);
        }
        
        // Wait for next update from any refiner
        match update_rx.recv() {
            Ok(Ok(update)) => {
                graph.apply_update(update)?;
            }
            Ok(Err(e)) => {
                stop_flag.stop();
                return Err(e);
            }
            Err(_) => {
                // All refiners have stopped
                stop_flag.stop();
                return graph.root.get_bounds();
            }
        }
    }
}
```

### Remove `RefineCommand` enum
The `RefineCommand::Step` / `RefineCommand::Stop` mechanism is no longer needed. Refiners check `StopFlag` directly instead of waiting for commands.

### Pros
- True continuous refinement as README describes
- Simple, easy to reason about
- Natural early stopping
- Removes command channel complexity

### Cons
- Refiners may do unnecessary work after precision is met (small window before they check stop flag)
- Still centralized (coordinator is single point)
- May want to add backoff when refiner has converged

### Optional enhancements
- Add exponential backoff when `refine_step` returns `Ok(false)` (no change)
- Track per-refiner convergence; stop individual refiners that can't improve further
- Use `recv_timeout` in coordinator to periodically check for external cancellation

---

## Option 3: Decentralized Pub/Sub with Subscriber Channels

### Summary
True pub/sub model matching the README. Each node maintains a list of subscribers (its parents). When a node's bounds update, it notifies all subscribers. Parent nodes have their own threads that listen for child updates, recompute, and propagate upward.

### How it works
- Each node maintains a list of subscriber channels (its parents)
- When a node's bounds update, it sends to all subscribers
- Parent nodes have their own thread that listens for child updates
- Parents recompute and propagate to *their* subscribers
- Root node notifies the main coordinator when it updates

### Node structure
```rust
use crossbeam_channel::{Sender, Receiver, unbounded};
use std::sync::RwLock;

struct SubscribableNode {
    id: usize,
    bounds: RwLock<Bounds>,
    subscribers: RwLock<Vec<Sender<BoundsUpdate>>>,
}

#[derive(Clone)]
struct BoundsUpdate {
    node_id: usize,
    bounds: Bounds,
}

impl SubscribableNode {
    /// Register as a subscriber to this node's updates.
    /// Returns a receiver that will get all future bounds updates.
    fn subscribe(&self) -> Receiver<BoundsUpdate> {
        let (tx, rx) = unbounded();
        self.subscribers.write().unwrap().push(tx);
        rx
    }
    
    /// Update bounds and notify all subscribers.
    fn publish(&self, bounds: Bounds) {
        *self.bounds.write().unwrap() = bounds.clone();
        let update = BoundsUpdate { node_id: self.id, bounds };
        
        // Send to all subscribers; ignore errors (subscriber may have dropped)
        let subscribers = self.subscribers.read().unwrap();
        for sub in subscribers.iter() {
            let _ = sub.send(update.clone());
        }
    }
    
    fn current_bounds(&self) -> Bounds {
        self.bounds.read().unwrap().clone()
    }
}
```

### Combinator node thread
Each combinator (Add, Mul, etc.) runs its own thread that listens for child updates:

```rust
fn combinator_thread(
    node: Arc<SubscribableNode>,
    left_child: Arc<SubscribableNode>,
    right_child: Arc<SubscribableNode>,
    compute_bounds: impl Fn(&Bounds, &Bounds) -> Result<Bounds, ComputableError>,
    stop: Arc<StopFlag>,
) {
    let left_rx = left_child.subscribe();
    let right_rx = right_child.subscribe();
    
    while !stop.is_stopped() {
        crossbeam_channel::select! {
            recv(left_rx) -> _ => {
                // Left child updated; recompute our bounds
                let left_bounds = left_child.current_bounds();
                let right_bounds = right_child.current_bounds();
                if let Ok(new_bounds) = compute_bounds(&left_bounds, &right_bounds) {
                    node.publish(new_bounds);
                }
            }
            recv(right_rx) -> _ => {
                // Right child updated; recompute our bounds
                let left_bounds = left_child.current_bounds();
                let right_bounds = right_child.current_bounds();
                if let Ok(new_bounds) = compute_bounds(&left_bounds, &right_bounds) {
                    node.publish(new_bounds);
                }
            }
        }
    }
}
```

### Refiner thread
```rust
fn refiner_thread(
    node: Arc<SubscribableNode>,
    refine_fn: impl Fn() -> Result<Bounds, ComputableError>,
    stop: Arc<StopFlag>,
) {
    while !stop.is_stopped() {
        match refine_fn() {
            Ok(new_bounds) => {
                node.publish(new_bounds);
            }
            Err(_) => break,
        }
    }
}
```

### Coordinator (minimal)
```rust
fn coordinate(
    root: Arc<SubscribableNode>,
    epsilon: &UBinary,
    stop: Arc<StopFlag>,
) -> Result<Bounds, ComputableError> {
    let root_rx = root.subscribe();
    
    // Check initial bounds
    let bounds = root.current_bounds();
    if bounds_width_leq(&bounds, epsilon) {
        stop.stop();
        return Ok(bounds);
    }
    
    // Wait for root to publish updates
    while let Ok(update) = root_rx.recv() {
        if bounds_width_leq(&update.bounds, epsilon) {
            stop.stop();
            return Ok(update.bounds);
        }
    }
    
    // Channel closed; return current bounds
    Ok(root.current_bounds())
}
```

### Thread spawning
Need to spawn threads for every node in the graph, not just refiners:

```rust
thread::scope(|scope| {
    let stop = Arc::new(StopFlag::new());
    
    // Spawn refiner threads for leaf nodes
    for refiner in &graph.refiners {
        let node = Arc::clone(refiner);
        let stop = Arc::clone(&stop);
        scope.spawn(move || refiner_thread(node, stop));
    }
    
    // Spawn combinator threads for internal nodes
    for combinator in &graph.combinators {
        let node = Arc::clone(&combinator.node);
        let left = Arc::clone(&combinator.left);
        let right = Arc::clone(&combinator.right);
        let stop = Arc::clone(&stop);
        scope.spawn(move || combinator_thread(node, left, right, combinator.compute, stop));
    }
    
    // Run coordinator on main thread
    coordinate(Arc::clone(&graph.root), epsilon, stop)
})
```

### Pros
- Matches README model closely (true pub/sub)
- Decentralized; updates flow naturally through graph
- Each node is independent; no central bottleneck
- Natural for DAG structure where nodes have multiple parents

### Cons
- More complex implementation
- Thread count scales with total node count, not just refiner count
- More synchronization overhead (each node has its own RwLock)
- Harder to coordinate shutdown cleanly
- May have thundering herd issues if many nodes subscribe to same child

### When to prefer this option
- Very deep graphs where centralized coordinator becomes a bottleneck
- Graphs with significant fan-out (nodes with many parents)
- When you want to add node-local optimizations (e.g., node stops subscribing when it can't improve)

---

## Comparison Summary

| Aspect | Option 1 | Option 2 | Option 3 |
|--------|----------|----------|----------|
| Complexity | Low | Low-Medium | High |
| Change from current | Minimal | Moderate | Significant |
| Matches README | Partial | Mostly | Fully |
| Thread count | Refiners only | Refiners only | All nodes |
| Coordinator | Central, event-driven | Central, event-driven | Minimal (root only) |
| Early stopping | Yes | Yes | Yes |
| Bottleneck risk | Coordinator | Coordinator | None |

## Recommendation

Start with **Option 2**. It provides the key benefits (continuous refinement, event-driven updates, early stopping) with moderate complexity. Option 3's full decentralization adds significant complexity that's likely not needed unless the coordinator proves to be a bottleneck.

Option 1 is a stepping stone—if you implement it, you'll likely want to move to Option 2 anyway to eliminate the command channel.
