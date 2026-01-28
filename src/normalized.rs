//! Normalized bounds representation for monotonic refinement.
//!
//! This module provides types for representing bounds in a "normalized" form where
//! refinement can only *extend* precision (add more "digits") rather than *change*
//! existing bounds. This makes refinement more like Haskell's lazy evaluation where
//! a thunk reveals its value incrementally.
//!
//! ## The Key Insight
//!
//! In the original design, refinement could change bounds in arbitrary ways (widening
//! lower, tightening upper, etc.) as long as the bounds didn't worsen. The normalized
//! approach instead views bounds as a "commitment" - once we've said "the value is in
//! [a, b]", future refinements can only narrow to something within [a, b].
//!
//! This is analogous to signed-digit stream representations of real numbers, where
//! each digit commits to a certain range and subsequent digits only provide more precision.
//! See [Exact real arithmetic - HaskellWiki](https://wiki.haskell.org/Exact_real_arithmetic)
//! and [Limits of real numbers in the binary signed digit representation](https://arxiv.org/abs/2103.15702).
//!
//! ## Architecture
//!
//! - `PrecisionLevel`: Tracks how many "digits" of precision have been computed
//! - `NormalizedBounds`: Bounds with a precision level, guaranteeing monotonic extension
//! - `RefinementBlackhole`: Coordinates refinement to prevent duplicate work
//!
//! ## Blackholing for Refinement
//!
//! Inspired by GHC's runtime system for Haskell, this module extends the blackholing
//! mechanism to refinement itself:
//!
//! - When a thread starts refining a node, it marks the node as "being refined"
//! - Other threads that want to refine the same node wait rather than duplicate work
//! - This prevents redundant computation and ensures consistent precision levels
//!
//! ## Integration with RefinementSync
//!
//! The `RefinementBlackhole` complements the existing `RefinementSync` in `node.rs`.
//! While `RefinementSync` coordinates at the graph level (which refiner thread is active),
//! `RefinementBlackhole` coordinates at the precision level (what precision is being computed).
//! This provides finer-grained control over concurrent refinement.

use std::sync::atomic::{AtomicU64, Ordering};

use num_bigint::BigInt;
use num_traits::{One, Zero};
use parking_lot::{Condvar, Mutex};

use crate::binary::{Bounds, UBinary, UXBinary, XBinary};
use crate::error::ComputableError;

/// Precision level representing how many "digits" of information have been computed.
///
/// This is analogous to the number of digits in a signed-digit stream representation
/// of a real number. Higher precision means narrower (more accurate) bounds.
///
/// The precision is measured in bits of accuracy. A precision of n bits means the
/// bounds width is approximately 2^(-n).
///
/// Special cases:
/// - Zero precision: No bounds computed yet (infinite width)
/// - Exact precision: Exact value known (zero width)
///
/// This design enables monotonic refinement: precision can only increase (more bits),
/// never decrease, mirroring how Haskell thunks reveal their values incrementally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecisionLevel(pub Option<BigInt>);

impl PrecisionLevel {
    /// Creates a precision level representing infinite bounds (no precision yet).
    pub fn infinite() -> Self {
        PrecisionLevel(None)
    }

    /// Creates a precision level from a number of bits.
    ///
    /// A precision of n bits means bounds width <= 2^(-n).
    pub fn from_bits(bits: impl Into<BigInt>) -> Self {
        PrecisionLevel(Some(bits.into()))
    }

    /// Creates a precision level from a width.
    ///
    /// The precision is computed as `-log2(width)` (approximately).
    pub fn from_width(width: &UXBinary) -> Self {
        match width {
            UXBinary::Inf => PrecisionLevel(None),
            UXBinary::Finite(w) if w.mantissa().is_zero() => {
                // Zero width = exact value = infinite precision
                // Represent as a very large precision value
                PrecisionLevel(Some(BigInt::from(i64::MAX)))
            }
            UXBinary::Finite(w) => {
                // precision = -exponent (approximately, ignoring mantissa bits)
                // A width of 2^e has precision -e
                let precision = -w.exponent();
                PrecisionLevel(Some(precision))
            }
        }
    }

    /// Returns true if this precision is at least as good as the target.
    pub fn at_least(&self, target: &PrecisionLevel) -> bool {
        match (&self.0, &target.0) {
            (None, _) => false,      // Infinite bounds never meet any target
            (Some(_), None) => true, // Any finite precision beats infinite
            (Some(self_prec), Some(target_prec)) => self_prec >= target_prec,
        }
    }

    /// Returns true if this represents infinite bounds.
    pub fn is_infinite(&self) -> bool {
        self.0.is_none()
    }

    /// Gets the precision value, or None if infinite.
    pub fn value(&self) -> Option<&BigInt> {
        self.0.as_ref()
    }

    /// Creates a precision level that is one step more refined.
    ///
    /// This increments the precision by 1, representing one more "digit" of information.
    pub fn increment(&self) -> Self {
        match &self.0 {
            None => PrecisionLevel(Some(BigInt::one())),
            Some(p) => PrecisionLevel(Some(p + BigInt::one())),
        }
    }

    /// Returns the required width for this precision level.
    ///
    /// Precision n means width <= 2^(-n).
    pub fn max_width(&self) -> Option<UBinary> {
        self.0.as_ref().map(|p| {
            use num_bigint::BigUint;
            // Width = 2^(-precision) = 1 * 2^(-p)
            UBinary::new(BigUint::one(), -p.clone())
        })
    }
}

impl Default for PrecisionLevel {
    fn default() -> Self {
        PrecisionLevel::infinite()
    }
}

/// State of a refinement operation, implementing the blackhole pattern.
///
/// This extends the bounds blackholing to the refinement process itself:
/// - `NotRefining`: No refinement is in progress
/// - `Refining`: A thread is actively refining (the "blackhole")
/// - `Refined`: Refinement completed with a new precision level
/// - `Failed`: Refinement failed with an error
///
/// The key insight from GHC's blackholing is that when multiple threads
/// want to refine the same value, only one should do the work while
/// others wait for the result. This prevents redundant computation and
/// ensures monotonic precision extension.
#[derive(Clone)]
pub enum RefinementBlackholeState {
    /// No refinement is in progress. Initial state.
    NotRefining,
    /// A thread is currently refining (the "blackhole").
    /// Other threads should wait rather than duplicate work.
    Refining {
        /// The precision level being refined toward.
        target_precision: PrecisionLevel,
    },
    /// Refinement completed successfully.
    Refined {
        /// The bounds achieved.
        bounds: Bounds,
        /// The precision level achieved.
        achieved_precision: PrecisionLevel,
    },
    /// Refinement failed with an error.
    /// Unlike GHC's <<loop>> detection, we propagate actual computation errors.
    Failed(ComputableError),
}

/// Synchronization wrapper for refinement blackhole state.
///
/// This coordinates refinement across threads to prevent duplicate work.
/// When a thread starts refining, it "blackholes" the node, and other threads
/// wait for the refinement to complete.
pub struct RefinementBlackhole {
    state: Mutex<RefinementBlackholeState>,
    condvar: Condvar,
    /// Current precision level achieved (atomically updated for quick checks).
    precision_epoch: AtomicU64,
}

impl RefinementBlackhole {
    /// Creates a new refinement blackhole in the NotRefining state.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(RefinementBlackholeState::NotRefining),
            condvar: Condvar::new(),
            precision_epoch: AtomicU64::new(0),
        }
    }

    /// Attempts to claim the refinement blackhole for refining.
    ///
    /// Returns:
    /// - `Ok(true)` if this thread should perform the refinement
    /// - `Ok(false)` if another thread is refining and we waited for it
    /// - `Err` if there was an error
    ///
    /// If `Ok(false)` is returned, the caller should re-check if more refinement
    /// is needed, as the other thread may have achieved sufficient precision.
    pub fn try_claim_or_wait(&self, target: &PrecisionLevel) -> Result<bool, ComputableError> {
        let mut state = self.state.lock();
        loop {
            match &*state {
                RefinementBlackholeState::NotRefining => {
                    // We're the first thread - claim the blackhole
                    *state = RefinementBlackholeState::Refining {
                        target_precision: target.clone(),
                    };
                    return Ok(true);
                }
                RefinementBlackholeState::Refining { .. } => {
                    // Another thread is refining - wait for it
                    self.condvar.wait(&mut state);
                    // Loop to re-check state after waking
                }
                RefinementBlackholeState::Refined {
                    achieved_precision, ..
                } => {
                    // Check if the achieved precision meets our target
                    if achieved_precision.at_least(target) {
                        return Ok(false); // No need to refine more
                    }
                    // Need more precision - claim for another round
                    *state = RefinementBlackholeState::Refining {
                        target_precision: target.clone(),
                    };
                    return Ok(true);
                }
                RefinementBlackholeState::Failed(err) => {
                    // Previous refinement failed - propagate the error
                    return Err(err.clone());
                }
            }
        }
    }

    /// Marks the refinement as not in progress (e.g., after an error).
    pub fn reset(&self) {
        let mut state = self.state.lock();
        *state = RefinementBlackholeState::NotRefining;
        self.condvar.notify_all();
    }

    /// Completes the refinement with an error.
    ///
    /// Transitions from Refining to Failed and wakes all waiters.
    /// This is similar to how GHC handles exceptions during thunk evaluation.
    pub fn fail(&self, err: ComputableError) {
        let mut state = self.state.lock();
        *state = RefinementBlackholeState::Failed(err);
        self.condvar.notify_all();
    }

    /// Returns the current precision epoch for quick change detection.
    ///
    /// This is incremented each time refinement completes successfully.
    /// Can be used to detect if precision has changed without locking.
    pub fn precision_epoch(&self) -> u64 {
        self.precision_epoch.load(Ordering::Acquire)
    }

    /// Returns the current achieved precision, if available.
    ///
    /// This is a non-blocking peek at the current state.
    pub fn peek_precision(&self) -> Option<PrecisionLevel> {
        let state = self.state.lock();
        match &*state {
            RefinementBlackholeState::Refined {
                achieved_precision, ..
            } => Some(achieved_precision.clone()),
            _ => None,
        }
    }

    /// Returns the current bounds, if available.
    ///
    /// This is a non-blocking peek at the current state.
    pub fn peek_bounds(&self) -> Option<Bounds> {
        let state = self.state.lock();
        match &*state {
            RefinementBlackholeState::Refined { bounds, .. } => Some(bounds.clone()),
            _ => None,
        }
    }

    /// Checks if the refinement is currently in progress.
    pub fn is_refining(&self) -> bool {
        let state = self.state.lock();
        matches!(&*state, RefinementBlackholeState::Refining { .. })
    }

    /// Returns the current precision level, if refined.
    ///
    /// Alias for `peek_precision()` for API compatibility.
    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        self.peek_precision()
    }

    /// Attempts to claim the refinement blackhole for a refinement step.
    ///
    /// Returns:
    /// - `Ok(RefinementClaimResult::AlreadyMeets(bounds))` if current precision meets the target
    /// - `Ok(RefinementClaimResult::Claimed { previous_precision })` if this thread should refine
    /// - `Err(e)` if a previous refinement failed
    ///
    /// This is the main entry point for coordinating refinement. It:
    /// 1. Checks if the target precision is already met
    /// 2. If not, attempts to claim the blackhole for refinement
    /// 3. If another thread is refining, waits for it to complete
    pub fn try_claim(
        &self,
        target: &PrecisionLevel,
    ) -> Result<RefinementClaimResult, ComputableError> {
        let mut state = self.state.lock();
        loop {
            match &*state {
                RefinementBlackholeState::NotRefining => {
                    // First refinement - claim it
                    *state = RefinementBlackholeState::Refining {
                        target_precision: target.clone(),
                    };
                    return Ok(RefinementClaimResult::Claimed {
                        previous_precision: PrecisionLevel::infinite(),
                    });
                }
                RefinementBlackholeState::Refining { .. } => {
                    // Another thread is refining - wait for it
                    self.condvar.wait(&mut state);
                    // Loop to re-check state after waking
                }
                RefinementBlackholeState::Refined {
                    bounds,
                    achieved_precision,
                } => {
                    if achieved_precision.at_least(target) {
                        // Already meets target - return the bounds
                        return Ok(RefinementClaimResult::AlreadyMeets(bounds.clone()));
                    }
                    // Need more precision - claim for refinement
                    let previous = achieved_precision.clone();
                    *state = RefinementBlackholeState::Refining {
                        target_precision: target.clone(),
                    };
                    return Ok(RefinementClaimResult::Claimed {
                        previous_precision: previous,
                    });
                }
                RefinementBlackholeState::Failed(err) => {
                    return Err(err.clone());
                }
            }
        }
    }

    /// Completes a refinement step with new bounds.
    ///
    /// The precision is computed from the bounds width.
    pub fn complete(&self, bounds: Bounds) -> Result<(), ComputableError> {
        let new_precision = PrecisionLevel::from_width(bounds.width());
        let mut state = self.state.lock();

        *state = RefinementBlackholeState::Refined {
            bounds,
            achieved_precision: new_precision,
        };
        self.precision_epoch.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
        Ok(())
    }

    /// Updates bounds without going through the full claim/complete cycle.
    ///
    /// Used for direct updates during refinement propagation.
    /// NOTE: This does NOT enforce monotonicity because derived nodes in the
    /// computation graph can have non-monotonic bounds changes during propagation
    /// (interval arithmetic can temporarily widen bounds before they narrow).
    /// The key correctness property is that the TRUE value is always contained
    /// in the bounds, which interval arithmetic guarantees.
    pub fn update(&self, bounds: Bounds) -> Result<(), ComputableError> {
        let new_precision = PrecisionLevel::from_width(bounds.width());
        let mut state = self.state.lock();

        *state = RefinementBlackholeState::Refined {
            bounds,
            achieved_precision: new_precision,
        };
        self.precision_epoch.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
        Ok(())
    }
}

impl Default for RefinementBlackhole {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of attempting to claim the refinement blackhole.
///
/// This enum represents the possible outcomes when a thread tries to
/// claim a refinement blackhole for refinement.
#[derive(Clone, Debug)]
pub enum RefinementClaimResult {
    /// The current precision already meets the target; includes the current bounds.
    AlreadyMeets(Bounds),
    /// This thread has claimed the blackhole and should perform refinement.
    Claimed { previous_precision: PrecisionLevel },
}

/// Normalized bounds with monotonic refinement guarantees.
///
/// This type wraps `Bounds` with additional tracking to ensure that refinement
/// only extends precision rather than changing existing committed values.
///
/// ## Invariants
///
/// 1. **Monotonic Narrowing**: Each refinement must produce bounds that are
///    contained within the previous bounds.
/// 2. **Precision Extension**: The precision level can only increase (or stay same).
/// 3. **Commitment**: Once a bound is established at a precision level, subsequent
///    refinements at higher precision must be consistent with it.
#[derive(Clone, Debug)]
pub struct NormalizedBounds {
    /// The current bounds.
    bounds: Bounds,
    /// The precision level of these bounds.
    precision: PrecisionLevel,
}

impl NormalizedBounds {
    /// Creates new normalized bounds from raw bounds.
    pub fn new(bounds: Bounds) -> Self {
        let precision = PrecisionLevel::from_width(bounds.width());
        Self { bounds, precision }
    }

    /// Creates normalized bounds representing infinite bounds (no information yet).
    pub fn infinite() -> Self {
        Self {
            bounds: Bounds::new(XBinary::NegInf, XBinary::PosInf),
            precision: PrecisionLevel::infinite(),
        }
    }

    /// Returns a reference to the underlying bounds.
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// Returns the current precision level.
    pub fn precision(&self) -> &PrecisionLevel {
        &self.precision
    }

    /// Attempts to refine these bounds with new, more precise bounds.
    ///
    /// Returns `Ok(())` if the refinement is valid (new bounds are contained
    /// within old bounds and have equal or better precision).
    ///
    /// Returns `Err` if:
    /// - The new bounds are not contained within the old bounds
    /// - The new precision is worse than the old precision
    pub fn refine(&mut self, new_bounds: Bounds) -> Result<(), ComputableError> {
        let new_precision = PrecisionLevel::from_width(new_bounds.width());

        // Check monotonicity: new bounds must be contained in old bounds
        if !self.contains_bounds(&new_bounds) {
            return Err(ComputableError::BoundsWorsened);
        }

        // Check precision: new precision must be at least as good
        if !new_precision.at_least(&self.precision) {
            // This shouldn't happen if containment holds, but check anyway
            return Err(ComputableError::BoundsWorsened);
        }

        self.bounds = new_bounds;
        self.precision = new_precision;
        Ok(())
    }

    /// Checks if this bounds contains another bounds.
    fn contains_bounds(&self, other: &Bounds) -> bool {
        // For containment: our lower <= their lower AND their upper <= our upper
        self.bounds.small() <= other.small() && other.large() <= self.bounds.large()
    }

    /// Returns true if these bounds meet the target precision.
    pub fn meets_precision(&self, target: &PrecisionLevel) -> bool {
        self.precision.at_least(target)
    }

    /// Returns true if these bounds have finite width <= epsilon.
    pub fn meets_epsilon(&self, epsilon: &UBinary) -> bool {
        match self.bounds.width() {
            UXBinary::Inf => false,
            UXBinary::Finite(w) => *w <= *epsilon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{ubin, xbin};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    // =========================================================================
    // PrecisionLevel Tests
    // =========================================================================

    #[test]
    fn precision_level_from_width() {
        // Width of 1 (2^0) -> precision 0
        let width = UXBinary::Finite(ubin(1, 0));
        let prec = PrecisionLevel::from_width(&width);
        assert_eq!(prec.0, Some(BigInt::from(0)));

        // Width of 0.5 (2^-1) -> precision 1
        let width = UXBinary::Finite(ubin(1, -1));
        let prec = PrecisionLevel::from_width(&width);
        assert_eq!(prec.0, Some(BigInt::from(1)));

        // Width of 0.125 (2^-3) -> precision 3
        let width = UXBinary::Finite(ubin(1, -3));
        let prec = PrecisionLevel::from_width(&width);
        assert_eq!(prec.0, Some(BigInt::from(3)));

        // Infinite width -> no precision
        let prec = PrecisionLevel::from_width(&UXBinary::Inf);
        assert!(prec.is_infinite());
    }

    #[test]
    fn precision_level_comparison() {
        let p0 = PrecisionLevel(Some(BigInt::from(0)));
        let p1 = PrecisionLevel(Some(BigInt::from(1)));
        let p_inf = PrecisionLevel::infinite();

        assert!(p1.at_least(&p0));
        assert!(p1.at_least(&p1));
        assert!(!p0.at_least(&p1));
        assert!(!p_inf.at_least(&p0));
        assert!(p0.at_least(&p_inf));
    }

    #[test]
    fn precision_level_increment() {
        let inf = PrecisionLevel::infinite();
        let p1 = inf.increment();
        assert_eq!(p1.0, Some(BigInt::from(1)));

        let p2 = p1.increment();
        assert_eq!(p2.0, Some(BigInt::from(2)));
    }

    #[test]
    fn precision_level_max_width() {
        let p0 = PrecisionLevel(Some(BigInt::from(0)));
        let p3 = PrecisionLevel(Some(BigInt::from(3)));
        let p_inf = PrecisionLevel::infinite();

        // Precision 0 -> width 2^0 = 1
        let w0 = p0.max_width().expect("should have width");
        assert_eq!(*w0.exponent(), BigInt::from(0));

        // Precision 3 -> width 2^-3 = 0.125
        let w3 = p3.max_width().expect("should have width");
        assert_eq!(*w3.exponent(), BigInt::from(-3));

        // Infinite precision has no max width
        assert!(p_inf.max_width().is_none());
    }

    // =========================================================================
    // NormalizedBounds Tests
    // =========================================================================

    #[test]
    fn normalized_bounds_refinement() {
        // Start with [0, 4]
        let initial = Bounds::new(xbin(0, 0), xbin(4, 0));
        let mut nb = NormalizedBounds::new(initial);

        // Refine to [1, 3] - should succeed (contained in [0, 4])
        let refined = Bounds::new(xbin(1, 0), xbin(3, 0));
        assert!(nb.refine(refined).is_ok());

        // Refine to [1.5, 2.5] - should succeed
        let more_refined = Bounds::new(xbin(3, -1), xbin(5, -1));
        assert!(nb.refine(more_refined).is_ok());

        // Try to widen to [0, 4] - should fail
        let widened = Bounds::new(xbin(0, 0), xbin(4, 0));
        assert!(nb.refine(widened).is_err());
    }

    #[test]
    fn normalized_bounds_shift_lower_fails() {
        // Start with [1, 3]
        let initial = Bounds::new(xbin(1, 0), xbin(3, 0));
        let mut nb = NormalizedBounds::new(initial);

        // Try to shift lower bound down to [0, 2] - should fail (0 < 1)
        let shifted = Bounds::new(xbin(0, 0), xbin(2, 0));
        assert!(nb.refine(shifted).is_err());
    }

    #[test]
    fn normalized_bounds_meets_epsilon() {
        let bounds = Bounds::new(xbin(1, 0), xbin(3, 0)); // width = 2
        let nb = NormalizedBounds::new(bounds);

        let epsilon_large = ubin(3, 0); // epsilon = 3
        let epsilon_exact = ubin(2, 0); // epsilon = 2
        let epsilon_small = ubin(1, 0); // epsilon = 1

        assert!(nb.meets_epsilon(&epsilon_large));
        assert!(nb.meets_epsilon(&epsilon_exact));
        assert!(!nb.meets_epsilon(&epsilon_small));
    }

    #[test]
    fn normalized_bounds_infinite_to_finite() {
        let mut nb = NormalizedBounds::infinite();
        assert!(nb.precision().is_infinite());

        // Refine to finite bounds
        let finite = Bounds::new(xbin(0, 0), xbin(10, 0));
        assert!(nb.refine(finite).is_ok());
        assert!(!nb.precision().is_infinite());
    }

    // =========================================================================
    // RefinementBlackhole Tests
    // =========================================================================

    /// Creates bounds with a specific precision level for testing.
    /// Precision p means width ~ 2^(-p), so we create bounds with width = 2^(-p).
    fn bounds_with_precision(precision: i64) -> Bounds {
        // Lower = 0, width = 2^(-precision)
        // Upper = 2^(-precision)
        Bounds::new(xbin(0, 0), xbin(1, -precision))
    }

    #[test]
    fn refinement_blackhole_single_thread() {
        let bh = RefinementBlackhole::new();
        let target = PrecisionLevel(Some(BigInt::from(10)));

        // First claim should succeed
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));

        // Complete the refinement with bounds that have precision 10
        let bounds = bounds_with_precision(10);
        bh.complete(bounds).expect("complete should succeed");

        // Next claim for same precision should return false (already achieved)
        assert!(!bh.try_claim_or_wait(&target).expect("should succeed"));

        // Next claim for higher precision should succeed
        let higher = PrecisionLevel(Some(BigInt::from(20)));
        assert!(bh.try_claim_or_wait(&higher).expect("should succeed"));
    }

    #[test]
    fn refinement_blackhole_fail_propagates_error() {
        let bh = RefinementBlackhole::new();
        let target = PrecisionLevel(Some(BigInt::from(10)));

        // Claim the blackhole
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));

        // Fail the refinement
        bh.fail(ComputableError::DomainError);

        // Next claim should get the error
        let result = bh.try_claim_or_wait(&target);
        assert!(matches!(result, Err(ComputableError::DomainError)));
    }

    #[test]
    fn refinement_blackhole_reset_clears_error() {
        let bh = RefinementBlackhole::new();
        let target = PrecisionLevel(Some(BigInt::from(10)));

        // Claim and fail
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));
        bh.fail(ComputableError::DomainError);

        // Reset
        bh.reset();

        // Now can claim again
        assert!(
            bh.try_claim_or_wait(&target)
                .expect("should succeed after reset")
        );
    }

    #[test]
    fn refinement_blackhole_peek_precision() {
        let bh = RefinementBlackhole::new();
        let target = PrecisionLevel(Some(BigInt::from(10)));

        // Before refinement
        assert!(bh.peek_precision().is_none());

        // Claim and complete
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));
        let bounds = bounds_with_precision(10);
        bh.complete(bounds).expect("complete should succeed");

        // After refinement, precision should be approximately 10
        let achieved = bh.peek_precision().expect("should have precision");
        assert!(achieved.at_least(&target));
    }

    #[test]
    fn refinement_blackhole_is_refining() {
        let bh = RefinementBlackhole::new();
        let target = PrecisionLevel(Some(BigInt::from(10)));

        assert!(!bh.is_refining());

        // Claim
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));
        assert!(bh.is_refining());

        // Complete
        let bounds = bounds_with_precision(10);
        bh.complete(bounds).expect("complete should succeed");
        assert!(!bh.is_refining());
    }

    // =========================================================================
    // Concurrent RefinementBlackhole Tests
    // =========================================================================

    #[test]
    fn refinement_blackhole_concurrent_claim_refines_once() {
        let bh = Arc::new(RefinementBlackhole::new());
        let refine_count = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let target = PrecisionLevel(Some(BigInt::from(10)));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let bh = Arc::clone(&bh);
                let count = Arc::clone(&refine_count);
                let bar = Arc::clone(&barrier);
                let target = target.clone();

                thread::spawn(move || {
                    bar.wait();

                    match bh.try_claim_or_wait(&target) {
                        Ok(true) => {
                            // We claimed it - "refine"
                            count.fetch_add(1, AtomicOrdering::SeqCst);
                            thread::sleep(Duration::from_millis(10));
                            let bounds = Bounds::new(xbin(0, 0), xbin(1, -10));
                            bh.complete(bounds).expect("complete should succeed");
                        }
                        Ok(false) => {
                            // Another thread did it
                        }
                        Err(_) => panic!("unexpected error"),
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("join");
        }

        // Refinement should happen exactly once
        assert_eq!(refine_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn refinement_blackhole_waiters_get_error_on_fail() {
        let bh = Arc::new(RefinementBlackhole::new());
        let barrier = Arc::new(Barrier::new(3));
        let target = PrecisionLevel(Some(BigInt::from(10)));

        // Claim the blackhole first
        assert!(bh.try_claim_or_wait(&target).expect("should succeed"));

        // Spawn waiters
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let bh = Arc::clone(&bh);
                let bar = Arc::clone(&barrier);
                let target = target.clone();

                thread::spawn(move || {
                    bar.wait();
                    bh.try_claim_or_wait(&target)
                })
            })
            .collect();

        // Give waiters time to start waiting
        barrier.wait();
        thread::sleep(Duration::from_millis(10));

        // Fail the refinement
        bh.fail(ComputableError::DomainError);

        // All waiters should get the error
        for handle in handles {
            let result = handle.join().expect("join");
            assert!(matches!(result, Err(ComputableError::DomainError)));
        }
    }

    #[test]
    fn refinement_blackhole_increasing_precision_serializes() {
        let bh = Arc::new(RefinementBlackhole::new());
        let precision_history = Arc::new(parking_lot::Mutex::new(Vec::new()));

        // Run multiple precision levels sequentially
        for p in [5i64, 10, 15, 20].iter() {
            let target = PrecisionLevel(Some(BigInt::from(*p)));
            if bh.try_claim_or_wait(&target).expect("should succeed") {
                precision_history.lock().push(*p);
                let bounds = Bounds::new(xbin(0, 0), xbin(1, -*p));
                bh.complete(bounds).expect("complete should succeed");
            }
        }

        let history = precision_history.lock();
        // Each increasing precision should have triggered a refinement
        assert_eq!(*history, vec![5, 10, 15, 20]);
    }
}
