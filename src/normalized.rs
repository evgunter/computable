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
//! - `BoundsBlackhole`: Unified blackhole for both initial computation AND refinement
//!
//! ## Blackholing (Unified)
//!
//! Inspired by GHC's runtime system for Haskell, this module implements a unified
//! blackholing mechanism that handles both:
//!
//! 1. **Initial computation**: When bounds are first requested, one thread computes
//!    while others wait
//! 2. **Refinement**: When higher precision is requested, one thread refines while
//!    others wait for the result
//!
//! This replaces the previous dual-blackhole design (separate `Blackhole` and
//! `RefinementBlackhole`) with a single `BoundsBlackhole` that tracks both the
//! cached bounds AND the precision level.

use std::sync::atomic::{AtomicU64, Ordering};

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use parking_lot::{Condvar, Mutex};

use crate::binary::{Bounds, UBinary, UXBinary, XBinary};
use crate::error::ComputableError;

// ============================================================================
// PrecisionLevel
// ============================================================================

/// Precision level representing how many "bits" of information have been computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrecisionLevel {
    /// No precision yet - initial state or infinite bounds.
    Zero,
    /// Finite precision measured in bits of accuracy.
    Bits(BigInt),
    /// Exact value - infinite precision (zero-width bounds).
    Exact,
}

impl PrecisionLevel {
    pub fn zero() -> Self { PrecisionLevel::Zero }
    pub fn infinite() -> Self { PrecisionLevel::Zero }
    pub fn exact() -> Self { PrecisionLevel::Exact }

    pub fn from_bits(bits: BigInt) -> Self {
        if bits <= BigInt::zero() { PrecisionLevel::Zero }
        else { PrecisionLevel::Bits(bits) }
    }

    pub fn from_width(width: &UXBinary) -> Self {
        match width {
            UXBinary::Inf => PrecisionLevel::Zero,
            UXBinary::Finite(w) if w.mantissa().is_zero() => PrecisionLevel::Exact,
            UXBinary::Finite(w) => {
                let mantissa_bits = w.mantissa().bits() as i64;
                let exponent_val = w.exponent().to_i64().unwrap_or(0);
                let precision = -exponent_val - mantissa_bits + 1;
                if precision <= 0 { PrecisionLevel::Zero }
                else { PrecisionLevel::Bits(BigInt::from(precision)) }
            }
        }
    }

    pub fn is_exact(&self) -> bool { matches!(self, PrecisionLevel::Exact) }
    pub fn is_zero(&self) -> bool { matches!(self, PrecisionLevel::Zero) }
    pub fn is_infinite(&self) -> bool { self.is_zero() }

    pub fn bits(&self) -> Option<&BigInt> {
        match self { PrecisionLevel::Bits(b) => Some(b), _ => None }
    }

    pub fn meets(&self, target: &PrecisionLevel) -> bool {
        match (self, target) {
            (PrecisionLevel::Exact, _) => true,
            (_, PrecisionLevel::Exact) => false,
            (PrecisionLevel::Zero, PrecisionLevel::Zero) => true,
            (PrecisionLevel::Zero, _) => false,
            (PrecisionLevel::Bits(_), PrecisionLevel::Zero) => true,
            (PrecisionLevel::Bits(s), PrecisionLevel::Bits(t)) => s >= t,
        }
    }

    pub fn at_least(&self, target: &PrecisionLevel) -> bool { self.meets(target) }

    pub fn increment(&self) -> Self {
        match self {
            PrecisionLevel::Zero => PrecisionLevel::Bits(BigInt::one()),
            PrecisionLevel::Bits(p) => PrecisionLevel::Bits(p + BigInt::one()),
            PrecisionLevel::Exact => PrecisionLevel::Exact,
        }
    }

    pub fn max_width(&self) -> Option<UBinary> {
        use num_bigint::BigUint;
        match self {
            PrecisionLevel::Zero => None,
            PrecisionLevel::Bits(p) => Some(UBinary::new(BigUint::one(), -p.clone())),
            PrecisionLevel::Exact => Some(UBinary::new(BigUint::zero(), BigInt::zero())),
        }
    }
}

impl Default for PrecisionLevel {
    fn default() -> Self { PrecisionLevel::zero() }
}

// ============================================================================
// BoundsBlackhole - Unified blackhole for computation AND refinement
// ============================================================================

/// State of a bounds computation/refinement, implementing the unified blackhole pattern.
///
/// This replaces the previous dual-blackhole design with a single state machine:
/// - `NotEvaluated`: Bounds have never been computed
/// - `BeingEvaluated`: A thread is computing/refining (the "blackhole")
/// - `Evaluated`: Bounds available at some precision level
/// - `Failed`: Computation failed with an error
#[derive(Clone)]
pub enum BoundsBlackholeState {
    /// Bounds have never been computed.
    NotEvaluated,
    /// A thread is currently computing or refining bounds.
    BeingEvaluated,
    /// Bounds have been computed/refined to some precision.
    Evaluated { bounds: Bounds, precision: PrecisionLevel },
    /// Computation failed with an error.
    Failed(ComputableError),
}

/// Result of attempting to claim a blackhole.
#[derive(Clone, Debug)]
pub enum ClaimResult {
    /// Target precision already met - return cached bounds.
    AlreadyMeets(Bounds),
    /// Claimed for computation/refinement - caller should compute.
    Claimed { current_bounds: Option<Bounds>, current_precision: PrecisionLevel },
}

/// Unified blackhole for both initial bounds computation AND refinement.
///
/// This combines the previous `Blackhole` (for initial computation) and
/// `RefinementBlackhole` (for precision tracking) into a single type.
///
/// ## Usage
///
/// ```ignore
/// // Initial computation
/// match blackhole.try_claim(&PrecisionLevel::zero())? {
///     ClaimResult::AlreadyMeets(bounds) => return Ok(bounds),
///     ClaimResult::Claimed { .. } => {
///         let bounds = compute_bounds()?;
///         blackhole.complete(bounds.clone());
///         return Ok(bounds);
///     }
/// }
///
/// // Refinement to higher precision
/// match blackhole.try_claim(&target_precision)? {
///     ClaimResult::AlreadyMeets(bounds) => return Ok(bounds),
///     ClaimResult::Claimed { current_bounds, .. } => {
///         let refined = refine(current_bounds)?;
///         blackhole.complete(refined.clone());
///         return Ok(refined);
///     }
/// }
/// ```
pub struct BoundsBlackhole {
    state: Mutex<BoundsBlackholeState>,
    condvar: Condvar,
    /// Epoch counter for detecting changes without locking.
    epoch: AtomicU64,
}

impl BoundsBlackhole {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(BoundsBlackholeState::NotEvaluated),
            condvar: Condvar::new(),
            epoch: AtomicU64::new(0),
        }
    }

    /// Creates a blackhole with pre-computed bounds (for constants).
    #[cfg(test)]
    pub fn with_value(bounds: Bounds) -> Self {
        let precision = PrecisionLevel::from_width(bounds.width());
        Self {
            state: Mutex::new(BoundsBlackholeState::Evaluated { bounds, precision }),
            condvar: Condvar::new(),
            epoch: AtomicU64::new(1),
        }
    }

    /// Attempts to claim the blackhole for computation or refinement.
    ///
    /// - If `NotEvaluated`: Claims for initial computation
    /// - If `BeingEvaluated`: Waits for the computing thread
    /// - If `Evaluated` with precision >= target: Returns `AlreadyMeets`
    /// - If `Evaluated` with precision < target: Claims for refinement
    /// - If `Failed`: Returns the cached error
    pub fn try_claim(&self, target: &PrecisionLevel) -> Result<ClaimResult, ComputableError> {
        let mut state = self.state.lock();
        loop {
            match &*state {
                BoundsBlackholeState::NotEvaluated => {
                    *state = BoundsBlackholeState::BeingEvaluated;
                    return Ok(ClaimResult::Claimed {
                        current_bounds: None,
                        current_precision: PrecisionLevel::zero(),
                    });
                }
                BoundsBlackholeState::BeingEvaluated => {
                    self.condvar.wait(&mut state);
                    // Loop to re-check state after waking
                }
                BoundsBlackholeState::Evaluated { bounds, precision } => {
                    if precision.meets(target) {
                        return Ok(ClaimResult::AlreadyMeets(bounds.clone()));
                    }
                    // Need higher precision - claim for refinement
                    let current = (bounds.clone(), precision.clone());
                    *state = BoundsBlackholeState::BeingEvaluated;
                    return Ok(ClaimResult::Claimed {
                        current_bounds: Some(current.0),
                        current_precision: current.1,
                    });
                }
                BoundsBlackholeState::Failed(err) => {
                    return Err(err.clone());
                }
            }
        }
    }

    /// Completes computation/refinement with the given bounds.
    pub fn complete(&self, bounds: Bounds) {
        let precision = PrecisionLevel::from_width(bounds.width());
        let mut state = self.state.lock();
        *state = BoundsBlackholeState::Evaluated { bounds, precision };
        self.epoch.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
    }

    /// Marks computation as failed.
    pub fn fail(&self, err: ComputableError) {
        let mut state = self.state.lock();
        *state = BoundsBlackholeState::Failed(err);
        self.condvar.notify_all();
    }

    /// Resets to NotEvaluated state (for error recovery).
    #[allow(dead_code)]
    pub fn reset(&self) {
        let mut state = self.state.lock();
        *state = BoundsBlackholeState::NotEvaluated;
        self.condvar.notify_all();
    }

    /// Returns the epoch counter for quick change detection.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Returns the current precision level, if evaluated.
    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        let state = self.state.lock();
        match &*state {
            BoundsBlackholeState::Evaluated { precision, .. } => Some(precision.clone()),
            _ => None,
        }
    }

    /// Returns the cached bounds without blocking, if available.
    pub fn peek(&self) -> Option<Bounds> {
        let state = self.state.lock();
        match &*state {
            BoundsBlackholeState::Evaluated { bounds, .. } => Some(bounds.clone()),
            _ => None,
        }
    }

    /// Updates bounds directly (for propagation during refinement).
    /// This is used when bounds are updated from outside (e.g., child refinement).
    pub fn update(&self, bounds: Bounds) {
        let precision = PrecisionLevel::from_width(bounds.width());
        let mut state = self.state.lock();
        *state = BoundsBlackholeState::Evaluated { bounds, precision };
        self.epoch.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
    }
}

impl Default for BoundsBlackhole {
    fn default() -> Self { Self::new() }
}

// Keep the old names as aliases for backward compatibility during transition
pub type RefinementBlackhole = BoundsBlackhole;
pub type RefinementBlackholeState = BoundsBlackholeState;
pub type RefinementClaimResult = ClaimResult;

// ============================================================================
// NormalizedBounds
// ============================================================================

#[derive(Clone, Debug)]
pub struct NormalizedBounds {
    bounds: Bounds,
    precision: PrecisionLevel,
}

impl NormalizedBounds {
    pub fn new(bounds: Bounds) -> Self {
        let precision = PrecisionLevel::from_width(bounds.width());
        Self { bounds, precision }
    }

    pub fn infinite() -> Self {
        Self {
            bounds: Bounds::new(XBinary::NegInf, XBinary::PosInf),
            precision: PrecisionLevel::zero(),
        }
    }

    pub fn bounds(&self) -> &Bounds { &self.bounds }
    pub fn precision(&self) -> &PrecisionLevel { &self.precision }

    /// Refines the bounds to a new, more precise value.
    ///
    /// # Invariant
    ///
    /// The new bounds should always be contained within the old bounds. This is
    /// enforced by debug_assert! because:
    /// - In correct implementations, refinement always narrows bounds
    /// - We're not 100% certain all code paths maintain this invariant yet
    /// - We want to catch violations during development without runtime cost in release
    pub fn refine(&mut self, new_bounds: Bounds) {
        let new_precision = PrecisionLevel::from_width(new_bounds.width());

        // This should never happen if the implementation is correct, but we're
        // not certain yet. Using debug_assert to catch issues during development.
        debug_assert!(
            self.contains_bounds(&new_bounds),
            "NormalizedBounds invariant violated: new bounds {:?} not contained in old {:?}",
            new_bounds, self.bounds
        );
        debug_assert!(
            new_precision.meets(&self.precision),
            "NormalizedBounds invariant violated: precision decreased from {:?} to {:?}",
            self.precision, new_precision
        );

        self.bounds = new_bounds;
        self.precision = new_precision;
    }

    fn contains_bounds(&self, other: &Bounds) -> bool {
        self.bounds.small() <= other.small() && other.large() <= self.bounds.large()
    }

    pub fn meets_precision(&self, target: &PrecisionLevel) -> bool { self.precision.meets(target) }

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
    use crate::test_utils::xbin;
    use num_bigint::BigUint;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn precision_level_from_width_zero_is_exact() {
        let zero_width = UXBinary::Finite(UBinary::new(BigUint::from(0u32), BigInt::from(0)));
        let precision = PrecisionLevel::from_width(&zero_width);
        assert!(precision.is_exact());
    }

    #[test]
    fn precision_level_from_width_infinite_is_zero() {
        let inf_width = UXBinary::Inf;
        let precision = PrecisionLevel::from_width(&inf_width);
        assert_eq!(precision, PrecisionLevel::zero());
    }

    #[test]
    fn precision_level_from_width_finite() {
        let width = UXBinary::Finite(UBinary::new(BigUint::from(1u32), BigInt::from(-8)));
        let precision = PrecisionLevel::from_width(&width);
        assert!(!precision.is_exact());
        assert_eq!(precision.bits(), Some(&BigInt::from(8)));
    }

    #[test]
    fn precision_level_meets_exact_meets_all() {
        let exact = PrecisionLevel::exact();
        let p8 = PrecisionLevel::from_bits(BigInt::from(8));
        let p16 = PrecisionLevel::from_bits(BigInt::from(16));
        assert!(exact.meets(&p8));
        assert!(exact.meets(&p16));
        assert!(exact.meets(&PrecisionLevel::zero()));
    }

    #[test]
    fn precision_level_meets_comparison() {
        let p8 = PrecisionLevel::from_bits(BigInt::from(8));
        let p16 = PrecisionLevel::from_bits(BigInt::from(16));
        assert!(p16.meets(&p8));
        assert!(!p8.meets(&p16));
        assert!(p8.meets(&p8));
    }

    #[test]
    fn normalized_bounds_refinement() {
        let initial = Bounds::new(xbin(0, 0), xbin(4, 0));
        let mut nb = NormalizedBounds::new(initial);
        let refined = Bounds::new(xbin(1, 0), xbin(3, 0));
        nb.refine(refined);
        let more_refined = Bounds::new(xbin(3, -1), xbin(5, -1));
        nb.refine(more_refined);
        // Widening would violate the invariant - caught by debug_assert in debug builds
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "NormalizedBounds invariant violated")]
    fn normalized_bounds_rejects_widening_in_debug() {
        let initial = Bounds::new(xbin(0, 0), xbin(4, 0));
        let mut nb = NormalizedBounds::new(initial);
        let refined = Bounds::new(xbin(1, 0), xbin(3, 0));
        nb.refine(refined);
        let widened = Bounds::new(xbin(0, 0), xbin(4, 0));
        nb.refine(widened); // Should panic in debug builds
    }

    fn test_bounds() -> Bounds { Bounds::new(xbin(1, 0), xbin(2, 0)) }

    #[test]
    fn bounds_blackhole_new_is_not_evaluated() {
        let bh = BoundsBlackhole::new();
        assert!(bh.current_precision().is_none());
        assert!(bh.peek().is_none());
    }

    #[test]
    fn bounds_blackhole_try_claim_on_new_returns_claimed() {
        let bh = BoundsBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let result = bh.try_claim(&target);
        assert!(matches!(
            result,
            Ok(ClaimResult::Claimed { current_bounds: None, current_precision })
            if current_precision == PrecisionLevel::zero()
        ));
    }

    #[test]
    fn bounds_blackhole_complete_updates_state() {
        let bh = BoundsBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let _ = bh.try_claim(&target).expect("should claim");
        let bounds = test_bounds();
        bh.complete(bounds.clone());
        assert!(bh.current_precision().is_some());
        assert_eq!(bh.peek(), Some(bounds));
    }

    #[test]
    fn bounds_blackhole_already_meets_returns_bounds() {
        let bh = BoundsBlackhole::new();
        let target_low = PrecisionLevel::from_bits(BigInt::from(4));
        let _ = bh.try_claim(&target_low).expect("should claim");
        let small_bounds = Bounds::new(xbin(1, 0), xbin(1, 0));
        bh.complete(small_bounds.clone());
        let result = bh.try_claim(&target_low);
        assert!(matches!(result, Ok(ClaimResult::AlreadyMeets(b)) if b == small_bounds));
    }

    #[test]
    fn bounds_blackhole_fail_propagates_error() {
        let bh = BoundsBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let _ = bh.try_claim(&target).expect("should claim");
        bh.fail(ComputableError::DomainError);
        let result = bh.try_claim(&target);
        assert!(matches!(result, Err(ComputableError::DomainError)));
    }

    #[test]
    fn bounds_blackhole_concurrent_claim_coordinates() {
        let bh = Arc::new(BoundsBlackhole::new());
        let complete_count = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4).map(|_| {
            let bh = Arc::clone(&bh);
            let count = Arc::clone(&complete_count);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                // Target precision of 8 bits = width of 2^(-8) = 1/256
                let target = PrecisionLevel::from_bits(BigInt::from(8));
                match bh.try_claim(&target) {
                    Ok(ClaimResult::Claimed { .. }) => {
                        count.fetch_add(1, AtomicOrdering::SeqCst);
                        thread::sleep(Duration::from_millis(10));
                        // Create bounds with width = 1 * 2^(-10) to meet precision 8
                        let bounds = Bounds::new(xbin(1, -10), xbin(2, -10));
                        bh.complete(bounds);
                    }
                    Ok(ClaimResult::AlreadyMeets(_)) => {}
                    Err(_) => panic!("unexpected error"),
                }
            })
        }).collect();

        for h in handles { h.join().expect("join"); }
        // Exactly one thread should have completed the refinement
        assert_eq!(complete_count.load(AtomicOrdering::SeqCst), 1);
    }
}
