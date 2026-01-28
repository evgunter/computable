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
// RefinementBlackholeState
// ============================================================================

#[derive(Clone)]
pub enum RefinementBlackholeState {
    NotRefining,
    Refining { target_precision: PrecisionLevel },
    Refined { achieved_precision: PrecisionLevel, bounds: Bounds },
    Failed(ComputableError),
}

// ============================================================================
// RefinementClaimResult
// ============================================================================

#[derive(Clone, Debug)]
pub enum RefinementClaimResult {
    AlreadyMeets(Bounds),
    Claimed { previous_precision: PrecisionLevel },
}

// ============================================================================
// RefinementBlackhole
// ============================================================================

pub struct RefinementBlackhole {
    state: Mutex<RefinementBlackholeState>,
    condvar: Condvar,
    precision_epoch: AtomicU64,
}

impl RefinementBlackhole {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(RefinementBlackholeState::NotRefining),
            condvar: Condvar::new(),
            precision_epoch: AtomicU64::new(0),
        }
    }

    pub fn try_claim(&self, target: &PrecisionLevel) -> Result<RefinementClaimResult, ComputableError> {
        let mut state = self.state.lock();
        loop {
            match &*state {
                RefinementBlackholeState::NotRefining => {
                    *state = RefinementBlackholeState::Refining { target_precision: target.clone() };
                    return Ok(RefinementClaimResult::Claimed { previous_precision: PrecisionLevel::zero() });
                }
                RefinementBlackholeState::Refining { .. } => {
                    self.condvar.wait(&mut state);
                }
                RefinementBlackholeState::Refined { achieved_precision, bounds } => {
                    if achieved_precision.meets(target) {
                        return Ok(RefinementClaimResult::AlreadyMeets(bounds.clone()));
                    }
                    let previous = achieved_precision.clone();
                    *state = RefinementBlackholeState::Refining { target_precision: target.clone() };
                    return Ok(RefinementClaimResult::Claimed { previous_precision: previous });
                }
                RefinementBlackholeState::Failed(err) => {
                    return Err(err.clone());
                }
            }
        }
    }

    pub fn complete(&self, bounds: Bounds) {
        let new_precision = PrecisionLevel::from_width(bounds.width());
        let mut state = self.state.lock();
        *state = RefinementBlackholeState::Refined { achieved_precision: new_precision, bounds };
        self.precision_epoch.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
    }

    pub fn fail(&self, err: ComputableError) {
        let mut state = self.state.lock();
        *state = RefinementBlackholeState::Failed(err);
        self.condvar.notify_all();
    }

    pub fn reset(&self) {
        let mut state = self.state.lock();
        *state = RefinementBlackholeState::NotRefining;
        self.condvar.notify_all();
    }

    pub fn precision_epoch(&self) -> u64 { self.precision_epoch.load(Ordering::Acquire) }

    pub fn current_precision(&self) -> Option<PrecisionLevel> {
        let state = self.state.lock();
        match &*state {
            RefinementBlackholeState::Refined { achieved_precision, .. } => Some(achieved_precision.clone()),
            _ => None,
        }
    }

    pub fn peek_bounds(&self) -> Option<Bounds> {
        let state = self.state.lock();
        match &*state {
            RefinementBlackholeState::Refined { bounds, .. } => Some(bounds.clone()),
            _ => None,
        }
    }

    pub fn is_refining(&self) -> bool {
        let state = self.state.lock();
        matches!(&*state, RefinementBlackholeState::Refining { .. })
    }
}

impl Default for RefinementBlackhole {
    fn default() -> Self { Self::new() }
}

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

    pub fn refine(&mut self, new_bounds: Bounds) -> Result<(), ComputableError> {
        let new_precision = PrecisionLevel::from_width(new_bounds.width());
        if !self.contains_bounds(&new_bounds) { return Err(ComputableError::BoundsWorsened); }
        if !new_precision.meets(&self.precision) { return Err(ComputableError::BoundsWorsened); }
        self.bounds = new_bounds;
        self.precision = new_precision;
        Ok(())
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
        assert!(nb.refine(refined).is_ok());
        let more_refined = Bounds::new(xbin(3, -1), xbin(5, -1));
        assert!(nb.refine(more_refined).is_ok());
        let widened = Bounds::new(xbin(0, 0), xbin(4, 0));
        assert!(nb.refine(widened).is_err());
    }

    fn test_bounds() -> Bounds { Bounds::new(xbin(1, 0), xbin(2, 0)) }

    #[test]
    fn refinement_blackhole_new_is_not_refined() {
        let rb = RefinementBlackhole::new();
        assert!(rb.current_precision().is_none());
        assert!(rb.peek_bounds().is_none());
    }

    #[test]
    fn refinement_blackhole_try_claim_on_new_returns_claimed() {
        let rb = RefinementBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let result = rb.try_claim(&target);
        assert!(matches!(result, Ok(RefinementClaimResult::Claimed { previous_precision }) if previous_precision == PrecisionLevel::zero()));
    }

    #[test]
    fn refinement_blackhole_complete_updates_state() {
        let rb = RefinementBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let _ = rb.try_claim(&target).expect("should claim");
        let bounds = test_bounds();
        rb.complete(bounds.clone());
        assert!(rb.current_precision().is_some());
        assert_eq!(rb.peek_bounds(), Some(bounds));
    }

    #[test]
    fn refinement_blackhole_already_meets_returns_bounds() {
        let rb = RefinementBlackhole::new();
        let target_low = PrecisionLevel::from_bits(BigInt::from(4));
        let _ = rb.try_claim(&target_low).expect("should claim");
        let small_bounds = Bounds::new(xbin(1, 0), xbin(1, 0));
        rb.complete(small_bounds.clone());
        let result = rb.try_claim(&target_low);
        assert!(matches!(result, Ok(RefinementClaimResult::AlreadyMeets(b)) if b == small_bounds));
    }

    #[test]
    fn refinement_blackhole_fail_propagates_error() {
        let rb = RefinementBlackhole::new();
        let target = PrecisionLevel::from_bits(BigInt::from(8));
        let _ = rb.try_claim(&target).expect("should claim");
        rb.fail(ComputableError::DomainError);
        let result = rb.try_claim(&target);
        assert!(matches!(result, Err(ComputableError::DomainError)));
    }

    #[test]
    fn refinement_blackhole_concurrent_claim_coordinates() {
        let rb = Arc::new(RefinementBlackhole::new());
        let complete_count = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4).map(|_| {
            let rb = Arc::clone(&rb);
            let count = Arc::clone(&complete_count);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                // Target precision of 8 bits = width of 2^(-8) = 1/256
                let target = PrecisionLevel::from_bits(BigInt::from(8));
                match rb.try_claim(&target) {
                    Ok(RefinementClaimResult::Claimed { .. }) => {
                        count.fetch_add(1, AtomicOrdering::SeqCst);
                        thread::sleep(Duration::from_millis(10));
                        // Create bounds with width = 1 * 2^(-10) to meet precision 8
                        // Width = 2^(-10), precision = -(-10) - 1 + 1 = 10 >= 8
                        let bounds = Bounds::new(xbin(1, -10), xbin(2, -10)); // Width = 2^(-10)
                        rb.complete(bounds);
                    }
                    Ok(RefinementClaimResult::AlreadyMeets(_)) => {}
                    Err(_) => panic!("unexpected error"),
                }
            })
        }).collect();

        for h in handles { h.join().expect("join"); }
        // Exactly one thread should have completed the refinement
        assert_eq!(complete_count.load(AtomicOrdering::SeqCst), 1);
    }
}
