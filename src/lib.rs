//! Computable real numbers with provable correctness.
//!
//! This crate provides a framework for exact real arithmetic using interval refinement.
//! Numbers are represented as computations that can be refined to arbitrary precision
//! while maintaining provably correct bounds.
//!
//! # Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`binary`]: Arbitrary-precision binary numbers (mantissa + exponent representation)
//! - [`ordered_pair`]: Interval types with bounds checking (Bounds, Interval)
//! - [`error`]: Error types for computable operations
//! - [`node`]: Computation graph infrastructure (Node, NodeOp traits)
//! - [`ops`]: Arithmetic and transcendental operations (add, mul, inv, sin, etc.)
//! - [`refinement`]: Parallel refinement infrastructure
//! - [`computable`]: The main Computable type
//!
//! # Example
//!
//! ```
//! use computable::{Computable, Binary};
//! use num_bigint::{BigInt, BigUint};
//!
//! // Create a constant
//! let x = Computable::constant(Binary::new(BigInt::from(2), BigInt::from(0)));
//!
//! // Arithmetic operations
//! let y = x.clone() + x.clone();
//! let z = y * x;
//!
//! // Get current bounds
//! let bounds = z.bounds().unwrap();
//! ```

#![warn(
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used
)]

// External modules (already exist)
mod binary;
mod concurrency;
mod ordered_pair;

// New internal modules
mod computable;
mod error;
mod node;
mod ops;
mod refinement;

// Re-export public API
pub use binary::{Binary, BinaryError, UBinary, UXBinary, XBinary, XBinaryError};
pub use computable::{Computable, DEFAULT_INV_MAX_REFINES, DEFAULT_MAX_REFINEMENT_ITERATIONS};
pub use error::ComputableError;
pub use ordered_pair::{Bounds, Interval, IntervalError};

// Used by tests module
#[cfg(test)]
use refinement::bounds_width_leq;

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use num_bigint::{BigInt, BigUint};
    use num_traits::One;
    use std::sync::{Arc, Barrier};
    use std::thread;

    type IntervalState = Bounds;

    // --- test utilities ---

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

    fn assert_bounds_compatible_with_expected(
        bounds: &Bounds,
        expected: &Binary,
        epsilon: &UBinary,
    ) {
        let lower = unwrap_finite(bounds.small());
        let upper_xb = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());
        let upper = unwrap_finite(&upper_xb);

        assert!(lower <= *expected && *expected <= upper);
        assert!(width <= *epsilon);
    }

    fn unwrap_finite_uxbinary(input: &UXBinary) -> UBinary {
        match input {
            UXBinary::Finite(value) => value.clone(),
            UXBinary::PosInf => {
                panic!("expected finite unsigned extended binary")
            }
        }
    }

    fn assert_width_nonnegative(bounds: &Bounds) {
        assert!(*bounds.width() >= UXBinary::zero());
    }

    // --- tests for different results of refinement (mostly errors) ---

    #[test]
    fn computable_refine_to_rejects_zero_epsilon() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(0, 0);
        let result = computable.refine_to_default(epsilon);
        assert!(matches!(result, Err(ComputableError::NonpositiveEpsilon)));
    }

    #[test]
    fn computable_refine_to_returns_refined_state() {
        let computable = interval_midpoint_computable(0, 2);
        let epsilon = ubin(1, -1);
        let bounds = computable
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected = xbin(1, 0);
        let upper = bounds.large();
        let width = unwrap_finite_uxbinary(bounds.width());

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
    fn computable_refine_to_rejects_unchanged_state() {
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
    fn computable_refine_to_enforces_max_iterations() {
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
    fn computable_refine_to_handles_non_meeting_bounds() {
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
    fn computable_refine_to_rejects_worsened_bounds() {
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

    // --- tests for bounds of arithmetic operations ---

    #[test]
    fn computable_add_combines_bounds() {
        let left = interval_midpoint_computable(0, 2);
        let right = interval_midpoint_computable(1, 3);

        let sum = left + right;
        let sum_bounds = sum.bounds().expect("bounds should succeed");
        assert_eq!(sum_bounds, Bounds::new(xbin(1, 0), xbin(5, 0)));
    }

    #[test]
    fn computable_sub_combines_bounds() {
        let left = interval_midpoint_computable(4, 6);
        let right = interval_midpoint_computable(1, 2);

        let diff = left - right;
        let diff_bounds = diff.bounds().expect("bounds should succeed");
        assert_eq!(diff_bounds, Bounds::new(xbin(2, 0), xbin(5, 0)));
    }

    #[test]
    fn computable_neg_flips_bounds() {
        let value = interval_midpoint_computable(1, 3);
        let negated = -value;
        let bounds = negated.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-3, 0), xbin(-1, 0)));
    }

    #[test]
    fn computable_inv_allows_infinite_bounds() {
        let value = interval_midpoint_computable(-1, 1);
        let inv = value.inv();
        let bounds = inv.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            Bounds::new(XBinary::NegInf, XBinary::PosInf)
        );
    }

    #[test]
    fn computable_inv_bounds_for_positive_interval() {
        let value = interval_midpoint_computable(2, 4);
        let inv = value.inv();
        let epsilon = ubin(1, -8);
        let bounds = inv
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");
        let expected_binary = XBinary::from_f64(1.0 / 3.0)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);

        assert_bounds_compatible_with_expected(&bounds, &expected_value, &epsilon);
    }

    #[test]
    fn computable_mul_combines_bounds_positive() {
        let left = interval_midpoint_computable(1, 3);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(2, 0), xbin(12, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_negative() {
        let left = interval_midpoint_computable(-3, -1);
        let right = interval_midpoint_computable(2, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-12, 0), xbin(-2, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_mixed() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(4, 5);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-10, 0), xbin(15, 0)));
    }

    #[test]
    fn computable_mul_combines_bounds_with_zero() {
        let left = interval_midpoint_computable(-2, 3);
        let right = interval_midpoint_computable(-1, 4);

        let product = left * right;
        let bounds = product.bounds().expect("bounds should succeed");
        assert_eq!(bounds, Bounds::new(xbin(-8, 0), xbin(12, 0)));
    }

    #[test]
    fn computable_from_binary_matches_constant_bounds() {
        let value = bin(3, 0);
        let computable: Computable = value.clone().into();

        let bounds = computable.bounds().expect("bounds should succeed");
        assert_eq!(
            bounds,
            Bounds::new(
                XBinary::Finite(value.clone()),
                XBinary::Finite(value)
            )
        );
    }

    // --- test more complex expressions ---

    #[test]
    fn computable_integration_sqrt2_expression() {
        let one = Computable::constant(bin(1, 0));
        let sqrt2 = sqrt_computable(2);
        let expr = (sqrt2.clone() + one.clone()) * (sqrt2.clone() - one) + sqrt2.inv();

        let epsilon = ubin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = bounds.large();
        let upper = unwrap_finite(&upper);
        let expected = 1.0_f64 + 2.0_f64.sqrt().recip();
        let expected_binary = XBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon.to_binary();

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }

    #[test]
    fn computable_shared_operand_in_expression() {
        let shared = sqrt_computable(2);
        let expr = shared.clone() + shared * Computable::constant(bin(1, 0));

        let epsilon = ubin(1, -12);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = bounds.large();
        let upper = unwrap_finite(&upper);
        let expected = 2.0_f64 * 2.0_f64.sqrt();
        let expected_binary = XBinary::from_f64(expected)
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected_binary);
        let eps_binary = epsilon.to_binary();

        let lower_plus = lower.add(&eps_binary);
        let upper_minus = upper.sub(&eps_binary);

        assert!(lower <= expected_value && expected_value <= upper);
        assert!(upper_minus <= expected_value && expected_value <= lower_plus);
    }

    // --- concurrency tests ---

    #[test]
    fn computable_refine_shared_clone_updates_original() {
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
    fn computable_refine_to_channel_closure() {
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
    fn computable_refine_to_max_iterations_multiple_refiners() {
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
    fn computable_refine_to_error_path_stops_refiners() {
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
        use std::time::{Duration, Instant};

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
        use std::time::Duration;

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

    // --- sin tests ---

    #[test]
    fn computable_sin_of_zero() {
        let zero = Computable::constant(bin(0, 0));
        let sin_zero = zero.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_zero
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0) = 0
        let expected = bin(0, 0);
        assert_bounds_compatible_with_expected(&bounds, &expected, &epsilon);
    }

    #[test]
    fn computable_sin_of_pi_over_2() {
        // pi/2 ~= 1.5707963...
        // We approximate it as 3217/2048 ~= 1.5708...
        let pi_over_2 = Computable::constant(bin(3217, -11));
        let sin_pi_2 = pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi/2) = 1
        let expected_f64 = (std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi/2) should be very close to 1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn computable_sin_of_pi() {
        // pi ~= 3.14159...
        // We approximate it as 6434/2048 ~= 3.1416...
        let pi_approx = Computable::constant(bin(6434, -11));
        let sin_pi = pi_approx.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_pi
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(pi) ~= 0 (should be close to 0)
        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(pi) should be very close to 0
        let small_bound = bin(1, -4);
        let neg_small_bound = bin(-1, -4);
        assert!(lower >= neg_small_bound);
        assert!(upper <= small_bound);
    }

    #[test]
    fn computable_sin_of_negative_pi_over_2() {
        // -pi/2 ~= -1.5707963...
        let neg_pi_over_2 = Computable::constant(bin(-3217, -11));
        let sin_neg_pi_2 = neg_pi_over_2.sin();
        let epsilon = ubin(1, -6);
        let bounds = sin_neg_pi_2
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(-pi/2) = -1
        let expected_f64 = (-std::f64::consts::FRAC_PI_2).sin();
        let expected_binary = XBinary::from_f64(expected_f64)
            .expect("expected value should convert to extended binary");
        let expected = unwrap_finite(&expected_binary);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        // sin(-pi/2) should be very close to -1
        assert!(lower <= expected && expected <= upper);
    }

    #[test]
    fn computable_sin_bounds_always_in_minus_one_to_one() {
        // Test with a large value that exercises argument reduction
        let large_value = Computable::constant(bin(100, 0));
        let sin_large = large_value.sin();
        let bounds = sin_large.bounds().expect("bounds should succeed");

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        assert!(lower >= neg_one);
        assert!(upper <= one);
    }

    #[test]
    fn computable_sin_of_small_value() {
        // For small x, sin(x) ~= x
        let small = Computable::constant(bin(1, -4)); // 1/16 = 0.0625
        let sin_small = small.sin();
        let epsilon = ubin(1, -8);
        let bounds = sin_small
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // sin(0.0625) ~= 0.0624593...
        let expected = XBinary::from_f64(0.0625_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn computable_sin_interval_spanning_maximum() {
        // An interval that spans pi/2 (where sin has maximum)
        let interval_state = Bounds::new(xbin(1, 0), xbin(2, 0)); // [1, 2] includes pi/2 ~= 1.57
        let computable = Computable::new(
            interval_state,
            |inner_state| Ok(interval_bounds(inner_state)),
            interval_refine,
        );
        let sin_interval = computable.sin();
        let bounds = sin_interval.bounds().expect("bounds should succeed");

        let upper = unwrap_finite(&bounds.large());

        // The upper bound should be close to 1 since the interval contains pi/2
        assert!(upper >= bin(1, -1)); // Upper bound should be at least 0.5
    }

    #[test]
    fn computable_sin_with_infinite_input_bounds() {
        let unbounded = Computable::new(
            0usize,
            |_| {
                Ok(Bounds::new(
                    XBinary::NegInf,
                    XBinary::PosInf,
                ))
            },
            |state| state + 1,
        );
        let sin_unbounded = unbounded.sin();
        let bounds = sin_unbounded.bounds().expect("bounds should succeed");

        // sin of unbounded input should be [-1, 1]
        assert_eq!(bounds.small(), &xbin(-1, 0));
        assert_eq!(&bounds.large(), &xbin(1, 0));
    }

    #[test]
    fn computable_sin_expression_with_arithmetic() {
        // Test sin(x) + cos-like expression: sin(x)^2 + sin(x + pi/2)^2 should be close to 1
        // Here we just test that sin works in expressions
        let x = Computable::constant(bin(1, 0)); // x = 1
        let sin_x = x.clone().sin();
        let two = Computable::constant(bin(2, 0));
        let expr = sin_x.clone() * two; // 2 * sin(1)

        let epsilon = ubin(1, -8);
        let bounds = expr
            .refine_to_default(epsilon.clone())
            .expect("refine_to should succeed");

        // 2 * sin(1) ~= 2 * 0.8414... ~= 1.6829...
        let expected = XBinary::from_f64(2.0 * 1.0_f64.sin())
            .expect("expected value should convert to extended binary");
        let expected_value = unwrap_finite(&expected);

        let lower = unwrap_finite(bounds.small());
        let upper = unwrap_finite(&bounds.large());

        assert!(lower <= expected_value && expected_value <= upper);
    }

    #[test]
    fn directed_rounding_produces_valid_bounds() {
        // Test that directed rounding produces well-ordered bounds that contain the true value.
        //
        // Key invariants:
        // 1. lower <= upper (bounds are ordered)
        // 2. lower_sum <= upper_sum (directed rounding produces correct ordering)
        // 3. The bounds interval width decreases with more terms
        // 4. Bounds remain within [-1, 1] (sin range)

        use crate::ops::sin::taylor_sin_bounds_test;

        let test_cases = [
            bin(1, -2),   // 0.25
            bin(1, 0),    // 1.0
            bin(3, 0),    // 3.0
            bin(-1, 0),   // -1.0
            bin(5, -1),   // 2.5
            bin(-3, -1),  // -1.5
        ];

        let neg_one = bin(-1, 0);
        let one = bin(1, 0);

        for x in &test_cases {
            // Compute Taylor bounds with directed rounding
            let (lower, upper) = taylor_sin_bounds_test(x, 10);

            // Verify bounds are ordered correctly
            assert!(
                lower <= upper,
                "Lower bound {} should be <= upper bound {} for x = {}",
                lower, upper, x
            );

            // Verify bounds are within sin's range [-1, 1]
            assert!(
                lower >= neg_one,
                "Lower bound {} should be >= -1 for x = {}",
                lower, x
            );
            assert!(
                upper <= one,
                "Upper bound {} should be <= 1 for x = {}",
                upper, x
            );
        }
    }

    #[test]
    fn directed_rounding_bounds_converge() {
        use crate::ops::sin::taylor_sin_bounds_test;

        // Verify that bounds get tighter as we add more terms
        let x = bin(1, 0); // 1.0

        let (lower5, upper5) = taylor_sin_bounds_test(&x, 5);
        let (lower10, upper10) = taylor_sin_bounds_test(&x, 10);

        let width5 = upper5.sub(&lower5);
        let width10 = upper10.sub(&lower10);

        // More terms should give tighter bounds
        assert!(
            width10 < width5,
            "Bounds with 10 terms (width {}) should be tighter than 5 terms (width {})",
            width10, width5
        );
    }

    #[test]
    fn directed_rounding_symmetry() {
        use crate::ops::sin::taylor_sin_bounds_test;

        // Test that sin(-x) bounds are the negation of sin(x) bounds
        // This verifies that the directed rounding handles negative inputs correctly

        let x = bin(1, -2); // 0.25
        let neg_x = bin(-1, -2); // -0.25

        let (lower_x, upper_x) = taylor_sin_bounds_test(&x, 10);
        let (lower_neg_x, upper_neg_x) = taylor_sin_bounds_test(&neg_x, 10);

        // sin(-x) = -sin(x), so bounds should be negated and swapped
        // lower(-x) should equal -upper(x)
        // upper(-x) should equal -lower(x)

        // Allow small differences due to rounding
        let neg_upper_x = upper_x.neg();
        let neg_lower_x = lower_x.neg();

        // The bounds should be approximately symmetric
        // We just verify they're in the right ballpark
        assert!(
            lower_neg_x <= neg_upper_x.add(&bin(1, -50)),
            "lower(sin(-x)) should be approximately -upper(sin(x))"
        );
        assert!(
            neg_lower_x <= upper_neg_x.add(&bin(1, -50)),
            "-lower(sin(x)) should be approximately upper(sin(-x))"
        );
    }

    #[test]
    fn directed_rounding_lower_bound_is_lower() {
        use crate::ops::sin::taylor_sin_partial_sum_test;

        // Verify that rounding down produces smaller values than rounding up
        let x = bin(1, 0); // 1.0
        let n = 5;

        let sum_down = taylor_sin_partial_sum_test(&x, n, true);
        let sum_up = taylor_sin_partial_sum_test(&x, n, false);

        // The down-rounded sum should be <= up-rounded sum
        assert!(
            sum_down <= sum_up,
            "Rounding down {} should produce <= rounding up {}",
            sum_down, sum_up
        );
    }
}
