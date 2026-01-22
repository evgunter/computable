use std::sync::atomic::{AtomicBool, Ordering};

/// Monotonic stop flag that can only transition from false to true.
///
/// Uses Release/Acquire ordering to ensure proper synchronization:
/// - `stop()` uses Release ordering so that any writes before stopping
///   are visible to threads that observe the stop
/// - `is_stopped()` uses Acquire ordering so that any reads after observing
///   the stop see the writes that happened before the stop
#[derive(Debug)]
pub struct StopFlag {
    inner: AtomicBool,
}

impl StopFlag {
    pub fn new() -> Self {
        Self {
            inner: AtomicBool::new(false),
        }
    }

    pub fn stop(&self) {
        self.inner.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }
}
