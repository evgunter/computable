use std::sync::atomic::{AtomicBool, Ordering};

/// Monotonic stop flag that can only transition from false to true.
///
/// Uses Release/Acquire ordering to ensure proper synchronization:
/// - `stop()` uses Release to make all prior writes visible to readers
/// - `is_stopped()` uses Acquire to see all writes made before `stop()`
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
