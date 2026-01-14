use std::sync::atomic::{AtomicBool, Ordering};

/// Monotonic stop flag that can only transition from false to true.
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
        self.inner.store(true, Ordering::Relaxed);
    }

    pub fn is_stopped(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }
}
