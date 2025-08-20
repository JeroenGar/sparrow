use std::time::Duration;
use jagua_rs::Instant;

/// Generic trait for any struct that can determine if the optimization process should terminate.
pub trait Terminator {
    /// Checks if the termination condition is met
    fn kill(&self) -> bool;

    /// Sets a new timeout duration
    fn new_timeout(&mut self, timeout: Duration);

    /// Returns the instant when a timeout was set, if any
    fn timeout_at(&self) -> Option<Instant>;
}

#[derive(Debug, Clone)]
pub struct BasicTerminator {
    pub timeout: Option<Instant>,
}

impl BasicTerminator {
    pub fn new() -> Self {
        Self { timeout: None }
    }
}

impl Terminator for BasicTerminator {
    fn kill(&self) -> bool {
        self.timeout.map_or(false, |timeout| Instant::now() > timeout)
    }

    fn new_timeout(&mut self, timeout: Duration){
        self.timeout = Some(Instant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<Instant> {
        self.timeout
    }
}