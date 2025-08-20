use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use jagua_rs::Instant;
use log::warn;
use crate::util::terminator::Terminator;

#[derive(Debug, Clone)]
pub struct CtrlCTerminator {
    pub timeout: Option<Instant>,
    pub ctrlc: Arc<AtomicBool>,
}

impl CtrlCTerminator {
    /// Sets up the handler for Ctrl-C (only call once)
    pub fn new() -> Self {
        let ctrlc = Arc::new(AtomicBool::new(false));
        let c = ctrlc.clone();

        ctrlc::set_handler(move || {
            warn!(" terminating...");
            c.store(true, Ordering::SeqCst);
        }).expect("Error setting Ctrl-C handler");

        Self {
            timeout: None,
            ctrlc,
        }
    }
}

impl Terminator for CtrlCTerminator {
    fn kill(&self) -> bool {
        self.timeout.map_or(false, |timeout| Instant::now() > timeout)
            || self.ctrlc.load(Ordering::SeqCst)
    }

    fn new_timeout(&mut self, timeout: Duration){
        // Reset the Ctrl-C flag and set a new timeout
        self.ctrlc.store(false, Ordering::SeqCst);
        self.timeout = Some(Instant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<Instant> {
        self.timeout
    }
}