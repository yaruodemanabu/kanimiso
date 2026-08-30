//! Destinations for quality events.

use crate::event::Event;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

/// Something that can receive a quality event.
pub trait Sink: Send + Sync {
    /// Persist or display one event.
    fn emit(&self, event: &Event);
}

/// In-memory sink used by tests and by [`crate::Ledger`].
#[derive(Debug, Default, Clone)]
pub struct MemorySink {
    events: Arc<Mutex<Vec<Event>>>,
}

impl MemorySink {
    /// Empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of recorded events.
    pub fn events(&self) -> Vec<Event> {
        self.events.lock().expect("memory sink poisoned").clone()
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.lock().expect("memory sink poisoned").len()
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Sink for MemorySink {
    fn emit(&self, event: &Event) {
        self.events
            .lock()
            .expect("memory sink poisoned")
            .push(event.clone());
    }
}

/// Writes each event as one line to stderr.
#[derive(Debug, Default, Clone, Copy)]
pub struct StderrSink;

impl Sink for StderrSink {
    fn emit(&self, event: &Event) {
        let mut stderr = io::stderr().lock();
        let _ = writeln!(stderr, "{}", event.render());
    }
}

/// Wrap a function as a sink.
pub struct FnSink<F>
where
    F: Fn(&Event) + Send + Sync,
{
    f: F,
}

impl<F> FnSink<F>
where
    F: Fn(&Event) + Send + Sync,
{
    /// Construct from a closure.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> Sink for FnSink<F>
where
    F: Fn(&Event) + Send + Sync,
{
    fn emit(&self, event: &Event) {
        (self.f)(event);
    }
}
