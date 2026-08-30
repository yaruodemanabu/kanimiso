//! Append-only in-process quality ledger.

use crate::event::{Event, EventKind};
use crate::sink::{MemorySink, Sink};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ordered store of quality events plus fan-out sinks.
#[derive(Clone)]
pub struct Ledger {
    seq: Arc<AtomicU64>,
    memory: MemorySink,
    sinks: Arc<Mutex<Vec<Arc<dyn Sink>>>>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Ledger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ledger")
            .field("n_events", &self.memory.len())
            .finish()
    }
}

impl Ledger {
    /// Empty ledger that always keeps an in-memory copy.
    pub fn new() -> Self {
        Self {
            seq: Arc::new(AtomicU64::new(0)),
            memory: MemorySink::new(),
            sinks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a fan-out sink (stderr, test spy, …).
    pub fn add_sink(&self, sink: Arc<dyn Sink>) {
        self.sinks.lock().expect("ledger sinks poisoned").push(sink);
    }

    /// Append an event, assigning a sequence number.
    pub fn append(&self, mut event: Event) -> Event {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        event.seq = seq;
        self.memory.emit(&event);
        let sinks = self.sinks.lock().expect("ledger sinks poisoned");
        for sink in sinks.iter() {
            sink.emit(&event);
        }
        event
    }

    /// Snapshot of all events.
    pub fn events(&self) -> Vec<Event> {
        self.memory.events()
    }

    /// Events of one kind.
    pub fn of_kind(&self, kind: EventKind) -> Vec<Event> {
        self.events()
            .into_iter()
            .filter(|e| e.kind == kind)
            .collect()
    }

    /// True when no events have been recorded. A finished `fit` must not
    /// leave the ledger empty.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.memory.len()
    }
}
