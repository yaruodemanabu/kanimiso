//! Extracted numerical-compromise journal.

use crate::event::{Event, EventKind};
use signlred::NumericalCompromise;

/// Time-ordered list of numerical compromises taken during a session.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompromiseJournal {
    entries: Vec<(u64, String, String, NumericalCompromise)>,
}

impl CompromiseJournal {
    /// Empty journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a compromise with the event sequence number and algorithm.
    pub fn push(
        &mut self,
        seq: u64,
        algorithm: impl Into<String>,
        operation: impl Into<String>,
        compromise: NumericalCompromise,
    ) {
        self.entries
            .push((seq, algorithm.into(), operation.into(), compromise));
    }

    /// Ingest from a quality event if it carries a compromise.
    pub fn ingest_event(&mut self, event: &Event) {
        if let Some(issue) = &event.issue {
            if let Some(c) = &issue.numerical_compromise {
                self.push(event.seq, &event.algorithm, &event.operation, c.clone());
            } else if event.kind == EventKind::NumericalCompromise {
                // issue without a structured compromise still belongs in the journal
                self.push(
                    event.seq,
                    &event.algorithm,
                    &event.operation,
                    NumericalCompromise::new(
                        "(unspecified intent)",
                        event.message.clone(),
                        "see event message",
                        "read the attached issue before interpreting numbers",
                    ),
                );
            }
        }
    }

    /// Number of compromises.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no compromise was recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries in order.
    pub fn entries(&self) -> &[(u64, String, String, NumericalCompromise)] {
        &self.entries
    }

    /// Multi-line dump.
    pub fn render(&self) -> String {
        let mut out = format!("ojizou-san compromise journal entries={}\n", self.len());
        for (seq, algo, op, c) in &self.entries {
            out.push_str(&format!("  seq={seq} {algo}::{op} {c}\n"));
        }
        out
    }
}
