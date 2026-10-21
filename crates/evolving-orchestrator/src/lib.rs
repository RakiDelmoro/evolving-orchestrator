//! Evolving Orchestrator: a workflow-evolution harness.
//!
//! The pipeline has one shape:
//!
//! ```text
//! L^(g) --design search--> d* --collection--> episodes --rank--> SFT --train--> L^(g+1)
//! ```
//!
//! Every external effect sits behind a port: the episode executor, the chat
//! model, the tokenizer, the trainer, and the exporter. The pure decisions —
//! the search loop, champion arbitration, the tournament schedule, the
//! Bradley–Terry fit, the split, and checkpoint selection — are ordinary
//! functions with no I/O.

pub mod champion;
pub mod chat;
pub mod collect;
pub mod config;
pub mod cycle;
pub mod design;
pub mod error;
pub mod exec;
pub mod export;
pub mod history;
pub mod io;
pub mod judge;
pub mod pi_session;
pub mod prompt;
pub mod ranking;
pub mod render;
pub mod search;
pub mod select;
pub mod train;

pub use error::{EvolveError, Result};
