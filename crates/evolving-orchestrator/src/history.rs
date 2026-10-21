//! Comparison history records.
//!
//! One row per judged pair. Keys are stable:
//! `v{k-1}_vs_v{k}`, `v{k}_vs_no_harness`, `champion_v{a}_vs_v{b}`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::io::{append_record, read_records};
use crate::judge::Winner;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryRow {
    pub key: String,
    pub comparison: String,
    pub winner: Winner,
    pub rationale: String,
    pub model: String,
    pub judged_at_unix_ms: u128,
    pub repo_a: String,
    pub repo_b: String,
}

impl HistoryRow {
    pub fn adjacent(
        previous: u32,
        current: u32,
        winner: Winner,
        rationale: impl Into<String>,
        model: impl Into<String>,
        repo_a: impl Into<String>,
        repo_b: impl Into<String>,
    ) -> Self {
        Self {
            key: format!("v{previous}_vs_v{current}"),
            comparison: format!("v{previous} vs v{current}"),
            winner,
            rationale: rationale.into(),
            model: model.into(),
            judged_at_unix_ms: now_ms(),
            repo_a: repo_a.into(),
            repo_b: repo_b.into(),
        }
    }

    pub fn baseline(
        version: u32,
        winner: Winner,
        rationale: impl Into<String>,
        model: impl Into<String>,
        repo_baseline: impl Into<String>,
        repo_version: impl Into<String>,
    ) -> Self {
        Self {
            key: format!("v{version}_vs_no_harness"),
            comparison: format!("v{version} vs no_harness"),
            winner,
            rationale: rationale.into(),
            model: model.into(),
            judged_at_unix_ms: now_ms(),
            repo_a: repo_baseline.into(),
            repo_b: repo_version.into(),
        }
    }

    pub fn champion(
        champion: u32,
        current: u32,
        winner: Winner,
        rationale: impl Into<String>,
        model: impl Into<String>,
        repo_a: impl Into<String>,
        repo_b: impl Into<String>,
    ) -> Self {
        Self {
            key: format!("champion_v{champion}_vs_v{current}"),
            comparison: format!("champion v{champion} vs v{current}"),
            winner,
            rationale: rationale.into(),
            model: model.into(),
            judged_at_unix_ms: now_ms(),
            repo_a: repo_a.into(),
            repo_b: repo_b.into(),
        }
    }
}

pub fn read_history(path: &Path) -> Result<Vec<HistoryRow>> {
    read_records(path)
}

pub fn append_history(path: &Path, row: &HistoryRow) -> Result<()> {
    append_record(path, row)
}

pub fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "evolve-history-{}-{}.jsonl",
            std::process::id(),
            now_ms()
        ));
        let row = HistoryRow::adjacent(0, 1, Winner::B, "better", "local", "a", "b");
        append_history(&path, &row).expect("append");
        let rows = read_history(&path).expect("read");
        assert_eq!(rows, vec![row]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn keys_are_stable() {
        assert_eq!(
            HistoryRow::adjacent(2, 3, Winner::A, "", "", "", "").key,
            "v2_vs_v3"
        );
        assert_eq!(
            HistoryRow::baseline(4, Winner::B, "", "", "", "").key,
            "v4_vs_no_harness"
        );
        assert_eq!(
            HistoryRow::champion(1, 4, Winner::B, "", "", "", "").key,
            "champion_v1_vs_v4"
        );
    }
}
