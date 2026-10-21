//! The design block: opaque text between two markers, plus the task catalog.
//!
//! The block is never parsed into roles or a graph. It is the artifact the
//! optimizer rewrites and the executor reads, exactly as written.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{EvolveError, Result};

pub const START_MARKER: &str =
    "Create an agent team with following agent candidates to solve this problem:";
pub const END_MARKER: &str = "Use uv for all Python workflows";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub query_id: u32,
    pub query_name: String,
    pub raw: String,
    pub before: String,
    pub design: String,
    pub after: String,
    pub task_text: String,
}

impl ParsedQuery {
    pub fn render_with_design(&self, design: &str) -> String {
        format!(
            "{}{}\n\n{}",
            self.before,
            design.trim(),
            self.after.trim_start()
        )
    }
}

/// Split a query file into task text, design block, and suffix.
///
/// The end marker is taken from the **last** occurrence because the phrase can
/// legitimately appear inside the design as an instruction line.
pub fn parse_query(path: &Path, start_marker: &str, end_marker: &str) -> Result<ParsedQuery> {
    let raw = fs::read_to_string(path)
        .map_err(|error| EvolveError::Io(format!("cannot read {}: {error}", path.display())))?;
    parse_query_text(&raw, path, start_marker, end_marker)
}

pub fn parse_query_text(
    raw: &str,
    path: &Path,
    start_marker: &str,
    end_marker: &str,
) -> Result<ParsedQuery> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            EvolveError::Design(format!("invalid query filename: {}", path.display()))
        })?;
    let query_id = query_id_from_stem(stem).ok_or_else(|| {
        EvolveError::Design(format!(
            "cannot parse a query id from filename {:?}",
            path.file_name()
        ))
    })?;

    let start = raw.find(start_marker).ok_or_else(|| {
        EvolveError::Design(format!("{}: start marker not found", path.display()))
    })?;
    let end = raw
        .rfind(end_marker)
        .ok_or_else(|| EvolveError::Design(format!("{}: end marker not found", path.display())))?;
    if end <= start {
        return Err(EvolveError::Design(format!(
            "{}: end marker occurs before start marker",
            path.display()
        )));
    }

    let before = raw[..start].to_string();
    let design = raw[start..end].trim().to_string();
    let after = raw[end..].to_string();

    let mut task_text = before.trim().to_string();
    if task_text.contains("\\n") {
        task_text = task_text.replace("\\n", "\n");
    }

    Ok(ParsedQuery {
        query_id,
        query_name: format!("query{query_id}_team"),
        raw: raw.to_string(),
        before,
        design,
        after,
        task_text,
    })
}

fn query_id_from_stem(stem: &str) -> Option<u32> {
    let rest = stem.strip_prefix("query")?;
    let digits: String = rest
        .chars()
        .take_while(|value| value.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: u32,
    pub domain: String,
    pub prompt: String,
    pub initial_design: String,
    pub task_text: String,
}

#[derive(Debug, Clone)]
pub struct TaskCatalog {
    pub directory: PathBuf,
    pub initial_suffix: String,
    pub start_marker: String,
    pub end_marker: String,
}

impl TaskCatalog {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            initial_suffix: "_team".into(),
            start_marker: START_MARKER.into(),
            end_marker: END_MARKER.into(),
        }
    }

    pub fn bare_path(&self, id: u32) -> PathBuf {
        self.directory.join("bare").join(format!("query{id}.txt"))
    }

    pub fn initial_path(&self, id: u32) -> PathBuf {
        self.directory
            .join("initial")
            .join(format!("query{id}{}.txt", self.initial_suffix))
    }

    pub fn parse(&self, id: u32) -> Result<ParsedQuery> {
        parse_query(&self.initial_path(id), &self.start_marker, &self.end_marker)
    }

    pub fn load(&self, id: u32, domain: impl Into<String>) -> Result<Task> {
        let parsed = self.parse(id)?;
        let prompt = fs::read_to_string(self.bare_path(id)).map_err(|error| {
            EvolveError::Io(format!(
                "cannot read bare prompt {}: {error}",
                self.bare_path(id).display()
            ))
        })?;
        Ok(Task {
            id,
            domain: domain.into(),
            prompt,
            initial_design: parsed.design.clone(),
            task_text: parsed.task_text.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Build a thing.\n\nCreate an agent team with following agent candidates to solve this problem:\n[{\"agent_id\":\"a\",\"role\":\"r\"}]\n\nUse uv for all Python workflows\n\nKeep this suffix.\n";

    #[test]
    fn splits_on_first_start_and_last_end() {
        let parsed = parse_query_text(
            SAMPLE,
            Path::new("query51_team.txt"),
            START_MARKER,
            END_MARKER,
        )
        .expect("parse");
        assert_eq!(parsed.query_id, 51);
        assert_eq!(parsed.task_text, "Build a thing.");
        assert!(parsed.design.starts_with("Create an agent team"));
        assert!(!parsed.design.ends_with(END_MARKER));
        assert!(parsed.after.starts_with(END_MARKER));
        assert!(parsed.after.contains("Keep this suffix."));
    }

    #[test]
    fn render_round_trips_the_design() {
        let parsed = parse_query_text(
            SAMPLE,
            Path::new("query51_team.txt"),
            START_MARKER,
            END_MARKER,
        )
        .expect("parse");
        let rendered = parsed.render_with_design(&parsed.design);
        assert_eq!(rendered.trim(), SAMPLE.trim());
    }

    #[test]
    fn escaped_newlines_are_decoded_in_task_text() {
        let text = "line one\\nline two\nCreate an agent team with following agent candidates to solve this problem:\nd\nUse uv for all Python workflows\n";
        let parsed = parse_query_text(text, Path::new("query7_team.txt"), START_MARKER, END_MARKER)
            .expect("parse");
        assert_eq!(parsed.task_text, "line one\nline two");
    }

    #[test]
    fn missing_marker_is_an_error() {
        let error = parse_query_text(
            "no markers here",
            Path::new("query1_team.txt"),
            START_MARKER,
            END_MARKER,
        )
        .expect_err("must fail");
        assert!(error.to_string().contains("start marker"));
    }
}
