//! Pairwise workspace judgment.
//!
//! The judge receives two workspace bundles and returns a winner and a
//! rationale. It never runs the project; it reads the deliverables.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{ChatClient, ChatOptions};
use crate::error::{EvolveError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Winner {
    A,
    B,
    Tie,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub winner: Winner,
    pub rationale: String,
}

#[derive(Debug, Clone)]
pub struct WorkspaceBundle {
    pub task_text: String,
    pub tree: String,
    pub excerpts: Vec<(String, String)>,
    pub log_excerpt: Option<String>,
}

impl WorkspaceBundle {
    pub fn render(&self, label: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Workspace for submission {label}\n\n"));
        out.push_str("## File tree\n\n```text\n");
        out.push_str(&self.tree);
        out.push_str("\n```\n\n");
        for (header, body) in &self.excerpts {
            out.push_str(header);
            out.push_str("\n\n```text\n");
            out.push_str(body);
            out.push_str("\n```\n\n");
        }
        if let Some(log) = &self.log_excerpt {
            out.push_str("## Execution log excerpt\n\n```text\n");
            out.push_str(log);
            out.push_str("\n```\n\n");
        }
        out
    }
}

pub trait PairwiseJudge {
    fn compare(
        &self,
        task_text: &str,
        a: &WorkspaceBundle,
        b: &WorkspaceBundle,
        seed: u64,
    ) -> Result<Verdict>;
}

pub struct ModelJudge<'a> {
    pub chat: &'a dyn ChatClient,
    pub options: ChatOptions,
}

impl PairwiseJudge for ModelJudge<'_> {
    fn compare(
        &self,
        task_text: &str,
        a: &WorkspaceBundle,
        b: &WorkspaceBundle,
        seed: u64,
    ) -> Result<Verdict> {
        let system = system_prompt();
        let user = user_prompt(task_text, a, b);
        let mut options = self.options.clone();
        options.seed = seed;
        let raw = self.chat.complete(&system, &user, &options)?;
        parse_verdict(&raw)
    }
}

pub fn system_prompt() -> String {
    "You are a strict senior reviewer comparing two completed submissions for the same assignment.\n\
Judge which submission better fulfills the task. Prefer concrete evidence from the workspaces: \
deliverable coverage, empirical rigor, reproducibility, presentation, engineering, and task alignment.\n\
Return only valid JSON with the keys \"winner\" (\"A\", \"B\", or \"tie\") and \"rationale\" (a string).\n"
        .into()
}

pub fn user_prompt(task_text: &str, a: &WorkspaceBundle, b: &WorkspaceBundle) -> String {
    format!(
        "# Task\n\n{task_text}\n\n{}\n{}\n\
## Comparison rubric\n\
1. Deliverable coverage: every explicitly required artifact is present and non-placeholder.\n\
2. Empirical rigor: appropriate baselines, honest limitations, consistent report and metrics.\n\
3. Reproducibility: dependencies, entry points, seeds, data or generation documented.\n\
4. Presentation: report structure, clarity, figure integration.\n\
5. Engineering: layout, modularity, readability.\n\
6. Task alignment: no wrong-problem or drift.\n\n\
## Output schema\n\
{{\n  \"winner\": \"A\" | \"B\" | \"tie\",\n  \"rationale\": \"<cite concrete evidence from both workspaces>\"\n}}\n\
Return JSON only.\n",
        a.render("A"),
        b.render("B")
    )
}

pub fn parse_verdict(raw: &str) -> Result<Verdict> {
    let candidate = extract_json_object(raw);
    let value: Value = serde_json::from_str(&candidate).map_err(|error| {
        EvolveError::Judgment(format!(
            "judge output is not valid JSON: {error}; raw={}",
            truncate(raw, 600)
        ))
    })?;
    let winner = match value.get("winner").and_then(Value::as_str) {
        Some("A") | Some("a") => Winner::A,
        Some("B") | Some("b") => Winner::B,
        Some("tie") | Some("Tie") | Some("TIE") => Winner::Tie,
        _ => {
            return Err(EvolveError::Judgment(
                "judge output has no valid winner".into(),
            ))
        }
    };
    let rationale = value
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if rationale.is_empty() {
        return Err(EvolveError::Judgment(
            "judge output has an empty rationale".into(),
        ));
    }
    Ok(Verdict { winner, rationale })
}

fn extract_json_object(raw: &str) -> String {
    let text = raw.trim();
    if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if end > start => text[start..=end].to_string(),
        _ => text.to_string(),
    }
}

fn truncate(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}

/// Build a bounded, deterministic evidence bundle from a workspace.
pub fn build_bundle(
    root: &Path,
    task_text: &str,
    log_path: Option<&Path>,
) -> Result<WorkspaceBundle> {
    let mut files = Vec::new();
    walk(root, 0, &mut files)?;
    files.sort();

    let tree = render_tree(root, &files, 400, 6);
    let mut excerpts = Vec::new();
    for path in files.iter().filter(|path| is_evidence_file(path)) {
        if excerpts.len() >= 4 {
            break;
        }
        let relative = path.strip_prefix(root).unwrap_or(path);
        let body = read_capped(path, 12_000)?;
        if body.trim().is_empty() {
            continue;
        }
        excerpts.push((format!("### File: `{}`", relative.display()), body));
    }

    let log_excerpt = match log_path {
        Some(path) if path.exists() => Some(read_tail(path, 16_000)?),
        _ => None,
    };

    Ok(WorkspaceBundle {
        task_text: task_text.to_string(),
        tree,
        excerpts,
        log_excerpt,
    })
}

fn walk(directory: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > 8 {
        return Ok(());
    }
    let entries = fs::read_dir(directory).map_err(|error| {
        EvolveError::Io(format!("cannot read {}: {error}", directory.display()))
    })?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(
            name.as_ref(),
            ".git" | "node_modules" | "target" | "__pycache__" | ".venv"
        ) {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            walk(&path, depth + 1, out)?;
        } else if metadata.is_file() && metadata.len() <= 2_000_000 {
            out.push(path);
        }
    }
    Ok(())
}

fn render_tree(root: &Path, files: &[PathBuf], max_files: usize, max_depth: usize) -> String {
    let mut lines = Vec::new();
    for path in files.iter().take(max_files) {
        let relative = path.strip_prefix(root).unwrap_or(path);
        if relative.components().count() > max_depth {
            continue;
        }
        lines.push(relative.display().to_string());
    }
    if files.len() > max_files {
        lines.push(format!("… {} more files", files.len() - max_files));
    }
    lines.join("\n")
}

fn is_evidence_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("md" | "json" | "py" | "rs" | "txt" | "csv" | "toml" | "yaml" | "yml")
    )
}

fn read_capped(path: &Path, cap: usize) -> Result<String> {
    let text = fs::read_to_string(path).unwrap_or_default();
    if text.len() <= cap {
        Ok(text)
    } else {
        Ok(format!(
            "{}\n… [truncated {} bytes]",
            &text[..cap],
            text.len() - cap
        ))
    }
}

fn read_tail(path: &Path, cap: usize) -> Result<String> {
    let text = fs::read_to_string(path).unwrap_or_default();
    if text.len() <= cap {
        Ok(text)
    } else {
        Ok(format!("… [head omitted]\n{}", &text[text.len() - cap..]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_json() {
        let verdict = parse_verdict("```json\n{\"winner\":\"B\",\"rationale\":\"better\"}\n```")
            .expect("parse");
        assert_eq!(verdict.winner, Winner::B);
        assert_eq!(verdict.rationale, "better");
    }

    #[test]
    fn rejects_unknown_winner() {
        let error = parse_verdict("{\"winner\":\"C\",\"rationale\":\"x\"}").expect_err("must fail");
        assert!(error.to_string().contains("winner"));
    }

    #[test]
    fn bundle_skips_vcs_and_binaries() {
        let root = std::env::temp_dir().join(format!(
            "evolve-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("reports")).unwrap();
        fs::write(root.join(".git/config"), "secret").unwrap();
        fs::write(root.join("reports/result.md"), "# Result\n").unwrap();
        fs::write(root.join("image.bin"), [0u8, 1, 2]).unwrap();
        let bundle = build_bundle(&root, "do it", None).expect("bundle");
        assert!(!bundle.tree.contains(".git"));
        assert!(bundle.tree.contains("reports/result.md"));
        assert_eq!(bundle.excerpts.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
