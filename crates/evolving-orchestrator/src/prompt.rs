//! The optimizer prompt and output normalization.
//!
//! The prompt is the optimizer's entire interface: task, current design, best
//! design, original design, workspace evidence, and comparison history. The
//! response is JSON with a full replacement design block.

use serde_json::Value;

use crate::error::{EvolveError, Result};
use crate::history::HistoryRow;

#[derive(Debug, Clone)]
pub struct BestDesign<'a> {
    pub version: u32,
    pub design: &'a str,
    pub rationale: &'a str,
}

#[derive(Debug, Clone)]
pub struct ProposalInput<'a> {
    pub task_text: &'a str,
    pub current_version: u32,
    pub current_design: &'a str,
    pub best: Option<BestDesign<'a>>,
    pub v0_design: Option<&'a str>,
    pub evidence: &'a str,
    pub execution_metadata: Option<&'a str>,
    pub history: &'a [HistoryRow],
    pub baseline_mode: bool,
    pub scoreboard: Option<&'a str>,
    pub start_marker: &'a str,
    pub end_marker: &'a str,
}

pub fn build_prompt(input: &ProposalInput<'_>) -> (String, String) {
    let system = format!(
        "You are a principal workflow engineer for autonomous coding agents.\n\
Improve only the multi-agent design block of the task prompt. Do not rewrite the task.\n\
Prioritize two things:\n\
1. stronger output contracts: each subagent must return the exact information later agents need;\n\
2. useful hops: allow the orchestrator to recall a subagent with a narrower follow-up when a check fails.\n\
Ground every change in concrete evidence from the workspace or the comparison history.\n\
Keep the design focused; do not add arbitrary numeric caps on agents, recalls, turns, tools, or time.\n\
Return valid JSON only. The design must start with {:?}.\n",
        input.start_marker
    );

    let mut sections = Vec::new();
    sections.push(format!("# Task\n\n{}", input.task_text.trim()));
    sections.push(format!(
        "# Current design (v{})\n\n{}",
        input.current_version,
        input.current_design.trim()
    ));

    if let Some(best) = &input.best {
        let rationale = if best.rationale.trim().is_empty() {
            String::new()
        } else {
            format!("\nJudge rationale: {}", best.rationale.trim())
        };
        sections.push(format!(
            "# Best design so far (v{})\n\n{}{}\n\nKeep what made it strong; merge only useful changes attempted since.",
            best.version,
            best.design.trim(),
            rationale
        ));
    }

    if let Some(v0) = input.v0_design {
        sections.push(format!(
            "# Original design (v0)\n\n{}\n\nUse it to notice drift, not to revert wholesale.",
            v0.trim()
        ));
    }

    sections.push(format!(
        "# Workspace evidence\n\n{}",
        if input.evidence.trim().is_empty() {
            "- No evidence was supplied."
        } else {
            input.evidence.trim()
        }
    ));
    sections.push(format!(
        "# Execution metadata\n\n{}",
        input.execution_metadata.unwrap_or("- None.").trim()
    ));

    let mut history = String::new();
    if input.history.is_empty() {
        history.push_str("- No prior comparisons yet.");
    } else {
        if let Some(scoreboard) = input.scoreboard {
            history.push_str(scoreboard.trim());
            history.push_str("\n\n");
        }
        for row in input.history {
            history.push_str(&format!(
                "- `{}` — winner {:?}\n  - rationale: {}\n  - judge: {}\n",
                row.key,
                row.winner,
                truncate(row.rationale.trim(), 2400),
                row.model
            ));
        }
    }
    sections.push(format!("# Comparison history\n\n{history}"));

    sections.push(format!(
        "# Instructions\n\n\
Improve the design from v{} to v{}.\n\
- Preserve the task's required deliverables and original intent.\n\
- Make output contracts explicit and machine-consumable.\n\
- Add recall/re-delegation where a check can fail and the work must be corrected.\n\
- Cite evidence in the JSON fields.\n\
- Do not change the task text.\n\n\
# Output schema\n\n\
{{\n  \"improved_multi_agent_design\": \"<full replacement block, starting with {:?}>\",\n  \"changes_from_previous\": [\"...\"],\n  \"why_it_should_improve\": [\"...\"],\n  \"evidence_used\": [\"...\"]\n}}\n",
        input.current_version,
        input.current_version + 1,
        input.start_marker
    ));

    if input.baseline_mode {
        sections.push(
            "The comparison target is the fixed bare-task baseline. The goal is to decisively outperform it, not merely the previous version."
                .into(),
        );
    }

    (system, sections.join("\n\n"))
}

/// Require the start marker and trim a copied suffix when the model runs past
/// the design block.
pub fn normalize_design(raw: &str, start_marker: &str, end_marker: &str) -> Result<String> {
    let text = raw.trim();
    if !text.starts_with(start_marker) {
        return Err(EvolveError::Judgment(format!(
            "optimizer output does not start with the design marker: {}",
            truncate(text, 200)
        )));
    }
    let mut normalized = text.to_string();
    if let Some(index) = normalized.rfind(end_marker) {
        if index * 2 >= normalized.len() {
            normalized.truncate(index);
        }
    }
    let normalized = normalized.trim().to_string();
    if normalized.len() <= start_marker.len() {
        return Err(EvolveError::Judgment(
            "optimizer returned an empty design".into(),
        ));
    }
    Ok(normalized)
}

/// Extract a required string field from a JSON object, accepting fenced JSON.
pub fn extract_json_field(raw: &str, field: &str) -> Result<String> {
    let text = raw.trim();
    let candidate = if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        match rest.find("```") {
            Some(end) => rest[..end].trim().to_string(),
            None => text.to_string(),
        }
    } else {
        match (text.find('{'), text.rfind('}')) {
            (Some(start), Some(end)) if end > start => text[start..=end].to_string(),
            _ => text.to_string(),
        }
    };
    let value: Value = serde_json::from_str(&candidate).map_err(|error| {
        EvolveError::Judgment(format!(
            "optimizer output is not valid JSON: {error}; raw={}",
            truncate(text, 400)
        ))
    })?;
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| EvolveError::Judgment(format!("optimizer output has no {field:?}")))
}

fn truncate(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        value.chars().take(limit).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::Winner;

    fn row(key: &str, winner: Winner, rationale: &str) -> HistoryRow {
        HistoryRow {
            key: key.into(),
            comparison: key.into(),
            winner,
            rationale: rationale.into(),
            model: "local".into(),
            judged_at_unix_ms: 0,
            repo_a: "a".into(),
            repo_b: "b".into(),
        }
    }

    #[test]
    fn prompt_has_the_required_sections() {
        let history = vec![row("v0_vs_v1", Winner::B, "better tests")];
        let (system, user) = build_prompt(&ProposalInput {
            task_text: "fix it",
            current_version: 1,
            current_design: "START\ndesign\nEND",
            best: Some(BestDesign {
                version: 1,
                design: "START\nbest\nEND",
                rationale: "passed",
            }),
            v0_design: Some("START\nv0\nEND"),
            evidence: "tests passed",
            execution_metadata: Some("exit 0"),
            history: &history,
            baseline_mode: false,
            scoreboard: None,
            start_marker: "START",
            end_marker: "END",
        });
        assert!(system.contains("Return valid JSON only"));
        assert!(user.contains("# Task"));
        assert!(user.contains("# Current design (v1)"));
        assert!(user.contains("# Best design so far (v1)"));
        assert!(user.contains("# Original design (v0)"));
        assert!(user.contains("# Workspace evidence"));
        assert!(user.contains("v0_vs_v1"));
        assert!(user.contains("improved_multi_agent_design"));
    }

    #[test]
    fn normalize_requires_the_marker() {
        let error = normalize_design("no marker", "START", "END").expect_err("reject");
        assert!(error.to_string().contains("marker"));
    }

    #[test]
    fn normalize_trims_a_copied_suffix() {
        let text = "START\ndesign\nEND\nrest";
        assert_eq!(
            normalize_design(text, "START", "END").expect("normalize"),
            "START\ndesign"
        );
    }

    #[test]
    fn extracts_a_fenced_field() {
        let raw = "```json\n{\"improved_multi_agent_design\":\"START\\nd\\nEND\"}\n```";
        assert_eq!(
            extract_json_field(raw, "improved_multi_agent_design").unwrap(),
            "START\nd\nEND"
        );
    }
}
