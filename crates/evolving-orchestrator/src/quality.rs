//! The software-quality contract.
//!
//! Hard gates are deterministic commands or scans. Scored dimensions are a
//! rubric judgment. A workspace is accepted only when every required gate
//! passes, every dimension clears its floor, and the weighted score clears the
//! policy threshold.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::chat::{ChatClient, ChatOptions};
use crate::error::{EvolveError, Result};
use crate::judge::build_bundle;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityPolicy {
    #[serde(default)]
    pub gates: Vec<Gate>,
    #[serde(default)]
    pub dimensions: Vec<DimensionRule>,
    #[serde(default)]
    pub threshold: f64,
    #[serde(default = "default_attempts")]
    pub max_attempts: u32,
}

fn default_attempts() -> u32 {
    1
}

impl Default for QualityPolicy {
    fn default() -> Self {
        Self {
            gates: Vec::new(),
            dimensions: Vec::new(),
            threshold: 0.0,
            max_attempts: 1,
        }
    }
}

impl QualityPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.max_attempts == 0 {
            return Err(EvolveError::Config(
                "quality.max_attempts must be positive".into(),
            ));
        }
        if !(0.0..=100.0).contains(&self.threshold) {
            return Err(EvolveError::Config(
                "quality.threshold must be in [0, 100]".into(),
            ));
        }
        for rule in &self.dimensions {
            if rule.name.trim().is_empty() {
                return Err(EvolveError::Config(
                    "quality dimension name must not be empty".into(),
                ));
            }
            if !(0.0..=100.0).contains(&rule.floor) {
                return Err(EvolveError::Config(format!(
                    "quality dimension {:?} floor must be in [0, 100]",
                    rule.name
                )));
            }
            if !rule.weight.is_finite() || rule.weight <= 0.0 {
                return Err(EvolveError::Config(format!(
                    "quality dimension {:?} weight must be positive",
                    rule.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Gate {
    Command {
        name: String,
        program: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default = "default_true")]
        required: bool,
        #[serde(default = "default_timeout")]
        timeout_seconds: u64,
    },
    Scan {
        name: String,
        scan: ScanKind,
        #[serde(default)]
        patterns: Vec<String>,
        #[serde(default = "default_true")]
        required: bool,
    },
}

fn default_true() -> bool {
    true
}
fn default_timeout() -> u64 {
    300
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScanKind {
    Secret,
    Unsafe,
    DebugArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub name: String,
    pub status: GateStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateStatus {
    Pass,
    Fail,
    Warn,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionRule {
    pub name: String,
    pub floor: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    pub name: String,
    pub score: f64,
    pub rationale: String,
}

pub trait RubricJudge {
    fn score(
        &self,
        task_text: &str,
        workspace: &Path,
        dimensions: &[DimensionRule],
    ) -> Result<Vec<DimensionScore>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub gates: Vec<GateResult>,
    pub dimensions: Vec<DimensionScore>,
    pub weighted_score: f64,
    pub accepted: bool,
    pub reasons: Vec<String>,
}

pub fn evaluate_quality(
    workspace: &Path,
    task_text: &str,
    policy: &QualityPolicy,
    rubric: &dyn RubricJudge,
) -> Result<QualityReport> {
    policy.validate()?;
    let gates = policy
        .gates
        .iter()
        .map(|gate| run_gate(workspace, gate))
        .collect::<Result<Vec<_>>>()?;

    let dimensions = if policy.dimensions.is_empty() {
        Vec::new()
    } else {
        let scores = rubric.score(task_text, workspace, &policy.dimensions)?;
        let mut by_name = BTreeMap::new();
        for score in scores {
            by_name.insert(score.name.clone(), score);
        }
        policy
            .dimensions
            .iter()
            .map(|rule| {
                by_name.remove(&rule.name).unwrap_or(DimensionScore {
                    name: rule.name.clone(),
                    score: 0.0,
                    rationale: "missing from rubric response".into(),
                })
            })
            .collect()
    };

    let mut reasons = Vec::new();
    for gate in &gates {
        if gate.status == GateStatus::Fail
            && policy
                .gates
                .iter()
                .find(|item| gate_name(item) == gate.name)
                .map(gate_required)
                .unwrap_or(true)
        {
            reasons.push(format!("gate {:?} failed: {}", gate.name, gate.detail));
        }
    }

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for rule in &policy.dimensions {
        let score = dimensions
            .iter()
            .find(|item| item.name == rule.name)
            .map(|item| item.score)
            .unwrap_or(0.0);
        if score < rule.floor {
            reasons.push(format!(
                "dimension {:?} scored {score:.1}, below floor {:.1}",
                rule.name, rule.floor
            ));
        }
        weighted_sum += score * rule.weight;
        weight_total += rule.weight;
    }
    let weighted_score = if weight_total > 0.0 {
        weighted_sum / weight_total
    } else {
        100.0
    };
    if weighted_score < policy.threshold {
        reasons.push(format!(
            "weighted score {weighted_score:.1} is below threshold {:.1}",
            policy.threshold
        ));
    }

    let accepted = reasons.is_empty();
    Ok(QualityReport {
        gates,
        dimensions,
        weighted_score,
        accepted,
        reasons,
    })
}

fn run_gate(workspace: &Path, gate: &Gate) -> Result<GateResult> {
    match gate {
        Gate::Command {
            name,
            program,
            args,
            required,
            timeout_seconds,
        } => {
            let _ = timeout_seconds;
            let output = Command::new(program)
                .args(args)
                .current_dir(workspace)
                .output()
                .map_err(|error| {
                    EvolveError::Training(format!(
                        "quality gate {:?} could not start {program:?}: {error}",
                        name
                    ))
                })?;
            let detail = format!(
                "exit {:?}\nstdout:\n{}\nstderr:\n{}",
                output.status.code(),
                truncate(&String::from_utf8_lossy(&output.stdout), 4000),
                truncate(&String::from_utf8_lossy(&output.stderr), 4000)
            );
            let status = if output.status.success() {
                GateStatus::Pass
            } else if *required {
                GateStatus::Fail
            } else {
                GateStatus::Warn
            };
            Ok(GateResult {
                name: name.clone(),
                status,
                detail,
            })
        }
        Gate::Scan {
            name,
            scan,
            patterns,
            required,
        } => {
            let mut hits = Vec::new();
            scan_workspace(workspace, *scan, patterns, &mut hits)?;
            let detail = if hits.is_empty() {
                "no matches".into()
            } else {
                truncate(&hits.join("\n"), 4000)
            };
            let status = if hits.is_empty() {
                GateStatus::Pass
            } else if *required {
                GateStatus::Fail
            } else {
                GateStatus::Warn
            };
            Ok(GateResult {
                name: name.clone(),
                status,
                detail,
            })
        }
    }
}

fn scan_workspace(
    root: &Path,
    kind: ScanKind,
    patterns: &[String],
    hits: &mut Vec<String>,
) -> Result<()> {
    let default_patterns: &[&str] = match kind {
        ScanKind::Secret => &["AKIA", "BEGIN PRIVATE KEY", "sk-", "password ="],
        ScanKind::Unsafe => &["unsafe {", "unsafe fn", "transmute("],
        ScanKind::DebugArtifact => &["dbg!(", "println!(\"debug", "console.log(", "TODO"],
    };
    let patterns: Vec<String> = if patterns.is_empty() {
        default_patterns
            .iter()
            .map(|value| value.to_string())
            .collect()
    } else {
        patterns.to_vec()
    };
    walk_text_files(root, &mut |path| {
        let text = fs::read_to_string(path).unwrap_or_default();
        for pattern in &patterns {
            if text.contains(pattern) {
                hits.push(format!("{}: contains {:?}", path.display(), pattern));
            }
        }
    })?;
    Ok(())
}

fn walk_text_files(root: &Path, visit: &mut dyn FnMut(&Path)) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(
            name.as_ref(),
            ".git" | "target" | "node_modules" | "__pycache__"
        ) {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            walk_text_files(&path, visit)?;
        } else if metadata.is_file() && metadata.len() <= 1_000_000 {
            visit(&path);
        }
    }
    Ok(())
}

fn gate_name(gate: &Gate) -> String {
    match gate {
        Gate::Command { name, .. } | Gate::Scan { name, .. } => name.clone(),
    }
}

fn gate_required(gate: &Gate) -> bool {
    match gate {
        Gate::Command { required, .. } | Gate::Scan { required, .. } => *required,
    }
}

fn truncate(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        value.to_string()
    } else {
        format!("{}…", &value[..limit])
    }
}

pub struct ModelRubricJudge<'a> {
    pub chat: &'a dyn ChatClient,
    pub options: ChatOptions,
}

impl RubricJudge for ModelRubricJudge<'_> {
    fn score(
        &self,
        task_text: &str,
        workspace: &Path,
        dimensions: &[DimensionRule],
    ) -> Result<Vec<DimensionScore>> {
        let bundle = build_bundle(workspace, task_text, None)?;
        let names: Vec<String> = dimensions.iter().map(|rule| rule.name.clone()).collect();
        let system =
            "You are a strict software-quality reviewer. Score each dimension from 0 to 100. \
Cite concrete evidence from the workspace. Return valid JSON only with a `scores` array; every \
requested dimension must appear exactly once.";
        let user = format!(
            "# Task\n\n{task_text}\n\n{}\n\n# Dimensions\n\n{}\n\n# Output schema\n\n\
{{\"scores\":[{{\"name\":\"architecture\",\"score\":0,\"rationale\":\"evidence\"}}]}}\n",
            bundle.render("candidate"),
            names.join("\n")
        );
        let raw = self.chat.complete(system, &user, &self.options)?;
        let json = extract_json(&raw)?;
        let value: serde_json::Value = serde_json::from_str(&json).map_err(|error| {
            EvolveError::Judgment(format!("rubric response is not JSON: {error}"))
        })?;
        let rows = value
            .get("scores")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| EvolveError::Judgment("rubric response has no scores".into()))?;
        let mut out = Vec::new();
        for row in rows {
            let name = row
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let score = row
                .get("score")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 100.0);
            let rationale = row
                .get("rationale")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                out.push(DimensionScore {
                    name: name.to_string(),
                    score,
                    rationale,
                });
            }
        }
        Ok(out)
    }
}

fn extract_json(raw: &str) -> Result<String> {
    let text = raw.trim();
    if let Some(start) = text.find("```") {
        let rest = &text[start + 3..];
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(end) = rest.find("```") {
            return Ok(rest[..end].trim().to_string());
        }
    }
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if end > start => Ok(text[start..=end].to_string()),
        _ => Err(EvolveError::Judgment(
            "rubric response has no JSON object".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct FixedRubric(f64);

    impl RubricJudge for FixedRubric {
        fn score(
            &self,
            _task: &str,
            _workspace: &Path,
            dimensions: &[DimensionRule],
        ) -> Result<Vec<DimensionScore>> {
            Ok(dimensions
                .iter()
                .map(|rule| DimensionScore {
                    name: rule.name.clone(),
                    score: self.0,
                    rationale: "fixed".into(),
                })
                .collect())
        }
    }

    fn workspace() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "evolve-quality-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
        root
    }

    #[test]
    fn a_passing_command_gate_and_rubric_accepts() {
        let root = workspace();
        let policy = QualityPolicy {
            gates: vec![Gate::Command {
                name: "true".into(),
                program: "/bin/true".into(),
                args: vec![],
                required: true,
                timeout_seconds: 5,
            }],
            dimensions: vec![DimensionRule {
                name: "architecture".into(),
                floor: 70.0,
                weight: 1.0,
            }],
            threshold: 70.0,
            max_attempts: 1,
        };
        let report = evaluate_quality(&root, "task", &policy, &FixedRubric(90.0)).unwrap();
        assert!(report.accepted);
        assert_eq!(report.weighted_score, 90.0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failing_required_gate_rejects() {
        let root = workspace();
        let policy = QualityPolicy {
            gates: vec![Gate::Command {
                name: "false".into(),
                program: "/bin/false".into(),
                args: vec![],
                required: true,
                timeout_seconds: 5,
            }],
            ..QualityPolicy::default()
        };
        let report = evaluate_quality(&root, "task", &policy, &FixedRubric(100.0)).unwrap();
        assert!(!report.accepted);
        assert!(report.reasons.iter().any(|reason| reason.contains("false")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_low_dimension_rejects_even_with_passing_gates() {
        let root = workspace();
        let policy = QualityPolicy {
            dimensions: vec![DimensionRule {
                name: "security".into(),
                floor: 80.0,
                weight: 1.0,
            }],
            threshold: 50.0,
            ..QualityPolicy::default()
        };
        let report = evaluate_quality(&root, "task", &policy, &FixedRubric(40.0)).unwrap();
        assert!(!report.accepted);
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("security")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_secret_scan_rejects() {
        let root = workspace();
        fs::write(root.join("config.txt"), "AKIAIOSFODNN7EXAMPLE\n").unwrap();
        let policy = QualityPolicy {
            gates: vec![Gate::Scan {
                name: "secrets".into(),
                scan: ScanKind::Secret,
                patterns: vec![],
                required: true,
            }],
            ..QualityPolicy::default()
        };
        let report = evaluate_quality(&root, "task", &policy, &FixedRubric(100.0)).unwrap();
        assert!(!report.accepted);
        fs::remove_dir_all(root).unwrap();
    }
}
