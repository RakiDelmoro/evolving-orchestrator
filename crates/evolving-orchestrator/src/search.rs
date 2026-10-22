//! The inner loop: run a design, judge it, revise it, keep the champion.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::champion::{derive_champion, derive_champion_baseline};
use crate::chat::{ChatClient, ChatOptions};
use crate::config::{CycleConfig, SelectionRule};
use crate::design::{ParsedQuery, Task, END_MARKER, START_MARKER};
use crate::error::{EvolveError, Result};
use crate::exec::{EpisodeRunner, EpisodeSpec};
use crate::history::{append_history, read_history, HistoryRow};
use crate::io::write_json;
use crate::judge::{build_bundle, PairwiseJudge, Winner};
use crate::prompt::{
    build_prompt, extract_json_field, normalize_design, BestDesign, ProposalInput,
};

#[derive(Debug, Clone)]
pub struct SearchPaths {
    pub run_dir: PathBuf,
    pub query_name: String,
}

impl SearchPaths {
    pub fn new(run_dir: impl Into<PathBuf>, query_name: impl Into<String>) -> Self {
        Self {
            run_dir: run_dir.into(),
            query_name: query_name.into(),
        }
    }

    pub fn search_dir(&self) -> PathBuf {
        self.run_dir.join("search").join(&self.query_name)
    }

    pub fn versions_dir(&self) -> PathBuf {
        self.search_dir().join("versions")
    }

    pub fn version_path(&self, version: u32) -> PathBuf {
        self.versions_dir()
            .join(format!("multi_agent_design_v{version}.txt"))
    }

    pub fn updated_dir(&self) -> PathBuf {
        self.search_dir().join("updated_queries")
    }

    pub fn updated_path(&self, version: u32) -> PathBuf {
        self.updated_dir()
            .join(format!("{}-v{version}.txt", self.query_name))
    }

    pub fn records_dir(&self) -> PathBuf {
        self.search_dir().join("records")
    }

    pub fn history_path(&self) -> PathBuf {
        self.records_dir().join("comparison_history.jsonl")
    }

    pub fn champion_history_path(&self) -> PathBuf {
        self.records_dir().join("champion_history.jsonl")
    }

    pub fn retained_path(&self) -> PathBuf {
        self.search_dir().join("retained_design.json")
    }

    pub fn workspace(&self, version: u32) -> PathBuf {
        self.run_dir
            .join("workspaces")
            .join(format!("v{version}"))
            .join(&self.query_name)
    }

    pub fn logs(&self, version: u32) -> PathBuf {
        self.run_dir
            .join("logs")
            .join("search")
            .join(format!("v{version}"))
            .join(&self.query_name)
    }

    pub fn reference_workspace(&self) -> PathBuf {
        self.run_dir.join("reference").join(&self.query_name)
    }

    pub fn reference_logs(&self) -> PathBuf {
        self.run_dir
            .join("logs")
            .join("reference")
            .join(&self.query_name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetainedDesign {
    pub version: u32,
    pub beats_reference: bool,
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub retained_version: u32,
    pub beats_reference: bool,
    pub champion_workspace: PathBuf,
    pub champion_design: String,
    pub versions: Vec<String>,
    pub history: Vec<HistoryRow>,
    pub champion_history: Vec<HistoryRow>,
}

pub trait Proposer {
    fn propose(&self, input: &ProposalInput<'_>) -> Result<String>;
}

pub struct ModelProposer<'a> {
    pub chat: &'a dyn ChatClient,
    pub options: ChatOptions,
}

impl Proposer for ModelProposer<'_> {
    fn propose(&self, input: &ProposalInput<'_>) -> Result<String> {
        let (system, user) = build_prompt(input);
        let raw = self.chat.complete(&system, &user, &self.options)?;
        let design = extract_json_field(&raw, "improved_multi_agent_design")?;
        normalize_design(&design, input.start_marker, input.end_marker)
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, Default)]
pub struct SearchSeed<'a> {
    pub initial_design: Option<&'a str>,
    pub base_workspace: Option<&'a Path>,
    pub gate_feedback: Option<&'a str>,
}

pub fn run_search(
    config: &CycleConfig,
    task: &Task,
    parsed: &ParsedQuery,
    paths: &SearchPaths,
    runner: &dyn EpisodeRunner,
    judge: &dyn PairwiseJudge,
    proposer: &dyn Proposer,
) -> Result<SearchOutcome> {
    run_search_seeded(
        config,
        task,
        parsed,
        paths,
        runner,
        judge,
        proposer,
        &SearchSeed::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_search_seeded(
    config: &CycleConfig,
    task: &Task,
    parsed: &ParsedQuery,
    paths: &SearchPaths,
    runner: &dyn EpisodeRunner,
    judge: &dyn PairwiseJudge,
    proposer: &dyn Proposer,
    seed: &SearchSeed<'_>,
) -> Result<SearchOutcome> {
    fs::create_dir_all(paths.versions_dir())?;
    fs::create_dir_all(paths.updated_dir())?;
    fs::create_dir_all(paths.records_dir())?;

    if !paths.version_path(0).is_file() {
        fs::write(paths.version_path(0), parsed.design.trim())?;
    }
    if !paths.updated_path(0).is_file() {
        fs::write(
            paths.updated_path(0),
            parsed.render_with_design(&parsed.design),
        )?;
    }

    let reference_condition = format!("t{}_bare", task.id);
    let reference = runner.run(&EpisodeSpec {
        prompt: &task.prompt,
        workspace: &paths.reference_workspace(),
        logs: &paths.reference_logs(),
        condition: &reference_condition,
        seed: config.search.seed_base + u64::from(task.id),
        model_id: &config.model_id,
        endpoint: &config.endpoint,
        base_workspace: seed.base_workspace,
    })?;
    if !reference.is_success() {
        return Err(EvolveError::Episode(format!(
            "task {}: baseline run failed ({:?})",
            task.id, reference.status
        )));
    }

    let mut versions: Vec<String> = Vec::new();
    let mut history: Vec<HistoryRow> = Vec::new();
    let mut champion_history: Vec<HistoryRow> = Vec::new();
    let mut champion_version = 0u32;
    let mut champion_has_win = false;

    for k in 0..=config.search.iterations {
        let design = if k == 0 {
            seed.initial_design
                .unwrap_or(&parsed.design)
                .trim()
                .to_string()
        } else {
            fs::read_to_string(paths.version_path(k)).map_err(|error| {
                EvolveError::Design(format!("missing design v{k} for task {}: {error}", task.id))
            })?
        };
        versions.push(design.clone());

        let prompt_text = if k == 0 {
            parsed.render_with_design(&design)
        } else {
            fs::read_to_string(paths.updated_path(k))
                .unwrap_or_else(|_| parsed.render_with_design(&design))
        };
        if !paths.updated_path(k).is_file() {
            fs::write(paths.updated_path(k), &prompt_text)?;
        }

        let condition = format!("t{}_v{k}", task.id);
        let outcome = runner.run(&EpisodeSpec {
            prompt: &prompt_text,
            workspace: &paths.workspace(k),
            logs: &paths.logs(k),
            condition: &condition,
            seed: config.search.seed_base + u64::from(task.id) + u64::from(k) * 1000,
            model_id: &config.model_id,
            endpoint: &config.endpoint,
            base_workspace: seed.base_workspace,
        })?;
        if !outcome.is_success() {
            return Err(EvolveError::Episode(format!(
                "task {}: design v{k} failed ({:?})",
                task.id, outcome.status
            )));
        }

        let judge_seed = config
            .ranking
            .seed
            .wrapping_add(u64::from(task.id) * 1000)
            .wrapping_add(u64::from(k));

        match config.search.rule {
            SelectionRule::RetainedBest => {
                if k > 0 {
                    let verdict = compare_workspaces(
                        judge,
                        &task.task_text,
                        &paths.workspace(k - 1),
                        &paths.logs(k - 1),
                        &paths.workspace(k),
                        &paths.logs(k),
                        judge_seed,
                    )?;
                    let row = HistoryRow::adjacent(
                        k - 1,
                        k,
                        verdict.winner,
                        verdict.rationale,
                        config.model_id.clone(),
                        paths.workspace(k - 1).display().to_string(),
                        paths.workspace(k).display().to_string(),
                    );
                    append_history(&paths.history_path(), &row)?;
                    history.push(row);
                }
                let (champion, needs) = derive_champion(&history, &champion_history, k);
                champion_version = champion;
                if needs {
                    let verdict = compare_workspaces(
                        judge,
                        &task.task_text,
                        &paths.workspace(champion_version),
                        &paths.logs(champion_version),
                        &paths.workspace(k),
                        &paths.logs(k),
                        judge_seed,
                    )?;
                    let row = HistoryRow::champion(
                        champion_version,
                        k,
                        verdict.winner,
                        verdict.rationale,
                        config.model_id.clone(),
                        paths.workspace(champion_version).display().to_string(),
                        paths.workspace(k).display().to_string(),
                    );
                    append_history(&paths.champion_history_path(), &row)?;
                    if verdict.winner == Winner::B {
                        champion_version = k;
                    }
                    champion_history.push(row);
                }
            }
            SelectionRule::FixedReference => {
                let verdict = compare_workspaces(
                    judge,
                    &task.task_text,
                    &paths.reference_workspace(),
                    &paths.reference_logs(),
                    &paths.workspace(k),
                    &paths.logs(k),
                    judge_seed,
                )?;
                let row = HistoryRow::baseline(
                    k,
                    verdict.winner,
                    verdict.rationale,
                    config.model_id.clone(),
                    paths.reference_workspace().display().to_string(),
                    paths.workspace(k).display().to_string(),
                );
                append_history(&paths.history_path(), &row)?;
                history.push(row);

                let (champion, has_win, needs) =
                    derive_champion_baseline(&history, &champion_history, k);
                champion_version = champion;
                champion_has_win = has_win;
                if needs {
                    let verdict = compare_workspaces(
                        judge,
                        &task.task_text,
                        &paths.workspace(champion_version),
                        &paths.logs(champion_version),
                        &paths.workspace(k),
                        &paths.logs(k),
                        judge_seed,
                    )?;
                    let row = HistoryRow::champion(
                        champion_version,
                        k,
                        verdict.winner,
                        verdict.rationale,
                        config.model_id.clone(),
                        paths.workspace(champion_version).display().to_string(),
                        paths.workspace(k).display().to_string(),
                    );
                    append_history(&paths.champion_history_path(), &row)?;
                    if verdict.winner == Winner::B {
                        champion_version = k;
                        champion_has_win = true;
                    }
                    champion_history.push(row);
                }
            }
        }

        if k < config.search.iterations {
            let bundle = build_bundle(&paths.workspace(k), &task.task_text, Some(&paths.logs(k)))?;
            let mut evidence = bundle.render("candidate");
            if let Some(feedback) = seed.gate_feedback {
                if !feedback.trim().is_empty() {
                    evidence.push_str("\n\n# Previous quality failures\n\n");
                    evidence.push_str(feedback.trim());
                }
            }
            let best_rationale = history
                .iter()
                .rev()
                .find(|row| row.key.contains(&format!("v{champion_version}")))
                .map(|row| row.rationale.as_str())
                .unwrap_or("");
            let best = if champion_version != k {
                Some(BestDesign {
                    version: champion_version,
                    design: versions[champion_version as usize].as_str(),
                    rationale: best_rationale,
                })
            } else {
                None
            };
            let input = ProposalInput {
                task_text: &task.task_text,
                current_version: k,
                current_design: &design,
                best,
                v0_design: if k > 0 {
                    Some(versions[0].as_str())
                } else {
                    None
                },
                evidence: &evidence,
                execution_metadata: Some(&format!(
                    "exit={:?} duration_ms={}",
                    outcome.exit_code,
                    outcome.duration.as_millis()
                )),
                history: &history,
                baseline_mode: config.search.rule == SelectionRule::FixedReference,
                scoreboard: None,
                start_marker: START_MARKER,
                end_marker: END_MARKER,
            };
            let next_design = proposer.propose(&input)?;
            fs::write(paths.version_path(k + 1), &next_design)?;
            fs::write(
                paths.updated_path(k + 1),
                parsed.render_with_design(&next_design),
            )?;
        }
    }

    let beats_reference = match config.search.rule {
        SelectionRule::RetainedBest => champion_version > 0,
        SelectionRule::FixedReference => champion_has_win,
    };
    write_json(
        &paths.retained_path(),
        &RetainedDesign {
            version: champion_version,
            beats_reference,
        },
    )?;

    Ok(SearchOutcome {
        retained_version: champion_version,
        beats_reference,
        champion_workspace: paths.workspace(champion_version),
        champion_design: versions
            .get(champion_version as usize)
            .cloned()
            .unwrap_or_default(),
        versions,
        history,
        champion_history,
    })
}

fn compare_workspaces(
    judge: &dyn PairwiseJudge,
    task_text: &str,
    a_workspace: &Path,
    a_logs: &Path,
    b_workspace: &Path,
    b_logs: &Path,
    seed: u64,
) -> Result<crate::judge::Verdict> {
    let a = build_bundle(a_workspace, task_text, Some(a_logs))?;
    let b = build_bundle(b_workspace, task_text, Some(b_logs))?;
    judge.compare(task_text, &a, &b, seed)
}

/// Read a retained design written by search.
pub fn read_retained(paths: &SearchPaths) -> Result<RetainedDesign> {
    crate::io::read_json(&paths.retained_path())
}

/// Read the comparison history, if any.
pub fn read_search_history(paths: &SearchPaths) -> Result<Vec<HistoryRow>> {
    read_history(&paths.history_path())
}

/// A convenience wrapper for tests and external callers that want the
/// champion's design text.
pub fn champion_design(paths: &SearchPaths) -> Result<String> {
    let retained = read_retained(paths)?;
    Ok(fs::read_to_string(paths.version_path(retained.version))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CollectionConfig, ExecutorConfig, JudgeConfig, RankingConfig, SearchConfig, TaskSplit,
        TrainingConfig,
    };
    use crate::design::parse_query_text;
    use crate::exec::{EpisodeOutcome, EpisodeStatus};
    use crate::judge::{Verdict, WorkspaceBundle};

    struct FakeRunner;

    impl EpisodeRunner for FakeRunner {
        fn run(&self, spec: &EpisodeSpec<'_>) -> Result<EpisodeOutcome> {
            fs::create_dir_all(spec.workspace)?;
            fs::create_dir_all(spec.logs)?;
            fs::write(spec.workspace.join("result.md"), "ok\n")?;
            Ok(EpisodeOutcome {
                status: EpisodeStatus::Success,
                exit_code: Some(0),
                duration: std::time::Duration::from_millis(1),
                prompt_path: spec.logs.join("prompt.txt"),
                stream_path: spec.logs.join("stream.jsonl"),
                stderr_path: spec.logs.join("stderr.log"),
                session_dir: spec.logs.join("sessions"),
                session_files: Vec::new(),
            })
        }
    }

    struct CurrentWins;

    impl PairwiseJudge for CurrentWins {
        fn compare(
            &self,
            _task: &str,
            _a: &WorkspaceBundle,
            _b: &WorkspaceBundle,
            _seed: u64,
        ) -> Result<Verdict> {
            Ok(Verdict {
                winner: Winner::B,
                rationale: "current is better".into(),
            })
        }
    }

    struct FakeProposer;

    impl Proposer for FakeProposer {
        fn propose(&self, input: &ProposalInput<'_>) -> Result<String> {
            Ok(format!(
                "{}\nnext v{}\n{}",
                input.start_marker,
                input.current_version + 1,
                input.end_marker
            ))
        }
    }

    fn config(rule: SelectionRule) -> CycleConfig {
        CycleConfig {
            generation: 0,
            run_dir: PathBuf::from("runs/L0"),
            model_id: "local".into(),
            parent_bf16: None,
            fp8_reference: None,
            next_bf16: None,
            next_fp8: None,
            endpoint: "http://127.0.0.1:8080/v1".into(),
            proxy_port: None,
            tasks: TaskSplit {
                train_candidates: vec![1],
                train: vec![1],
                test: vec![],
                excluded: vec![],
            },
            search: SearchConfig {
                iterations: 2,
                seed_base: 100,
                rule,
            },
            collection: CollectionConfig {
                candidates: 2,
                train_ranks: 1,
                validation_rank: 2,
                seed_base: 200,
            },
            ranking: RankingConfig {
                opponents_per_episode: 2,
                seed: 7,
            },
            training: TrainingConfig {
                window: 64,
                context_overlap: 8,
                steps: 1,
                lr: 1e-5,
                warmup_steps: 1,
                min_lr_ratio: 0.1,
                lora_r: 4,
                lora_alpha: 8,
                lora_dropout: 0.05,
                save_every: 1,
                val_every: 1,
                replicas: 1,
                gpus_per_rank: 1,
                grad_accum: 3,
                ce_chunk: 8,
                seed: 1,
            },
            executor: ExecutorConfig::default(),
            judge: JudgeConfig::default(),
            quality: Default::default(),
        }
    }

    fn task() -> Task {
        Task {
            id: 1,
            domain: "software".into(),
            prompt: "fix the test".into(),
            initial_design: "START\ndesign\nEND".into(),
            task_text: "fix the test".into(),
        }
    }

    fn parsed() -> ParsedQuery {
        parse_query_text(
            "fix the test\n\nCreate an agent team with following agent candidates to solve this problem:\ndesign\n\nUse uv for all Python workflows\n",
            Path::new("query1_team.txt"),
            START_MARKER,
            END_MARKER,
        )
        .expect("parse")
    }

    #[test]
    fn searches_three_designs_and_retains_the_last() {
        let root = std::env::temp_dir().join(format!(
            "evolve-search-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = config(SelectionRule::RetainedBest);
        let paths = SearchPaths::new(&root, "query1_team");
        let outcome = run_search(
            &config,
            &task(),
            &parsed(),
            &paths,
            &FakeRunner,
            &CurrentWins,
            &FakeProposer,
        )
        .expect("search");
        assert_eq!(outcome.versions.len(), 3);
        assert_eq!(outcome.retained_version, 2);
        assert_eq!(outcome.history.len(), 2);
        assert!(outcome.beats_reference);
        assert!(outcome.champion_design.contains("next v2"));
        assert!(paths.retained_path().is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fixed_reference_counts_a_baseline_win() {
        let root = std::env::temp_dir().join(format!(
            "evolve-search-fixed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = config(SelectionRule::FixedReference);
        let paths = SearchPaths::new(&root, "query1_team");
        let outcome = run_search(
            &config,
            &task(),
            &parsed(),
            &paths,
            &FakeRunner,
            &CurrentWins,
            &FakeProposer,
        )
        .expect("search");
        assert!(outcome.beats_reference);
        assert_eq!(outcome.history.len(), 3);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paths_are_stable() {
        let paths = SearchPaths::new("runs/L0", "query1_team");
        assert_eq!(
            paths.version_path(2),
            PathBuf::from("runs/L0/search/query1_team/versions/multi_agent_design_v2.txt")
        );
        assert_eq!(
            paths.workspace(1),
            PathBuf::from("runs/L0/workspaces/v1/query1_team")
        );
    }
}
