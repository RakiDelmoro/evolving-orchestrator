//! One cycle: search, collect, rank, select, render, train, export.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::collect::collect;
use crate::config::CycleConfig;
use crate::design::TaskCatalog;
use crate::error::{EvolveError, Result};
use crate::exec::EpisodeRunner;
use crate::export::{ExportRequest, Exporter};
use crate::io::write_json;
use crate::judge::PairwiseJudge;
use crate::pi_session::load_sessions;
use crate::render::{
    render_sessions, write_windows, PromptSubstitution, RenderConfig, SftWindow, Tokenizer,
};
use crate::search::{run_search, Proposer, SearchOutcome, SearchPaths};
use crate::select::{rank_candidates, select_rows, Candidate, RankedEpisode};
use crate::train::{Checkpoint, TrainRequest, Trainer};

#[derive(Debug, Clone)]
pub struct CyclePaths {
    pub run_dir: PathBuf,
    pub tasks_dir: PathBuf,
    pub teacher_directive: PathBuf,
}

pub struct CyclePorts<'a> {
    pub runner: &'a dyn EpisodeRunner,
    pub judge: &'a dyn PairwiseJudge,
    pub proposer: &'a dyn Proposer,
    pub tokenizer: &'a dyn Tokenizer,
    pub trainer: &'a dyn Trainer,
    pub exporter: &'a dyn Exporter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleOptions {
    pub train: bool,
}

impl Default for CycleOptions {
    fn default() -> Self {
        Self { train: true }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CycleSummary {
    pub generation: u32,
    pub tasks: Vec<TaskSummary>,
    pub episodes: usize,
    pub windows: usize,
    pub checkpoint_step: Option<u32>,
    pub exported: bool,
    pub trained: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSummary {
    pub task: u32,
    pub retained_version: u32,
    pub beats_reference: bool,
    pub collected: usize,
    pub ranked: usize,
}

pub struct CycleOutcome {
    pub searches: Vec<(u32, SearchOutcome)>,
    pub selections: Vec<(u32, Vec<RankedEpisode>)>,
    pub train_rows: Vec<RankedEpisode>,
    pub validation_rows: Vec<RankedEpisode>,
    pub windows: usize,
    pub checkpoint: Option<Checkpoint>,
    pub exported: bool,
    pub summary: CycleSummary,
}

pub fn run_cycle(
    config: &CycleConfig,
    catalog: &TaskCatalog,
    paths: &CyclePaths,
    ports: &CyclePorts<'_>,
) -> Result<CycleOutcome> {
    run_cycle_with_options(config, catalog, paths, ports, CycleOptions::default())
}

pub fn run_cycle_with_options(
    config: &CycleConfig,
    catalog: &TaskCatalog,
    paths: &CyclePaths,
    ports: &CyclePorts<'_>,
    options: CycleOptions,
) -> Result<CycleOutcome> {
    let teacher_directive = fs::read_to_string(&paths.teacher_directive).map_err(|error| {
        EvolveError::Config(format!(
            "cannot read teacher directive {}: {error}",
            paths.teacher_directive.display()
        ))
    })?;

    let mut searches = Vec::new();
    let mut selections = Vec::new();
    let mut task_summaries = Vec::new();
    let mut train_rows: Vec<RankedEpisode> = Vec::new();
    let mut validation_rows: Vec<RankedEpisode> = Vec::new();
    let mut train_windows: Vec<SftWindow> = Vec::new();
    let mut validation_windows: Vec<SftWindow> = Vec::new();
    let mut episode_count = 0usize;

    for id in &config.tasks.train {
        let task = catalog.load(*id, "software")?;
        let parsed = catalog.parse(*id)?;
        let search_paths = SearchPaths::new(&paths.run_dir, parsed.query_name.clone());
        let outcome = run_search(
            config,
            &task,
            &parsed,
            &search_paths,
            ports.runner,
            ports.judge,
            ports.proposer,
        )?;
        searches.push((*id, outcome.clone()));
        task_summaries.push(TaskSummary {
            task: *id,
            retained_version: outcome.retained_version,
            beats_reference: outcome.beats_reference,
            collected: 0,
            ranked: 0,
        });
        if !outcome.beats_reference {
            continue;
        }

        let collection = collect(
            config,
            &task,
            &outcome.champion_design,
            &teacher_directive,
            &paths.run_dir,
            ports.runner,
        )?;
        episode_count += collection.rows.len();
        if let Some(summary) = task_summaries.last_mut() {
            summary.collected = collection.rows.len();
        }

        let candidates: Vec<Candidate> = collection
            .rows
            .iter()
            .map(|row| Candidate {
                name: row.episode.clone(),
                task: row.task,
                condition: row.condition.clone(),
                seed: row.seed,
                workspace: row.workspace.clone(),
                logs_dir: row.logs_dir.clone(),
                teacher_prompt: Some(row.teacher_prompt.clone()),
            })
            .collect();
        let ranked = rank_candidates(
            ports.judge,
            &task.task_text,
            &candidates,
            config.ranking.opponents_per_episode as usize,
            config.ranking.seed.wrapping_add(u64::from(*id) * 1000),
            500,
        )?;
        let selection = select_rows(
            &ranked,
            config.collection.train_ranks,
            config.collection.validation_rank,
        )?;
        if let Some(summary) = task_summaries.last_mut() {
            summary.ranked = ranked.len();
        }
        let selection_dir = paths.run_dir.join("selection");
        fs::create_dir_all(&selection_dir)?;
        write_json(
            &selection_dir.join(format!("task{id}_ranked.json")),
            &ranked,
        )?;

        if let Some(best) = ranked.first() {
            publish_workspace(&paths.run_dir, *id, best, outcome.retained_version)?;
        }

        train_rows.extend(selection.train.iter().cloned());
        validation_rows.push(selection.validation.clone());
        selections.push((*id, ranked));

        if options.train {
            let render_config = RenderConfig {
                window: config.training.window as usize,
                context_overlap: config.training.context_overlap as usize,
            };
            for row in selection
                .train
                .iter()
                .chain(std::iter::once(&selection.validation))
            {
                let teacher_prompt = row
                    .teacher_prompt
                    .as_ref()
                    .ok_or_else(|| {
                        EvolveError::Training(format!("episode {} has no teacher prompt", row.name))
                    })
                    .and_then(|path| fs::read_to_string(path).map_err(EvolveError::from))?;
                let substitution = PromptSubstitution {
                    from: teacher_prompt,
                    to: task.prompt.trim().to_string(),
                };
                let sessions = load_sessions(&row.logs_dir.join("sessions"))?;
                let windows = render_sessions(
                    &sessions,
                    ports.tokenizer,
                    render_config,
                    Some(&substitution),
                    &row.name,
                    *id,
                )?;
                if windows.is_empty() {
                    return Err(EvolveError::Training(format!(
                        "episode {} produced no windows",
                        row.name
                    )));
                }
                if selection
                    .train
                    .iter()
                    .any(|candidate| candidate.name == row.name)
                {
                    train_windows.extend(windows);
                } else {
                    validation_windows.extend(windows);
                }
            }
        }
    }

    if train_rows.is_empty() || validation_rows.is_empty() {
        return Err(EvolveError::Training(
            "no eligible training or validation episodes".into(),
        ));
    }

    let selection_dir = paths.run_dir.join("selection");
    write_json(&selection_dir.join("train.json"), &train_rows)?;
    write_json(&selection_dir.join("val.json"), &validation_rows)?;

    if !options.train {
        let summary = CycleSummary {
            generation: config.generation,
            tasks: task_summaries,
            episodes: episode_count,
            windows: 0,
            checkpoint_step: None,
            exported: false,
            trained: false,
        };
        write_json(&paths.run_dir.join("cycle_summary.json"), &summary)?;
        return Ok(CycleOutcome {
            searches,
            selections,
            train_rows,
            validation_rows,
            windows: 0,
            checkpoint: None,
            exported: false,
            summary,
        });
    }

    let data_dir = paths.run_dir.join("data");
    fs::create_dir_all(&data_dir)?;
    let train_file = data_dir.join("train.jsonl");
    let validation_file = data_dir.join("val.jsonl");
    write_windows(&train_file, &train_windows)?;
    write_windows(&validation_file, &validation_windows)?;

    let outcome = ports.trainer.train(&TrainRequest {
        run_dir: paths.run_dir.clone(),
        train_file,
        validation_file,
        parent_model: config.parent_bf16.clone(),
        config: config.training.clone(),
    })?;
    if let Some(checkpoint) = &outcome.checkpoint {
        write_json(&paths.run_dir.join("selected_checkpoint.json"), checkpoint)?;
    }

    let mut exported = false;
    if let Some(checkpoint) = &outcome.checkpoint {
        let base_model = config
            .parent_bf16
            .clone()
            .ok_or_else(|| EvolveError::Training("parent_bf16 is required for export".into()))?;
        ports.exporter.export(&ExportRequest {
            base_model,
            adapter: checkpoint.path.clone(),
            out_dir: paths.run_dir.join("exported"),
            lora_r: config.training.lora_r,
            lora_alpha: config.training.lora_alpha,
            quantize: true,
        })?;
        exported = true;
    }

    let summary = CycleSummary {
        generation: config.generation,
        tasks: task_summaries,
        episodes: episode_count,
        windows: train_windows.len() + validation_windows.len(),
        checkpoint_step: outcome.checkpoint.as_ref().map(|value| value.step),
        exported,
        trained: true,
    };
    write_json(&paths.run_dir.join("cycle_summary.json"), &summary)?;

    Ok(CycleOutcome {
        searches,
        selections,
        train_rows,
        validation_rows,
        windows: train_windows.len() + validation_windows.len(),
        checkpoint: outcome.checkpoint,
        exported,
        summary,
    })
}

fn publish_workspace(
    run_dir: &Path,
    task: u32,
    best: &RankedEpisode,
    retained_version: u32,
) -> Result<()> {
    let output = run_dir.join("outputs").join(format!("query{task}"));
    let workspace = output.join("workspace");
    if workspace.exists() {
        fs::remove_dir_all(&workspace)?;
    }
    copy_dir(&best.workspace, &workspace)?;
    write_json(
        &output.join("manifest.json"),
        &serde_json::json!({
            "task": task,
            "source_episode": best.name,
            "rank": best.rank,
            "score": best.score,
            "retained_version": retained_version,
            "workspace": workspace,
        }),
    )?;
    Ok(())
}

fn copy_dir(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let destination = target.join(entry.file_name());
        if entry.metadata()?.is_dir() {
            copy_dir(&path, &destination)?;
        } else {
            fs::copy(&path, &destination)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatOptions;
    use crate::design::{parse_query_text, Task};
    use crate::exec::{EpisodeOutcome, EpisodeSpec, EpisodeStatus};
    use crate::export::Exporter;
    use crate::judge::{Verdict, Winner, WorkspaceBundle};
    use crate::prompt::ProposalInput;
    use crate::train::{TrainOutcome, TrainRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeRunner;
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    impl EpisodeRunner for FakeRunner {
        fn run(&self, spec: &EpisodeSpec<'_>) -> Result<EpisodeOutcome> {
            fs::create_dir_all(spec.workspace)?;
            fs::create_dir_all(spec.logs.join("sessions"))?;
            fs::write(spec.workspace.join("result.md"), "ok\n")?;
            let session = r#"{"type":"session","version":3,"id":"s"}
{"type":"message","message":{"role":"system","content":"sys"}}
{"type":"message","message":{"role":"user","content":"teacher prompt"}}
{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"done"}],"stopReason":"stop"}}
"#;
            fs::write(spec.logs.join("sessions/s.jsonl"), session)?;
            COUNTER.fetch_add(1, Ordering::SeqCst);
            Ok(EpisodeOutcome {
                status: EpisodeStatus::Success,
                exit_code: Some(0),
                duration: std::time::Duration::from_millis(1),
                prompt_path: spec.logs.join("prompt.txt"),
                stream_path: spec.logs.join("stream.jsonl"),
                stderr_path: spec.logs.join("stderr.log"),
                session_dir: spec.logs.join("sessions"),
                session_files: vec![spec.logs.join("sessions/s.jsonl")],
            })
        }
    }

    struct FakeJudge;

    impl PairwiseJudge for FakeJudge {
        fn compare(
            &self,
            _task: &str,
            _a: &WorkspaceBundle,
            _b: &WorkspaceBundle,
            _seed: u64,
        ) -> Result<Verdict> {
            Ok(Verdict {
                winner: Winner::B,
                rationale: "b".into(),
            })
        }
    }

    struct FakeProposer;

    impl Proposer for FakeProposer {
        fn propose(&self, input: &ProposalInput<'_>) -> Result<String> {
            Ok(format!(
                "{}\nnext\n{}",
                input.start_marker, input.end_marker
            ))
        }
    }

    struct FakeTrainer;

    impl Trainer for FakeTrainer {
        fn train(&self, request: &TrainRequest) -> Result<TrainOutcome> {
            let training = request.run_dir.join("training");
            let checkpoint = training.join("step_00001");
            fs::create_dir_all(&checkpoint)?;
            fs::write(checkpoint.join("adapter.safetensors"), "adapter")?;
            fs::write(
                training.join("train_log.jsonl"),
                "{\"step\":1,\"val_loss\":0.5,\"val_tokens\":10,\"checkpoint\":\"step_00001\"}\n",
            )?;
            Ok(TrainOutcome {
                entries: Vec::new(),
                checkpoint: Some(Checkpoint {
                    step: 1,
                    val_loss: 0.5,
                    path: checkpoint,
                }),
            })
        }
    }

    struct FakeExporter;

    impl Exporter for FakeExporter {
        fn export(&self, request: &ExportRequest) -> Result<()> {
            fs::create_dir_all(&request.out_dir)?;
            fs::write(request.out_dir.join("READY"), "ok")?;
            Ok(())
        }
    }

    fn config(root: &Path, parent: &Path) -> CycleConfig {
        let value = serde_json::json!({
            "generation": 0,
            "run_dir": root.display().to_string(),
            "model_id": "local",
            "parent_bf16": parent.display().to_string(),
            "endpoint": "http://127.0.0.1:8080/v1",
            "tasks": {"train_candidates": [1], "train": [1], "test": [], "excluded": []},
            "search": {"iterations": 1, "seed_base": 1, "rule": "retained-best"},
            "collection": {"candidates": 2, "train_ranks": 1, "validation_rank": 2, "seed_base": 100},
            "ranking": {"opponents_per_episode": 2, "seed": 1},
            "training": {
                "window": 256, "context_overlap": 8, "steps": 1, "lr": 0.0001,
                "warmup_steps": 1, "min_lr_ratio": 0.1, "lora_r": 4, "lora_alpha": 8,
                "lora_dropout": 0.05, "save_every": 1, "val_every": 1,
                "replicas": 1, "gpus_per_rank": 1, "grad_accum": 3,
                "ce_chunk": 8, "seed": 1
            }
        });
        CycleConfig::parse(value, "test").expect("config")
    }

    #[test]
    fn runs_one_cycle_end_to_end_with_fakes() {
        let root = std::env::temp_dir().join(format!(
            "evolve-cycle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let tasks_dir = root.join("tasks");
        fs::create_dir_all(tasks_dir.join("bare")).unwrap();
        fs::create_dir_all(tasks_dir.join("initial")).unwrap();
        fs::write(tasks_dir.join("bare/query1.txt"), "teacher prompt\n").unwrap();
        fs::write(
            tasks_dir.join("initial/query1_team.txt"),
            "teacher prompt\n\nCreate an agent team with following agent candidates to solve this problem:\ndesign\n\nUse uv for all Python workflows\n",
        )
        .unwrap();
        fs::write(root.join("directive.txt"), "run the team").unwrap();
        fs::create_dir_all(root.join("parent")).unwrap();
        let config = config(&root.join("run"), &root.join("parent"));
        let catalog = TaskCatalog::new(&tasks_dir);
        let steps = crate::render::ByteTokenizer;
        let outcome = run_cycle(
            &config,
            &catalog,
            &CyclePaths {
                run_dir: config.run_dir.clone(),
                tasks_dir,
                teacher_directive: root.join("directive.txt"),
            },
            &CyclePorts {
                runner: &FakeRunner,
                judge: &FakeJudge,
                proposer: &FakeProposer,
                tokenizer: &steps,
                trainer: &FakeTrainer,
                exporter: &FakeExporter,
            },
        )
        .expect("cycle");
        assert!(outcome.exported);
        assert!(outcome.checkpoint.is_some());
        assert!(config.run_dir.join("outputs/query1/workspace").is_dir());
        assert!(config.run_dir.join("data/train.jsonl").is_file());
        assert!(config.run_dir.join("cycle_summary.json").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn placeholder_task_imports() {
        let _ = parse_query_text(
            "t\nCreate an agent team with following agent candidates to solve this problem:\nd\nUse uv for all Python workflows\n",
            Path::new("query1_team.txt"),
            crate::design::START_MARKER,
            crate::design::END_MARKER,
        )
        .unwrap();
        let _: Task = Task {
            id: 1,
            domain: "software".into(),
            prompt: "t".into(),
            initial_design: "d".into(),
            task_text: "t".into(),
        };
        let _ = ChatOptions::default();
    }
}
