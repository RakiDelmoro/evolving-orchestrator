//! Teacher collection: run the retained design several times.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::CycleConfig;
use crate::design::Task;
use crate::error::{EvolveError, Result};
use crate::exec::{EpisodeRunner, EpisodeSpec};
use crate::io::{read_json, write_json};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolRow {
    pub episode: String,
    pub task: u32,
    pub condition: String,
    pub seed: u64,
    pub workspace: PathBuf,
    pub logs_dir: PathBuf,
    pub teacher_prompt: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CollectionFailure {
    pub episode: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct CollectionOutcome {
    pub rows: Vec<PoolRow>,
    pub failures: Vec<CollectionFailure>,
}

pub fn collect(
    config: &CycleConfig,
    task: &Task,
    retained_design: &str,
    teacher_directive: &str,
    run_dir: &Path,
    runner: &dyn EpisodeRunner,
) -> Result<CollectionOutcome> {
    let teacher_dir = run_dir.join("teacher_prompts");
    fs::create_dir_all(&teacher_dir)?;
    let teacher_prompt = teacher_dir.join(format!("query{}.txt", task.id));
    let text = format!(
        "{}\n\n{}",
        retained_design.trim_end(),
        teacher_directive.trim()
    );
    if teacher_prompt.is_file() {
        let existing = fs::read_to_string(&teacher_prompt)?;
        if existing != text {
            return Err(EvolveError::Episode(format!(
                "teacher prompt changed: {}",
                teacher_prompt.display()
            )));
        }
    } else {
        fs::write(&teacher_prompt, &text)?;
    }

    let pool_path = run_dir.join("pool.json");
    let mut existing: Vec<PoolRow> = if pool_path.is_file() {
        read_json(&pool_path)?
    } else {
        Vec::new()
    };
    existing.retain(|row| row.task != task.id);

    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for index in 0..config.collection.candidates {
        let seed = config.collection.seed_base + 100 * u64::from(task.id) + u64::from(index);
        let condition = format!("t{}_L{}", task.id, config.generation);
        let episode = format!("{condition}_s{seed}");
        let workspace = run_dir.join("pool").join("workspaces").join(&episode);
        let logs_dir = run_dir.join("pool").join("runlogs").join(&episode);

        let outcome = runner.run(&EpisodeSpec {
            prompt: &text,
            workspace: &workspace,
            logs: &logs_dir,
            condition: &condition,
            seed,
            model_id: &config.model_id,
            endpoint: &config.endpoint,
        });
        match outcome {
            Ok(result) if result.is_success() => {
                rows.push(PoolRow {
                    episode,
                    task: task.id,
                    condition,
                    seed,
                    workspace,
                    logs_dir,
                    teacher_prompt: teacher_prompt.clone(),
                });
            }
            Ok(result) => failures.push(CollectionFailure {
                episode,
                message: format!("episode ended {:?}", result.status),
            }),
            Err(error) => failures.push(CollectionFailure {
                episode,
                message: error.to_string(),
            }),
        }
    }

    existing.extend(rows.iter().cloned());
    existing.sort_by(|left, right| left.episode.cmp(&right.episode));
    write_json(&pool_path, &existing)?;

    Ok(CollectionOutcome { rows, failures })
}

/// Read the pool, optionally filtered to one task.
pub fn read_pool(run_dir: &Path, task: Option<u32>) -> Result<Vec<PoolRow>> {
    let path = run_dir.join("pool.json");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let rows: Vec<PoolRow> = read_json(&path)?;
    Ok(match task {
        Some(task) => rows.into_iter().filter(|row| row.task == task).collect(),
        None => rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{EpisodeOutcome, EpisodeStatus};

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

    fn config(root: &Path) -> CycleConfig {
        let mut config: CycleConfig = serde_json::from_value(serde_json::json!({
            "generation": 0,
            "run_dir": root,
            "model_id": "local",
            "endpoint": "http://127.0.0.1:8080/v1",
            "tasks": {"train_candidates": [1], "train": [1], "test": [], "excluded": []},
            "search": {"iterations": 1, "seed_base": 1},
            "collection": {"candidates": 2, "train_ranks": 1, "validation_rank": 2, "seed_base": 100},
            "ranking": {"opponents_per_episode": 2, "seed": 1},
            "training": {
                "window": 64, "context_overlap": 8, "steps": 1, "lr": 0.0001,
                "warmup_steps": 1, "min_lr_ratio": 0.1, "lora_r": 4, "lora_alpha": 8,
                "lora_dropout": 0.05, "save_every": 1, "val_every": 1,
                "replicas": 1, "gpus_per_rank": 1, "grad_accum": 3,
                "ce_chunk": 8, "seed": 1
            }
        }))
        .expect("config");
        config.run_dir = root.to_path_buf();
        config
    }

    fn task() -> Task {
        Task {
            id: 1,
            domain: "software".into(),
            prompt: "fix it".into(),
            initial_design: "START\nd\nEND".into(),
            task_text: "fix it".into(),
        }
    }

    #[test]
    fn collects_candidates_with_the_reference_seed_formula() {
        let root = std::env::temp_dir().join(format!(
            "evolve-collect-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = config(&root);
        let outcome = collect(
            &config,
            &task(),
            "START\nd\nEND",
            "run the team",
            &root,
            &FakeRunner,
        )
        .expect("collect");
        assert_eq!(outcome.rows.len(), 2);
        assert_eq!(outcome.rows[0].seed, 200);
        assert_eq!(outcome.rows[1].seed, 201);
        assert_eq!(outcome.rows[0].condition, "t1_L0");
        assert!(root.join("pool.json").is_file());
        assert!(root.join("teacher_prompts/query1.txt").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }
}
