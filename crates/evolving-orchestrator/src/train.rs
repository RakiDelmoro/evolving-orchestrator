//! Training orchestration and checkpoint selection.
//!
//! The tensor math is an external command. The decisions that must be
//! reproducible live here: the request, the log schema, and the rule that
//! selects the next model.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;
use crate::error::{EvolveError, Result};
use crate::io::{read_json, read_records, write_json};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainLogEntry {
    pub step: u32,
    #[serde(default)]
    pub val_loss: Option<f64>,
    #[serde(default)]
    pub val_tokens: u64,
    #[serde(default)]
    pub checkpoint: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub step: u32,
    pub val_loss: f64,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TrainRequest {
    pub run_dir: PathBuf,
    pub train_file: PathBuf,
    pub validation_file: PathBuf,
    pub parent_model: Option<PathBuf>,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone)]
pub struct TrainOutcome {
    pub entries: Vec<TrainLogEntry>,
    pub checkpoint: Option<Checkpoint>,
}

pub trait Trainer {
    fn train(&self, request: &TrainRequest) -> Result<TrainOutcome>;
}

/// Minimum finite validation loss, ignoring step 0 and unsaved checkpoints.
pub fn best_checkpoint(entries: &[TrainLogEntry]) -> Result<Option<Checkpoint>> {
    let mut seen = std::collections::BTreeSet::new();
    let mut best: Option<Checkpoint> = None;
    for entry in entries {
        if !seen.insert(entry.step) {
            return Err(EvolveError::Training(format!(
                "train log contains step {} twice",
                entry.step
            )));
        }
        if entry.step == 0 {
            continue;
        }
        let Some(loss) = entry.val_loss else {
            continue;
        };
        if !loss.is_finite() || entry.val_tokens == 0 {
            continue;
        }
        let Some(path) = &entry.checkpoint else {
            continue;
        };
        let candidate = Checkpoint {
            step: entry.step,
            val_loss: loss,
            path: path.clone(),
        };
        match &best {
            Some(current)
                if (current.val_loss, current.step) <= (candidate.val_loss, candidate.step) => {}
            _ => best = Some(candidate),
        }
    }
    Ok(best)
}

pub fn best_saved_checkpoint(entries: &[TrainLogEntry]) -> Result<Option<Checkpoint>> {
    let best = best_checkpoint(entries)?;
    if let Some(checkpoint) = &best {
        if !checkpoint.path.exists() {
            return Err(EvolveError::Training(format!(
                "selected checkpoint does not exist: {}",
                checkpoint.path.display()
            )));
        }
    }
    Ok(best)
}

#[derive(Debug, Clone)]
pub struct CommandTrainer {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandTrainer {
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    /// Build from `EVOLVE_TRAIN_COMMAND`, run through `/bin/sh -c`.
    pub fn from_env() -> Option<Self> {
        std::env::var("EVOLVE_TRAIN_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| Self::new("/bin/sh", vec!["-c".into(), value]))
    }
}

impl Trainer for CommandTrainer {
    fn train(&self, request: &TrainRequest) -> Result<TrainOutcome> {
        if !request.train_file.is_file() {
            return Err(EvolveError::Training(format!(
                "training file missing: {}",
                request.train_file.display()
            )));
        }
        if !request.validation_file.is_file() {
            return Err(EvolveError::Training(format!(
                "validation file missing: {}",
                request.validation_file.display()
            )));
        }

        let train_dir = request.run_dir.join("training");
        std::fs::create_dir_all(&train_dir)?;
        let config_path = train_dir.join("config.json");
        write_json(&config_path, &request.config)?;

        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .current_dir(&request.run_dir)
            .env("EVOLVE_TRAIN_DIR", &train_dir)
            .env("EVOLVE_TRAIN_FILE", &request.train_file)
            .env("EVOLVE_VALIDATION_FILE", &request.validation_file)
            .env("EVOLVE_TRAIN_CONFIG", &config_path)
            .env("EVOLVE_STEPS", request.config.steps.to_string())
            .env("EVOLVE_LORA_R", request.config.lora_r.to_string())
            .env("EVOLVE_LORA_ALPHA", request.config.lora_alpha.to_string())
            .env("EVOLVE_SEED", request.config.seed.to_string());
        if let Some(parent) = &request.parent_model {
            command.env("EVOLVE_PARENT_MODEL", parent);
        }

        let status = command.status().map_err(|error| {
            EvolveError::Training(format!("cannot start trainer {:?}: {error}", self.program))
        })?;
        if !status.success() {
            return Err(EvolveError::Training(format!(
                "trainer exited with {status}"
            )));
        }

        let log_path = train_dir.join("train_log.jsonl");
        if !log_path.is_file() {
            return Err(EvolveError::Training(format!(
                "trainer did not write {}",
                log_path.display()
            )));
        }
        let mut entries: Vec<TrainLogEntry> = read_records(&log_path)?;
        for entry in &mut entries {
            if let Some(path) = &entry.checkpoint {
                if !path.is_absolute() {
                    entry.checkpoint = Some(train_dir.join(path));
                }
            }
        }
        let checkpoint = best_saved_checkpoint(&entries)?;
        Ok(TrainOutcome {
            entries,
            checkpoint,
        })
    }
}

/// Read a training log without running the trainer.
pub fn read_train_log(path: &Path) -> Result<Vec<TrainLogEntry>> {
    read_records(path)
}

/// Read the frozen training config written for the trainer.
pub fn read_train_config(path: &Path) -> Result<TrainingConfig> {
    read_json(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(step: u32, loss: Option<f64>, tokens: u64, checkpoint: Option<&str>) -> TrainLogEntry {
        TrainLogEntry {
            step,
            val_loss: loss,
            val_tokens: tokens,
            checkpoint: checkpoint.map(PathBuf::from),
        }
    }

    #[test]
    fn selects_the_lowest_finite_loss_after_step_zero() {
        let entries = vec![
            entry(0, Some(0.1), 100, Some("step_00000")),
            entry(10, Some(0.5), 100, Some("step_00010")),
            entry(20, Some(0.3), 100, Some("step_00020")),
        ];
        let best = best_checkpoint(&entries).expect("select").expect("some");
        assert_eq!(best.step, 20);
        assert_eq!(best.val_loss, 0.3);
    }

    #[test]
    fn ignores_zero_tokens_and_non_finite_losses() {
        let entries = vec![
            entry(10, Some(f64::NAN), 100, Some("a")),
            entry(20, Some(0.2), 0, Some("b")),
            entry(30, Some(0.4), 10, Some("c")),
        ];
        assert_eq!(best_checkpoint(&entries).unwrap().unwrap().step, 30);
    }

    #[test]
    fn rejects_duplicate_steps() {
        let entries = vec![
            entry(10, Some(0.5), 100, Some("a")),
            entry(10, Some(0.4), 100, Some("b")),
        ];
        assert!(best_checkpoint(&entries).is_err());
    }

    #[test]
    fn runs_a_command_trainer() {
        let root = std::env::temp_dir().join(format!(
            "evolve-train-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let train = root.join("train.jsonl");
        let val = root.join("val.jsonl");
        std::fs::write(&train, "{}\n").unwrap();
        std::fs::write(&val, "{}\n").unwrap();

        let script = r#"
mkdir -p "$EVOLVE_TRAIN_DIR/step_00010"
printf '%s\n' \
  '{"step":0,"val_loss":0.1,"val_tokens":10,"checkpoint":"step_00000"}' \
  '{"step":10,"val_loss":0.4,"val_tokens":10,"checkpoint":"step_00010"}' \
  > "$EVOLVE_TRAIN_DIR/train_log.jsonl"
touch "$EVOLVE_TRAIN_DIR/step_00010/adapter.safetensors"
"#;
        let trainer = CommandTrainer::new("/bin/sh", vec!["-c".into(), script.into()]);
        let config = TrainingConfig {
            window: 1024,
            context_overlap: 128,
            steps: 10,
            lr: 2e-4,
            warmup_steps: 1,
            min_lr_ratio: 0.1,
            lora_r: 16,
            lora_alpha: 32,
            lora_dropout: 0.05,
            save_every: 10,
            val_every: 10,
            replicas: 1,
            gpus_per_rank: 1,
            grad_accum: 3,
            ce_chunk: 128,
            seed: 1,
        };
        let outcome = trainer
            .train(&TrainRequest {
                run_dir: root.clone(),
                train_file: train,
                validation_file: val,
                parent_model: None,
                config,
            })
            .expect("train");
        assert_eq!(outcome.checkpoint.unwrap().step, 10);
        std::fs::remove_dir_all(root).unwrap();
    }
}
