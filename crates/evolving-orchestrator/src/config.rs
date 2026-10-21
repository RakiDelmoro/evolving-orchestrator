//! Cycle configuration and boundary validation.
//!
//! The configuration is deliberately data-only: it names the current model,
//! the task split, the search budget, the collection and ranking recipe, the
//! training hyperparameters, and the two command adapters. It never holds a
//! credential.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{EvolveError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionRule {
    /// Compare each proposal with the best-so-far output directly.
    #[default]
    RetainedBest,
    /// Compare each proposal with a fixed bare-task reference, then arbitrate
    /// the champion. This is the more conservative variant.
    FixedReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSplit {
    pub train_candidates: Vec<u32>,
    pub train: Vec<u32>,
    pub test: Vec<u32>,
    #[serde(default)]
    pub excluded: Vec<u32>,
}

impl TaskSplit {
    pub fn validate(&self) -> Result<()> {
        let candidates = unique(&self.train_candidates, "train_candidates")?;
        let train = unique(&self.train, "train")?;
        let test = unique(&self.test, "test")?;
        let excluded = unique(&self.excluded, "excluded")?;

        if !train.is_disjoint(&test) {
            return Err(EvolveError::Config(
                "train and test tasks must be disjoint".into(),
            ));
        }
        if !train.is_disjoint(&excluded) {
            return Err(EvolveError::Config(
                "train and excluded tasks must be disjoint".into(),
            ));
        }
        if !test.is_disjoint(&excluded) {
            return Err(EvolveError::Config(
                "test and excluded tasks must be disjoint".into(),
            ));
        }

        let combined: std::collections::BTreeSet<u32> = train.union(&excluded).copied().collect();
        if combined != candidates {
            return Err(EvolveError::Config(
                "train_candidates must equal train + excluded".into(),
            ));
        }
        Ok(())
    }
}

fn unique(ids: &[u32], name: &str) -> Result<std::collections::BTreeSet<u32>> {
    let set: std::collections::BTreeSet<u32> = ids.iter().copied().collect();
    if set.len() != ids.len() {
        return Err(EvolveError::Config(format!(
            "{name} contains a duplicate task id"
        )));
    }
    Ok(set)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    pub iterations: u32,
    pub seed_base: u64,
    #[serde(default)]
    pub rule: SelectionRule,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionConfig {
    pub candidates: u32,
    pub train_ranks: u32,
    pub validation_rank: u32,
    pub seed_base: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankingConfig {
    #[serde(alias = "opponents_per_trial")]
    pub opponents_per_episode: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub window: u32,
    pub context_overlap: u32,
    pub steps: u32,
    pub lr: f64,
    pub warmup_steps: u32,
    pub min_lr_ratio: f64,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub lora_dropout: f64,
    pub save_every: u32,
    pub val_every: u32,
    pub replicas: u32,
    pub gpus_per_rank: u32,
    pub grad_accum: u32,
    pub ce_chunk: u32,
    pub seed: u64,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.window == 0 || self.window <= self.context_overlap {
            return Err(EvolveError::Config(
                "training.window must exceed context_overlap".into(),
            ));
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(EvolveError::Config(
                "training.lr must be finite and positive".into(),
            ));
        }
        if !self.min_lr_ratio.is_finite() || !(0.0..1.0).contains(&self.min_lr_ratio) {
            return Err(EvolveError::Config(
                "training.min_lr_ratio must be in [0, 1)".into(),
            ));
        }
        if !self.lora_dropout.is_finite() || !(0.0..1.0).contains(&self.lora_dropout) {
            return Err(EvolveError::Config(
                "training.lora_dropout must be in [0, 1)".into(),
            ));
        }
        for (name, value) in [
            ("steps", self.steps),
            ("lora_r", self.lora_r),
            ("lora_alpha", self.lora_alpha),
            ("save_every", self.save_every),
            ("val_every", self.val_every),
            ("replicas", self.replicas),
            ("gpus_per_rank", self.gpus_per_rank),
            ("grad_accum", self.grad_accum),
            ("ce_chunk", self.ce_chunk),
        ] {
            if value == 0 {
                return Err(EvolveError::Config(format!(
                    "training.{name} must be positive"
                )));
            }
        }
        if self.replicas * self.grad_accum != 3 {
            return Err(EvolveError::Config(
                "training.replicas * training.grad_accum must equal 3".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    /// Program to run for one episode. It receives the prompt on stdin unless
    /// `stdin_prompt` is false, in which case the prompt path is in the
    /// environment as `EVOLVE_PROMPT_FILE`.
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_true")]
    pub stdin_prompt: bool,
    /// Append the rendered prompt as the final argv element instead of
    /// writing it to stdin. Use this for `pi`.
    #[serde(default)]
    pub prompt_arg: bool,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

fn default_true() -> bool {
    true
}
fn default_timeout() -> u64 {
    3600
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            program: "qwen".into(),
            args: vec![
                "--approval-mode".into(),
                "yolo".into(),
                "--output-format".into(),
                "stream-json".into(),
            ],
            stdin_prompt: true,
            prompt_arg: false,
            timeout_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_judge_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_judge_tokens")]
    pub max_output_tokens: u32,
}

fn default_api_key_env() -> String {
    "EVOLVE_API_KEY".into()
}
fn default_judge_temperature() -> f64 {
    0.6
}
fn default_top_p() -> f64 {
    0.95
}
fn default_judge_tokens() -> u32 {
    4096
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            api_key_env: default_api_key_env(),
            temperature: default_judge_temperature(),
            top_p: default_top_p(),
            max_output_tokens: default_judge_tokens(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CycleConfig {
    pub generation: u32,
    pub run_dir: PathBuf,
    pub model_id: String,
    #[serde(default)]
    pub parent_bf16: Option<PathBuf>,
    #[serde(default)]
    pub fp8_reference: Option<PathBuf>,
    #[serde(default)]
    pub next_bf16: Option<PathBuf>,
    #[serde(default)]
    pub next_fp8: Option<PathBuf>,
    pub endpoint: String,
    #[serde(default)]
    pub proxy_port: Option<u16>,
    pub tasks: TaskSplit,
    pub search: SearchConfig,
    pub collection: CollectionConfig,
    pub ranking: RankingConfig,
    pub training: TrainingConfig,
    #[serde(default)]
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub judge: JudgeConfig,
}

impl CycleConfig {
    pub fn parse(value: serde_json::Value, source: &str) -> Result<Self> {
        let config: Self = serde_json::from_value(value)
            .map_err(|error| EvolveError::Config(format!("{source}: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.run_dir.as_os_str().is_empty() {
            return Err(EvolveError::Config("run_dir must not be empty".into()));
        }
        if self.model_id.trim().is_empty() {
            return Err(EvolveError::Config("model_id must not be empty".into()));
        }
        let endpoint = parse_endpoint(&self.endpoint)?;
        if let Some(proxy) = self.proxy_port {
            if proxy == endpoint.port {
                return Err(EvolveError::Config(
                    "endpoint port must differ from proxy_port".into(),
                ));
            }
        }
        self.tasks.validate()?;
        if self.search.iterations < 1 {
            return Err(EvolveError::Config(
                "search.iterations must be at least 1".into(),
            ));
        }
        if self.collection.validation_rank != self.collection.train_ranks + 1 {
            return Err(EvolveError::Config(
                "collection.validation_rank must immediately follow train_ranks".into(),
            ));
        }
        if self.collection.candidates < self.collection.validation_rank {
            return Err(EvolveError::Config(
                "collection.candidates must be at least validation_rank".into(),
            ));
        }
        if self.ranking.opponents_per_episode < 2
            || !self.ranking.opponents_per_episode.is_multiple_of(2)
        {
            return Err(EvolveError::Config(
                "ranking.opponents_per_episode must be even and at least 2".into(),
            ));
        }
        if self.executor.program.trim().is_empty() {
            return Err(EvolveError::Config(
                "executor.program must not be empty".into(),
            ));
        }
        if self.executor.stdin_prompt == self.executor.prompt_arg {
            return Err(EvolveError::Config(
                "exactly one of executor.stdin_prompt and executor.prompt_arg must be true".into(),
            ));
        }
        self.training.validate()?;
        Ok(())
    }

    pub fn stages() -> &'static [&'static str] {
        &[
            "search", "collect", "rank", "select", "render", "train", "export",
        ]
    }

    /// Resolve a configured path relative to the configuration file.
    pub fn resolve(base: &Path, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            base.join(path)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoint {
    pub port: u16,
}

/// Parse the loopback OpenAI-compatible endpoint the executor and judge share.
pub fn parse_endpoint(value: &str) -> Result<Endpoint> {
    let rest = value.strip_prefix("http://").ok_or_else(|| {
        EvolveError::Config("endpoint must start with http://127.0.0.1:PORT/v1".into())
    })?;
    if rest.contains('?') || rest.contains('#') || rest.contains('@') || rest.contains("://") {
        return Err(EvolveError::Config(
            "endpoint must have no credentials, query, or fragment".into(),
        ));
    }
    let (authority, path) = rest
        .split_once('/')
        .ok_or_else(|| EvolveError::Config("endpoint must include the /v1 path".into()))?;
    if path != "v1" && path != "v1/" {
        return Err(EvolveError::Config("endpoint path must be /v1".into()));
    }
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or_else(|| EvolveError::Config("endpoint must include an explicit port".into()))?;
    if host != "127.0.0.1" && host != "localhost" {
        return Err(EvolveError::Config(
            "endpoint must be loopback (127.0.0.1 or localhost)".into(),
        ));
    }
    let port: u16 = port
        .parse()
        .map_err(|_| EvolveError::Config("endpoint port is not a number".into()))?;
    Ok(Endpoint { port })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn value() -> serde_json::Value {
        json!({
            "generation": 0,
            "run_dir": "runs/L0",
            "model_id": "local/model",
            "parent_bf16": "models/L0-bf16",
            "fp8_reference": "models/L0-fp8",
            "next_bf16": "models/L1-bf16",
            "next_fp8": "models/L1-fp8",
            "endpoint": "http://127.0.0.1:8001/v1",
            "proxy_port": 9001,
            "tasks": {
                "train_candidates": [51, 53],
                "train": [51],
                "test": [60],
                "excluded": [53]
            },
            "search": {"iterations": 11, "seed_base": 450000},
            "collection": {"candidates": 18, "train_ranks": 15, "validation_rank": 16, "seed_base": 710000},
            "ranking": {"opponents_per_episode": 6, "seed": 1234},
            "training": {
                "window": 49152, "context_overlap": 8192, "steps": 1624,
                "lr": 0.00003, "warmup_steps": 5, "min_lr_ratio": 0.1,
                "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
                "save_every": 50, "val_every": 150, "replicas": 3,
                "gpus_per_rank": 2, "grad_accum": 1, "ce_chunk": 512,
                "seed": 1234
            }
        })
    }

    #[test]
    fn parses_the_reference_recipe() {
        let config = CycleConfig::parse(value(), "fixture.json").expect("valid");
        assert_eq!(config.search.iterations, 11);
        assert_eq!(config.collection.train_ranks, 15);
        assert_eq!(config.training.lora_r, 64);
        assert_eq!(config.search.rule, SelectionRule::RetainedBest);
    }

    #[test]
    fn rejects_overlapping_splits() {
        let mut value = value();
        value["tasks"]["test"] = json!([51]);
        let error = CycleConfig::parse(value, "fixture").expect_err("must reject");
        assert!(error.to_string().contains("disjoint"));
    }

    #[test]
    fn rejects_bad_endpoint() {
        for endpoint in [
            "https://127.0.0.1:8001/v1",
            "http://10.0.0.1:8001/v1",
            "http://127.0.0.1:8001/v2",
            "http://127.0.0.1/v1",
        ] {
            let mut value = value();
            value["endpoint"] = json!(endpoint);
            assert!(CycleConfig::parse(value, "fixture").is_err(), "{endpoint}");
        }
    }

    #[test]
    fn rejects_wrong_validation_rank() {
        let mut value = value();
        value["collection"]["validation_rank"] = json!(17);
        let error = CycleConfig::parse(value, "fixture").expect_err("must reject");
        assert!(error.to_string().contains("validation_rank"));
    }
}
