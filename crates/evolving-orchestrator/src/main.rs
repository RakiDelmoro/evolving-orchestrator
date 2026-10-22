//! Command-line entry point.

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use evolving_orchestrator::chat::{ChatOptions, OpenAiChat};
use evolving_orchestrator::collect::{collect, read_pool};
use evolving_orchestrator::config::CycleConfig;
use evolving_orchestrator::cycle::{run_cycle_with_options, CycleOptions, CyclePaths, CyclePorts};
use evolving_orchestrator::design::TaskCatalog;
use evolving_orchestrator::error::{EvolveError, Result};
use evolving_orchestrator::exec::ProcessRunner;
use evolving_orchestrator::export::{CommandExporter, ExportRequest, Exporter};
use evolving_orchestrator::io::{append_record, read_json, write_json};
use evolving_orchestrator::judge::ModelJudge;
use evolving_orchestrator::pi_session::load_sessions;
use evolving_orchestrator::quality::{GateStatus, ModelRubricJudge, QualityReport};
use evolving_orchestrator::render::{
    write_windows, ByteTokenizer, CommandTokenizer, PromptSubstitution, RenderConfig, SftWindow,
    Tokenizer,
};
use evolving_orchestrator::search::{
    champion_design, read_retained, run_search, ModelProposer, SearchPaths,
};
use evolving_orchestrator::select::{rank_candidates, select_rows, Candidate, RankedEpisode};
use evolving_orchestrator::train::{CommandTrainer, TrainRequest, Trainer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(
    name = "evolving-orchestrator",
    version,
    about = "Workflow-evolution harness: design search, trajectory ranking, and model updates"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate the cycle configuration and the task files it names.
    Validate {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Print the cycle plan, stage order, and resolved paths.
    Plan {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Search designs for one task or all configured tasks.
    Search {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
        #[arg(long)]
        task: Option<u32>,
    },
    /// Run teacher collection under retained designs.
    Collect {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
        #[arg(long)]
        task: Option<u32>,
    },
    /// Rank collected episodes and write `task{id}_ranked.json`.
    Rank {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
        #[arg(long)]
        task: Option<u32>,
    },
    /// Split ranked episodes into train and validation.
    Select {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Render selected episodes into masked windows.
    Render {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Run the configured trainer and select a checkpoint.
    Train {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Merge and export the selected checkpoint.
    Export {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
    },
    /// Long-running task loop: repeat attempts until every quality gate passes.
    Task {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
        /// Base repository copied into the first attempt. Later attempts start
        /// from the best rejected workspace.
        #[arg(long)]
        repo: Option<PathBuf>,
        /// 0 means no attempt cap; the stagnation guard still applies.
        #[arg(long, default_value_t = 0)]
        max_attempts: u32,
        /// Stop after this many attempts with no score or gate progress.
        #[arg(long, default_value_t = 5)]
        stagnation_limit: u32,
    },
    /// Run one cycle. `--task-only` stops after ranking and publishing.
    Cycle {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
        /// Stop after search, collection, ranking, selection, and publishing.
        #[arg(long)]
        task_only: bool,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("evolve: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Validate { config, tasks } => validate(&config, &tasks),
        Command::Plan { config, tasks } => plan(&config, &tasks),
        Command::Search {
            config,
            tasks,
            task,
        } => search(&config, &tasks, task),
        Command::Collect {
            config,
            tasks,
            task,
        } => collect_stage(&config, &tasks, task),
        Command::Rank {
            config,
            tasks,
            task,
        } => rank_stage(&config, &tasks, task),
        Command::Select { config, tasks } => select_stage(&config, &tasks),
        Command::Render { config, tasks } => render_stage(&config, &tasks),
        Command::Train { config, tasks } => train_stage(&config, &tasks),
        Command::Export { config, tasks } => export_stage(&config, &tasks),
        Command::Task {
            config,
            tasks,
            repo,
            max_attempts,
            stagnation_limit,
        } => task_stage(
            &config,
            &tasks,
            repo.as_deref(),
            max_attempts,
            stagnation_limit,
        ),
        Command::Cycle {
            config,
            tasks,
            task_only,
        } => cycle_stage(&config, &tasks, task_only),
    }
}

fn load(config_path: &Path) -> Result<CycleConfig> {
    let value: serde_json::Value = read_json(config_path)?;
    CycleConfig::parse(value, &config_path.display().to_string())
}

fn validate(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let mut ids: Vec<u32> = config
        .tasks
        .train_candidates
        .iter()
        .chain(config.tasks.test.iter())
        .copied()
        .collect();
    ids.sort_unstable();
    ids.dedup();

    for id in &ids {
        if !catalog.bare_path(*id).is_file() {
            return Err(EvolveError::Config(format!(
                "missing bare task file {}",
                catalog.bare_path(*id).display()
            )));
        }
        if !catalog.initial_path(*id).is_file() {
            return Err(EvolveError::Config(format!(
                "missing initial design file {}",
                catalog.initial_path(*id).display()
            )));
        }
        let parsed = catalog.parse(*id)?;
        if parsed.task_text.trim().is_empty() {
            return Err(EvolveError::Config(format!(
                "task {id} has an empty prompt before the design marker"
            )));
        }
    }

    println!(
        "config ok: model {}, {} train / {} test task(s), {} stage(s)",
        config.model_id,
        config.tasks.train.len(),
        config.tasks.test.len(),
        CycleConfig::stages().len()
    );
    Ok(())
}

fn plan(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    println!("model: {}", config.model_id);
    println!("endpoint: {}", config.endpoint);
    println!("generation: L{}", config.generation);
    println!("run_dir: {}", config.run_dir.display());
    println!("tasks_dir: {}", tasks_dir.display());
    println!(
        "tasks: train={:?} test={:?} excluded={:?}",
        config.tasks.train, config.tasks.test, config.tasks.excluded
    );
    println!("selection rule: {:?}", config.search.rule);
    println!(
        "search: {} iterations, seed_base {}",
        config.search.iterations, config.search.seed_base
    );
    println!(
        "collection: {} candidates, ranks 1..{} train, rank {} validation, seed_base {}",
        config.collection.candidates,
        config.collection.train_ranks,
        config.collection.validation_rank,
        config.collection.seed_base
    );
    println!(
        "ranking: {} opponents, seed {}",
        config.ranking.opponents_per_episode, config.ranking.seed
    );
    println!(
        "training: window {} / overlap {}, LoRA r={} alpha={}, {} steps, {} replica(s) x {} GPU(s)",
        config.training.window,
        config.training.context_overlap,
        config.training.lora_r,
        config.training.lora_alpha,
        config.training.steps,
        config.training.replicas,
        config.training.gpus_per_rank
    );
    println!("stages: {}", CycleConfig::stages().join(" -> "));
    println!();
    println!("execution is sequential: one episode at a time, one inference server.");
    println!(
        "executor: {} {:?} (stdin prompt: {})",
        config.executor.program, config.executor.args, config.executor.stdin_prompt
    );
    Ok(())
}

fn chat(config: &CycleConfig) -> OpenAiChat {
    let api_key = std::env::var(&config.judge.api_key_env).unwrap_or_else(|_| "local".into());
    OpenAiChat::new(config.endpoint.clone(), config.model_id.clone(), api_key)
}

fn options(config: &CycleConfig, seed: u64) -> ChatOptions {
    ChatOptions {
        temperature: config.judge.temperature,
        top_p: config.judge.top_p,
        max_output_tokens: config.judge.max_output_tokens,
        seed,
    }
}

fn task_ids(config: &CycleConfig, task: Option<u32>, train_only: bool) -> Result<Vec<u32>> {
    let ids: Vec<u32> = match task {
        Some(id) => {
            if !config
                .tasks
                .train_candidates
                .iter()
                .chain(config.tasks.test.iter())
                .any(|candidate| *candidate == id)
            {
                return Err(EvolveError::Config(format!(
                    "task {id} is not in the configured split"
                )));
            }
            vec![id]
        }
        None if train_only => config.tasks.train.clone(),
        None => config
            .tasks
            .train_candidates
            .iter()
            .chain(config.tasks.test.iter())
            .copied()
            .collect(),
    };
    Ok(ids)
}

fn search(config_path: &Path, tasks_dir: &Path, task: Option<u32>) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let chat = chat(&config);
    let judge = ModelJudge {
        chat: &chat,
        options: options(&config, 0),
    };
    let proposer = ModelProposer {
        chat: &chat,
        options: options(&config, 1),
    };
    let runner = ProcessRunner::new(config.executor.clone());

    for id in task_ids(&config, task, false)? {
        let task = catalog.load(id, "software")?;
        let parsed = catalog.parse(id)?;
        let paths = SearchPaths::new(&config.run_dir, parsed.query_name.clone());
        let outcome = run_search(&config, &task, &parsed, &paths, &runner, &judge, &proposer)?;
        println!(
            "task {id}: retained v{}, beats_reference {}, comparisons {}",
            outcome.retained_version,
            outcome.beats_reference,
            outcome.history.len()
        );
    }
    Ok(())
}

fn collect_stage(config_path: &Path, tasks_dir: &Path, task: Option<u32>) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let runner = ProcessRunner::new(config.executor.clone());
    let directive = read_teacher_directive(tasks_dir)?;
    for id in task_ids(&config, task, true)? {
        let task = catalog.load(id, "software")?;
        let parsed = catalog.parse(id)?;
        let paths = SearchPaths::new(&config.run_dir, parsed.query_name.clone());
        let retained = read_retained(&paths)?;
        if !retained.beats_reference {
            println!("task {id}: no retained design; skipped");
            continue;
        }
        let design = champion_design(&paths)?;
        let outcome = collect(
            &config,
            &task,
            &design,
            &directive,
            &config.run_dir,
            &runner,
        )?;
        println!(
            "task {id}: {} collected, {} failed",
            outcome.rows.len(),
            outcome.failures.len()
        );
    }
    Ok(())
}

fn rank_stage(config_path: &Path, tasks_dir: &Path, task: Option<u32>) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let chat = chat(&config);
    let judge = ModelJudge {
        chat: &chat,
        options: options(&config, 0),
    };
    let selection_dir = config.run_dir.join("selection");
    std::fs::create_dir_all(&selection_dir)?;
    for id in task_ids(&config, task, true)? {
        let task = catalog.load(id, "software")?;
        let rows = read_pool(&config.run_dir, Some(id))?;
        let candidates: Vec<Candidate> = rows
            .into_iter()
            .map(|row| Candidate {
                name: row.episode,
                task: row.task,
                condition: row.condition,
                seed: row.seed,
                workspace: row.workspace,
                logs_dir: row.logs_dir,
                teacher_prompt: Some(row.teacher_prompt),
            })
            .collect();
        let ranked = rank_candidates(
            &judge,
            &task.task_text,
            &candidates,
            config.ranking.opponents_per_episode as usize,
            config.ranking.seed.wrapping_add(u64::from(id) * 1000),
            500,
        )?;
        write_json(
            &selection_dir.join(format!("task{id}_ranked.json")),
            &ranked,
        )?;
        println!("task {id}: ranked {} episodes", ranked.len());
    }
    Ok(())
}

fn select_stage(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    let selection_dir = config.run_dir.join("selection");
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for id in &config.tasks.train {
        let path = selection_dir.join(format!("task{id}_ranked.json"));
        if !path.is_file() {
            return Err(EvolveError::Ranking(format!(
                "missing ranked file {}; run `evolve rank` first",
                path.display()
            )));
        }
        let ranked: Vec<RankedEpisode> = read_json(&path)?;
        let split = select_rows(
            &ranked,
            config.collection.train_ranks,
            config.collection.validation_rank,
        )?;
        train.extend(split.train);
        validation.push(split.validation);
    }
    if train.is_empty() || validation.is_empty() {
        return Err(EvolveError::Ranking("empty selection".into()));
    }
    write_json(&selection_dir.join("train.json"), &train)?;
    write_json(&selection_dir.join("val.json"), &validation)?;
    println!(
        "selected {} training and {} validation episodes from {} task(s)",
        train.len(),
        validation.len(),
        config.tasks.train.len()
    );
    let _ = tasks_dir;
    Ok(())
}

fn render_stage(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let selection_dir = config.run_dir.join("selection");
    let train: Vec<RankedEpisode> = read_json(&selection_dir.join("train.json"))?;
    let validation: Vec<RankedEpisode> = read_json(&selection_dir.join("val.json"))?;
    let tokenizer: Box<dyn Tokenizer> = match CommandTokenizer::from_env() {
        Some(tokenizer) => Box::new(tokenizer),
        None => {
            eprintln!("warning: EVOLVE_TOKENIZE_COMMAND is not set; using byte tokens");
            Box::new(ByteTokenizer)
        }
    };
    let render_config = RenderConfig {
        window: config.training.window as usize,
        context_overlap: config.training.context_overlap as usize,
    };

    let mut train_windows: Vec<SftWindow> = Vec::new();
    let mut validation_windows: Vec<SftWindow> = Vec::new();
    for (is_train, rows) in [(true, train), (false, validation)] {
        for row in rows {
            let task = catalog.load(row.task, "software")?;
            let teacher_prompt = row
                .teacher_prompt
                .as_ref()
                .ok_or_else(|| EvolveError::Training("missing teacher prompt".into()))
                .and_then(|path| std::fs::read_to_string(path).map_err(EvolveError::from))?;
            let substitution = PromptSubstitution {
                from: teacher_prompt,
                to: task.prompt.trim().to_string(),
            };
            let sessions = load_sessions(&row.logs_dir.join("sessions"))?;
            let windows = evolving_orchestrator::render::render_sessions(
                &sessions,
                tokenizer.as_ref(),
                render_config,
                Some(&substitution),
                &row.name,
                row.task,
            )?;
            if windows.is_empty() {
                return Err(EvolveError::Training(format!(
                    "episode {} produced no windows",
                    row.name
                )));
            }
            if is_train {
                train_windows.extend(windows);
            } else {
                validation_windows.extend(windows);
            }
        }
    }

    let data = config.run_dir.join("data");
    write_windows(&data.join("train.jsonl"), &train_windows)?;
    write_windows(&data.join("val.jsonl"), &validation_windows)?;
    println!(
        "rendered {} training and {} validation windows",
        train_windows.len(),
        validation_windows.len()
    );
    Ok(())
}

fn train_stage(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    let trainer = CommandTrainer::from_env()
        .ok_or_else(|| EvolveError::Training("EVOLVE_TRAIN_COMMAND is not set".into()))?;
    let outcome = trainer.train(&TrainRequest {
        run_dir: config.run_dir.clone(),
        train_file: config.run_dir.join("data/train.jsonl"),
        validation_file: config.run_dir.join("data/val.jsonl"),
        parent_model: config.parent_bf16.clone(),
        config: config.training.clone(),
    })?;
    if let Some(checkpoint) = &outcome.checkpoint {
        write_json(&config.run_dir.join("selected_checkpoint.json"), checkpoint)?;
        println!("selected checkpoint step {}", checkpoint.step);
    } else {
        println!("no saved checkpoint with a finite validation loss");
    }
    let _ = tasks_dir;
    Ok(())
}

fn export_stage(config_path: &Path, tasks_dir: &Path) -> Result<()> {
    let config = load(config_path)?;
    let checkpoint: evolving_orchestrator::train::Checkpoint =
        read_json(&config.run_dir.join("selected_checkpoint.json"))?;
    let base_model = config
        .parent_bf16
        .clone()
        .ok_or_else(|| EvolveError::Training("parent_bf16 is required for export".into()))?;
    let exporter = CommandExporter::from_env()
        .ok_or_else(|| EvolveError::Training("EVOLVE_EXPORT_COMMAND is not set".into()))?;
    exporter.export(&ExportRequest {
        base_model,
        adapter: checkpoint.path,
        out_dir: config.run_dir.join("exported"),
        lora_r: config.training.lora_r,
        lora_alpha: config.training.lora_alpha,
        quantize: true,
    })?;
    println!("export complete");
    let _ = tasks_dir;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TaskLoopState {
    attempt: u32,
    accepted: bool,
    best_score: f64,
    best_gates_passed: usize,
    best_design: Option<String>,
    best_workspace: Option<PathBuf>,
    last_reasons: Vec<String>,
    stagnation: u32,
}

#[derive(Debug, Clone, Serialize)]
struct AttemptRecord {
    attempt: u32,
    accepted: bool,
    score: f64,
    gates_passed: usize,
    reasons: Vec<String>,
    error: Option<String>,
    duration_seconds: u64,
}

fn task_stage(
    config_path: &Path,
    tasks_dir: &Path,
    repo: Option<&Path>,
    max_attempts: u32,
    stagnation_limit: u32,
) -> Result<()> {
    let config = load(config_path)?;
    if config.quality.gates.is_empty() && config.quality.dimensions.is_empty() {
        return Err(EvolveError::Config(
            "quality policy is empty; the task loop would accept immediately".into(),
        ));
    }
    let catalog = TaskCatalog::new(tasks_dir);
    let chat = chat(&config);
    let judge = ModelJudge {
        chat: &chat,
        options: options(&config, 0),
    };
    let proposer = ModelProposer {
        chat: &chat,
        options: options(&config, 1),
    };
    let rubric = ModelRubricJudge {
        chat: &chat,
        options: options(&config, 2),
    };
    let runner = ProcessRunner::new(config.executor.clone());
    let tokenizer: Box<dyn Tokenizer> = match CommandTokenizer::from_env() {
        Some(tokenizer) => Box::new(tokenizer),
        None => {
            eprintln!("warning: EVOLVE_TOKENIZE_COMMAND is not set; using byte tokens");
            Box::new(ByteTokenizer)
        }
    };
    let trainer = CommandTrainer::new("/bin/false", Vec::new());
    let exporter = CommandExporter::new("/bin/false", Vec::new());

    let state_path = config.run_dir.join("task-loop.json");
    let attempts_path = config.run_dir.join("attempts.jsonl");
    fs::create_dir_all(&config.run_dir)?;
    let mut state: TaskLoopState = if state_path.is_file() {
        read_json(&state_path)?
    } else {
        TaskLoopState::default()
    };
    if state.accepted {
        println!("task loop already accepted at attempt {}", state.attempt);
        return Ok(());
    }

    loop {
        let attempt = state.attempt + 1;
        if max_attempts > 0 && attempt > max_attempts {
            return Err(EvolveError::Training(format!(
                "no accepted workspace after {max_attempts} attempts; best score {:.1}",
                state.best_score
            )));
        }
        let attempt_dir = config
            .run_dir
            .join("attempts")
            .join(format!("{attempt:04}"));
        fs::create_dir_all(&attempt_dir)?;
        let mut attempt_config = config.clone();
        attempt_config.run_dir = attempt_dir.clone();

        let base_workspace = state
            .best_workspace
            .clone()
            .or_else(|| repo.map(Path::to_path_buf));
        let options = CycleOptions {
            train: false,
            base_workspace,
            initial_design: state.best_design.clone(),
            gate_feedback: if state.last_reasons.is_empty() {
                None
            } else {
                Some(state.last_reasons.join("\n"))
            },
        };

        let started = std::time::Instant::now();
        let outcome = run_cycle_with_options(
            &attempt_config,
            &catalog,
            &CyclePaths {
                run_dir: attempt_dir.clone(),
                tasks_dir: tasks_dir.to_path_buf(),
                teacher_directive: tasks_dir.join("teacher_directive.txt"),
            },
            &CyclePorts {
                runner: &runner,
                judge: &judge,
                proposer: &proposer,
                tokenizer: tokenizer.as_ref(),
                trainer: &trainer,
                exporter: &exporter,
                quality_judge: &rubric,
            },
            options,
        );

        let (accepted, score, gates_passed, reasons, error) = match &outcome {
            Ok(outcome) => {
                let mut reports = Vec::new();
                for task in &attempt_config.tasks.train {
                    let path = attempt_dir
                        .join("outputs")
                        .join(format!("query{task}"))
                        .join("quality.json");
                    if path.is_file() {
                        reports.push(read_json::<QualityReport>(&path)?);
                    }
                }
                let score = reports
                    .iter()
                    .map(|report| report.weighted_score)
                    .fold(f64::INFINITY, f64::min);
                let score = if score.is_finite() { score } else { 0.0 };
                let gates_passed = reports
                    .iter()
                    .flat_map(|report| report.gates.iter())
                    .filter(|gate| gate.status == GateStatus::Pass)
                    .count();
                let reasons: Vec<String> = reports
                    .iter()
                    .flat_map(|report| report.reasons.iter().cloned())
                    .collect();
                (outcome.summary.accepted, score, gates_passed, reasons, None)
            }
            Err(error) => (
                false,
                state.best_score,
                state.best_gates_passed,
                Vec::new(),
                Some(error.to_string()),
            ),
        };

        let duration_seconds = started.elapsed().as_secs();
        let improved =
            accepted || score > state.best_score + 0.5 || gates_passed > state.best_gates_passed;
        if accepted {
            state.accepted = true;
        } else if improved {
            state.best_score = score;
            state.best_gates_passed = gates_passed;
            state.stagnation = 0;
            if let Ok(outcome) = &outcome {
                if let Some((_, rows)) = outcome.selections.first() {
                    if let Some(best) = rows.first() {
                        state.best_workspace = Some(best.workspace.clone());
                    }
                }
                if let Some((_, search)) = outcome.searches.first() {
                    state.best_design = Some(search.champion_design.clone());
                }
            }
        } else {
            state.stagnation += 1;
        }
        state.attempt = attempt;
        state.last_reasons = reasons.clone();
        write_json(&state_path, &state)?;
        append_record(
            &attempts_path,
            &AttemptRecord {
                attempt,
                accepted,
                score,
                gates_passed,
                reasons: reasons.clone(),
                error: error.clone(),
                duration_seconds,
            },
        )?;

        println!(
            "attempt {attempt}: accepted={accepted} score={score:.1} gates_passed={gates_passed} stagnation={}",
            state.stagnation
        );
        if let Some(error) = error {
            eprintln!("attempt {attempt} error: {error}");
        }

        if accepted {
            if let Ok(outcome) = &outcome {
                publish_attempt_outputs(&config.run_dir, &attempt_dir, outcome)?;
            }
            println!(
                "accepted after {attempt} attempt(s); outputs at {}",
                config.run_dir.join("outputs").display()
            );
            return Ok(());
        }
        if stagnation_limit > 0 && state.stagnation >= stagnation_limit {
            return Err(EvolveError::Training(format!(
                "stopping after {} attempts without measurable progress; best score {:.1}, last reasons: {}",
                state.stagnation,
                state.best_score,
                state.last_reasons.join("; ")
            )));
        }
    }
}

fn publish_attempt_outputs(
    final_run_dir: &Path,
    attempt_dir: &Path,
    outcome: &evolving_orchestrator::cycle::CycleOutcome,
) -> Result<()> {
    for (task, _) in &outcome.selections {
        let source = attempt_dir.join("outputs").join(format!("query{task}"));
        let target = final_run_dir.join("outputs").join(format!("query{task}"));
        if source.is_dir() {
            copy_tree(&source, &target)?;
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let destination = target.join(entry.file_name());
        if entry.metadata()?.is_dir() {
            copy_tree(&path, &destination)?;
        } else {
            fs::copy(&path, &destination)?;
        }
    }
    Ok(())
}

fn cycle_stage(config_path: &Path, tasks_dir: &Path, task_only: bool) -> Result<()> {
    let config = load(config_path)?;
    let catalog = TaskCatalog::new(tasks_dir);
    let chat = chat(&config);
    let judge = ModelJudge {
        chat: &chat,
        options: options(&config, 0),
    };
    let proposer = ModelProposer {
        chat: &chat,
        options: options(&config, 1),
    };
    let runner = ProcessRunner::new(config.executor.clone());
    let rubric = ModelRubricJudge {
        chat: &chat,
        options: options(&config, 2),
    };
    let trainer = CommandTrainer::from_env();
    let exporter = CommandExporter::from_env();
    if !task_only && (trainer.is_none() || exporter.is_none()) {
        return Err(EvolveError::Training(
            "EVOLVE_TRAIN_COMMAND and EVOLVE_EXPORT_COMMAND are required unless --task-only is set"
                .into(),
        ));
    }
    let trainer = trainer.unwrap_or_else(|| CommandTrainer::new("/bin/false", Vec::new()));
    let exporter = exporter.unwrap_or_else(|| CommandExporter::new("/bin/false", Vec::new()));
    let tokenizer: Box<dyn Tokenizer> = match CommandTokenizer::from_env() {
        Some(tokenizer) => Box::new(tokenizer),
        None => {
            eprintln!("warning: EVOLVE_TOKENIZE_COMMAND is not set; using byte tokens");
            Box::new(ByteTokenizer)
        }
    };

    let outcome = run_cycle_with_options(
        &config,
        &catalog,
        &CyclePaths {
            run_dir: config.run_dir.clone(),
            tasks_dir: tasks_dir.to_path_buf(),
            teacher_directive: tasks_dir.join("teacher_directive.txt"),
        },
        &CyclePorts {
            runner: &runner,
            judge: &judge,
            proposer: &proposer,
            tokenizer: tokenizer.as_ref(),
            trainer: &trainer,
            exporter: &exporter,
            quality_judge: &rubric,
        },
        CycleOptions {
            train: !task_only,
            ..CycleOptions::default()
        },
    )?;
    println!(
        "cycle {}: {} episodes, {} windows, checkpoint {}, exported {}, trained {}",
        outcome.summary.generation,
        outcome.summary.episodes,
        outcome.summary.windows,
        outcome
            .summary
            .checkpoint_step
            .map(|step| step.to_string())
            .unwrap_or_else(|| "none".into()),
        outcome.exported,
        outcome.summary.trained
    );
    Ok(())
}

fn read_teacher_directive(tasks_dir: &Path) -> Result<String> {
    let path = tasks_dir.join("teacher_directive.txt");
    std::fs::read_to_string(&path).map_err(|error| {
        EvolveError::Config(format!(
            "cannot read teacher directive {}: {error}",
            path.display()
        ))
    })
}
