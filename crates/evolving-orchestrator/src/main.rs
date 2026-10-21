//! Command-line entry point.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use evolving_orchestrator::chat::{ChatOptions, OpenAiChat};
use evolving_orchestrator::collect::{collect, read_pool};
use evolving_orchestrator::config::CycleConfig;
use evolving_orchestrator::cycle::{run_cycle, CyclePaths, CyclePorts};
use evolving_orchestrator::design::TaskCatalog;
use evolving_orchestrator::error::{EvolveError, Result};
use evolving_orchestrator::exec::ProcessRunner;
use evolving_orchestrator::export::{CommandExporter, ExportRequest, Exporter};
use evolving_orchestrator::io::{read_json, write_json};
use evolving_orchestrator::judge::ModelJudge;
use evolving_orchestrator::pi_session::load_sessions;
use evolving_orchestrator::render::{
    write_windows, ByteTokenizer, CommandTokenizer, PromptSubstitution, RenderConfig, SftWindow,
    Tokenizer,
};
use evolving_orchestrator::search::{
    champion_design, read_retained, run_search, ModelProposer, SearchPaths,
};
use evolving_orchestrator::select::{rank_candidates, select_rows, Candidate, RankedEpisode};
use evolving_orchestrator::train::{CommandTrainer, TrainRequest, Trainer};

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
    /// Run one complete cycle.
    Cycle {
        #[arg(long, default_value = "configs/evolve/local-3b.json")]
        config: PathBuf,
        #[arg(long, default_value = "configs/evolve/tasks")]
        tasks: PathBuf,
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
        Command::Cycle { config, tasks } => cycle_stage(&config, &tasks),
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

fn cycle_stage(config_path: &Path, tasks_dir: &Path) -> Result<()> {
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
    let trainer = CommandTrainer::from_env()
        .ok_or_else(|| EvolveError::Training("EVOLVE_TRAIN_COMMAND is not set".into()))?;
    let exporter = CommandExporter::from_env()
        .ok_or_else(|| EvolveError::Training("EVOLVE_EXPORT_COMMAND is not set".into()))?;
    let tokenizer: Box<dyn Tokenizer> = match CommandTokenizer::from_env() {
        Some(tokenizer) => Box::new(tokenizer),
        None => {
            eprintln!("warning: EVOLVE_TOKENIZE_COMMAND is not set; using byte tokens");
            Box::new(ByteTokenizer)
        }
    };

    let outcome = run_cycle(
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
        },
    )?;
    println!(
        "cycle {}: {} episodes, {} windows, checkpoint {}, exported {}",
        outcome.summary.generation,
        outcome.summary.episodes,
        outcome.summary.windows,
        outcome
            .summary
            .checkpoint_step
            .map(|step| step.to_string())
            .unwrap_or_else(|| "none".into()),
        outcome.exported
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
