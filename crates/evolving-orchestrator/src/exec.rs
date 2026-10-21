//! Episode execution.
//!
//! One episode is one process in one workspace. The runner writes the prompt,
//! invokes the configured program, captures stdout/stderr, records a status
//! file, and collects the session files the executor produced.
//!
//! Pi is the reference executor. When the program is `pi`, the runner adds
//! `--session-dir` and `--session-id` so the conversation is persisted as
//! JSONL for the renderer. Other programs are invoked exactly as configured.

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ExecutorConfig;
use crate::error::{EvolveError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeStatus {
    Success,
    Failed,
    Timeout,
}

impl EpisodeStatus {
    pub fn is_success(self) -> bool {
        self == EpisodeStatus::Success
    }
}

#[derive(Debug, Clone)]
pub struct EpisodeSpec<'a> {
    pub prompt: &'a str,
    pub workspace: &'a Path,
    pub logs: &'a Path,
    pub condition: &'a str,
    pub seed: u64,
    pub model_id: &'a str,
    pub endpoint: &'a str,
}

#[derive(Debug, Clone)]
pub struct EpisodeOutcome {
    pub status: EpisodeStatus,
    pub exit_code: Option<i32>,
    pub duration: Duration,
    pub prompt_path: PathBuf,
    pub stream_path: PathBuf,
    pub stderr_path: PathBuf,
    pub session_dir: PathBuf,
    pub session_files: Vec<PathBuf>,
}

impl EpisodeOutcome {
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }
}

pub trait EpisodeRunner {
    fn run(&self, spec: &EpisodeSpec<'_>) -> Result<EpisodeOutcome>;
}

#[derive(Debug, Clone)]
pub struct ProcessRunner {
    pub config: ExecutorConfig,
}

impl ProcessRunner {
    pub fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    pub fn is_pi(&self) -> bool {
        Path::new(&self.config.program)
            .file_name()
            .and_then(OsStr::to_str)
            == Some("pi")
    }
}

impl EpisodeRunner for ProcessRunner {
    fn run(&self, spec: &EpisodeSpec<'_>) -> Result<EpisodeOutcome> {
        let prompt_path = spec.logs.join("prompt.txt");
        let stream_path = spec.logs.join("stream.jsonl");
        let stderr_path = spec.logs.join("stderr.log");
        let session_dir = spec.logs.join("sessions");

        // A completed episode is reused; a partial one is preserved and
        // refused, so infrastructure failures are never silently overwritten.
        if let Some(outcome) = read_completed(
            spec.logs,
            &prompt_path,
            &stream_path,
            &stderr_path,
            &session_dir,
        )? {
            return Ok(outcome);
        }

        if spec.workspace.exists() {
            let mut entries = fs::read_dir(spec.workspace).map_err(|error| {
                EvolveError::Episode(format!(
                    "cannot read workspace {}: {error}",
                    spec.workspace.display()
                ))
            })?;
            if entries.next().is_some() {
                return Err(EvolveError::Episode(format!(
                    "refusing to run in non-empty workspace {}",
                    spec.workspace.display()
                )));
            }
        } else {
            fs::create_dir_all(spec.workspace)?;
        }

        fs::create_dir_all(spec.logs)?;
        if spec.logs.read_dir()?.next().is_some() {
            return Err(EvolveError::Episode(format!(
                "logs directory already contains partial output: {}",
                spec.logs.display()
            )));
        }
        fs::create_dir_all(&session_dir)?;
        fs::write(&prompt_path, spec.prompt)?;

        let is_pi = self.is_pi();
        let mut args = self.config.args.clone();
        if is_pi {
            add_pi_session_args(&mut args, &session_dir, spec);
        }
        if self.config.prompt_arg {
            if spec.prompt.len() > 500_000 {
                return Err(EvolveError::Episode(
                    "prompt is too large for a command-line argument; use stdin_prompt".into(),
                ));
            }
            args.push(spec.prompt.to_string());
        }

        let started = Instant::now();
        let started_unix_ms = now_ms();
        let stdout = File::create(&stream_path)?;
        let stderr = File::create(&stderr_path)?;

        let mut command = Command::new(&self.config.program);
        command
            .args(&args)
            .current_dir(spec.workspace)
            .env("EVOLVE_MODEL_ID", spec.model_id)
            .env("EVOLVE_ENDPOINT", spec.endpoint)
            .env("EVOLVE_PROMPT_FILE", &prompt_path)
            .env("EVOLVE_WORKSPACE", spec.workspace)
            .env("EVOLVE_LOGS", spec.logs)
            .env("EVOLVE_SEED", spec.seed.to_string())
            .env("EVOLVE_CONDITION", spec.condition)
            .env("EVOLVE_SUBAGENT_DEPTH", "0")
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));

        let stdin_pipe = self.config.stdin_prompt;
        command.stdin(if stdin_pipe {
            Stdio::piped()
        } else {
            Stdio::null()
        });

        let mut child = command.spawn().map_err(|error| {
            EvolveError::Episode(format!(
                "cannot start executor {:?}: {error}",
                self.config.program
            ))
        })?;

        if stdin_pipe {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(spec.prompt.as_bytes())?;
            }
        }

        let timeout = Duration::from_secs(self.config.timeout_seconds.max(1));
        let mut timed_out = false;
        let exit_code = loop {
            match child.try_wait()? {
                Some(status) => break status.code(),
                None if started.elapsed() >= timeout => {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                None => thread::sleep(Duration::from_millis(200)),
            }
        };

        let duration = started.elapsed();
        let stream_success = if is_pi {
            Some(pi_trace_success(&stream_path))
        } else {
            None
        };
        let trace_has_success_result = match stream_success {
            Some(value) => value,
            None => exit_code == Some(0) && !timed_out,
        };
        let status = if timed_out {
            EpisodeStatus::Timeout
        } else if exit_code == Some(0) && trace_has_success_result {
            EpisodeStatus::Success
        } else {
            EpisodeStatus::Failed
        };

        let session_files = collect_sessions(&session_dir)?;
        let stderr_tail = read_tail(&stderr_path, 4_000);
        let record = RunStatus {
            condition: spec.condition.to_string(),
            seed: spec.seed,
            model_id: spec.model_id.to_string(),
            exit_code,
            trace_has_success_result,
            status,
            started_unix_ms,
            finished_unix_ms: now_ms(),
            duration_ms: duration.as_millis(),
            prompt_file: prompt_path.display().to_string(),
            workspace: spec.workspace.display().to_string(),
            session_dir: session_dir.display().to_string(),
            session_files: session_files
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
            timed_out,
            stderr_tail,
        };
        let status_path = spec.logs.join("run_status.json");
        fs::write(&status_path, serde_json::to_string_pretty(&record)? + "\n")?;

        Ok(EpisodeOutcome {
            status,
            exit_code,
            duration,
            prompt_path,
            stream_path,
            stderr_path,
            session_dir,
            session_files,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RunStatus {
    condition: String,
    seed: u64,
    model_id: String,
    exit_code: Option<i32>,
    trace_has_success_result: bool,
    status: EpisodeStatus,
    started_unix_ms: u128,
    finished_unix_ms: u128,
    duration_ms: u128,
    prompt_file: String,
    workspace: String,
    session_dir: String,
    session_files: Vec<String>,
    timed_out: bool,
    stderr_tail: String,
}

fn read_completed(
    logs: &Path,
    prompt_path: &Path,
    stream_path: &Path,
    stderr_path: &Path,
    session_dir: &Path,
) -> Result<Option<EpisodeOutcome>> {
    let status_path = logs.join("run_status.json");
    if !status_path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&status_path)?;
    let record: RunStatus = serde_json::from_str(&text).map_err(|error| {
        EvolveError::Episode(format!(
            "{}: invalid run status: {error}",
            status_path.display()
        ))
    })?;
    if !record.trace_has_success_result || !record.status.is_success() {
        return Err(EvolveError::Episode(format!(
            "logs contain an unsuccessful episode ({}): {}",
            record.status_str(),
            logs.display()
        )));
    }
    Ok(Some(EpisodeOutcome {
        status: record.status,
        exit_code: record.exit_code,
        duration: Duration::from_millis(record.duration_ms as u64),
        prompt_path: prompt_path.to_path_buf(),
        stream_path: stream_path.to_path_buf(),
        stderr_path: stderr_path.to_path_buf(),
        session_dir: session_dir.to_path_buf(),
        session_files: collect_sessions(session_dir)?,
    }))
}

impl RunStatus {
    fn status_str(&self) -> &'static str {
        match self.status {
            EpisodeStatus::Success => "success",
            EpisodeStatus::Failed => "failed",
            EpisodeStatus::Timeout => "timeout",
        }
    }
}

fn add_pi_session_args(args: &mut Vec<String>, session_dir: &Path, spec: &EpisodeSpec<'_>) {
    if args.iter().any(|value| value == "--no-session") {
        return;
    }
    let has_session_dir = args.iter().any(|value| value == "--session-dir");
    if !has_session_dir {
        args.push("--session-dir".into());
        args.push(session_dir.display().to_string());
    }
    if !args.iter().any(|value| value == "--session-id") {
        args.push("--session-id".into());
        args.push(format!("{}-{}", sanitize(spec.condition), spec.seed));
    }
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

/// Pi reports success with a terminal assistant message and `agent_settled`.
/// A final `error` or `aborted` stop reason is a failure even when the
/// process exits zero.
fn pi_trace_success(path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let mut last_stop: Option<String> = None;
    let mut settled = false;
    let mut saw_event = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        saw_event = true;
        match value.get("type").and_then(Value::as_str) {
            Some("message_end") => {
                let message = &value["message"];
                if message.get("role").and_then(Value::as_str) == Some("assistant") {
                    if let Some(reason) = message.get("stopReason").and_then(Value::as_str) {
                        last_stop = Some(reason.to_string());
                    }
                }
            }
            Some("agent_settled") => settled = true,
            _ => {}
        }
    }
    if !saw_event {
        return false;
    }
    match last_stop.as_deref() {
        Some("error") | Some("aborted") => false,
        Some("stop") | Some("toolUse") | Some("length") => true,
        _ => settled,
    }
}

fn collect_sessions(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !directory.exists() {
        return Ok(out);
    }
    collect_sessions_inner(directory, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_sessions_inner(directory: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_sessions_inner(&path, out)?;
        } else if path.extension().and_then(OsStr::to_str) == Some("jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

fn read_tail(path: &Path, limit: usize) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return String::new();
    };
    if text.len() <= limit {
        return text;
    }
    let mut start = text.len() - limit;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    format!("… [truncated]\n{}", &text[start..])
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("evolve-{label}-{}-{stamp}", std::process::id()))
    }

    fn shell_config(script: &str, timeout: u64) -> ExecutorConfig {
        ExecutorConfig {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            stdin_prompt: true,
            prompt_arg: false,
            timeout_seconds: timeout,
        }
    }

    #[test]
    fn runs_a_process_and_collects_a_pi_style_session() {
        let root = unique_dir("exec-ok");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        let script = r#"
cat >/dev/null
mkdir -p "$EVOLVE_LOGS/sessions"
printf '%s\n' '{"type":"session","version":3,"id":"s1","cwd":"/tmp"}' > "$EVOLVE_LOGS/sessions/s1.jsonl"
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"done"}],"stopReason":"stop"}}' '{"type":"agent_settled"}' > "$EVOLVE_LOGS/stream.jsonl"
"#;
        let runner = ProcessRunner::new(shell_config(script, 10));
        let outcome = runner
            .run(&EpisodeSpec {
                prompt: "fix it",
                workspace: &workspace,
                logs: &logs,
                condition: "t1_v0",
                seed: 42,
                model_id: "local",
                endpoint: "http://127.0.0.1:8080/v1",
            })
            .expect("run");
        assert_eq!(outcome.status, EpisodeStatus::Success);
        assert_eq!(outcome.session_files.len(), 1);
        assert_eq!(fs::read_to_string(&outcome.prompt_path).unwrap(), "fix it");
        assert!(logs.join("run_status.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reuses_a_completed_episode() {
        let root = unique_dir("exec-reuse");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        let script = r#"
cat >/dev/null
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}' '{"type":"agent_settled"}' > "$EVOLVE_LOGS/stream.jsonl"
"#;
        let runner = ProcessRunner::new(shell_config(script, 10));
        let spec = EpisodeSpec {
            prompt: "p",
            workspace: &workspace,
            logs: &logs,
            condition: "t1_v0",
            seed: 1,
            model_id: "local",
            endpoint: "http://127.0.0.1:8080/v1",
        };
        runner.run(&spec).expect("first");
        runner.run(&spec).expect("cached");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refuses_a_non_empty_workspace() {
        let root = unique_dir("exec-nonempty");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        fs::create_dir_all(&workspace).unwrap();
        fs::write(workspace.join("file.txt"), "x").unwrap();
        let runner = ProcessRunner::new(shell_config("cat >/dev/null", 5));
        let error = runner
            .run(&EpisodeSpec {
                prompt: "p",
                workspace: &workspace,
                logs: &logs,
                condition: "t1_v0",
                seed: 1,
                model_id: "local",
                endpoint: "http://127.0.0.1:8080/v1",
            })
            .expect_err("must refuse");
        assert!(error.to_string().contains("non-empty"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    fn pi_config(root: &Path, script: &str, timeout: u64) -> ExecutorConfig {
        use std::os::unix::fs::PermissionsExt;

        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let pi = bin_dir.join("pi");
        fs::write(&pi, format!("#!/bin/sh\n{script}")).unwrap();
        fs::set_permissions(&pi, fs::Permissions::from_mode(0o755)).unwrap();
        ExecutorConfig {
            program: pi.display().to_string(),
            args: Vec::new(),
            stdin_prompt: true,
            prompt_arg: false,
            timeout_seconds: timeout,
        }
    }

    #[cfg(unix)]
    #[test]
    fn treats_a_pi_error_stop_reason_as_failed() {
        let root = unique_dir("exec-pi-error");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        let script = r#"
cat >/dev/null
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","stopReason":"error"}}' '{"type":"agent_settled"}' > "$EVOLVE_LOGS/stream.jsonl"
"#;
        let runner = ProcessRunner::new(pi_config(&root, script, 10));
        let outcome = runner
            .run(&EpisodeSpec {
                prompt: "p",
                workspace: &workspace,
                logs: &logs,
                condition: "t1_v0",
                seed: 1,
                model_id: "local",
                endpoint: "http://127.0.0.1:8080/v1",
            })
            .expect("run");
        assert_eq!(outcome.status, EpisodeStatus::Failed);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn adds_pi_session_flags_for_a_pi_executable() {
        let root = unique_dir("exec-pi-args");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        let script = r#"
printf '%s\n' "$@" > "$EVOLVE_LOGS/args.txt"
cat >/dev/null
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}' '{"type":"agent_settled"}' > "$EVOLVE_LOGS/stream.jsonl"
"#;
        let config = pi_config(&root, script, 10);
        ProcessRunner::new(config)
            .run(&EpisodeSpec {
                prompt: "p",
                workspace: &workspace,
                logs: &logs,
                condition: "t1_v0",
                seed: 7,
                model_id: "local",
                endpoint: "http://127.0.0.1:8080/v1",
            })
            .expect("run");

        let args = fs::read_to_string(logs.join("args.txt")).unwrap();
        assert!(args.contains("--session-dir"));
        assert!(args.contains("sessions"));
        assert!(args.contains("--session-id"));
        assert!(args.contains("t1_v0-7"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn kills_a_timed_out_process() {
        let root = unique_dir("exec-timeout");
        let workspace = root.join("workspace");
        let logs = root.join("logs");
        let runner = ProcessRunner::new(shell_config("sleep 5", 1));
        let outcome = runner
            .run(&EpisodeSpec {
                prompt: "p",
                workspace: &workspace,
                logs: &logs,
                condition: "t1_v0",
                seed: 1,
                model_id: "local",
                endpoint: "http://127.0.0.1:8080/v1",
            })
            .expect("run");
        assert_eq!(outcome.status, EpisodeStatus::Timeout);
        fs::remove_dir_all(root).unwrap();
    }
}
