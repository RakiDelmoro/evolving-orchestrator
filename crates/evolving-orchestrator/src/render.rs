//! Render Pi sessions into masked supervised windows.
//!
//! The tokenizer is a port: production uses the model's tokenizer through
//! `EVOLVE_TOKENIZE_COMMAND`; tests use one byte per token. The masking rule is
//! fixed: assistant reasoning, text, and tool calls are targets; system, user,
//! tool results, and overlap context are masked.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::error::{EvolveError, Result};
use crate::pi_session::{PiMessage, PiRole, PiSession};

pub const IGNORE: i64 = -100;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }
}

#[derive(Debug, Clone)]
pub struct CommandTokenizer {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandTokenizer {
    pub fn from_env() -> Option<Self> {
        std::env::var("EVOLVE_TOKENIZE_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| Self {
                program: "/bin/sh".into(),
                args: vec!["-c".into(), value],
            })
    }
}

impl Tokenizer for CommandTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| EvolveError::Training(format!("cannot start tokenizer: {error}")))?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(text.as_bytes())?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(EvolveError::Training(format!(
                "tokenizer exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let text = String::from_utf8(output.stdout).map_err(|error| {
            EvolveError::Training(format!("tokenizer output is not UTF-8: {error}"))
        })?;
        let ids: Vec<u32> = serde_json::from_str(text.trim()).map_err(|error| {
            EvolveError::Training(format!("tokenizer output is not a JSON array: {error}"))
        })?;
        Ok(ids)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderConfig {
    pub window: usize,
    pub context_overlap: usize,
}

impl RenderConfig {
    pub fn validate(self) -> Result<()> {
        if self.window == 0 {
            return Err(EvolveError::Training(
                "render window must be positive".into(),
            ));
        }
        if self.context_overlap >= self.window {
            return Err(EvolveError::Training(
                "context_overlap must be smaller than window".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSubstitution {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SftWindow {
    pub episode: String,
    pub task: u32,
    pub thread: String,
    pub role: String,
    pub input_ids: Vec<u32>,
    pub labels: Vec<i64>,
    pub n_tokens: usize,
    pub n_loss_tokens: usize,
}

#[derive(Debug, Clone)]
struct Block {
    role: String,
    ids: Vec<u32>,
    loss: bool,
}

pub fn render_session(
    session: &PiSession,
    tokenizer: &dyn Tokenizer,
    config: RenderConfig,
    substitution: Option<&PromptSubstitution>,
    episode: &str,
    task: u32,
    thread: &str,
) -> Result<Vec<SftWindow>> {
    config.validate()?;

    let mut blocks: Vec<Block> = Vec::new();
    let mut first_user = true;
    for message in &session.messages {
        let mut text = message_text(message);
        if first_user && message.role == PiRole::User {
            first_user = false;
            if let Some(substitution) = substitution {
                if !substitution.from.is_empty() {
                    text = text.replacen(&substitution.from, &substitution.to, 1);
                }
            }
        }
        if text.trim().is_empty() {
            continue;
        }
        let ids = tokenizer.encode(&text)?;
        if ids.is_empty() {
            continue;
        }
        blocks.push(Block {
            role: message.role.as_str().to_string(),
            ids,
            loss: message.loss(),
        });
    }

    let header_len = blocks
        .iter()
        .take_while(|block| !block.loss)
        .take(2)
        .map(|block| block.ids.len())
        .sum::<usize>();
    let header_count = blocks
        .iter()
        .take_while(|block| !block.loss)
        .take(2)
        .count();
    if header_len > config.window {
        return Err(EvolveError::Training(format!(
            "header is larger than the window: {header_len} > {}",
            config.window
        )));
    }

    let body = &blocks[header_count..];
    let mut windows = Vec::new();
    let mut index = 0usize;
    while index < body.len() {
        if !body[index].loss {
            index += 1;
            continue;
        }

        let mut context_start = index;
        let mut context_tokens = 0usize;
        while context_start > 0
            && context_tokens + body[context_start - 1].ids.len() <= config.context_overlap
        {
            context_start -= 1;
            context_tokens += body[context_start].ids.len();
        }

        let budget = config.window.saturating_sub(header_len + context_tokens);
        let mut end = index;
        let mut used = 0usize;
        while end < body.len() && used + body[end].ids.len() <= budget {
            used += body[end].ids.len();
            end += 1;
            if end > index && body[end - 1].loss {
                // Keep extending while there is room; the loop continues.
            }
        }
        if end == index {
            // The first loss block does not fit; take its prefix so the
            // episode is represented rather than silently dropped.
            let take = budget.min(body[index].ids.len());
            if take == 0 {
                index += 1;
                continue;
            }
            let mut input_ids = Vec::new();
            let mut labels = Vec::new();
            for block in &blocks[..header_count] {
                input_ids.extend(&block.ids);
                labels.extend(std::iter::repeat_n(IGNORE, block.ids.len()));
            }
            for block in &body[context_start..index] {
                input_ids.extend(&block.ids);
                labels.extend(std::iter::repeat_n(IGNORE, block.ids.len()));
            }
            input_ids.extend(&body[index].ids[..take]);
            labels.extend(body[index].ids[..take].iter().map(|id| *id as i64));
            windows.push(SftWindow {
                episode: episode.to_string(),
                task,
                thread: thread.to_string(),
                role: body[index].role.clone(),
                n_tokens: input_ids.len(),
                n_loss_tokens: take,
                input_ids,
                labels,
            });
            index += 1;
            continue;
        }

        let mut input_ids = Vec::new();
        let mut labels = Vec::new();
        for block in &blocks[..header_count] {
            input_ids.extend(&block.ids);
            labels.extend(std::iter::repeat_n(IGNORE, block.ids.len()));
        }
        for block in &body[context_start..index] {
            input_ids.extend(&block.ids);
            labels.extend(std::iter::repeat_n(IGNORE, block.ids.len()));
        }
        for block in &body[index..end] {
            input_ids.extend(&block.ids);
            if block.loss {
                labels.extend(block.ids.iter().map(|id| *id as i64));
            } else {
                labels.extend(std::iter::repeat_n(IGNORE, block.ids.len()));
            }
        }
        if input_ids.len() > config.window {
            index = end;
            continue;
        }
        let n_loss_tokens = labels.iter().filter(|label| **label != IGNORE).count();
        windows.push(SftWindow {
            episode: episode.to_string(),
            task,
            thread: thread.to_string(),
            role: "mixed".into(),
            n_tokens: input_ids.len(),
            n_loss_tokens,
            input_ids,
            labels,
        });
        index = end;
    }

    Ok(windows)
}

fn message_text(message: &PiMessage) -> String {
    match message.role {
        PiRole::System => message
            .sections
            .as_ref()
            .map(|sections| sections.values().cloned().collect::<Vec<_>>().join("\n"))
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| message.text()),
        PiRole::Assistant => {
            let mut text = String::new();
            if !message.thinking().is_empty() {
                text.push_str(&message.thinking());
                text.push('\n');
            }
            text.push_str(&message.text());
            for (id, name, arguments) in message.tool_calls() {
                text.push_str(&format!(
                    "\n[tool call {name} id={id}]\n{}\n",
                    serde_json::to_string(arguments).unwrap_or_default()
                ));
            }
            text
        }
        _ => message.text(),
    }
}

pub fn render_sessions(
    sessions: &[PiSession],
    tokenizer: &dyn Tokenizer,
    config: RenderConfig,
    substitution: Option<&PromptSubstitution>,
    episode: &str,
    task: u32,
) -> Result<Vec<SftWindow>> {
    let mut windows = Vec::new();
    for session in sessions {
        let thread = session
            .path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("session");
        windows.extend(render_session(
            session,
            tokenizer,
            config,
            substitution,
            episode,
            task,
            thread,
        )?);
    }
    Ok(windows)
}

pub fn write_windows(path: &Path, windows: &[SftWindow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for window in windows {
        writeln!(file, "{}", serde_json::to_string(window)?)?;
    }
    Ok(())
}

pub fn read_tokenizer_command(command: &str) -> CommandTokenizer {
    CommandTokenizer {
        program: "/bin/sh".into(),
        args: vec!["-c".into(), command.into()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pi_session::parse_session;
    use std::path::Path;

    const SESSION: &str = r#"{"type":"session","version":3,"id":"s"}
{"type":"message","message":{"role":"system","content":"sys"}}
{"type":"message","message":{"role":"user","content":"fix the test"}}
{"type":"message","message":{"role":"assistant","content":[{"type":"thinking","thinking":"think"},{"type":"text","text":"done"}],"stopReason":"stop"}}
"#;

    #[test]
    fn masks_user_and_keeps_assistant_targets() {
        let session = parse_session(Path::new("s.jsonl"), SESSION).expect("parse");
        let windows = render_session(
            &session,
            &ByteTokenizer,
            RenderConfig {
                window: 256,
                context_overlap: 8,
            },
            None,
            "ep",
            1,
            "main",
        )
        .expect("render");
        assert_eq!(windows.len(), 1);
        let window = &windows[0];
        assert!(window.n_loss_tokens > 0);
        // The header is system + user and is masked.
        assert_eq!(window.labels[0], IGNORE);
        // At least one assistant token is a target.
        assert!(window.labels.iter().any(|label| *label != IGNORE));
    }

    #[test]
    fn substitutes_the_teacher_prompt_once() {
        let session = parse_session(Path::new("s.jsonl"), SESSION).expect("parse");
        let windows = render_session(
            &session,
            &ByteTokenizer,
            RenderConfig {
                window: 256,
                context_overlap: 8,
            },
            Some(&PromptSubstitution {
                from: "fix the test".into(),
                to: "bare".into(),
            }),
            "ep",
            1,
            "main",
        )
        .expect("render");
        let ids = &windows[0].input_ids;
        let text: String = ids.iter().map(|id| *id as u8 as char).collect();
        assert!(text.contains("bare"));
        assert!(!text.contains("fix the test"));
    }

    #[test]
    fn rejects_an_overlap_larger_than_the_window() {
        let error = RenderConfig {
            window: 8,
            context_overlap: 8,
        }
        .validate()
        .expect_err("reject");
        assert!(error.to_string().contains("overlap"));
    }
}
