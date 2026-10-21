//! Parse Pi session JSONL into messages and content blocks.
//!
//! Pi stores a session as JSONL: one `session` header followed by tree
//! entries. This module reads the `message` entries, preserving text,
//! thinking, tool calls, and tool results. The renderer consumes this shape;
//! it never reads Pi's JSON directly.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{EvolveError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiRole {
    System,
    User,
    Assistant,
    ToolResult,
    BashExecution,
    Custom,
    Unknown,
}

impl PiRole {
    pub fn as_str(self) -> &'static str {
        match self {
            PiRole::System => "system",
            PiRole::User => "user",
            PiRole::Assistant => "assistant",
            PiRole::ToolResult => "toolResult",
            PiRole::BashExecution => "bashExecution",
            PiRole::Custom => "custom",
            PiRole::Unknown => "unknown",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "system" => PiRole::System,
            "user" => PiRole::User,
            "assistant" => PiRole::Assistant,
            "toolResult" => PiRole::ToolResult,
            "bashExecution" => PiRole::BashExecution,
            "custom" => PiRole::Custom,
            _ => PiRole::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    Image {
        mime_type: String,
    },
    Other {
        value: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiMessage {
    pub role: PiRole,
    pub content: Vec<ContentBlock>,
    pub sections: Option<BTreeMap<String, String>>,
    pub timestamp: Option<u64>,
    pub model: Option<String>,
    pub stop_reason: Option<String>,
    pub is_error: bool,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub usage: Option<Value>,
}

impl PiMessage {
    /// Assistant reasoning, text, and tool calls are training targets.
    pub fn loss(&self) -> bool {
        self.role == PiRole::Assistant
    }

    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn thinking(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Thinking { thinking } => Some(thinking.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.as_str(), name.as_str(), arguments)),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiSession {
    pub path: PathBuf,
    pub id: String,
    pub version: u32,
    pub cwd: Option<String>,
    pub parent_session: Option<String>,
    pub messages: Vec<PiMessage>,
}

impl PiSession {
    pub fn final_assistant_text(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == PiRole::Assistant)
            .map(PiMessage::text)
            .unwrap_or_default()
    }

    pub fn is_success(&self) -> bool {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == PiRole::Assistant)
            .map(|message| {
                !matches!(
                    message.stop_reason.as_deref(),
                    Some("error") | Some("aborted") | Some("pending")
                )
            })
            .unwrap_or(false)
    }
}

pub fn load_session(path: &Path) -> Result<PiSession> {
    let text = fs::read_to_string(path).map_err(|error| {
        EvolveError::Episode(format!("cannot read session {}: {error}", path.display()))
    })?;
    parse_session(path, &text)
}

pub fn parse_session(path: &Path, text: &str) -> Result<PiSession> {
    let mut session = PiSession {
        path: path.to_path_buf(),
        id: String::new(),
        version: 0,
        cwd: None,
        parent_session: None,
        messages: Vec::new(),
    };

    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|error| {
            EvolveError::Episode(format!(
                "{}:{}: invalid session JSON: {error}",
                path.display(),
                line_number + 1
            ))
        })?;
        match value.get("type").and_then(Value::as_str) {
            Some("session") => {
                session.id = value
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                session.version = value.get("version").and_then(Value::as_u64).unwrap_or(0) as u32;
                session.cwd = value.get("cwd").and_then(Value::as_str).map(str::to_string);
                session.parent_session = value
                    .get("parentSession")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            Some("message") => {
                if let Some(message) = value.get("message") {
                    session.messages.push(parse_message(message));
                }
            }
            _ => {}
        }
    }

    Ok(session)
}

fn parse_message(value: &Value) -> PiMessage {
    let role = PiRole::parse(value.get("role").and_then(Value::as_str).unwrap_or(""));
    let content = parse_content(value.get("content"));
    let sections = value.get("sections").and_then(Value::as_object).map(|map| {
        map.iter()
            .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
            .collect()
    });
    PiMessage {
        role,
        content,
        sections,
        timestamp: value.get("timestamp").and_then(Value::as_u64),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        stop_reason: value
            .get("stopReason")
            .and_then(Value::as_str)
            .map(str::to_string),
        is_error: value
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        tool_call_id: value
            .get("toolCallId")
            .and_then(Value::as_str)
            .map(str::to_string),
        tool_name: value
            .get("toolName")
            .and_then(Value::as_str)
            .map(str::to_string),
        usage: value.get("usage").cloned(),
    }
}

fn parse_content(value: Option<&Value>) -> Vec<ContentBlock> {
    match value {
        Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
        Some(Value::Array(items)) => items.iter().map(parse_block).collect(),
        _ => Vec::new(),
    }
}

fn parse_block(value: &Value) -> ContentBlock {
    match value.get("type").and_then(Value::as_str) {
        Some("text") => ContentBlock::Text {
            text: value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        Some("thinking") => ContentBlock::Thinking {
            thinking: value
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        Some("toolCall") => ContentBlock::ToolCall {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: value.get("arguments").cloned().unwrap_or(Value::Null),
        },
        Some("image") => ContentBlock::Image {
            mime_type: value
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        _ => ContentBlock::Other {
            value: value.clone(),
        },
    }
}

/// Load every `.jsonl` session under a directory, sorted by path.
pub fn load_sessions(directory: &Path) -> Result<Vec<PiSession>> {
    let mut paths = Vec::new();
    if directory.exists() {
        collect_jsonl(directory, &mut paths)?;
    }
    paths.sort();
    paths.iter().map(|path| load_session(path)).collect()
}

fn collect_jsonl(directory: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_jsonl(&path, out)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = r#"{"type":"session","version":3,"id":"sess-1","cwd":"/work"}
{"type":"message","id":"a","parentId":null,"message":{"role":"system","content":"sys","sections":{"preamble":"You are a coder."},"timestamp":1}}
{"type":"message","id":"b","parentId":"a","message":{"role":"user","content":"fix the test","timestamp":2}}
{"type":"message","id":"c","parentId":"b","message":{"role":"assistant","content":[{"type":"thinking","thinking":"look first"},{"type":"text","text":"I will edit."},{"type":"toolCall","id":"call1","name":"read","arguments":{"path":"src/lib.rs"}}],"model":"local","stopReason":"toolUse","usage":{"input":10,"output":5,"totalTokens":15},"timestamp":3}}
{"type":"message","id":"d","parentId":"c","message":{"role":"toolResult","toolCallId":"call1","toolName":"read","content":[{"type":"text","text":"fn range() {}"}],"isError":false,"timestamp":4}}
{"type":"message","id":"e","parentId":"d","message":{"role":"assistant","content":[{"type":"text","text":"Done."}],"model":"local","stopReason":"stop","timestamp":5}}
"#;

    #[test]
    fn parses_messages_and_blocks() {
        let session = parse_session(Path::new("s.jsonl"), SESSION).expect("parse");
        assert_eq!(session.id, "sess-1");
        assert_eq!(session.messages.len(), 5);
        assert_eq!(session.messages[0].role, PiRole::System);
        assert!(session.messages[0]
            .sections
            .as_ref()
            .unwrap()
            .contains_key("preamble"));
        assert_eq!(session.messages[2].role, PiRole::Assistant);
        assert_eq!(session.messages[2].thinking(), "look first");
        assert_eq!(session.messages[2].text(), "I will edit.");
        assert_eq!(session.messages[2].tool_calls()[0].1, "read");
        assert_eq!(session.messages[3].role, PiRole::ToolResult);
        assert_eq!(session.messages[3].tool_name.as_deref(), Some("read"));
        assert_eq!(session.messages[4].stop_reason.as_deref(), Some("stop"));
        assert_eq!(session.final_assistant_text(), "Done.");
        assert!(session.messages[2].loss());
        assert!(!session.messages[3].loss());
        assert!(session.is_success());
    }

    #[test]
    fn detects_an_errored_final_message() {
        let text = r#"{"type":"session","version":3,"id":"s"}
{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"boom"}],"stopReason":"error"}}
"#;
        let session = parse_session(Path::new("e.jsonl"), text).expect("parse");
        assert!(!session.is_success());
    }

    #[test]
    fn loads_sessions_recursively() {
        let root = std::env::temp_dir().join(format!(
            "evolve-sessions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub/s.jsonl"), SESSION).unwrap();
        let sessions = load_sessions(&root).expect("load");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "sess-1");
        fs::remove_dir_all(root).unwrap();
    }
}
