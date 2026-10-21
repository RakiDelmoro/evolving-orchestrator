//! An OpenAI-compatible chat client.
//!
//! The client is deliberately synchronous: the harness is a batch pipeline,
//! and one request at a time is easier to make reproducible and to resume.

use std::time::Duration;

use serde_json::{json, Value};

use crate::error::{EvolveError, Result};

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub temperature: f64,
    pub top_p: f64,
    pub max_output_tokens: u32,
    pub seed: u64,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            max_output_tokens: 4096,
            seed: 0,
        }
    }
}

pub trait ChatClient {
    fn complete(&self, system: &str, user: &str, options: &ChatOptions) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct OpenAiChat {
    pub endpoint: String,
    pub model: String,
    pub api_key: String,
    pub timeout: Duration,
}

impl OpenAiChat {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(1800),
        }
    }
}

impl ChatClient for OpenAiChat {
    fn complete(&self, system: &str, user: &str, options: &ChatOptions) -> Result<String> {
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_output_tokens,
            "seed": options.seed
        });

        let agent = ureq::AgentBuilder::new().timeout(self.timeout).build();
        let response = agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body);

        let value = match response {
            Ok(response) => response.into_json::<Value>().map_err(|error| {
                EvolveError::Judgment(format!("model response is not JSON: {error}"))
            })?,
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                return Err(EvolveError::Judgment(format!(
                    "model request failed with HTTP {code}: {}",
                    truncate(&body, 800)
                )));
            }
            Err(error) => {
                return Err(EvolveError::Judgment(format!(
                    "model request failed: {error}"
                )))
            }
        };

        let content = value.pointer("/choices/0/message/content").ok_or_else(|| {
            EvolveError::Judgment("model response has no choices[0].message.content".into())
        })?;

        let text = match content {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => {
                return Err(EvolveError::Judgment(
                    "model content is neither text nor text parts".into(),
                ))
            }
        };

        if text.trim().is_empty() {
            return Err(EvolveError::Judgment("model returned empty content".into()));
        }
        Ok(text)
    }
}

fn truncate(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        value.to_string()
    } else {
        format!("{}…", &value[..limit])
    }
}
