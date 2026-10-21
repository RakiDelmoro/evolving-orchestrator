//! Small file helpers. JSON is written with two-space indentation and a
//! trailing newline; record files are one compact JSON object per line.

use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{EvolveError, Result};

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path)
        .map_err(|error| EvolveError::Io(format!("cannot read {}: {error}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|error| EvolveError::Json(format!("{}: {error}", path.display())))
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("json")
    ));
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn read_records<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str(line).map_err(|error| {
            EvolveError::Json(format!("{}:{}: {error}", path.display(), line_number + 1))
        })?;
        rows.push(value);
    }
    Ok(rows)
}

pub fn append_record<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(value)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    use std::io::Write as _;
    writeln!(file, "{line}")?;
    Ok(())
}
