//! Export the selected adapter into a deployable model.
//!
//! The merge and quantization are external commands. The request and the
//! environment are the contract.

use std::path::PathBuf;
use std::process::Command;

use crate::error::{EvolveError, Result};

#[derive(Debug, Clone)]
pub struct ExportRequest {
    pub base_model: PathBuf,
    pub adapter: PathBuf,
    pub out_dir: PathBuf,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub quantize: bool,
}

pub trait Exporter {
    fn export(&self, request: &ExportRequest) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct CommandExporter {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandExporter {
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    pub fn from_env() -> Option<Self> {
        std::env::var("EVOLVE_EXPORT_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| Self::new("/bin/sh", vec!["-c".into(), value]))
    }
}

impl Exporter for CommandExporter {
    fn export(&self, request: &ExportRequest) -> Result<()> {
        if !request.base_model.is_dir() {
            return Err(EvolveError::Training(format!(
                "base model directory missing: {}",
                request.base_model.display()
            )));
        }
        if !request.adapter.exists() {
            return Err(EvolveError::Training(format!(
                "adapter missing: {}",
                request.adapter.display()
            )));
        }
        std::fs::create_dir_all(&request.out_dir)?;

        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .env("EVOLVE_EXPORT_BASE", &request.base_model)
            .env("EVOLVE_EXPORT_ADAPTER", &request.adapter)
            .env("EVOLVE_EXPORT_OUT", &request.out_dir)
            .env("EVOLVE_LORA_R", request.lora_r.to_string())
            .env("EVOLVE_LORA_ALPHA", request.lora_alpha.to_string())
            .env("EVOLVE_QUANTIZE", if request.quantize { "1" } else { "0" });

        let status = command
            .status()
            .map_err(|error| EvolveError::Training(format!("cannot start exporter: {error}")))?;
        if !status.success() {
            return Err(EvolveError::Training(format!(
                "exporter exited with {status}"
            )));
        }
        if !request.out_dir.join("READY").is_file() {
            return Err(EvolveError::Training(format!(
                "exporter did not write {}/READY",
                request.out_dir.display()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_an_exporter_command() {
        let root = std::env::temp_dir().join(format!(
            "evolve-export-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let base = root.join("base");
        let adapter = root.join("adapter");
        let out = root.join("out");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(&adapter, "adapter").unwrap();
        let exporter = CommandExporter::new(
            "/bin/sh",
            vec![
                "-c".into(),
                "printf 'ok\\n' > \"$EVOLVE_EXPORT_OUT/READY\"".into(),
            ],
        );
        exporter
            .export(&ExportRequest {
                base_model: base,
                adapter,
                out_dir: out.clone(),
                lora_r: 16,
                lora_alpha: 32,
                quantize: true,
            })
            .expect("export");
        assert!(out.join("READY").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }
}
