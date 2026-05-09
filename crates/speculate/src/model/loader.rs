//! Hugging Face / local-file model loader (skeleton).
//!
//! Phase 1 will wire this up to candle's `quantized_llama` / `qwen2` builders.
//! For now we expose only the resolution layer — turning a model identifier into
//! a concrete on-disk path — so the engine builder can validate inputs early.

use std::path::{Path, PathBuf};

/// Where a model checkpoint lives.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Hugging Face Hub repo id, e.g. `"meta-llama/Llama-3.1-8B-Instruct"`.
    HuggingFace {
        /// Repo id in `"org/name"` form.
        repo_id: String,
        /// Optional revision (branch / commit / tag); defaults to `main`.
        revision: Option<String>,
    },
    /// Local directory containing `config.json`, `tokenizer.json`, and shards.
    Local(PathBuf),
}

impl ModelSource {
    /// Construct from a string the user passed in.
    ///
    /// Heuristic:
    /// - starts with `/`, `./`, or `~`  → Local
    /// - contains exactly one `/`        → HuggingFace
    /// - otherwise                       → HuggingFace under `abyo-software/<id>` (preset alias)
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.starts_with('/') || s.starts_with("./") || s.starts_with("~") {
            return ModelSource::Local(PathBuf::from(s));
        }
        let slash_count = s.matches('/').count();
        if slash_count == 1 {
            return ModelSource::HuggingFace {
                repo_id: s.to_string(),
                revision: None,
            };
        }
        ModelSource::HuggingFace {
            repo_id: format!("abyo-software/{s}"),
            revision: None,
        }
    }

    /// Local path of the checkpoint, if it has been resolved on disk.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            ModelSource::Local(p) => Some(p.as_path()),
            ModelSource::HuggingFace { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_local_paths() {
        assert!(matches!(
            ModelSource::parse("/tmp/model"),
            ModelSource::Local(_)
        ));
        assert!(matches!(
            ModelSource::parse("./checkpoints/foo"),
            ModelSource::Local(_)
        ));
    }

    #[test]
    fn parse_hf_repo() {
        match ModelSource::parse("meta-llama/Llama-3.1-8B-Instruct") {
            ModelSource::HuggingFace { repo_id, .. } => {
                assert_eq!(repo_id, "meta-llama/Llama-3.1-8B-Instruct");
            }
            _ => panic!("expected HuggingFace"),
        }
    }

    #[test]
    fn parse_alias_falls_back_to_org() {
        match ModelSource::parse("llama-3.1-8b-instruct") {
            ModelSource::HuggingFace { repo_id, .. } => {
                assert_eq!(repo_id, "abyo-software/llama-3.1-8b-instruct");
            }
            _ => panic!("expected HuggingFace alias"),
        }
    }
}
