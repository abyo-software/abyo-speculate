//! Hugging Face Hub download helpers.
//!
//! Wraps `hf-hub` with abyo-speculate's `Result` type and a small convenience
//! API for fetching multiple files from one repo. The cache is whatever
//! `hf-hub` defaults to (typically `~/.cache/huggingface/hub`).

use crate::{Error, Result};
use std::path::PathBuf;

/// Download every file in `files` from `repo_id` and return their local paths
/// in the same order. Re-uses the existing cache if files are already
/// downloaded — only the first call pays the network cost.
///
/// Network errors (offline, transient 5xx, etc.) are surfaced as
/// `Error::Other`. Callers running in CI should gate on something like
/// `cfg!(test) && std::env::var("ABYO_SPECULATE_OFFLINE_TESTS").is_ok()` to
/// skip when the network isn't available.
pub fn download_files(repo_id: &str, files: &[&str]) -> Result<Vec<PathBuf>> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Other(anyhow::anyhow!("hf-hub api init failed: {e}")))?;
    let repo = api.model(repo_id.to_string());
    files
        .iter()
        .map(|f| {
            repo.get(f)
                .map_err(|e| Error::Other(anyhow::anyhow!("hf-hub get {repo_id}/{f}: {e}")))
        })
        .collect()
}

/// Convenience: download the standard trio of files for a Qwen-family model
/// (`config.json`, `tokenizer.json`, `model.safetensors`). Returns
/// `(config, tokenizer, weights)` paths.
///
/// For sharded checkpoints (e.g. 7B+ models), use [`download_files`] directly
/// with the shard list from the repo's `model.safetensors.index.json`.
pub fn download_qwen2_single_shard(repo_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let files = download_files(
        repo_id,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    Ok((files[0].clone(), files[1].clone(), files[2].clone()))
}
