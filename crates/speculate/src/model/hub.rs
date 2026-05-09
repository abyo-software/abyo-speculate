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
/// For sharded checkpoints (e.g. 7B+ models), use [`download_qwen2`] which
/// auto-detects single vs multi-shard via `model.safetensors.index.json`.
pub fn download_qwen2_single_shard(repo_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let files = download_files(
        repo_id,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    Ok((files[0].clone(), files[1].clone(), files[2].clone()))
}

/// Auto-detecting Qwen-family downloader.
///
/// Tries `model.safetensors` (single-shard, ≤ 4 GB-ish models). If that 404s,
/// falls back to fetching `model.safetensors.index.json` and every shard it
/// references.
///
/// Returns `(config_path, tokenizer_path, weight_paths)` where `weight_paths`
/// is `Vec<PathBuf>` ordered as the index file lists them (or a single-element
/// vec for single-shard).
pub fn download_qwen2(repo_id: &str) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Other(anyhow::anyhow!("hf-hub api init failed: {e}")))?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| Error::Other(anyhow::anyhow!("get config.json: {e}")))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| Error::Other(anyhow::anyhow!("get tokenizer.json: {e}")))?;

    // Try single-shard first.
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok((config_path, tokenizer_path, vec![p]));
    }

    // Fall back to sharded layout.
    let index_path = repo
        .get("model.safetensors.index.json")
        .map_err(|e| Error::Other(anyhow::anyhow!("get index.json: {e}")))?;
    let index_json: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)
        .map_err(|e| Error::Other(anyhow::anyhow!("parse index.json: {e}")))?;
    let weight_map = index_json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            Error::Other(anyhow::anyhow!(
                "model.safetensors.index.json has no `weight_map`"
            ))
        })?;
    // Collect distinct shard filenames.
    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shards.sort();
    shards.dedup();

    let mut paths = Vec::with_capacity(shards.len());
    for s in &shards {
        let p = repo
            .get(s)
            .map_err(|e| Error::Other(anyhow::anyhow!("get shard {s}: {e}")))?;
        paths.push(p);
    }
    Ok((config_path, tokenizer_path, paths))
}
