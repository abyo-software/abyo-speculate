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

/// Download every PyTorch shard listed in a repo's
/// `pytorch_model.bin.index.json`. Returns the local paths of the shards
/// (cached) plus the path of the index file itself, so callers can build a
/// multi-shard [`MultiPthBackend`].
///
/// Used to load checkpoints that ship `.bin`-only and have not been
/// converted to safetensors (e.g. lmsys/vicuna-7b-v1.5,
/// FasterDecoding/medusa-1.0-vicuna-7b-v1.5).
pub fn download_pth_sharded(
    repo_id: &str,
) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Other(anyhow::anyhow!("hf-hub api init failed: {e}")))?;
    let repo = api.model(repo_id.to_string());

    let index_path = repo
        .get("pytorch_model.bin.index.json")
        .map_err(|e| Error::Other(anyhow::anyhow!("get pytorch_model.bin.index.json: {e}")))?;
    let index_json: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)
        .map_err(|e| Error::Other(anyhow::anyhow!("parse pth index: {e}")))?;
    let weight_map = index_json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            Error::Other(anyhow::anyhow!(
                "pytorch_model.bin.index.json has no `weight_map`"
            ))
        })?;
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
            .map_err(|e| Error::Other(anyhow::anyhow!("get pth shard {s}: {e}")))?;
        paths.push(p);
    }
    Ok((index_path, paths))
}

/// SimpleBackend that resolves tensor lookups across multiple PyTorch
/// shards. Built from [`download_pth_sharded`]'s output via
/// [`MultiPthBackend::from_paths`].
pub struct MultiPthBackend {
    /// Per-shard PthTensors, in deterministic order.
    shards: Vec<candle_core::pickle::PthTensors>,
    /// `tensor_name → shard_idx`, populated from `pytorch_model.bin.index.json`.
    tensor_to_shard: std::collections::HashMap<String, usize>,
}

impl MultiPthBackend {
    /// Build from the index json + ordered shard paths returned by
    /// [`download_pth_sharded`].
    pub fn from_paths(
        index_path: impl AsRef<std::path::Path>,
        shard_paths: &[impl AsRef<std::path::Path>],
    ) -> Result<Self> {
        let index_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(index_path.as_ref())?)
                .map_err(|e| Error::Other(anyhow::anyhow!("parse pth index: {e}")))?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Other(anyhow::anyhow!("missing weight_map")))?;

        let shards: Vec<candle_core::pickle::PthTensors> = shard_paths
            .iter()
            .map(|p| candle_core::pickle::PthTensors::new(p.as_ref(), None).map_err(Error::Candle))
            .collect::<Result<Vec<_>>>()?;

        // Build filename → shard_idx using the basenames of the loaded shards.
        let basename_to_idx: std::collections::HashMap<String, usize> = shard_paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.as_ref()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| (n.to_string(), i))
            })
            .collect();

        let mut tensor_to_shard = std::collections::HashMap::new();
        for (tensor_name, shard_filename) in weight_map.iter() {
            if let Some(name_str) = shard_filename.as_str() {
                let basename = std::path::Path::new(name_str)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(name_str);
                if let Some(&idx) = basename_to_idx.get(basename) {
                    tensor_to_shard.insert(tensor_name.clone(), idx);
                } else if let Some(&idx) = basename_to_idx.get(name_str) {
                    tensor_to_shard.insert(tensor_name.clone(), idx);
                }
            }
        }

        if tensor_to_shard.is_empty() {
            return Err(Error::Other(anyhow::anyhow!(
                "no tensors mapped — shard filename mismatch between index and downloaded files"
            )));
        }

        Ok(Self {
            shards,
            tensor_to_shard,
        })
    }
}

impl candle_nn::var_builder::SimpleBackend for MultiPthBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: candle_core::DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let idx =
            self.tensor_to_shard
                .get(name)
                .ok_or_else(|| candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                })?;
        let tensor = match self.shards[*idx].get(name)? {
            Some(t) => t,
            None => {
                return Err(candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                });
            }
        };
        let tensor = tensor.to_device(dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            });
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }
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
