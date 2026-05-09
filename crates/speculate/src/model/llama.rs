//! [`Decoder`] impl for Llama 1 / 2 / 3.x models loaded via candle.
//!
//! Mirrors [`crate::model::qwen2::Qwen2Decoder`]'s API surface — the public
//! shapes (`from_paths`, `next_logits`, `batched_logits`, `tree_logits`,
//! `last_hidden_state`, `rollback_to`) are intentionally identical so a
//! caller can swap `Qwen2Decoder` for `LlamaDecoder` without changing
//! anything else.
//!
//! Implementation differences vs. Qwen2:
//! - Llama's KV cache is centralised (one `Cache` struct, indexed per layer).
//!   We hold it inside the decoder and call `cache.truncate_to(len)` for
//!   fast rollback.
//! - Llama 3 supports rope-frequency scaling; the vendored
//!   [`crate::model::llama_local::Cache::new`] handles that via the
//!   `rope_scaling` field on `Config`.

use crate::model::llama_local::{Cache, Config, Llama};
use crate::model::Decoder;
use crate::tree::DraftTree;
use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use std::path::Path;
use tokenizers::Tokenizer;

/// A Llama-family decoder usable as either a target or draft in SD.
pub struct LlamaDecoder {
    model: Llama,
    cache: Cache,
    lm_head: Linear,
    tokenizer: Tokenizer,
    history: Vec<u32>,
    device: Device,
    dtype: DType,
    vocab_size: usize,
    hidden_size: usize,
    cache_len: usize,
}

impl std::fmt::Debug for LlamaDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaDecoder")
            .field("vocab_size", &self.vocab_size)
            .field("hidden_size", &self.hidden_size)
            .field("history_len", &self.history.len())
            .field("cache_len", &self.cache_len)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl LlamaDecoder {
    /// Load from local files.
    pub fn from_paths(
        config: &Config,
        safetensor_paths: &[impl AsRef<Path>],
        tokenizer_path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let paths: Vec<_> = safetensor_paths
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, dtype, &device).map_err(Error::Candle)?
        };
        let model = Llama::load(vb.clone(), config).map_err(Error::Candle)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens_weight().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
                .map_err(Error::Candle)?
        };
        let cache = Cache::new(dtype, config, &device).map_err(Error::Candle)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self {
            model,
            cache,
            lm_head,
            tokenizer,
            history: Vec::new(),
            device,
            dtype,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            cache_len: 0,
        })
    }

    /// Device the model is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tensor dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Hidden dim.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Tokenize.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Detokenize.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    fn forward_advance(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Sampling("forward_advance with empty tokens".into()));
        }
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward(&input, self.cache_len, &mut self.cache)
            .map_err(Error::Candle)?;
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;
        self.cache_len += tokens.len();
        Ok(logits)
    }

    /// See [`crate::model::qwen2::Qwen2Decoder::last_hidden_state`].
    pub fn last_hidden_state(&mut self) -> Result<Tensor> {
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "last_hidden_state with empty history".into(),
            ));
        }
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.cache.truncate_to(target_len).map_err(Error::Candle)?;
        self.cache_len = target_len;
        let input = Tensor::from_slice(&[last], (1, 1), &self.device).map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward(&input, self.cache_len, &mut self.cache)
            .map_err(Error::Candle)?;
        self.cache_len += 1;
        hidden.i((0, 0, ..)).map_err(Error::Candle)
    }

    fn row_to_vec(&self, t: &Tensor) -> Result<Vec<f32>> {
        let t = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            t.to_dtype(DType::F32).map_err(Error::Candle)?
        };
        t.to_vec1::<f32>().map_err(Error::Candle)
    }

    /// See [`crate::model::qwen2::Qwen2Decoder::tree_logits`].
    pub fn tree_logits(&mut self, tree: &DraftTree) -> Result<Vec<Vec<f32>>> {
        if self.history.is_empty() {
            return Err(Error::Sampling("tree_logits with empty history".into()));
        }
        let last_committed = *self.history.last().unwrap();
        if tree.token_at(0) != last_committed {
            return Err(Error::Sampling(format!(
                "tree root ({}) must equal last committed token ({})",
                tree.token_at(0),
                last_committed
            )));
        }

        let pre_cache_len = self.cache_len;
        debug_assert_eq!(pre_cache_len, self.history.len());
        let prefix_len = pre_cache_len - 1;

        self.cache.truncate_to(prefix_len).map_err(Error::Candle)?;
        self.cache_len = prefix_len;

        let positions: Vec<u32> = (0..tree.len())
            .map(|i| (prefix_len + tree.depth_of(i)) as u32)
            .collect();
        let position_tensor =
            Tensor::from_vec(positions, (tree.len(),), &self.device).map_err(Error::Candle)?;
        let bias = tree.full_attention_bias_4d(prefix_len, 1, 1, &self.device, self.dtype)?;
        let input_ids = Tensor::from_slice(tree.tokens(), (1, tree.len()), &self.device)
            .map_err(Error::Candle)?;

        let hidden = self
            .model
            .forward_with_positions(&input_ids, &position_tensor, &bias, &mut self.cache)
            .map_err(Error::Candle)?;
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;

        let mut out = Vec::with_capacity(tree.len());
        for i in 0..tree.len() {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out.push(self.row_to_vec(&row)?);
        }

        // Restore: drop tree, re-feed the root.
        self.cache.truncate_to(prefix_len).map_err(Error::Candle)?;
        self.cache_len = prefix_len;
        let _ = self.forward_advance(&[last_committed])?;
        debug_assert_eq!(self.cache_len, pre_cache_len);

        Ok(out)
    }
}

impl crate::model::TreeDecoder for LlamaDecoder {
    fn last_hidden_state(&mut self) -> Result<Tensor> {
        LlamaDecoder::last_hidden_state(self)
    }

    fn tree_logits(&mut self, tree: &DraftTree) -> Result<Vec<Vec<f32>>> {
        LlamaDecoder::tree_logits(self, tree)
    }
}

impl Decoder for LlamaDecoder {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        LlamaDecoder::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        LlamaDecoder::decode(self, ids, skip_special_tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn history(&self) -> &[u32] {
        &self.history
    }

    fn reset(&mut self) {
        self.history.clear();
        self.cache.clear();
        self.cache_len = 0;
    }

    fn observe(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let _ = self.forward_advance(ids)?;
        self.history.extend_from_slice(ids);
        Ok(())
    }

    fn next_logits(&mut self) -> Result<Vec<f32>> {
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "next_logits called with empty history".into(),
            ));
        }
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.cache.truncate_to(target_len).map_err(Error::Candle)?;
        self.cache_len = target_len;
        let logits = self.forward_advance(&[last])?;
        let last_row = logits
            .i((logits.dim(0).map_err(Error::Candle)? - 1, ..))
            .map_err(Error::Candle)?;
        self.row_to_vec(&last_row)
    }

    fn batched_logits(&mut self, drafts: &[u32]) -> Result<Vec<Vec<f32>>> {
        if drafts.is_empty() {
            let logits = self.next_logits()?;
            return Ok(vec![logits]);
        }
        if self.history.is_empty() {
            return Err(Error::Sampling("batched_logits with empty history".into()));
        }
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.cache.truncate_to(target_len).map_err(Error::Candle)?;
        self.cache_len = target_len;
        let mut combined: Vec<u32> = Vec::with_capacity(1 + drafts.len());
        combined.push(last);
        combined.extend_from_slice(drafts);
        let logits = self.forward_advance(&combined)?;
        let n_rows = logits.dim(0).map_err(Error::Candle)?;
        debug_assert_eq!(n_rows, drafts.len() + 1);
        let mut out = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out.push(self.row_to_vec(&row)?);
        }
        self.history.extend_from_slice(drafts);
        Ok(out)
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        if len > self.history.len() {
            return Err(Error::CacheRollback(format!(
                "rollback target {len} > history length {}",
                self.history.len()
            )));
        }
        self.history.truncate(len);
        self.cache.truncate_to(len).map_err(Error::Candle)?;
        self.cache_len = len;
        Ok(())
    }
}
