//! [`Decoder`] impl for Qwen 2 / Qwen 2.5 models loaded via candle.
//!
//! ## Why this lives here
//!
//! candle's `qwen2::ModelForCausalLM::forward` slices to the last position
//! before applying the LM head, which discards exactly the per-position logits
//! that SD verification needs. We instead drive `qwen2::Model` directly (which
//! returns full hidden states `[batch, seq, hidden]`) and apply our own
//! `lm_head` Linear afterward, so `batched_logits(drafts)` can return one
//! distribution per draft position from a single forward pass.
//!
//! ## Phase 1 limitations (intentional)
//!
//! - **Rollback is not yet KV-cache-aware.** `rollback_to` calls
//!   `model.clear_kv_cache()` and re-observes the surviving prefix. That's
//!   O(history) per rollback rather than O(rollback distance), which means
//!   Phase-1a SD on real models is roughly 1× autoregressive speed (no
//!   speedup) but is verifiably correct. Optimizing this is a Phase-1c task.
//! - **CPU only is the supported path** for unit testing. CUDA / Metal builds
//!   compile but require the corresponding cargo feature.
//! - **Model loading via [`Qwen2Decoder::from_paths`] is offline.** Use
//!   [`crate::model::loader::ModelSource`] + your own download step (or
//!   `hf-hub`) to fetch the files first; this struct does not download.

use crate::model::Decoder;
use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use candle_transformers::models::qwen2::{Config, Model};
use std::path::Path;
use tokenizers::Tokenizer;

/// A Qwen-family decoder usable as either a target or draft in SD.
pub struct Qwen2Decoder {
    model: Model,
    lm_head: Linear,
    tokenizer: Tokenizer,
    history: Vec<u32>,
    device: Device,
    dtype: DType,
    vocab_size: usize,
    /// All tokens we have advanced the KV cache over. May be longer than
    /// `history` between `batched_logits` and the subsequent `rollback_to`.
    cache_len: usize,
}

impl std::fmt::Debug for Qwen2Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen2Decoder")
            .field("vocab_size", &self.vocab_size)
            .field("history_len", &self.history.len())
            .field("cache_len", &self.cache_len)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl Qwen2Decoder {
    /// Load from local files. `safetensor_paths` should be the model shards in
    /// load order (most checkpoints have just one). `tokenizer_path` points to
    /// `tokenizer.json`.
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
        // Safety: candle's `from_mmaped_safetensors` mmaps the file. The file
        // must remain valid for the lifetime of the VarBuilder. Standard
        // pattern for candle model loading.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, dtype, &device).map_err(Error::Candle)?
        };
        let model = Model::new(config, vb.clone()).map_err(Error::Candle)?;

        // Tied vs untied embeddings — Qwen2 0.5B ties, 7B does not.
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
                .map_err(Error::Candle)?
        } else {
            // Tied: re-use embed_tokens weights. We need them out of the model;
            // load them via vb.pp("model.embed_tokens").
            let embed = vb
                .pp("model")
                .pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")
                .map_err(Error::Candle)?;
            Linear::new(embed, None)
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self {
            model,
            lm_head,
            tokenizer,
            history: Vec::new(),
            device,
            dtype,
            vocab_size: config.vocab_size,
            cache_len: 0,
        })
    }

    /// Encode a string to token ids via the bundled tokenizer.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Run a forward pass over `tokens` starting at the current cache length.
    /// Returns logits `[seq_len, vocab]` (batch dim squeezed). Updates the
    /// internal cache_len bookkeeping.
    fn forward_advance(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Sampling("forward_advance with empty tokens".into()));
        }
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(Error::Candle)?;
        // qwen2::Model::forward returns hidden states [b, seq, hidden] for ALL positions.
        let hidden = self
            .model
            .forward(&input, self.cache_len, None)
            .map_err(Error::Candle)?;
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        // Drop the batch dim → [seq, vocab].
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;
        self.cache_len += tokens.len();
        Ok(logits)
    }

    /// Convert a single [vocab] tensor row to host f32. Materializes from
    /// device → host; only call from non-hot paths.
    fn row_to_vec(&self, t: &Tensor) -> Result<Vec<f32>> {
        let t = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            t.to_dtype(DType::F32).map_err(Error::Candle)?
        };
        t.to_vec1::<f32>().map_err(Error::Candle)
    }

    /// Truncate the KV cache by clearing it and re-running the prefix. Slow but
    /// correct. Phase-1c will replace this with a partial-truncation primitive.
    fn cache_clear_and_replay(&mut self, prefix: &[u32]) -> Result<()> {
        self.model.clear_kv_cache();
        self.cache_len = 0;
        if !prefix.is_empty() {
            // Re-observe prefix without producing logits we use.
            let _ = self.forward_advance(prefix)?;
        }
        Ok(())
    }
}

impl Decoder for Qwen2Decoder {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn history(&self) -> &[u32] {
        &self.history
    }

    fn reset(&mut self) {
        self.history.clear();
        self.model.clear_kv_cache();
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
        // To produce the distribution over the *next* token without committing
        // it to the cache permanently, we forward a single sentinel-free pass
        // would normally need a no-op token. Instead we expose the simpler
        // behavior: callers should `observe` an actual chosen token between
        // `next_logits` calls, and the SD loop does exactly this.
        //
        // We compute logits for "what comes after history.last()" by re-using
        // the most recent forward's last row. To avoid keeping that around
        // statefully, we re-forward the last token if needed.
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "next_logits called with empty history (need a prompt first)".into(),
            ));
        }
        // We need the logits at position history_len - 1. The cache currently
        // ends at cache_len (== history_len after observe). The last forward
        // already produced these logits but we discarded them. Re-forward the
        // last token after rolling the cache back by 1.
        let last = *self.history.last().unwrap();
        // Roll back the cache by one position, then forward the last token to
        // recover the logits for the next position.
        self.model.clear_kv_cache();
        self.cache_len = 0;
        let prefix = self.history.clone();
        let _ = self.forward_advance(&prefix[..prefix.len() - 1])?;
        let logits = self.forward_advance(&[last])?;
        let last_row = logits
            .i((logits.dim(0).map_err(Error::Candle)? - 1, ..))
            .map_err(Error::Candle)?;
        self.row_to_vec(&last_row)
    }

    fn batched_logits(&mut self, drafts: &[u32]) -> Result<Vec<Vec<f32>>> {
        if drafts.is_empty() {
            // Just return the next-position logits as a single-element Vec.
            let logits = self.next_logits()?;
            return Ok(vec![logits]);
        }
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "batched_logits called with empty history".into(),
            ));
        }

        // We need k+1 logit vectors, one for each prefix:
        //   history, history+drafts[0], ..., history+drafts
        // Strategy: forward the last_history_token + drafts together, take all
        // k+1 output rows.
        let last = *self.history.last().unwrap();
        // Reset cache, replay everything except the last history token, then
        // forward [last, drafts...] in one pass to get k+1 rows.
        self.model.clear_kv_cache();
        self.cache_len = 0;
        let prefix = self.history.clone();
        let _ = self.forward_advance(&prefix[..prefix.len() - 1])?;
        let mut combined: Vec<u32> = Vec::with_capacity(1 + drafts.len());
        combined.push(last);
        combined.extend_from_slice(drafts);
        let logits = self.forward_advance(&combined)?; // [k+1, vocab]

        let n_rows = logits.dim(0).map_err(Error::Candle)?;
        debug_assert_eq!(n_rows, drafts.len() + 1);
        let mut out = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out.push(self.row_to_vec(&row)?);
        }

        // Per the trait contract, leave history positioned as if we observed `drafts`.
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
        let prefix = self.history.clone();
        self.cache_clear_and_replay(&prefix)?;
        Ok(())
    }
}
