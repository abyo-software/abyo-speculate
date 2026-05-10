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
    eos_token_ids: Vec<u32>,
    cache_len: usize,
    /// Cached next-token logits from the most recent `forward_advance` /
    /// `observe` call. The previous AR loop pattern (observe → next_logits)
    /// did 2 model forwards per token because next_logits would
    /// truncate-and-replay the last committed token; caching the logits
    /// produced as a side-effect of `observe` lets `next_logits` return
    /// them directly. Invalidated on `reset`, `rollback_to`, and
    /// tree_logits / commit_tree_path (those mutate cache).
    last_logits: Option<Vec<f32>>,
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
        Self::from_var_builder(config, vb, tokenizer_path, device, dtype)
    }

    /// Load from a caller-supplied [`VarBuilder`], e.g. when the weights
    /// come from a non-safetensors source like
    /// [`crate::model::hub::MultiPthBackend`] (sharded PyTorch .bin) or a
    /// custom test-only [`candle_nn::var_builder::SimpleBackend`].
    pub fn from_var_builder(
        config: &Config,
        vb: VarBuilder<'_>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
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

        let eos_token_ids = match &config.eos_token_id {
            Some(crate::model::llama_local::LlamaEosToks::Single(id)) => vec![*id],
            Some(crate::model::llama_local::LlamaEosToks::Multiple(v)) => v.clone(),
            None => Vec::new(),
        };

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
            eos_token_ids,
            cache_len: 0,
            last_logits: None,
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
        // Cache the LAST row's logits — that's the next-token prediction
        // for `cache_len + 1` (i.e. one past the last appended token).
        // Subsequent `next_logits()` calls return this without a redundant
        // truncate-and-replay forward (~2× AR speedup).
        let n_rows = logits.dim(0).map_err(Error::Candle)?;
        let last_row = logits.i((n_rows - 1, ..)).map_err(Error::Candle)?;
        self.last_logits = Some(self.row_to_vec(&last_row)?);
        Ok(logits)
    }

    /// Like [`Decoder::observe`], but additionally returns the last
    /// position's hidden state from the same forward pass.
    pub fn observe_returning_last_hidden(&mut self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(Error::Sampling(
                "observe_returning_last_hidden with empty ids".into(),
            ));
        }
        let input = Tensor::new(ids, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward(&input, self.cache_len, &mut self.cache)
            .map_err(Error::Candle)?;
        self.cache_len += ids.len();
        self.history.extend_from_slice(ids);
        let last_idx = hidden.dim(1).map_err(Error::Candle)? - 1;
        hidden.i((0, last_idx, ..)).map_err(Error::Candle)
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

        // Restore: drop tree, re-feed the root. The restoration also gives
        // us the GEMV-path logits for the root token; we use them to
        // overwrite per_node_logits[0]. See `LlamaQuantDecoder::tree_logits`
        // for the full v0.2.2 root-replacement rationale (single-position
        // forward goes through GEMV, multi-position through GEMM, and the
        // two have different FP accumulation orders that flip borderline
        // argmax).
        self.cache.truncate_to(prefix_len).map_err(Error::Candle)?;
        self.cache_len = prefix_len;
        let restore_logits = self.forward_advance(&[last_committed])?;
        let restore_row = restore_logits
            .i((restore_logits.dim(0).map_err(Error::Candle)? - 1, ..))
            .map_err(Error::Candle)?;
        out[0] = self.row_to_vec(&restore_row)?;
        debug_assert_eq!(self.cache_len, pre_cache_len);

        Ok(out)
    }

    /// Hidden states of the most recent committed token at multiple layers
    /// (residual stream after each requested layer's MLP). Used by EAGLE-3
    /// to fetch low/mid/high target features in one quantized forward.
    pub fn last_hidden_states_multi(
        &mut self,
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "last_hidden_states_multi with empty history".into(),
            ));
        }
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.cache.truncate_to(target_len).map_err(Error::Candle)?;
        self.cache_len = target_len;
        let input = Tensor::from_slice(&[last], (1, 1), &self.device).map_err(Error::Candle)?;
        let (final_h, mids) = self
            .model
            .forward_with_layer_hooks(&input, self.cache_len, &mut self.cache, layers)
            .map_err(Error::Candle)?;
        self.cache_len += 1;
        let mids_last: Vec<Tensor> = mids
            .into_iter()
            .map(|t| t.i((0, 0, ..)).map_err(Error::Candle))
            .collect::<Result<_>>()?;
        let final_last = final_h.i((0, 0, ..)).map_err(Error::Candle)?;
        Ok((final_last, mids_last))
    }

    /// Apply the model's lm_head — exposed so EAGLE / EAGLE-3 can re-use
    /// the target's vocab projection without owning a separate copy.
    /// Auto-promotes the input dtype to match the lm_head weight if needed
    /// (EAGLE's run loop sometimes passes F32 because the Q4 path requires
    /// F32 inputs — silently converting here means the same call works
    /// for both BF16 and Q4 targets).
    pub fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let want = self.dtype;
        let hidden = if hidden.dtype() != want {
            hidden.to_dtype(want).map_err(Error::Candle)?
        } else {
            hidden.clone()
        };
        self.lm_head.forward(&hidden).map_err(Error::Candle)
    }

    /// Embed token ids via the target's tied embedding (used by EAGLE-3).
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.embed_tokens(input_ids).map_err(Error::Candle)
    }

    /// Number of transformer layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.model.num_hidden_layers()
    }

    /// EAGLE-optimised tree forward: returns hidden states alongside
    /// logits, leaves the KV cache populated with the tree (no
    /// restoration), skips the v0.2.2 GEMV root-fix. Use [`Self::commit_tree_path`]
    /// to commit the accepted path without re-forwarding.
    pub fn tree_logits_keep_kv(
        &mut self,
        tree: &DraftTree,
    ) -> Result<(Vec<Vec<f32>>, Vec<Tensor>)> {
        // Cache state will diverge from history; invalidate the cached
        // next-token logits so a subsequent next_logits doesn't read stale.
        self.last_logits = None;
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
        self.cache_len = prefix_len + tree.len();
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;

        let mut out_logits = Vec::with_capacity(tree.len());
        let mut out_hidden = Vec::with_capacity(tree.len());
        for i in 0..tree.len() {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out_logits.push(self.row_to_vec(&row)?);
            out_hidden.push(hidden.i((0, i, ..)).map_err(Error::Candle)?);
        }
        Ok((out_logits, out_hidden))
    }

    /// Commit the accepted tree path to permanent KV state via index_select
    /// reordering of the cache. See `LlamaQuantDecoder::commit_tree_path`
    /// for the index-arithmetic explanation.
    pub fn commit_tree_path(
        &mut self,
        tree: &DraftTree,
        accepted_indices: &[usize],
    ) -> Result<()> {
        // KV reorder doesn't run a forward, so the previously cached
        // next-token logits no longer correspond to the new last_committed.
        self.last_logits = None;
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "commit_tree_path with empty history".into(),
            ));
        }
        debug_assert!(!accepted_indices.is_empty() && accepted_indices[0] == 0);
        let last_committed = *self.history.last().unwrap();
        let prefix_len = self.history.len() - 1;
        debug_assert_eq!(tree.token_at(0), last_committed);

        let mut keep: Vec<u32> = Vec::with_capacity(prefix_len + accepted_indices.len());
        for i in 0..prefix_len {
            keep.push(i as u32);
        }
        for &ti in accepted_indices {
            keep.push((prefix_len + ti) as u32);
        }
        self.cache
            .keep_kv_indices(&keep)
            .map_err(Error::Candle)?;
        self.cache_len = keep.len();

        for &ti in accepted_indices.iter().skip(1) {
            self.history.push(tree.token_at(ti));
        }
        Ok(())
    }
}

impl crate::model::TreeDecoder for LlamaDecoder {
    fn last_hidden_state(&mut self) -> Result<Tensor> {
        LlamaDecoder::last_hidden_state(self)
    }

    fn tree_logits(&mut self, tree: &DraftTree) -> Result<Vec<Vec<f32>>> {
        LlamaDecoder::tree_logits(self, tree)
    }

    fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        LlamaDecoder::apply_lm_head(self, hidden)
    }

    fn last_hidden_states_multi(
        &mut self,
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        LlamaDecoder::last_hidden_states_multi(self, layers)
    }

    fn num_hidden_layers(&self) -> usize {
        LlamaDecoder::num_hidden_layers(self)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        LlamaDecoder::embed_tokens(self, input_ids)
    }

    fn tree_logits_keep_kv(
        &mut self,
        tree: &DraftTree,
    ) -> Result<(Vec<Vec<f32>>, Vec<Tensor>)> {
        LlamaDecoder::tree_logits_keep_kv(self, tree)
    }

    fn observe_returning_last_hidden(&mut self, ids: &[u32]) -> Result<Tensor> {
        LlamaDecoder::observe_returning_last_hidden(self, ids)
    }

    fn commit_tree_path(
        &mut self,
        tree: &DraftTree,
        accepted_indices: &[usize],
    ) -> Result<()> {
        LlamaDecoder::commit_tree_path(self, tree, accepted_indices)
    }
}

impl Decoder for LlamaDecoder {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        LlamaDecoder::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        LlamaDecoder::decode(self, ids, skip_special_tokens)
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        self.eos_token_ids.clone()
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
        self.last_logits = None;
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
        // Fast path: the last `forward_advance` (or `observe`, which calls
        // it) cached the next-token logits as a side effect. Use them.
        if let Some(cached) = &self.last_logits {
            return Ok(cached.clone());
        }
        // Slow path: cache was invalidated (post-rollback or post-tree
        // operation). Truncate-and-replay the last token to recompute.
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
        // We keep the truncate-and-replay of `last` here on purpose: a
        // "skip last via cached_logits" optimization (analogous to v0.4.1
        // next_logits) measurably *hurts* SD acceptance because last
        // committed token's KV from observe (GEMV path) differs from the
        // GEMM-path KV the drafts attend against in a single batched
        // forward. The drift lowers acceptance enough to net 25-35%
        // worse on chat/translation tasks.
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
        self.last_logits = None;
        Ok(())
    }
}
