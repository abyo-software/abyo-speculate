//! [`Decoder`] impl for Llama-family GGUF (Q4 / Q5 / Q8) checkpoints.
//!
//! Mirrors [`crate::model::quantized_qwen2::Qwen2QuantDecoder`] but for
//! Llama 1/2/3.x (and Vicuna, code-llama, mistral GGUF). Use this when a
//! 7B+ Llama-family target needs to fit alongside a draft model on a
//! commodity GPU — Q4_K_M Llama 3 8B is ~4.5 GB, leaving room for the
//! 1.5 GB EAGLE-LLaMA3 draft on a 16 GB card.
//!
//! Tokenizer: GGUF embeds a vocab description but it isn't directly
//! compatible with the [`tokenizers`](https://docs.rs/tokenizers) crate's
//! JSON format. Pass the upstream `tokenizer.json` (e.g.
//! `meta-llama/Meta-Llama-3-8B-Instruct/tokenizer.json`).

#![allow(missing_docs)]

use crate::model::quantized_llama_local::ModelWeights;
use crate::model::{Decoder, TreeDecoder};
use crate::tree::DraftTree;
use crate::{Error, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct LlamaQuantDecoder {
    model: ModelWeights,
    tokenizer: Tokenizer,
    history: Vec<u32>,
    device: Device,
    vocab_size: usize,
    hidden_size: usize,
    /// Cached next-token logits set by every `forward_advance_logits` call;
    /// returned by `next_logits` to skip the redundant truncate-and-replay
    /// (~2× AR speedup). Invalidated on reset/rollback/tree mutations.
    last_logits: Option<Vec<f32>>,
    eos_token_ids: Vec<u32>,
    cache_len: usize,
}

impl std::fmt::Debug for LlamaQuantDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaQuantDecoder")
            .field("vocab_size", &self.vocab_size)
            .field("hidden_size", &self.hidden_size)
            .field("history_len", &self.history.len())
            .field("cache_len", &self.cache_len)
            .field("device", &self.device)
            .finish()
    }
}

impl LlamaQuantDecoder {
    /// Load from a single `.gguf` + matching `tokenizer.json`.
    ///
    /// `eos_token_ids` is caller-supplied — Llama 3 uses 128001 (`<|end_of_text|>`)
    /// and 128009 (`<|eot_id|>`); Llama 2 uses 2 (`</s>`).
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
        eos_token_ids: Vec<u32>,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path.as_ref())
            .map_err(|e| Error::Other(anyhow::anyhow!("open gguf: {e}")))?;
        let content = gguf_file::Content::read(&mut file).map_err(Error::Candle)?;

        // Re-read for metadata after Content::read consumes the metadata.
        // (Content::read doesn't consume — but to be safe and explicit, re-read.)
        let hidden_size = content
            .metadata
            .get("llama.embedding_length")
            .ok_or_else(|| Error::Other(anyhow::anyhow!("missing llama.embedding_length")))?
            .to_u32()
            .map_err(Error::Candle)? as usize;
        let vocab_size = content
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|v| v.len())
            .unwrap_or(128256); // Llama 3 default

        let model = ModelWeights::from_gguf(content, &mut file, &device).map_err(Error::Candle)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self {
            model,
            tokenizer,
            history: Vec::new(),
            device,
            vocab_size,
            hidden_size,
            eos_token_ids,
            cache_len: 0,
            last_logits: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Apply the model's quantized lm_head — exposed so EAGLE's draft
    /// loop can re-use the target's vocab projection without owning a
    /// separate copy. Auto-promotes the input to F32 (the dtype the Q4
    /// QMatMul kernel expects) so callers can pass BF16/F16 draft
    /// hiddens directly.
    pub fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden_owned;
        let hidden_use: &Tensor = if hidden.dtype() != DType::F32 {
            hidden_owned = hidden.to_dtype(DType::F32).map_err(Error::Candle)?;
            &hidden_owned
        } else {
            hidden
        };
        self.model.apply_lm_head(hidden_use).map_err(Error::Candle)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    fn forward_advance_logits(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Sampling("forward_advance with empty tokens".into()));
        }
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward_hidden(&input, self.cache_len)
            .map_err(Error::Candle)?;
        let logits = self.model.apply_lm_head(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;
        self.cache_len += tokens.len();
        // Cache the LAST row's logits as the next-token prediction so a
        // subsequent next_logits doesn't re-forward the last committed
        // token (~2× AR speedup on Q4 paths).
        let n_rows = logits.dim(0).map_err(Error::Candle)?;
        let last_row = logits.i((n_rows - 1, ..)).map_err(Error::Candle)?;
        self.last_logits = Some(self.row_to_vec(&last_row)?);
        Ok(logits)
    }

    fn row_to_vec(&self, t: &Tensor) -> Result<Vec<f32>> {
        let t = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            t.to_dtype(DType::F32).map_err(Error::Candle)?
        };
        t.to_vec1::<f32>().map_err(Error::Candle)
    }

    /// Like [`Decoder::observe`], but additionally returns the last
    /// position's hidden state from the same forward pass. EAGLE's run
    /// loop uses this to chain the deepest committed token's hidden into
    /// the next round's draft input without a separate
    /// [`Self::last_hidden_state`] forward.
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
            .forward_hidden(&input, self.cache_len)
            .map_err(Error::Candle)?;
        self.cache_len += ids.len();
        self.history.extend_from_slice(ids);
        let last_idx = hidden.dim(1).map_err(Error::Candle)? - 1;
        hidden.i((0, last_idx, ..)).map_err(Error::Candle)
    }

    pub fn last_hidden_state(&mut self) -> Result<Tensor> {
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "last_hidden_state with empty history".into(),
            ));
        }
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
        self.cache_len = target_len;
        let input = Tensor::from_slice(&[last], (1, 1), &self.device).map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward_hidden(&input, self.cache_len)
            .map_err(Error::Candle)?;
        self.cache_len += 1;
        hidden.i((0, 0, ..)).map_err(Error::Candle)
    }

    /// Hidden states of the most recent committed token at multiple layers.
    /// Returns `(final_norm_output, [layer_outputs in `layers` order])`. Used
    /// by EAGLE-3 (low/mid/high feature concat).
    ///
    /// Layer indices are 0-based and refer to the residual output *after*
    /// `layers[i]`'s MLP — i.e. the input to `layers[i+1]` (or to `norm` if
    /// `i == n_layers - 1`).
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
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
        self.cache_len = target_len;
        let input = Tensor::from_slice(&[last], (1, 1), &self.device).map_err(Error::Candle)?;
        let (final_h, mids) = self
            .model
            .forward_hidden_with_layers(&input, self.cache_len, layers)
            .map_err(Error::Candle)?;
        self.cache_len += 1;
        // Slice each to the last position only (seq=1 here, so position 0).
        let mids_last: Vec<Tensor> = mids
            .into_iter()
            .map(|t| t.i((0, 0, ..)).map_err(Error::Candle))
            .collect::<Result<_>>()?;
        let final_last = final_h.i((0, 0, ..)).map_err(Error::Candle)?;
        Ok((final_last, mids_last))
    }

    /// Number of transformer layers (used by EAGLE-3 to pick low/mid/high).
    pub fn num_hidden_layers(&self) -> usize {
        self.model.num_hidden_layers()
    }

    /// Embed token ids via the target's tied embedding. EAGLE-3 reuses
    /// this — the draft checkpoint ships without embed_tokens.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.embed_tokens(input_ids).map_err(Error::Candle)
    }

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

        self.model
            .truncate_kv_cache_to(prefix_len)
            .map_err(Error::Candle)?;
        self.cache_len = prefix_len;

        let positions: Vec<u32> = (0..tree.len())
            .map(|i| (prefix_len + tree.depth_of(i)) as u32)
            .collect();
        let position_tensor =
            Tensor::from_vec(positions, (tree.len(),), &self.device).map_err(Error::Candle)?;
        // Quantized intermediates are F32 in candle.
        let bias = tree.full_attention_bias_4d(prefix_len, 1, 1, &self.device, DType::F32)?;
        let input_ids = Tensor::from_slice(tree.tokens(), (1, tree.len()), &self.device)
            .map_err(Error::Candle)?;

        let hidden = self
            .model
            .forward_with_positions(&input_ids, &position_tensor, &bias)
            .map_err(Error::Candle)?;
        let logits = self.model.apply_lm_head(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;

        let mut out = Vec::with_capacity(tree.len());
        for i in 0..tree.len() {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out.push(self.row_to_vec(&row)?);
        }

        self.model
            .truncate_kv_cache_to(prefix_len)
            .map_err(Error::Candle)?;
        self.cache_len = prefix_len;
        // Restoration step also gives us the GEMV-path logits for the
        // root token. The multi-position GEMM path used inside the tree
        // forward returns slightly different values for position 0 than
        // a single-position forward would (independent FP accumulation
        // order across kernel sizes — verified to drift by ~0.01-0.05
        // across tree sizes 4 / 9 / 16, enough to flip a borderline
        // argmax). Overwriting per_node_logits[0] with the restoration
        // logits guarantees the invariant `tree_logits[0] == next_logits`.
        let restore_logits = self.forward_advance_logits(&[last_committed])?;
        let restore_row = restore_logits
            .i((restore_logits.dim(0).map_err(Error::Candle)? - 1, ..))
            .map_err(Error::Candle)?;
        out[0] = self.row_to_vec(&restore_row)?;
        debug_assert_eq!(self.cache_len, pre_cache_len);
        Ok(out)
    }

    /// Tree forward optimised for the EAGLE inner loop.
    ///
    /// Differences from [`Self::tree_logits`]:
    /// 1. Returns hidden states per node alongside the logits (the deepest
    ///    accepted node's hidden state becomes the next round's draft input).
    /// 2. Does **not** restore the KV cache to `prefix_len + 1` after the
    ///    call. Cache state is left at `prefix_len + tree.len()` with all
    ///    tree nodes' KVs in BFS order. The caller commits the accepted
    ///    path via [`Self::commit_tree_path`] (one O(layers) index_select
    ///    per layer, no extra forward).
    /// 3. Skips the GEMV root-fix from v0.2.2 — non-strict mode trades a
    ///    tiny per-token logit drift for a 2× target-side throughput gain.
    ///
    /// Returns `(per_node_logits, per_node_hidden)`.
    pub fn tree_logits_keep_kv(
        &mut self,
        tree: &DraftTree,
    ) -> Result<(Vec<Vec<f32>>, Vec<Tensor>)> {
        // Cache state will diverge from history; invalidate cached logits.
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

        self.model
            .truncate_kv_cache_to(prefix_len)
            .map_err(Error::Candle)?;
        self.cache_len = prefix_len;

        let positions: Vec<u32> = (0..tree.len())
            .map(|i| (prefix_len + tree.depth_of(i)) as u32)
            .collect();
        let position_tensor =
            Tensor::from_vec(positions, (tree.len(),), &self.device).map_err(Error::Candle)?;
        let bias = tree.full_attention_bias_4d(prefix_len, 1, 1, &self.device, DType::F32)?;
        let input_ids = Tensor::from_slice(tree.tokens(), (1, tree.len()), &self.device)
            .map_err(Error::Candle)?;

        let hidden = self
            .model
            .forward_with_positions(&input_ids, &position_tensor, &bias)
            .map_err(Error::Candle)?;
        // Cache is now `prefix_len + tree.len()` (we deliberately don't
        // restore — `commit_tree_path` handles the KV consolidation).
        self.cache_len = prefix_len + tree.len();
        let logits = self.model.apply_lm_head(&hidden).map_err(Error::Candle)?;
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

    /// After a [`Self::tree_logits_keep_kv`] call, commit the accepted tree
    /// path to permanent KV cache state by reordering the cache to keep
    /// only the accepted positions, and update history. **No target
    /// forward is needed** — the accepted nodes' KVs are already in cache
    /// from the tree forward, we just keep them and drop the rest.
    ///
    /// `accepted_indices` are tree node indices in BFS order from the root
    /// (e.g. `[0, 1, 4]` = root → child[1] → grandchild[0] of child[1]).
    /// Index 0 (root) is always present and equals the previous
    /// `last_committed` token.
    ///
    /// To add a bonus / non-tree token after committing the path, follow
    /// up with a normal [`Decoder::observe`] of the extra token(s).
    pub fn commit_tree_path(
        &mut self,
        tree: &DraftTree,
        accepted_indices: &[usize],
    ) -> Result<()> {
        // KV reorder doesn't run a forward, so cached logits are stale.
        self.last_logits = None;
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "commit_tree_path with empty history".into(),
            ));
        }
        debug_assert!(!accepted_indices.is_empty() && accepted_indices[0] == 0);
        let last_committed = *self.history.last().unwrap();
        debug_assert_eq!(tree.token_at(0), last_committed);
        let prefix_len = self.history.len() - 1;

        let mut keep: Vec<u32> = Vec::with_capacity(prefix_len + accepted_indices.len());
        for i in 0..prefix_len {
            keep.push(i as u32);
        }
        for &ti in accepted_indices {
            keep.push((prefix_len + ti) as u32);
        }
        self.model.keep_kv_indices(&keep).map_err(Error::Candle)?;
        self.cache_len = keep.len();

        for &ti in accepted_indices.iter().skip(1) {
            self.history.push(tree.token_at(ti));
        }
        Ok(())
    }
}

impl TreeDecoder for LlamaQuantDecoder {
    fn last_hidden_state(&mut self) -> Result<Tensor> {
        LlamaQuantDecoder::last_hidden_state(self)
    }

    fn tree_logits(&mut self, tree: &DraftTree) -> Result<Vec<Vec<f32>>> {
        LlamaQuantDecoder::tree_logits(self, tree)
    }

    fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        LlamaQuantDecoder::apply_lm_head(self, hidden)
    }

    fn last_hidden_states_multi(
        &mut self,
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        LlamaQuantDecoder::last_hidden_states_multi(self, layers)
    }

    fn num_hidden_layers(&self) -> usize {
        LlamaQuantDecoder::num_hidden_layers(self)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        LlamaQuantDecoder::embed_tokens(self, input_ids)
    }

    fn tree_logits_keep_kv(
        &mut self,
        tree: &DraftTree,
    ) -> Result<(Vec<Vec<f32>>, Vec<Tensor>)> {
        LlamaQuantDecoder::tree_logits_keep_kv(self, tree)
    }

    fn observe_returning_last_hidden(&mut self, ids: &[u32]) -> Result<Tensor> {
        LlamaQuantDecoder::observe_returning_last_hidden(self, ids)
    }

    fn commit_tree_path(
        &mut self,
        tree: &DraftTree,
        accepted_indices: &[usize],
    ) -> Result<()> {
        LlamaQuantDecoder::commit_tree_path(self, tree, accepted_indices)
    }
}

impl Decoder for LlamaQuantDecoder {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        LlamaQuantDecoder::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        LlamaQuantDecoder::decode(self, ids, skip_special_tokens)
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
        self.model.clear_kv_cache();
        self.cache_len = 0;
        self.last_logits = None;
    }

    fn observe(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let _ = self.forward_advance_logits(ids)?;
        self.history.extend_from_slice(ids);
        Ok(())
    }

    fn next_logits(&mut self) -> Result<Vec<f32>> {
        if self.history.is_empty() {
            return Err(Error::Sampling(
                "next_logits called with empty history".into(),
            ));
        }
        // Fast path: prior `forward_advance_logits` (called from observe)
        // already cached the next-token logits.
        if let Some(cached) = &self.last_logits {
            return Ok(cached.clone());
        }
        // Slow path: truncate-and-replay (used after tree mutations).
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
        self.cache_len = target_len;
        let logits = self.forward_advance_logits(&[last])?;
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
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
        self.cache_len = target_len;
        let mut combined: Vec<u32> = Vec::with_capacity(1 + drafts.len());
        combined.push(last);
        combined.extend_from_slice(drafts);
        let logits = self.forward_advance_logits(&combined)?;
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
        self.last_logits = None;
        self.model
            .truncate_kv_cache_to(len)
            .map_err(Error::Candle)?;
        self.cache_len = len;
        Ok(())
    }
}
