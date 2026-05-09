//! [`Decoder`] impl for Qwen 2 / Qwen 2.5 models loaded via candle.
//!
//! ## Why this lives here
//!
//! candle's `qwen2::ModelForCausalLM::forward` slices to the last position
//! before applying the LM head, which discards exactly the per-position logits
//! that SD verification needs. We instead drive [`crate::model::qwen2_local::Model`]
//! (our vendored variant — see that module's rustdoc) directly and apply our
//! own `lm_head` Linear afterward, so `batched_logits(drafts)` can return one
//! distribution per draft position from a single forward pass.
//!
//! ## Phase 1c (this revision)
//!
//! - **Tree decoding** via [`Qwen2Decoder::tree_logits`]: a single forward
//!   over `DraftTree::tokens()` with the per-position RoPE + 4D attention
//!   bias built by [`crate::tree::DraftTree::full_attention_bias_4d`].
//! - **Fast rollback** via the vendored
//!   [`crate::model::qwen2_local::Model::truncate_kv_cache_to`]: O(1) tensor
//!   slice per layer instead of clear-and-replay.
//! - `next_logits` / `batched_logits` similarly truncate-and-replay-one-token
//!   instead of clearing the entire cache.

use crate::model::qwen2_local::{Config, Model};
use crate::model::Decoder;
use crate::tree::DraftTree;
use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
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
    hidden_size: usize,
    /// Mirrors `model.kv_cache_len()`. Maintained explicitly so we can avoid
    /// querying the model on hot paths.
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
            // Tied: re-use embed_tokens weights.
            Linear::new(model.embed_tokens_weight().clone(), None)
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
            hidden_size: config.hidden_size,
            cache_len: 0,
        })
    }

    /// Device the model is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tensor dtype the model uses internally.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Hidden dim of the model — the input dim a Medusa head expects.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
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
    /// Returns logits `[seq_len, vocab]` (batch dim squeezed). Updates
    /// `cache_len`.
    fn forward_advance(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Sampling("forward_advance with empty tokens".into()));
        }
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(Error::Candle)?;
        let hidden = self
            .model
            .forward(&input, self.cache_len)
            .map_err(Error::Candle)?;
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?;
        self.cache_len += tokens.len();
        Ok(logits)
    }

    /// Compute the hidden state for the *next* position (i.e. the one a
    /// Medusa head would consume). Returns a `[hidden_size]` tensor on the
    /// model's device, with the cache state restored to its pre-call value.
    ///
    /// Implementation: truncate the cache by 1, re-forward the last committed
    /// token, take the model's hidden state output (pre-LM-head). Same
    /// truncate-and-replay trick as `next_logits` but stops one step earlier.
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
            .forward(&input, self.cache_len)
            .map_err(Error::Candle)?;
        self.cache_len += 1;
        // hidden shape: [1, 1, hidden_size] → squeeze to [hidden_size].
        hidden.i((0, 0, ..)).map_err(Error::Candle)
    }

    /// Materialize a single `[vocab]` tensor row to host `f32`.
    fn row_to_vec(&self, t: &Tensor) -> Result<Vec<f32>> {
        let t = if t.dtype() == DType::F32 {
            t.clone()
        } else {
            t.to_dtype(DType::F32).map_err(Error::Candle)?
        };
        t.to_vec1::<f32>().map_err(Error::Candle)
    }

    /// Tree-decoding forward.
    ///
    /// Preconditions:
    /// - `tree.token_at(0)` must equal `self.history.last()` (the tree is
    ///   rooted at the most recently committed token).
    /// - `cache_len == history.len()`.
    ///
    /// Postconditions:
    /// - Returns one `Vec<f32>` per tree node — `out[i]` is the target
    ///   distribution at the position *after* the path-from-root-to-node-i.
    /// - `cache_len` and `history` are unchanged. The caller decides which
    ///   path to commit and calls [`Decoder::observe`] to advance the state.
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
        let pre_history_len = self.history.len();
        debug_assert_eq!(pre_cache_len, pre_history_len);
        let prefix_len = pre_cache_len - 1; // cache before the root

        // 1. Drop the root from the cache; we'll feed it back as the first
        //    tree-input position so its hidden state is computed alongside the
        //    tree-tail nodes.
        self.model
            .truncate_kv_cache_to(prefix_len)
            .map_err(Error::Candle)?;
        self.cache_len = prefix_len;

        // 2. Build position_ids: depth-based absolute positions.
        let positions: Vec<u32> = (0..tree.len())
            .map(|i| (prefix_len + tree.depth_of(i)) as u32)
            .collect();
        let position_tensor =
            Tensor::from_vec(positions, (tree.len(),), &self.device).map_err(Error::Candle)?;

        // 3. Build attention bias over [prefix | tree].
        let bias = tree.full_attention_bias_4d(prefix_len, 1, 1, &self.device, self.dtype)?;

        // 4. Build input ids [1, n_tree].
        let input_ids = Tensor::from_slice(tree.tokens(), (1, tree.len()), &self.device)
            .map_err(Error::Candle)?;

        // 5. Tree-aware forward.
        let hidden = self
            .model
            .forward_with_positions(&input_ids, &position_tensor, &bias)
            .map_err(Error::Candle)?;
        // The forward_with_positions path appended n_tree entries to the cache.
        let logits = self.lm_head.forward(&hidden).map_err(Error::Candle)?;
        let logits = logits.i((0, .., ..)).map_err(Error::Candle)?; // [n_tree, vocab]

        // 6. Snapshot per-node distributions.
        let mut out = Vec::with_capacity(tree.len());
        for i in 0..tree.len() {
            let row = logits.i((i, ..)).map_err(Error::Candle)?;
            out.push(self.row_to_vec(&row)?);
        }

        // 7. Restore cache to its pre-call state: drop the tree, re-feed only
        //    the root so cache_len returns to pre_cache_len.
        self.model
            .truncate_kv_cache_to(prefix_len)
            .map_err(Error::Candle)?;
        self.cache_len = prefix_len;
        let _ = self.forward_advance(&[last_committed])?;
        debug_assert_eq!(self.cache_len, pre_cache_len);

        Ok(out)
    }
}

impl crate::model::TreeDecoder for Qwen2Decoder {
    fn last_hidden_state(&mut self) -> Result<Tensor> {
        Qwen2Decoder::last_hidden_state(self)
    }

    fn tree_logits(&mut self, tree: &DraftTree) -> Result<Vec<Vec<f32>>> {
        Qwen2Decoder::tree_logits(self, tree)
    }
}

impl Decoder for Qwen2Decoder {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Qwen2Decoder::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Qwen2Decoder::decode(self, ids, skip_special_tokens)
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
        // Truncate the cache by 1 (drop the last committed token), then
        // re-forward it to recover its next-token logits. O(1) tensor slice
        // + 1 forward pass instead of full clear+replay.
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
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

        // Truncate cache by 1, re-feed [last_committed, drafts...] in one
        // forward pass to get k+1 logit rows.
        let last = *self.history.last().unwrap();
        let target_len = self.history.len() - 1;
        self.model
            .truncate_kv_cache_to(target_len)
            .map_err(Error::Candle)?;
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

        // Trait contract: history advances by drafts; caller rolls back what
        // it doesn't keep.
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
        self.model
            .truncate_kv_cache_to(len)
            .map_err(Error::Candle)?;
        self.cache_len = len;
        Ok(())
    }
}
