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
    /// separate copy.
    pub fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.model.apply_lm_head(hidden).map_err(Error::Candle)
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
        let _ = self.forward_advance_logits(&[last_committed])?;
        debug_assert_eq!(self.cache_len, pre_cache_len);
        Ok(out)
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
        self.model
            .truncate_kv_cache_to(len)
            .map_err(Error::Candle)?;
        self.cache_len = len;
        Ok(())
    }
}
