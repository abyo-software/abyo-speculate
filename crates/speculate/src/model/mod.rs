//! Model abstraction over candle decoders.
//!
//! Phase 1 only wires up the loader skeleton — the actual `forward()` glue is
//! filled in alongside the engine, and lives behind the [`Decoder`] trait so
//! future work (real models, Medusa heads, EAGLE feature draft) can plug in
//! without disturbing the SD verification loop.
//!
//! [`TreeDecoder`] is the additional capability a decoder needs to participate
//! in tree-attention SD methods (Medusa, EAGLE). Both `Qwen2Decoder` and
//! `LlamaDecoder` implement it; mock decoders do not.

pub mod hub;
pub mod llama;
pub mod llama_local;
pub mod loader;
pub mod mock;
pub mod phi3;
pub mod phi3_local;
pub mod quantized_llama;
pub mod quantized_llama_local;
pub mod quantized_qwen2;
pub mod quantized_qwen2_local;
pub mod qwen2;
pub mod qwen2_local;

use crate::Result;

/// The contract every concrete decoder must satisfy to participate in
/// abyo-speculate's SD loops.
///
/// The trait is shape-agnostic on purpose — it talks in `Vec<f32>` logit slabs
/// at the API boundary. Real implementations are free to keep tensors on-device
/// and only materialize logits when the caller asks for them.
///
/// Implementations are expected to be **stateful**: a `Decoder` carries its
/// observed token history and any associated KV-cache. Methods that mutate
/// state (`observe`, `next_logits`, `batched_logits`, `rollback_to`) must
/// leave the decoder in a self-consistent state if they return `Err`.
pub trait Decoder {
    /// Vocabulary size in tokens.
    fn vocab_size(&self) -> usize;

    /// Tokens consumed so far (prompt + generated).
    fn history(&self) -> &[u32];

    /// Number of tokens currently in the history.
    fn history_len(&self) -> usize {
        self.history().len()
    }

    /// Clear all state — KV cache, history, position counters.
    fn reset(&mut self);

    /// Append `ids` to the history, advancing the underlying KV cache.
    fn observe(&mut self, ids: &[u32]) -> Result<()>;

    /// Logits for the *next* token, given current history. Does **not** mutate
    /// state; calling this twice in a row must yield the same result.
    fn next_logits(&mut self) -> Result<Vec<f32>>;

    /// Speculative parallel forward.
    ///
    /// Returns `drafts.len() + 1` logit vectors. Slot `i` holds the predicted
    /// distribution after the prefix `history ++ drafts[..i]`. Slot `0` is
    /// therefore the same as [`Self::next_logits`].
    ///
    /// Implementations should evaluate this in **one** forward pass when
    /// possible (that is the whole reason SD is faster than autoregressive).
    /// State after the call must be equivalent to having observed `drafts` —
    /// callers will use [`Self::rollback_to`] to discard the parts they don't
    /// commit.
    fn batched_logits(&mut self, drafts: &[u32]) -> Result<Vec<Vec<f32>>>;

    /// Truncate history to the given length. For mock decoders this is a
    /// `Vec::truncate`; for real models it requires KV-cache rollback (or, in
    /// the simple Phase-1a path, a `clear_kv_cache` followed by re-observation
    /// of the prefix).
    fn rollback_to(&mut self, len: usize) -> Result<()>;

    /// Tokenize `text` to model token ids. Default impl returns
    /// `Error::UnsupportedMethod` — implementations that bundle a tokenizer
    /// (Qwen2Decoder, LlamaDecoder) override this.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let _ = (text, add_special_tokens);
        Err(crate::Error::UnsupportedMethod {
            method: "encode",
            reason: "this decoder has no bundled tokenizer".into(),
        })
    }

    /// Detokenize ids back to a string. Same default-error behaviour as
    /// [`Self::encode`].
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let _ = (ids, skip_special_tokens);
        Err(crate::Error::UnsupportedMethod {
            method: "decode",
            reason: "this decoder has no bundled tokenizer".into(),
        })
    }

    /// EOS / stop token ids for this decoder (e.g. `</s>`, `<|im_end|>`).
    /// Default empty — implementations that know their EOS override this so
    /// engine-level [`generate`](crate::SpeculateEngine::generate) can stop
    /// at the natural end of a response.
    fn eos_token_ids(&self) -> Vec<u32> {
        Vec::new()
    }
}

/// Capability trait for decoders that support tree-attention SD methods
/// (Medusa, EAGLE). Implementations must:
///
/// - Expose the *next-position* hidden state (the input a Medusa head consumes).
/// - Verify a [`crate::tree::DraftTree`] in a single forward pass and return
///   one logit row per node. The decoder's `history` and KV cache must be
///   restored to their pre-call state so the caller can commit the winning
///   path via [`Decoder::observe`].
pub trait TreeDecoder: Decoder {
    /// Hidden state at position `history.len()` (i.e. what comes *after* the
    /// last committed token). Shape `[hidden_size]`. Cache state restored.
    fn last_hidden_state(&mut self) -> Result<candle_core::Tensor>;

    /// Per-node next-token logit distributions for `tree`. Output length =
    /// `tree.len()`. Cache state restored.
    fn tree_logits(&mut self, tree: &crate::tree::DraftTree) -> Result<Vec<Vec<f32>>>;

    /// Project a hidden state `[batch, seq, hidden]` (or `[batch, hidden]`)
    /// to logits over the model's vocab. Used by EAGLE drafts that share the
    /// target's lm_head. Default impl returns `UnsupportedMethod` — only
    /// real-model decoders need to implement this.
    fn apply_lm_head(&self, hidden: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let _ = hidden;
        Err(crate::Error::UnsupportedMethod {
            method: "apply_lm_head",
            reason: "this TreeDecoder does not expose its lm_head".into(),
        })
    }

    /// Hidden states at multiple layer indices for the most recent committed
    /// token. Returns `(final_layer_hidden, layer_hiddens_for(layers))`. Used
    /// by EAGLE-3's low/mid/high concat input. Default impl returns
    /// `UnsupportedMethod`.
    fn last_hidden_states_multi(
        &mut self,
        layers: &[usize],
    ) -> Result<(candle_core::Tensor, Vec<candle_core::Tensor>)> {
        let _ = layers;
        Err(crate::Error::UnsupportedMethod {
            method: "last_hidden_states_multi",
            reason: "this TreeDecoder does not expose intermediate layers".into(),
        })
    }

    /// Number of transformer layers in this decoder. Default `0` — implement
    /// when callers (EAGLE-3) need to compute layer indices like `n/2` or
    /// `n - 2`.
    fn num_hidden_layers(&self) -> usize {
        0
    }

    /// Embed token ids via this decoder's tied embedding. EAGLE-3 needs
    /// this because the draft checkpoint does not ship its own
    /// embed_tokens — the official inference flow reuses the target's.
    /// Default impl returns `UnsupportedMethod`.
    fn embed_tokens(&self, input_ids: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let _ = input_ids;
        Err(crate::Error::UnsupportedMethod {
            method: "embed_tokens",
            reason: "this TreeDecoder does not expose its embedding table".into(),
        })
    }

    /// EAGLE-optimised tree forward. Returns `(per_node_logits,
    /// per_node_hidden)` and **leaves the KV cache populated with the
    /// tree** — no restoration. Use [`Self::commit_tree_path`] to commit
    /// the accepted path. Default impl falls back to [`Self::tree_logits`]
    /// without hidden states (for decoders that haven't implemented the
    /// optimised path yet).
    fn tree_logits_keep_kv(
        &mut self,
        tree: &crate::tree::DraftTree,
    ) -> Result<(Vec<Vec<f32>>, Vec<candle_core::Tensor>)> {
        let _ = tree;
        Err(crate::Error::UnsupportedMethod {
            method: "tree_logits_keep_kv",
            reason: "this TreeDecoder hasn't implemented the EAGLE fast path yet".into(),
        })
    }

    /// Like [`Decoder::observe`], but additionally returns the last
    /// position's hidden state from the same forward pass. EAGLE chains
    /// this into the next round's draft input. Default impl falls back
    /// to `observe` + `last_hidden_state` (two forwards instead of one).
    fn observe_returning_last_hidden(
        &mut self,
        ids: &[u32],
    ) -> Result<candle_core::Tensor> {
        self.observe(ids)?;
        self.last_hidden_state()
    }

    /// Commit the accepted tree path produced by
    /// [`Self::tree_logits_keep_kv`] without re-running the target — KV
    /// reordering only. The `accepted_indices` are tree node indices in
    /// BFS order from the root (e.g. `[0, 1, 4]`). Index 0 (root) is
    /// always present. To add a bonus token after, follow up with a
    /// normal [`Decoder::observe`] call.
    fn commit_tree_path(
        &mut self,
        tree: &crate::tree::DraftTree,
        accepted_indices: &[usize],
    ) -> Result<()> {
        let _ = (tree, accepted_indices);
        Err(crate::Error::UnsupportedMethod {
            method: "commit_tree_path",
            reason: "this TreeDecoder hasn't implemented the EAGLE fast path yet".into(),
        })
    }
}
