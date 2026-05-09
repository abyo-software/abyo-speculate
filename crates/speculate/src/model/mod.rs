//! Model abstraction over candle decoders.
//!
//! Phase 1 only wires up the loader skeleton — the actual `forward()` glue is
//! filled in alongside the engine, and lives behind the [`Decoder`] trait so
//! future work (real models, Medusa heads, EAGLE feature draft) can plug in
//! without disturbing the SD verification loop.

pub mod loader;
pub mod mock;
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
}
