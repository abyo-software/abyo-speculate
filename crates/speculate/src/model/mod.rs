//! Model loading and abstraction over candle decoders.
//!
//! Phase 1 only wires up the loader skeleton — the actual `forward()` glue is
//! filled in alongside the engine, and lives behind the [`DecoderModel`] trait
//! so future work (Medusa heads, EAGLE feature draft) can plug in without
//! disturbing the SD verification loop.

pub mod loader;

use crate::Result;
use candle_core::Tensor;

/// Generic interface implemented by every model abyo-speculate can run as
/// either a *target* or a *draft*.
pub trait DecoderModel {
    /// Run a forward pass over `input_ids` (shape `[batch=1, seq]`) starting
    /// at the given absolute position.
    ///
    /// Returns logits of shape `[batch=1, seq, vocab]`. Implementations are
    /// expected to update their internal KV cache as a side effect.
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>;

    /// Tokenizer vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Reset all internal state (KV caches, position counters).
    fn reset(&mut self);
}
