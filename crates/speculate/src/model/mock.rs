//! Mock decoder used by the SD-correctness test suite.
//!
//! The mock has *no* learned weights and *no* dependence on tensor shapes — it
//! returns whatever logits a caller-supplied closure says it should. This lets
//! us verify the correctness of the rejection-sampling loop in `methods/vanilla.rs`
//! against a hand-controlled "target distribution" without ever touching candle.
//!
//! It is **not** part of the public API; gated behind `#[cfg(any(test, feature = "test-util"))]`
//! so it can be reused from integration tests in `tests/`.

#![cfg(any(test, feature = "test-util"))]

use crate::Result;
use std::sync::Arc;

/// Function `(prefix, position) → logits over vocab`.
pub type LogitFn = Arc<dyn Fn(&[u32], usize) -> Vec<f32> + Send + Sync>;

/// A toy decoder that produces logits according to a caller-provided closure.
///
/// The closure receives the full token history seen so far (including the
/// initial prompt) plus the absolute position of the next token, so callers can
/// model context-dependent distributions (e.g. n-gram dependence) without
/// allocating a real model.
#[derive(Clone)]
pub struct MockDecoder {
    vocab_size: usize,
    history: Vec<u32>,
    logits_fn: LogitFn,
}

impl std::fmt::Debug for MockDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockDecoder")
            .field("vocab_size", &self.vocab_size)
            .field("history_len", &self.history.len())
            .finish()
    }
}

impl MockDecoder {
    /// Build a decoder with the given vocabulary size and logit function.
    pub fn new(vocab_size: usize, logits_fn: LogitFn) -> Self {
        Self {
            vocab_size,
            history: Vec::new(),
            logits_fn,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Forget all observed tokens (equivalent to clearing the KV cache on a
    /// real model).
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Read-only view of tokens consumed so far.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Append `tokens` to the history *without* sampling — used to seed the
    /// prompt before a generation loop starts.
    pub fn observe(&mut self, tokens: &[u32]) {
        self.history.extend_from_slice(tokens);
    }

    /// Roll the history back to length `len`.
    pub fn rollback_to(&mut self, len: usize) {
        debug_assert!(len <= self.history.len(), "cannot roll forward");
        self.history.truncate(len);
    }

    /// Compute logits for the *next* token given the current history.
    ///
    /// Does **not** mutate state — the SD loop must call [`Self::observe`]
    /// after a token is committed.
    pub fn next_logits(&self) -> Vec<f32> {
        let pos = self.history.len();
        let raw = (self.logits_fn)(&self.history, pos);
        debug_assert_eq!(
            raw.len(),
            self.vocab_size,
            "logit_fn returned {} logits but vocab_size is {}",
            raw.len(),
            self.vocab_size
        );
        raw
    }

    /// Compute logits for `k` future positions assuming the supplied draft
    /// tokens were appended after the current history.
    ///
    /// Returns `k+1` logit vectors: one for each prefix length
    /// `[history.len(), history.len()+1, ..., history.len()+k]`. This is the
    /// *parallel verify* shape produced by a real target model evaluating
    /// `committed + drafts` in one forward pass.
    pub fn batched_logits(&self, drafts: &[u32]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(drafts.len() + 1);
        let mut prefix: Vec<u32> = self.history.clone();
        out.push((self.logits_fn)(&prefix, prefix.len()));
        for &d in drafts {
            prefix.push(d);
            out.push((self.logits_fn)(&prefix, prefix.len()));
        }
        out
    }
}

impl super::Decoder for MockDecoder {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn history(&self) -> &[u32] {
        &self.history
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn observe(&mut self, ids: &[u32]) -> Result<()> {
        self.history.extend_from_slice(ids);
        Ok(())
    }

    fn next_logits(&mut self) -> Result<Vec<f32>> {
        Ok(MockDecoder::next_logits(self))
    }

    fn batched_logits(&mut self, drafts: &[u32]) -> Result<Vec<Vec<f32>>> {
        let out = MockDecoder::batched_logits(self, drafts);
        // Match the contract: leave the decoder positioned as if it had
        // observed `drafts`. The caller will roll back what it doesn't keep.
        self.history.extend_from_slice(drafts);
        Ok(out)
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        if len > self.history.len() {
            return Err(crate::Error::CacheRollback(format!(
                "rollback target {len} > history length {}",
                self.history.len()
            )));
        }
        self.history.truncate(len);
        Ok(())
    }
}

/// Convenience constructor: a decoder whose distribution is a fixed categorical
/// over the vocabulary, independent of context.
pub fn fixed_distribution(probs: Vec<f32>) -> MockDecoder {
    let vocab = probs.len();
    let logits: Vec<f32> = probs
        .iter()
        .map(|&p| {
            // Convert probabilities back to logits up to a constant.
            // Clamp to avoid -inf for zero-mass tokens.
            (p.max(1e-30)).ln()
        })
        .collect();
    let logits = Arc::new(logits);
    let f: LogitFn = {
        let logits = Arc::clone(&logits);
        Arc::new(move |_history, _pos| (*logits).clone())
    };
    MockDecoder::new(vocab, f)
}
