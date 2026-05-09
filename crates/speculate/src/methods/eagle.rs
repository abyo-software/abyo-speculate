//! EAGLE-2 / EAGLE-3 (Li et al. 2024 / 2025) — *skeleton* for v0.2.0.
//!
//! ## What's here in v0.1.0
//!
//! Type stubs + the architectural shape we'll fill in. The verification
//! math is already complete via [`crate::methods::medusa::run_medusa_real`]:
//! EAGLE differs from Medusa primarily in *how* the draft tree is
//! generated, not in *how* the target verifies it. Once
//! [`EagleDraftCandle`]'s forward + dynamic tree construction land, the
//! existing tree-attention + tree_logits plumbing handles the rest.
//!
//! ## What EAGLE-2 changes vs Medusa
//!
//! - **Draft model is autoregressive** (1-layer transformer) rather than
//!   a flat set of heads. Input: `concat(hidden_state, token_embedding)`;
//!   output: a draft hidden state from which the next token is sampled.
//! - **Tree is dynamic.** Each iteration, the draft samples top-`k` per
//!   position; the tree expands the most confident branches up to a
//!   budget. Static Cartesian-product trees are a degenerate special case.
//! - **Acceptance** uses target's distribution directly (same modified
//!   rejection rule as Vanilla SD); we re-use the same primitives.
//!
//! ## What EAGLE-3 adds vs EAGLE-2
//!
//! - Multi-layer feature aggregation: instead of just the target's last
//!   hidden state, the draft consumes a fused projection of multiple
//!   target layers' hidden states. Improves draft quality at the cost
//!   of running additional projections per round.
//!
//! ## Status: NOT YET IMPLEMENTED
//!
//! The forward pass is a TODO. Calling [`run_eagle_real`] will return
//! `Error::UnsupportedMethod`. v0.2.0 will fill in:
//!
//! - [`EagleDraftConfig`] population from a published checkpoint's
//!   `config.json`.
//! - [`EagleDraftModule::forward`]: 1-layer transformer over
//!   `concat(hidden, token_emb)`.
//! - [`EagleDraftCandle::sample_tree`]: dynamic top-k tree expansion.
//! - End-to-end `run_eagle_real` that orchestrates target + draft + verify.

#![allow(dead_code)] // intentional during the skeleton phase
#![allow(missing_docs)] // most fields are placeholders until v0.2.0

use crate::model::TreeDecoder;
use crate::tree::DraftTree;
use crate::{Error, Result};

/// Hyper-parameters for an EAGLE draft model.
///
/// Populated from a published EAGLE checkpoint's `config.json` (see e.g.
/// `yuhuili/EAGLE-LLaMA3-Instruct-8B`).
#[derive(Debug, Clone)]
pub struct EagleDraftConfig {
    /// Hidden dim of both the target and the draft (must match).
    pub hidden_size: usize,
    /// Vocabulary size — typically tied to the target's lm_head.
    pub vocab_size: usize,
    /// Number of attention heads in the draft transformer layer.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA) in the draft.
    pub num_key_value_heads: usize,
    /// Intermediate dim of the draft's MLP.
    pub intermediate_size: usize,
    /// Whether the draft uses EAGLE-3's multi-layer feature fusion.
    pub multi_layer_features: bool,
}

/// One draft model loaded from a published EAGLE checkpoint.
///
/// Skeleton — actual weights + forward pending v0.2.0.
#[derive(Debug, Clone)]
pub struct EagleDraftCandle {
    config: EagleDraftConfig,
    // weights: TODO — q/k/v/o projections, MLP gate/up/down, RMSNorms,
    // optional input projection if hidden + token_emb concatenation
    // requires reshaping. Plus the lm_head (typically tied with target's).
}

impl EagleDraftCandle {
    /// Placeholder — returns the config that would have been loaded.
    pub fn from_random(config: EagleDraftConfig) -> Self {
        Self { config }
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &EagleDraftConfig {
        &self.config
    }

    /// Sample a draft tree given the target's most recent hidden state.
    ///
    /// Skeleton — returns `UnsupportedMethod` until v0.2.0.
    pub fn sample_tree(
        &self,
        target_hidden: &candle_core::Tensor,
        committed_root: u32,
        _budget: usize,
    ) -> Result<DraftTree> {
        let _ = (target_hidden, committed_root);
        Err(Error::UnsupportedMethod {
            method: "eagle::sample_tree",
            reason: "EAGLE draft sampling lands in v0.2.0".into(),
        })
    }
}

/// Run-loop config for EAGLE-2 / EAGLE-3.
#[derive(Debug, Clone)]
pub struct EagleRunConfig {
    /// Top-`k` per draft autoregressive step (used to build the tree).
    pub top_k_per_step: usize,
    /// Total tree-node budget (caps the dynamic-expansion depth).
    pub tree_budget: usize,
    /// Sampling temperature applied at the target side. Match the target's
    /// own settings to preserve the output distribution.
    pub temperature: f32,
    /// Top-p nucleus for the target's sampling.
    pub top_p: f32,
    /// Use EAGLE-3 (multi-layer feature fusion) if the draft supports it.
    pub use_eagle3: bool,
}

impl Default for EagleRunConfig {
    fn default() -> Self {
        Self {
            top_k_per_step: 8,
            tree_budget: 60,
            temperature: 0.7,
            top_p: 0.95,
            use_eagle3: false,
        }
    }
}

/// End-to-end EAGLE loop against any [`TreeDecoder`] target with a
/// candle-backed draft.
///
/// **Not yet implemented in v0.1.0.** Returns `UnsupportedMethod`. Once
/// `EagleDraftCandle::sample_tree` lands the loop becomes:
///
/// 1. `target.last_hidden_state()` → seed the draft.
/// 2. `draft.sample_tree(hidden, root, budget)` → a [`DraftTree`].
/// 3. `target.tree_logits(&tree)` → per-node target distributions.
/// 4. Walk root-to-leaf paths, accept via Vanilla-SD rejection sampling.
/// 5. Commit longest-accepted prefix + bonus from target's deepest node.
pub fn run_eagle_real<T, R>(
    target: &mut T,
    draft: &EagleDraftCandle,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &EagleRunConfig,
    rng: &mut R,
) -> Result<Vec<u32>>
where
    T: TreeDecoder + ?Sized,
    R: rand::Rng + ?Sized,
{
    let _ = (target, draft, prompt, max_new_tokens, config, rng);
    Err(Error::UnsupportedMethod {
        method: "eagle::run_eagle_real",
        reason: "EAGLE run loop lands in v0.2.0; the verification math is \
                 ready (run_medusa_real reuses the same primitives) but the \
                 EAGLE-specific draft forward + dynamic tree construction \
                 are still TODO."
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_returns_unsupported() {
        let cfg = EagleDraftConfig {
            hidden_size: 4096,
            vocab_size: 32000,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 11008,
            multi_layer_features: false,
        };
        let draft = EagleDraftCandle::from_random(cfg.clone());
        assert_eq!(draft.config().hidden_size, 4096);

        // sample_tree currently errors — keeps users away from broken paths
        // while signalling the v0.2.0 API surface.
        let dummy_hidden =
            candle_core::Tensor::zeros((4096,), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap();
        assert!(draft.sample_tree(&dummy_hidden, 0, 16).is_err());
    }
}
