//! Medusa multi-head speculative decoding (Cai et al. 2024).
//!
//! ## What Medusa is
//!
//! A "Medusa-augmented" target model has the usual LM head **plus** `N`
//! auxiliary heads. Where the LM head predicts the next token, head `k`
//! (`k = 1..=N`) predicts the token `k` positions ahead. Combined with the
//! base LM output, one forward pass gives you a draft of length `N + 1`.
//!
//! The verifier then runs the same target model over a *tree* of candidates
//! (the cross-product of each head's top-`k` predictions) and selects the
//! longest accepted path. Tree-attention amortises this — see [`crate::tree`].
//!
//! ## Phase 1b skeleton (this module today)
//!
//! - [`MedusaConfig`] / [`MedusaHead`] / [`MedusaHeads`] data structures.
//! - [`MedusaHeads::build_draft_tree`]: turn per-head top-k token candidates
//!   into a [`crate::tree::DraftTree`] using either a greedy chain or a
//!   Cartesian-product topology.
//! - Unit tests against synthetic head outputs.
//!
//! ## Phase 1b real-model (also in this module)
//!
//! - [`MedusaHeadModule`]: one head's residual MLP + projection forward,
//!   with a `from_random` constructor for synthetic-init testing and a
//!   `from_var_builder` constructor for loading published checkpoints.
//! - [`MedusaHeadsCandle`]: bundle of `N` heads + `top_k_per_head` helper
//!   that turns a target hidden state into per-head candidate tokens.
//! - [`run_medusa_real`]: end-to-end Medusa loop against a real
//!   [`crate::model::qwen2::Qwen2Decoder`] target — uses `tree_logits` for
//!   single-pass tree verification (Phase 1c) instead of the per-path
//!   walk in [`run_medusa`].
//!
//! ## What is **not** yet here
//!
//! - Loaders for specific community Medusa checkpoints (`FasterDecoding/...`).
//!   The forward path is in place; pointing it at a public checkpoint is a
//!   follow-up once a Qwen-compatible head is available (most published
//!   heads target Vicuna / Llama 2).

#![allow(clippy::needless_range_loop)]

use crate::model::Decoder;
use crate::tree::DraftTree;
use crate::{Error, Result};

/// Hyper-parameters for a Medusa-augmented model.
///
/// `hidden_size` and `vocab_size` must match the underlying target. Vary
/// `n_heads` per checkpoint — vicuna-7b-v1.3 ships with 4, vicuna-13b with 5.
#[derive(Debug, Clone)]
pub struct MedusaConfig {
    /// Number of auxiliary draft heads (one extra prediction per head).
    pub n_heads: usize,
    /// Hidden dimension of the target model (== input dim of each head).
    pub hidden_size: usize,
    /// Vocabulary size (== output dim of each head's projection).
    pub vocab_size: usize,
    /// Number of residual MLP layers stacked inside each head. The released
    /// vicuna checkpoints use 1; some experimental setups use 2–3.
    pub residual_layers: usize,
}

impl MedusaConfig {
    /// Reasonable defaults for the released vicuna-7b Medusa heads.
    pub fn vicuna_7b_defaults() -> Self {
        Self {
            n_heads: 4,
            hidden_size: 4096,
            vocab_size: 32000,
            residual_layers: 1,
        }
    }
}

/// A single Medusa head as a tag/shape descriptor — the actual weight tensors
/// are owned by the model-side struct that loads them.
///
/// This is the abstract shape of one head; the concrete `Linear` weights live
/// in [`MedusaHeads`] (or a future `MedusaHeadsCandle` variant).
#[derive(Debug, Clone)]
pub struct MedusaHead {
    /// Which future position this head predicts (1 = "next-but-one", etc).
    pub offset: usize,
}

/// A bundle of Medusa heads attached to a target model.
///
/// Phase 1b skeleton: only structural metadata + the per-head top-`k`
/// candidate plumbing. Once the loader lands, this struct will also carry
/// the residual + projection weight tensors.
#[derive(Debug, Clone)]
pub struct MedusaHeads {
    config: MedusaConfig,
    heads: Vec<MedusaHead>,
}

impl MedusaHeads {
    /// Construct a placeholder bundle with `config.n_heads` heads.
    pub fn from_config(config: MedusaConfig) -> Self {
        let heads = (1..=config.n_heads)
            .map(|offset| MedusaHead { offset })
            .collect();
        Self { config, heads }
    }

    /// Number of heads.
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Whether the bundle contains zero heads (degenerate — equivalent to plain
    /// autoregressive).
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Read-only view of the underlying [`MedusaConfig`].
    pub fn config(&self) -> &MedusaConfig {
        &self.config
    }

    /// Build a draft tree from per-head top-`k` candidate token IDs.
    ///
    /// `committed_root` is the last *accepted* token (becomes the tree root).
    /// `head_top_k[h]` is the list of candidate token IDs from head `h`
    /// (ordered by head logit, highest first). For greedy mode, only
    /// `head_top_k[h][0]` is used.
    pub fn build_draft_tree(
        &self,
        committed_root: u32,
        head_top_k: &[Vec<u32>],
        topology: TreeTopology,
    ) -> Result<DraftTree> {
        if head_top_k.len() != self.heads.len() {
            return Err(Error::Sampling(format!(
                "head_top_k has {} entries, expected {} (one per head)",
                head_top_k.len(),
                self.heads.len()
            )));
        }
        for (h, candidates) in head_top_k.iter().enumerate() {
            if candidates.is_empty() {
                return Err(Error::Sampling(format!("head {h} has no candidates")));
            }
        }

        match topology {
            TreeTopology::Greedy => Ok(build_greedy_chain(committed_root, head_top_k)),
            TreeTopology::CartesianProduct => Ok(build_cartesian_tree(committed_root, head_top_k)),
        }
    }
}

/// How to assemble a draft tree from the per-head top-`k` lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeTopology {
    /// One path: each head's top-1 forms a linear chain rooted at the
    /// committed token. Equivalent to vanilla SD with `k = n_heads`.
    Greedy,
    /// Full Cartesian product: every combination of one candidate per head
    /// becomes a path. Tree size = `1 + Σ_{i=0..n_heads} Π_{j<=i} k_j`.
    /// Use small `k` (2–3) per head to keep this manageable.
    CartesianProduct,
}

fn build_greedy_chain(root: u32, head_top_k: &[Vec<u32>]) -> DraftTree {
    let chain: Vec<u32> = head_top_k.iter().map(|cands| cands[0]).collect();
    DraftTree::linear(root, &chain)
}

fn build_cartesian_tree(root: u32, head_top_k: &[Vec<u32>]) -> DraftTree {
    // Build BFS-order parent table. Layer 0 = root. Layer h+1 = each candidate
    // of head h attached under every node in layer h.
    let mut nodes: Vec<(usize, u32)> = vec![(0, root)];
    let mut prev_layer_indices: Vec<usize> = vec![0];

    for cands in head_top_k {
        let mut next_layer_indices = Vec::with_capacity(prev_layer_indices.len() * cands.len());
        for &parent_idx in &prev_layer_indices {
            for &cand in cands {
                let new_idx = nodes.len();
                nodes.push((parent_idx, cand));
                next_layer_indices.push(new_idx);
            }
        }
        prev_layer_indices = next_layer_indices;
    }

    // The constructor enforces parent < self, which our BFS layout already
    // guarantees. unwrap is safe — we built `nodes` with at least the root.
    DraftTree::from_parent_table(&nodes).expect("Cartesian builder produces valid tree")
}

/// Configuration for the Medusa generation loop.
#[derive(Debug, Clone)]
pub struct MedusaRunConfig {
    /// Tree topology to use when assembling head outputs.
    pub topology: TreeTopology,
    /// Top-`k` taken from each head when topology is `CartesianProduct`.
    /// Ignored under `Greedy`.
    pub top_k_per_head: usize,
    /// Acceptance rule (controls how a target distribution decides whether a
    /// drafted token "passes").
    pub acceptance: Acceptance,
}

impl Default for MedusaRunConfig {
    fn default() -> Self {
        Self {
            topology: TreeTopology::CartesianProduct,
            top_k_per_head: 2,
            acceptance: Acceptance::Greedy,
        }
    }
}

/// How to decide whether the target accepts a drafted token at a given tree
/// position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Acceptance {
    /// Accept iff `argmax(p_target) == draft_token`. Provably correct only
    /// when sampling at temperature 0; matches the greedy-eval used by most
    /// open Medusa demos.
    Greedy,
    /// Cai et al. §3.2 typical acceptance:
    /// `accept iff p_target(x) >= max(epsilon, delta * exp(-H(p_target)))`.
    /// `epsilon` is the hard floor, `delta` is the entropy-scaled threshold.
    Typical {
        /// Hard probability floor below which a token is always rejected.
        epsilon: f32,
        /// Entropy-scaled threshold multiplier.
        delta: f32,
    },
}

/// Function the run loop calls to obtain per-head top-`k` candidate tokens
/// given the current committed history.
///
/// For real Medusa this comes from the heads' forward pass. For the mock-based
/// correctness tests it's a closure the test supplies.
pub type HeadDraftFn = Box<dyn FnMut(&[u32]) -> Vec<Vec<u32>>>;

/// Run one Medusa generation loop against an arbitrary [`Decoder`].
///
/// This is the **reference correctness implementation** for Medusa, modelled
/// after [`crate::methods::vanilla::run_vanilla_sd`] (and unit-tested the
/// same way: mock decoder + analytic checks).
///
/// Algorithm per round:
/// 1. Ask `head_draft` for per-head top-`k` candidates.
/// 2. Build a [`DraftTree`] using the configured [`TreeTopology`].
/// 3. For every node in the tree, fetch the target's next-token distribution
///    *given the path from root to that node* (in real Medusa this is one
///    tree-attention forward pass; in this reference impl we walk paths).
/// 4. Walk every root-to-leaf path. For each path, greedily accept tokens
///    while the [`Acceptance`] rule passes. Track the longest accepted prefix
///    across paths.
/// 5. Commit the longest accepted prefix to the target's history; if the
///    final accepted node has further-ahead next-token logits, sample one
///    bonus token from them.
pub fn run_medusa<T, R>(
    target: &mut T,
    heads: &MedusaHeads,
    mut head_draft: HeadDraftFn,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &MedusaRunConfig,
    rng: &mut R,
) -> Result<Vec<u32>>
where
    T: Decoder + ?Sized,
    R: rand::Rng + ?Sized,
{
    target.reset();
    target.observe(prompt)?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

    while generated.len() < max_new_tokens {
        let root_token = *target
            .history()
            .last()
            .ok_or_else(|| Error::Sampling("Medusa requires non-empty prompt".into()))?;

        // 1. Per-head candidates.
        let head_top_k = head_draft(target.history());

        // 2. Build tree.
        let tree = heads.build_draft_tree(root_token, &head_top_k, config.topology)?;

        // 3. Per-node next-token distributions. For each node `i`, this is the
        //    target distribution AFTER observing prompt + ... + path-to-i.
        let pre_target_len = target.history_len();
        let per_node_logits = evaluate_tree(target, &tree, pre_target_len)?;

        // 4. Walk each root-to-leaf path; track the longest accepted prefix.
        let mut best_path: Vec<usize> = vec![0]; // root is always implicitly accepted
        for path in tree.paths() {
            let accepted_len =
                walk_and_accept(&path, &tree, &per_node_logits, &config.acceptance, rng);
            // path[..accepted_len+1] = root + accepted draft nodes. Pick the
            // path whose accepted prefix is longest (ties: first wins).
            if accepted_len + 1 > best_path.len() {
                best_path = path[..=accepted_len].to_vec();
            }
        }

        // 5. Commit. The root is already in history. Append the accepted
        //    draft tokens, then optionally one bonus token sampled from the
        //    distribution at the deepest accepted node.
        let mut committed: Vec<u32> = best_path
            .iter()
            .skip(1)
            .map(|&i| tree.token_at(i))
            .collect();

        // Bonus token from deepest accepted node's distribution.
        let deepest_idx = *best_path.last().unwrap();
        if generated.len() + committed.len() < max_new_tokens {
            let bonus_logits = &per_node_logits[deepest_idx];
            let bonus = sample_argmax_or_categorical(bonus_logits, &config.acceptance, rng)?;
            committed.push(bonus);
        }

        // Re-anchor target: it currently still has only the original prefix
        // (evaluate_tree restored it).
        debug_assert_eq!(target.history_len(), pre_target_len);
        target.observe(&committed)?;
        generated.extend_from_slice(&committed);

        if committed.is_empty() {
            // No bonus + no accepted draft: would loop forever. Defensive
            // guard; in practice this shouldn't happen because we always
            // append at least the bonus.
            return Err(Error::Sampling(
                "Medusa round committed zero tokens — would loop forever".into(),
            ));
        }
    }

    // We may have over-shot by 1 due to the bonus token; truncate.
    generated.truncate(max_new_tokens);
    Ok(generated)
}

/// For each tree node, compute the *next-token* distribution that the target
/// assigns after consuming `prefix + path-to-node`. The target's history is
/// restored to `pre_target_len` before this returns.
///
/// Reference impl: walks paths separately. A real model with tree-attention
/// can do this in a single forward pass (see [`DraftTree::full_attention_bias`]).
fn evaluate_tree<T: Decoder + ?Sized>(
    target: &mut T,
    tree: &DraftTree,
    pre_target_len: usize,
) -> Result<Vec<Vec<f32>>> {
    let n = tree.len();
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); n];

    // Node 0 (root) is the last committed token; the distribution "after the
    // root" is just the target's current next_logits.
    out[0] = target.next_logits()?;

    // For deeper nodes, observe the path tokens (excluding the root, which is
    // already in history), get next_logits, then roll back.
    for i in 1..n {
        let path = tree.path_to(i); // [0, ..., i]
        let path_tokens_after_root: Vec<u32> =
            path.iter().skip(1).map(|&idx| tree.token_at(idx)).collect();
        target.observe(&path_tokens_after_root)?;
        out[i] = target.next_logits()?;
        target.rollback_to(pre_target_len)?;
    }
    Ok(out)
}

/// Walk a single root-to-leaf `path` and count how many tokens (after the
/// root) the [`Acceptance`] rule accepts.
fn walk_and_accept<R: rand::Rng + ?Sized>(
    path: &[usize],
    tree: &DraftTree,
    per_node_logits: &[Vec<f32>],
    acceptance: &Acceptance,
    rng: &mut R,
) -> usize {
    let _ = rng; // typical-acceptance with sampling not yet wired
                 // path[0] = root (always implicitly accepted, contributes nothing)
                 // path[i] for i >= 1: token tree.token_at(path[i]) is the candidate. The
                 // distribution we judge it against is per_node_logits[path[i-1]] (the
                 // distribution at the *parent*, predicting what comes next).
    let mut accepted = 0;
    for w in path.windows(2) {
        let parent = w[0];
        let child = w[1];
        let candidate = tree.token_at(child);
        let parent_dist = &per_node_logits[parent];
        if accept_one(parent_dist, candidate, acceptance) {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

fn accept_one(target_logits: &[f32], candidate: u32, acceptance: &Acceptance) -> bool {
    match acceptance {
        Acceptance::Greedy => {
            // argmax with stable tie-break (lowest index wins).
            let argmax = target_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0;
            argmax == candidate as usize
        }
        Acceptance::Typical { epsilon, delta } => {
            // Convert logits → probabilities via softmax (numerically stable).
            let max = target_logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = target_logits.iter().map(|&l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            if sum <= 0.0 || !sum.is_finite() {
                return false;
            }
            let probs: Vec<f32> = exps.iter().map(|p| p / sum).collect();
            let entropy: f32 = probs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();
            let threshold = epsilon.max(delta * (-entropy).exp());
            probs[candidate as usize] >= threshold
        }
    }
}

/// For the bonus token: sample from the deepest-accepted-node's distribution.
/// Greedy mode picks argmax; typical mode also picks argmax (the "bonus" in
/// Medusa is always the target's deterministic prediction at the verified
/// position). We expose this function so it can be specialised later.
fn sample_argmax_or_categorical<R: rand::Rng + ?Sized>(
    logits: &[f32],
    acceptance: &Acceptance,
    _rng: &mut R,
) -> Result<u32> {
    if logits.is_empty() {
        return Err(Error::Sampling("empty logits for bonus token".into()));
    }
    let _ = acceptance;
    let argmax = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0;
    Ok(argmax as u32)
}

/// Pick the top-`k` token indices from a logit slice, highest-logit first.
///
/// Stable on ties (lowest index wins). `k` is clamped to `logits.len()`.
pub fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    // Sort descending by value, ascending by index on ties.
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ============================================================================
// Phase 1b real-model: candle-backed Medusa heads + Qwen2 integration loop.
// ============================================================================

use crate::model::TreeDecoder;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use std::path::Path;

/// One Medusa head: a stack of residual MLP layers followed by a projection
/// onto the vocabulary.
///
/// The architecture mirrors Cai et al. 2024: each residual block is
/// `x -> x + SiLU(linear(x))`, then a final `linear_no_bias(hidden -> vocab)`.
///
/// Two constructors:
/// - [`Self::from_random`] — synthetic initialization for plumbing tests.
/// - [`Self::from_var_builder`] — load published checkpoints via candle's
///   safetensors VarBuilder.
#[derive(Debug, Clone)]
pub struct MedusaHeadModule {
    res_blocks: Vec<Linear>,
    output_proj: Linear,
}

impl MedusaHeadModule {
    /// Load one head's weights from a candle [`VarBuilder`] sub-namespace.
    ///
    /// Expected key layout (matches the FasterDecoding Medusa convention):
    /// - `res.<i>.weight` / `res.<i>.bias` for `i in 0..residual_layers`
    /// - `output.weight` for the final projection
    pub fn from_var_builder(cfg: &MedusaConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(cfg.residual_layers);
        for i in 0..cfg.residual_layers {
            let l = linear(cfg.hidden_size, cfg.hidden_size, vb.pp(format!("res.{i}")))
                .map_err(Error::Candle)?;
            res_blocks.push(l);
        }
        let output_proj = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("output"))
            .map_err(Error::Candle)?;
        Ok(Self {
            res_blocks,
            output_proj,
        })
    }

    /// Build a head with random Gaussian-init weights.
    ///
    /// Predictions from such a head are near-uniform — useful only for
    /// verifying the forward pipeline and shape contracts. Use
    /// [`Self::from_var_builder`] for published checkpoints.
    pub fn from_random(cfg: &MedusaConfig, device: &Device, dtype: DType) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(cfg.residual_layers);
        for _ in 0..cfg.residual_layers {
            let w = Tensor::randn(0f32, 0.02, (cfg.hidden_size, cfg.hidden_size), device)
                .map_err(Error::Candle)?
                .to_dtype(dtype)
                .map_err(Error::Candle)?;
            let b = Tensor::zeros(cfg.hidden_size, dtype, device).map_err(Error::Candle)?;
            res_blocks.push(Linear::new(w, Some(b)));
        }
        let w = Tensor::randn(0f32, 0.02, (cfg.vocab_size, cfg.hidden_size), device)
            .map_err(Error::Candle)?
            .to_dtype(dtype)
            .map_err(Error::Candle)?;
        let output_proj = Linear::new(w, None);
        Ok(Self {
            res_blocks,
            output_proj,
        })
    }

    /// Apply the head to a hidden state. The input may be `[hidden]`,
    /// `[seq, hidden]`, or `[batch, seq, hidden]`; the output's leading dims
    /// match the input, with the trailing dim swapped to `vocab_size`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let needs_squeeze = hidden.dims().len() == 1;
        let mut x = if needs_squeeze {
            hidden.unsqueeze(0).map_err(Error::Candle)?
        } else {
            hidden.clone()
        };
        for rb in &self.res_blocks {
            // Residual block: x' = x + SiLU(linear(x))
            let y = candle_nn::ops::silu(&rb.forward(&x).map_err(Error::Candle)?)
                .map_err(Error::Candle)?;
            x = (y + &x).map_err(Error::Candle)?;
        }
        let logits = self.output_proj.forward(&x).map_err(Error::Candle)?;
        if needs_squeeze {
            logits.squeeze(0).map_err(Error::Candle)
        } else {
            Ok(logits)
        }
    }
}

/// Bundle of `N` Medusa heads attached to a target. Wraps
/// [`MedusaHeadModule`]s plus a `top_k_per_head` materialization helper.
#[derive(Debug, Clone)]
pub struct MedusaHeadsCandle {
    config: MedusaConfig,
    heads: Vec<MedusaHeadModule>,
}

impl MedusaHeadsCandle {
    /// Build N heads with random init.
    pub fn from_random(cfg: &MedusaConfig, device: &Device, dtype: DType) -> Result<Self> {
        let mut heads = Vec::with_capacity(cfg.n_heads);
        for _ in 0..cfg.n_heads {
            heads.push(MedusaHeadModule::from_random(cfg, device, dtype)?);
        }
        Ok(Self {
            config: cfg.clone(),
            heads,
        })
    }

    /// Load N heads from local safetensors files.
    ///
    /// Expected top-level key layout (FasterDecoding convention):
    /// `medusa_head.<i>.{res.<j>.{weight,bias}, output.weight}`.
    pub fn from_safetensors(
        cfg: &MedusaConfig,
        paths: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let owned: Vec<_> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
        // Safety: same as Qwen2Decoder::from_paths — files must outlive VB.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&owned, dtype, device).map_err(Error::Candle)?
        };
        let mut heads = Vec::with_capacity(cfg.n_heads);
        for i in 0..cfg.n_heads {
            heads.push(MedusaHeadModule::from_var_builder(
                cfg,
                vb.pp(format!("medusa_head.{i}")),
            )?);
        }
        Ok(Self {
            config: cfg.clone(),
            heads,
        })
    }

    /// Load N heads from a FasterDecoding-style PyTorch pickle file
    /// (`medusa_lm_head.pt`).
    ///
    /// FasterDecoding's convention is **not** `medusa_head.<i>....`. Their
    /// key layout, observed from `FasterDecoding/medusa-vicuna-7b-v1.3`, is:
    ///
    /// ```text
    /// <i>.<j>.linear.weight   shape (hidden, hidden)
    /// <i>.<j>.linear.bias     shape (hidden,)
    /// <i>.<num_layers>.weight shape (vocab, hidden)        # output proj, no bias
    /// ```
    ///
    /// where `i in 0..n_heads` and `j in 0..num_layers`. The output projection
    /// sits at index `num_layers` (one past the last residual block).
    ///
    /// `cfg.n_heads` selects how many of the file's heads to load (some
    /// checkpoints store more heads than the corresponding `config.json`
    /// claims — trust this argument over the config file).
    pub fn from_fasterdecoding_pt(
        cfg: &MedusaConfig,
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(path.as_ref(), dtype, device).map_err(Error::Candle)?;
        let mut heads = Vec::with_capacity(cfg.n_heads);
        for i in 0..cfg.n_heads {
            let head_vb = vb.pp(i.to_string());
            let mut res_blocks = Vec::with_capacity(cfg.residual_layers);
            for j in 0..cfg.residual_layers {
                let l = linear(
                    cfg.hidden_size,
                    cfg.hidden_size,
                    head_vb.pp(j.to_string()).pp("linear"),
                )
                .map_err(Error::Candle)?;
                res_blocks.push(l);
            }
            let output_proj = linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                head_vb.pp(cfg.residual_layers.to_string()),
            )
            .map_err(Error::Candle)?;
            heads.push(MedusaHeadModule {
                res_blocks,
                output_proj,
            });
        }
        Ok(Self {
            config: cfg.clone(),
            heads,
        })
    }

    /// Read-only access to the [`MedusaConfig`].
    pub fn config(&self) -> &MedusaConfig {
        &self.config
    }

    /// Apply each head to the same hidden state, returning one logit tensor
    /// per head.
    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        self.heads.iter().map(|h| h.forward(hidden)).collect()
    }

    /// Compute per-head top-`k` token IDs from a hidden state.
    ///
    /// Convenience wrapper around [`Self::forward`] + [`top_k_indices`].
    pub fn top_k_per_head(&self, hidden: &Tensor, k: usize) -> Result<Vec<Vec<u32>>> {
        let logits_per_head = self.forward(hidden)?;
        let mut out = Vec::with_capacity(self.heads.len());
        for logits in logits_per_head {
            // logits dims: same as hidden's leading + vocab. We expect [vocab]
            // after squeezing the per-position dim — caller is responsible
            // for passing in a 1D hidden state.
            let v = logits
                .to_dtype(DType::F32)
                .map_err(Error::Candle)?
                .to_vec1::<f32>()
                .map_err(Error::Candle)?;
            let top: Vec<u32> = top_k_indices(&v, k).into_iter().map(|i| i as u32).collect();
            out.push(top);
        }
        Ok(out)
    }
}

/// End-to-end Medusa loop against any [`TreeDecoder`] target with
/// candle-backed heads. Uses `target.tree_logits` for single-pass tree
/// verification — much faster than the path-by-path simulation in
/// [`run_medusa`].
pub fn run_medusa_real<T, R>(
    target: &mut T,
    heads: &MedusaHeadsCandle,
    skeleton: &MedusaHeads,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &MedusaRunConfig,
    rng: &mut R,
) -> Result<Vec<u32>>
where
    T: TreeDecoder + ?Sized,
    R: rand::Rng + ?Sized,
{
    if heads.config().n_heads != skeleton.len() {
        return Err(Error::Sampling(format!(
            "head bundle size ({}) does not match skeleton ({})",
            heads.config().n_heads,
            skeleton.len()
        )));
    }

    target.reset();
    Decoder::observe(target, prompt)?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    while generated.len() < max_new_tokens {
        let root = *Decoder::history(target)
            .last()
            .ok_or_else(|| Error::Sampling("Medusa requires non-empty prompt".into()))?;

        // 1. Per-head top-k from the target's most recent hidden state.
        let hidden = target.last_hidden_state()?;
        let head_top_k = heads.top_k_per_head(&hidden, config.top_k_per_head)?;

        // 2. Tree from the skeleton's topology rule.
        let tree = skeleton.build_draft_tree(root, &head_top_k, config.topology)?;

        // 3. Tree verification: single forward, k+1 logit rows.
        let per_node_logits = target.tree_logits(&tree)?;

        // 4. Walk every root-to-leaf path; track the longest accepted prefix.
        let mut best_path: Vec<usize> = vec![0];
        for path in tree.paths() {
            let accepted_len =
                walk_and_accept(&path, &tree, &per_node_logits, &config.acceptance, rng);
            if accepted_len + 1 > best_path.len() {
                best_path = path[..=accepted_len].to_vec();
            }
        }

        // 5. Commit + bonus.
        let mut committed: Vec<u32> = best_path
            .iter()
            .skip(1)
            .map(|&i| tree.token_at(i))
            .collect();
        let deepest_idx = *best_path.last().unwrap();
        if generated.len() + committed.len() < max_new_tokens {
            let bonus = sample_argmax_or_categorical(
                &per_node_logits[deepest_idx],
                &config.acceptance,
                rng,
            )?;
            committed.push(bonus);
        }

        if committed.is_empty() {
            return Err(Error::Sampling(
                "Medusa round committed zero tokens — would loop forever".into(),
            ));
        }

        Decoder::observe(target, &committed)?;
        generated.extend_from_slice(&committed);
    }

    generated.truncate(max_new_tokens);
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_sensible() {
        let c = MedusaConfig::vicuna_7b_defaults();
        assert_eq!(c.n_heads, 4);
        assert_eq!(c.hidden_size, 4096);
    }

    #[test]
    fn heads_bundle_matches_n_heads() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 5,
            hidden_size: 256,
            vocab_size: 1000,
            residual_layers: 1,
        });
        assert_eq!(h.len(), 5);
        for (i, head) in h.heads.iter().enumerate() {
            assert_eq!(
                head.offset,
                i + 1,
                "head {i} should target offset {}",
                i + 1
            );
        }
    }

    #[test]
    fn top_k_picks_highest_with_stable_tie_break() {
        let logits = [0.1, 0.5, 0.5, 0.3, 0.5];
        // Three tied at 0.5 → indices 1,2,4. Top-3 should be 1,2,4.
        let idx = top_k_indices(&logits, 3);
        assert_eq!(idx, vec![1, 2, 4]);
    }

    #[test]
    fn top_k_clamps_to_vocab_size() {
        let logits = [0.5, 0.4];
        assert_eq!(top_k_indices(&logits, 100), vec![0, 1]);
    }

    #[test]
    fn greedy_topology_makes_linear_tree() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 3,
            hidden_size: 16,
            vocab_size: 100,
            residual_layers: 1,
        });
        let tree = h
            .build_draft_tree(7, &[vec![10], vec![20], vec![30]], TreeTopology::Greedy)
            .unwrap();
        assert_eq!(tree.tokens(), &[7, 10, 20, 30]);
        assert_eq!(tree.paths(), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn cartesian_topology_branches_at_each_head() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 2,
            hidden_size: 16,
            vocab_size: 100,
            residual_layers: 1,
        });
        // 2 candidates per head → 1 root, 2 layer-1 nodes, 4 layer-2 nodes.
        let tree = h
            .build_draft_tree(
                0,
                &[vec![10, 11], vec![20, 21]],
                TreeTopology::CartesianProduct,
            )
            .unwrap();
        assert_eq!(tree.len(), 1 + 2 + 4);
        // Four root-to-leaf paths, each of depth 2.
        let mut paths = tree.paths();
        paths.sort();
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert_eq!(p.len(), 3);
        }
    }

    #[test]
    fn cartesian_tree_attention_mask_blocks_cross_branches() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 2,
            hidden_size: 16,
            vocab_size: 100,
            residual_layers: 1,
        });
        let tree = h
            .build_draft_tree(0, &[vec![10, 11], vec![20]], TreeTopology::CartesianProduct)
            .unwrap();
        // Layout:
        //   0 = root (token 0)
        //   1 = head1 cand 10
        //   2 = head1 cand 11
        //   3 = head2 cand 20 under 1
        //   4 = head2 cand 20 under 2
        let mask = tree.attention_mask_bool();
        // Node 3's ancestors: {0, 1, 3}
        assert!(mask[3][0] && mask[3][1] && mask[3][3]);
        assert!(!mask[3][2], "node 3 must not see sibling-branch ancestor");
        assert!(!mask[3][4]);
        // Node 4's ancestors: {0, 2, 4}
        assert!(mask[4][0] && mask[4][2] && mask[4][4]);
        assert!(!mask[4][1] && !mask[4][3]);
    }

    #[test]
    fn build_rejects_wrong_head_count() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 3,
            hidden_size: 16,
            vocab_size: 100,
            residual_layers: 1,
        });
        let err = h
            .build_draft_tree(0, &[vec![1], vec![2]], TreeTopology::Greedy)
            .unwrap_err();
        assert!(matches!(err, Error::Sampling(_)));
    }

    #[test]
    fn build_rejects_empty_candidate_list() {
        let h = MedusaHeads::from_config(MedusaConfig {
            n_heads: 2,
            hidden_size: 16,
            vocab_size: 100,
            residual_layers: 1,
        });
        let err = h
            .build_draft_tree(0, &[vec![1], vec![]], TreeTopology::Greedy)
            .unwrap_err();
        assert!(matches!(err, Error::Sampling(_)));
    }

    // ======================================================================
    // run_medusa correctness tests — Phase 1b reference loop.
    // ======================================================================

    use crate::model::mock::fixed_distribution;
    use rand::SeedableRng;

    /// Build a `HeadDraftFn` that always returns the same per-head top-`k`
    /// candidates regardless of history. Useful for deterministic tests.
    fn fixed_head_draft(per_head: Vec<Vec<u32>>) -> HeadDraftFn {
        Box::new(move |_history| per_head.clone())
    }

    fn vocab_peak_at(vocab_size: usize, peak_idx: usize) -> Vec<f32> {
        let mut p = vec![0.001f32; vocab_size];
        // Reserve most of the mass for the peak so argmax is unambiguous.
        let remainder = 1.0 - 0.001 * (vocab_size as f32 - 1.0);
        p[peak_idx] = remainder;
        p
    }

    #[test]
    fn medusa_greedy_oracle_head_accepts_all() {
        // Target peaks at token 5. Head proposes 5 every position. We expect
        // all draft tokens accepted; the loop should commit n_heads + 1
        // tokens per round (4 accepted + 1 bonus = 5).
        let vocab = 16;
        let mut target = fixed_distribution(vocab_peak_at(vocab, 5));
        let heads = MedusaHeads::from_config(MedusaConfig {
            n_heads: 4,
            hidden_size: 1,
            vocab_size: vocab,
            residual_layers: 1,
        });
        let head_draft = fixed_head_draft(vec![vec![5], vec![5], vec![5], vec![5]]);
        let cfg = MedusaRunConfig {
            topology: TreeTopology::Greedy,
            top_k_per_head: 1,
            acceptance: Acceptance::Greedy,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let out = run_medusa(&mut target, &heads, head_draft, &[7u32], 20, &cfg, &mut rng).unwrap();
        assert_eq!(out.len(), 20);
        // Every output token should be 5 (target's argmax).
        for &t in &out {
            assert_eq!(t, 5, "expected target argmax (5), got {t}");
        }
    }

    #[test]
    fn medusa_greedy_wrong_head_falls_back_to_bonus_only() {
        // Target peaks at 5; head proposes 7. Greedy rejects every draft, but
        // each round still commits the bonus token (target's argmax = 5).
        // So we should still produce exactly max_new_tokens tokens, all = 5.
        let vocab = 16;
        let mut target = fixed_distribution(vocab_peak_at(vocab, 5));
        let heads = MedusaHeads::from_config(MedusaConfig {
            n_heads: 3,
            hidden_size: 1,
            vocab_size: vocab,
            residual_layers: 1,
        });
        let head_draft = fixed_head_draft(vec![vec![7], vec![7], vec![7]]);
        let cfg = MedusaRunConfig {
            topology: TreeTopology::Greedy,
            top_k_per_head: 1,
            acceptance: Acceptance::Greedy,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let out = run_medusa(&mut target, &heads, head_draft, &[1u32], 12, &cfg, &mut rng).unwrap();
        assert_eq!(out.len(), 12);
        for &t in &out {
            assert_eq!(t, 5);
        }
    }

    #[test]
    fn medusa_cartesian_picks_correct_branch() {
        // Cartesian product with one branch matching target, one not.
        // Target peaks at 5. Head 0 proposes [5, 99]. Head 1 proposes [5, 99].
        // The 4 paths are (5,5), (5,99), (99,5), (99,99).
        // Greedy acceptance: (5,5) accepts both, others fail at first node.
        // Best path = (5,5), so we commit 2 + 1 bonus = 3 tokens per round.
        let vocab = 128;
        let mut target = fixed_distribution(vocab_peak_at(vocab, 5));
        let heads = MedusaHeads::from_config(MedusaConfig {
            n_heads: 2,
            hidden_size: 1,
            vocab_size: vocab,
            residual_layers: 1,
        });
        let head_draft = fixed_head_draft(vec![vec![5, 99], vec![5, 99]]);
        let cfg = MedusaRunConfig {
            topology: TreeTopology::CartesianProduct,
            top_k_per_head: 2,
            acceptance: Acceptance::Greedy,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let out = run_medusa(&mut target, &heads, head_draft, &[1u32], 9, &cfg, &mut rng).unwrap();
        assert_eq!(out.len(), 9);
        for &t in &out {
            assert_eq!(t, 5);
        }
    }

    #[test]
    fn medusa_typical_acceptance_threshold_blocks_low_mass_token() {
        // Target distribution: mass spread thinly over 50 tokens, peak at 0.
        // Head proposes 25 (a low-mass token). With strict typical threshold
        // (epsilon=0.5), the candidate's probability < 0.5 → rejected.
        let vocab = 50;
        let mut probs = vec![0.01f32; vocab];
        probs[0] = 1.0 - 0.01 * (vocab as f32 - 1.0);
        let mut target = fixed_distribution(probs);
        let heads = MedusaHeads::from_config(MedusaConfig {
            n_heads: 2,
            hidden_size: 1,
            vocab_size: vocab,
            residual_layers: 1,
        });
        let head_draft = fixed_head_draft(vec![vec![25], vec![25]]);
        let cfg = MedusaRunConfig {
            topology: TreeTopology::Greedy,
            top_k_per_head: 1,
            acceptance: Acceptance::Typical {
                epsilon: 0.5,
                delta: 1.0,
            },
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let out = run_medusa(&mut target, &heads, head_draft, &[1u32], 6, &cfg, &mut rng).unwrap();
        // All draft tokens rejected → only bonus per round (= argmax = 0).
        assert_eq!(out.len(), 6);
        for &t in &out {
            assert_eq!(t, 0);
        }
    }

    #[test]
    fn evaluate_tree_restores_target_history() {
        let vocab = 8;
        let mut target = fixed_distribution(vocab_peak_at(vocab, 3));
        Decoder::observe(&mut target, &[0u32, 1, 2]).unwrap();
        let pre = Decoder::history_len(&target);
        let tree = DraftTree::linear(2, &[5, 6, 7]);
        let _ = evaluate_tree(&mut target, &tree, pre).unwrap();
        assert_eq!(
            Decoder::history_len(&target),
            pre,
            "evaluate_tree must restore history"
        );
    }
}
