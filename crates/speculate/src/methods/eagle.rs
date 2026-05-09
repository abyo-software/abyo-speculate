//! EAGLE-2 (Li et al. 2024) — speculative decoding with target-hidden-state
//! conditioned drafts.
//!
//! ## What EAGLE adds vs Medusa / Vanilla SD
//!
//! Vanilla SD pairs the target with a *separate* small target-shaped draft
//! model — both run autoregressively. Medusa attaches `N` heads that
//! predict directly from the target's hidden state, no draft autoregression.
//! EAGLE sits in between: a tiny **1-layer transformer** runs autoregressively
//! over a sequence of `(target_hidden, token_embedding)` pairs, propagating
//! its own KV cache. The result is higher acceptance rates than Medusa
//! (because the draft sees real target context, not just the last hidden)
//! and lower draft cost than vanilla SD (because the draft is 1 layer, not
//! ~30).
//!
//! ## Reference checkpoint
//!
//! `yuhuili/EAGLE-LLaMA3-Instruct-8B` ships the draft for Llama 3 8B in a
//! 1.5 GB `pytorch_model.bin`. Key layout:
//!
//! ```text
//! embed_tokens.weight                       (vocab,  hidden)
//! fc.weight                                 (hidden, 2*hidden)   # concat input projection
//! layers.0.self_attn.{q,k,v,o}_proj.weight  Llama attention (no biases, GQA)
//! layers.0.mlp.{gate,up,down}_proj.weight   Llama SwiGLU MLP
//! layers.0.post_attention_layernorm.weight  RmsNorm before MLP
//! ```
//!
//! Notably absent: `input_layernorm`. The draft's input is the target's
//! last-layer norm output, so we skip the second normalisation.
//!
//! ## What's still v0.2.0 follow-up
//!
//! - Dynamic confidence-based tree expansion (this v0.1.0 lands a fixed
//!   Cartesian-product tree, like Medusa's `TreeTopology::CartesianProduct`).
//!   The dynamic version improves acceptance rates by ~10-20% on the EAGLE
//!   paper's benchmarks.
//! - EAGLE-3's multi-layer feature aggregation. The 1-layer draft sees
//!   only the target's last hidden state today.
//! - Real-GPU end-to-end speedup measurement (needs `last_hidden_state`
//!   exposed for *each* draft step, not just the most recent commit —
//!   this loop calls `last_hidden_state` once per round and grows the
//!   tree from there with the draft's own forward; bench numbers land in
//!   v0.2.0 alongside the dynamic-tree improvement).

#![allow(missing_docs)]

use crate::model::TreeDecoder;
use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use std::path::Path;

/// Hyper-parameters for an EAGLE draft model.
#[derive(Debug, Clone)]
pub struct EagleDraftConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl EagleDraftConfig {
    /// Defaults for `yuhuili/EAGLE-LLaMA3-Instruct-8B`.
    pub fn eagle_llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            vocab_size: 128256,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0, // Llama 3 uses 500k, not 10k
            max_position_embeddings: 2048,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct DraftAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    cos: Tensor,
    sin: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl DraftAttention {
    fn load(
        cfg: &EagleDraftConfig,
        vb: VarBuilder<'_>,
        dev: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear_no_bias(h, n * head_dim, vb.pp("q_proj")).map_err(Error::Candle)?;
        let k_proj = linear_no_bias(h, n_kv * head_dim, vb.pp("k_proj")).map_err(Error::Candle)?;
        let v_proj = linear_no_bias(h, n_kv * head_dim, vb.pp("v_proj")).map_err(Error::Candle)?;
        let o_proj = linear_no_bias(n * head_dim, h, vb.pp("o_proj")).map_err(Error::Candle)?;

        // Precompute cos/sin tables.
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq.clone(), (1, inv_freq.len()), dev)
            .map_err(Error::Candle)?
            .to_dtype(dtype)
            .map_err(Error::Candle)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)
            .map_err(Error::Candle)?
            .to_dtype(dtype)
            .map_err(Error::Candle)?
            .reshape((cfg.max_position_embeddings, 1))
            .map_err(Error::Candle)?;
        let freqs = t.matmul(&inv_freq_t).map_err(Error::Candle)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cos: freqs.cos().map_err(Error::Candle)?,
            sin: freqs.sin().map_err(Error::Candle)?,
            n_head: n,
            n_kv_head: n_kv,
            head_dim,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, position: usize) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3().map_err(Error::Candle)?;
        let q = self
            .q_proj
            .forward(xs)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_head, self.head_dim))
            .map_err(Error::Candle)?
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;
        let k = self
            .k_proj
            .forward(xs)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_kv_head, self.head_dim))
            .map_err(Error::Candle)?
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;
        let v = self
            .v_proj
            .forward(xs)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_kv_head, self.head_dim))
            .map_err(Error::Candle)?
            .transpose(1, 2)
            .map_err(Error::Candle)?;

        // RoPE.
        let cos = self.cos.narrow(0, position, q_len).map_err(Error::Candle)?;
        let sin = self.sin.narrow(0, position, q_len).map_err(Error::Candle)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin).map_err(Error::Candle)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin).map_err(Error::Candle)?;

        // KV cache.
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2).map_err(Error::Candle)?,
                Tensor::cat(&[pv, &v], 2).map_err(Error::Candle)?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA repeat.
        let n_rep = self.n_head / self.n_kv_head;
        let k = candle_transformers::utils::repeat_kv(k, n_rep)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;
        let v = candle_transformers::utils::repeat_kv(v, n_rep)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q
            .matmul(&k.t().map_err(Error::Candle)?)
            .map_err(Error::Candle)?
            * scale)
            .map_err(Error::Candle)?;
        // Build a causal mask for the new positions vs the cached prefix.
        let prev_len = self
            .kv_cache
            .as_ref()
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
            - q_len;
        let attn = if q_len <= 1 {
            attn
        } else {
            let total = prev_len + q_len;
            let mut data = vec![0f32; q_len * total];
            for i in 0..q_len {
                for j in 0..q_len {
                    if j > i {
                        data[i * total + prev_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let bias = Tensor::from_slice(&data, (q_len, total), xs.device())
                .map_err(Error::Candle)?
                .to_dtype(xs.dtype())
                .map_err(Error::Candle)?
                .reshape((1, 1, q_len, total))
                .map_err(Error::Candle)?;
            attn.broadcast_add(&bias).map_err(Error::Candle)?
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn).map_err(Error::Candle)?;
        let y = attn.matmul(&v).map_err(Error::Candle)?;
        let y = y
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_head * self.head_dim))
            .map_err(Error::Candle)?;
        self.o_proj.forward(&y).map_err(Error::Candle)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct DraftMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DraftMlp {
    fn load(cfg: &EagleDraftConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate: linear_no_bias(h, i, vb.pp("gate_proj")).map_err(Error::Candle)?,
            up: linear_no_bias(h, i, vb.pp("up_proj")).map_err(Error::Candle)?,
            down: linear_no_bias(i, h, vb.pp("down_proj")).map_err(Error::Candle)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(xs).map_err(Error::Candle)?)
            .map_err(Error::Candle)?;
        let u = self.up.forward(xs).map_err(Error::Candle)?;
        self.down
            .forward(&(g * u).map_err(Error::Candle)?)
            .map_err(Error::Candle)
    }
}

/// EAGLE draft model loaded from a published checkpoint.
///
/// The draft consumes `concat(target_last_hidden, token_embedding)` per
/// position, projects through `fc` to hidden, then through a single Llama
/// block (no `input_layernorm` — the target's last RmsNorm already
/// normalises the input). Output hidden goes through the *target's*
/// `lm_head` for vocab logits (no separate draft head).
pub struct EagleDraftCandle {
    config: EagleDraftConfig,
    embed_tokens: Embedding,
    fc: Linear,
    attn: DraftAttention,
    post_attention_layernorm: RmsNorm,
    mlp: DraftMlp,
}

impl std::fmt::Debug for EagleDraftCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EagleDraftCandle")
            .field("hidden_size", &self.config.hidden_size)
            .field("vocab_size", &self.config.vocab_size)
            .finish()
    }
}

impl EagleDraftCandle {
    /// Read-only view of the config.
    pub fn config(&self) -> &EagleDraftConfig {
        &self.config
    }

    /// Load from a single PyTorch pickle file
    /// (`yuhuili/EAGLE-...`'s `pytorch_model.bin`).
    pub fn from_pth(
        config: &EagleDraftConfig,
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(path.as_ref(), dtype, device).map_err(Error::Candle)?;
        Self::from_var_builder(config, vb, device, dtype)
    }

    /// Load from a caller-supplied [`VarBuilder`].
    pub fn from_var_builder(
        config: &EagleDraftConfig,
        vb: VarBuilder<'_>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))
                .map_err(Error::Candle)?;
        let fc = linear_no_bias(2 * config.hidden_size, config.hidden_size, vb.pp("fc"))
            .map_err(Error::Candle)?;
        let attn = DraftAttention::load(config, vb.pp("layers.0.self_attn"), device, dtype)?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("layers.0.post_attention_layernorm"),
        )
        .map_err(Error::Candle)?;
        let mlp = DraftMlp::load(config, vb.pp("layers.0.mlp"))?;
        Ok(Self {
            config: config.clone(),
            embed_tokens,
            fc,
            attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Reset the draft's KV cache between rounds.
    pub fn reset(&mut self) {
        self.attn.clear_kv_cache();
    }

    /// Run one forward step.
    ///
    /// Inputs:
    /// - `target_hidden`: shape `[1, seq, hidden]` — target's last-layer
    ///   norm output for the same `seq` positions.
    /// - `token_ids`: shape `[1, seq]` — the token ids at those positions
    ///   (used for `embed_tokens`).
    /// - `position`: absolute position offset for RoPE (typically the
    ///   number of tokens already in the target prefix when starting a
    ///   round, then incremented).
    ///
    /// Returns: shape `[1, seq, hidden]` — the draft's hidden state output,
    /// suitable for feeding to the target's `lm_head` for next-token logits.
    pub fn forward(
        &mut self,
        target_hidden: &Tensor,
        token_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let token_emb = self
            .embed_tokens
            .forward(token_ids)
            .map_err(Error::Candle)?;
        // The target may dequantize to F32 (quantized_llama path) while
        // the draft was loaded in F16. Promote target_hidden to the
        // draft's dtype before concat.
        let target_hidden_owned;
        let target_hidden_use: &Tensor = if target_hidden.dtype() != token_emb.dtype() {
            target_hidden_owned = target_hidden
                .to_dtype(token_emb.dtype())
                .map_err(Error::Candle)?;
            &target_hidden_owned
        } else {
            target_hidden
        };
        let combined =
            Tensor::cat(&[target_hidden_use, &token_emb], D::Minus1).map_err(Error::Candle)?;
        let xs = self.fc.forward(&combined).map_err(Error::Candle)?;
        // EAGLE block: attention (no pre-LN) + post_attention_layernorm + mlp.
        let res = xs;
        let attn = self.attn.forward(&res, position)?;
        let xs = (attn + &res).map_err(Error::Candle)?;
        let res = &xs;
        let xs_n = self
            .post_attention_layernorm
            .forward(&xs)
            .map_err(Error::Candle)?;
        let m = self.mlp.forward(&xs_n)?;
        (m + res).map_err(Error::Candle)
    }
}

/// Run-loop config for EAGLE.
#[derive(Debug, Clone)]
pub struct EagleRunConfig {
    /// Top-`k` per draft autoregressive step. Each step's top-k forms a
    /// branching factor in the static Cartesian-product tree.
    pub top_k_per_step: usize,
    /// Number of draft autoregressive steps per round (= tree depth).
    pub draft_depth: usize,
    /// If set, prune the Cartesian-product tree (1 + Σ k^d ≈ k^depth nodes)
    /// down to this many nodes by keeping the top-N path-scored nodes plus
    /// every ancestor needed to keep them connected. v0.2.0-3 dynamic tree.
    /// `None` keeps the full Cartesian tree.
    pub max_tree_nodes: Option<usize>,
    /// Sampling temperature applied at the target side.
    pub temperature: f32,
    /// Top-p nucleus.
    pub top_p: f32,
}

impl Default for EagleRunConfig {
    fn default() -> Self {
        Self {
            top_k_per_step: 2,
            draft_depth: 4,
            max_tree_nodes: None,
            temperature: 0.0, // greedy by default — strictest acceptance
            top_p: 1.0,
        }
    }
}

/// End-to-end EAGLE-2 loop.
///
/// Algorithm per round:
/// 1. Get the target's last hidden state for the most recent committed
///    token via [`TreeDecoder::last_hidden_state`].
/// 2. Run the draft forward `draft_depth` times. Each step:
///    - Input: that hidden state + the most recent token's id.
///    - Output: draft hidden → target's lm_head → vocab logits → top-k
///      (we cheat slightly by using the target's lm_head externally; full
///      EAGLE shares it through tied embeddings — close enough for a
///      static tree).
///    - For each top-k branch, the next iteration's input is the draft's
///      *own* output hidden + the candidate token's embedding.
/// 3. Build a Cartesian-product DraftTree from the per-step top-k.
/// 4. Verify the tree via `target.tree_logits` (Phase 2a tree attention).
/// 5. Walk paths, accept via greedy match (temperature 0 default).
/// 6. Commit the longest accepted prefix + bonus.
///
/// The per-step "draft logits" path uses [`TreeDecoder::apply_lm_head`]
/// directly — EAGLE shares the target's vocab head via tied embeddings, so
/// we don't keep a separate copy on the draft.
pub fn run_eagle<T, R>(
    target: &mut T,
    draft: &mut EagleDraftCandle,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &EagleRunConfig,
    rng: &mut R,
) -> Result<Vec<u32>>
where
    T: TreeDecoder + ?Sized,
    R: rand::Rng + ?Sized,
{
    use crate::methods::medusa::top_k_indices;

    target.reset();
    target.observe(prompt)?;

    let mut generated = Vec::with_capacity(max_new_tokens);
    while generated.len() < max_new_tokens {
        let root_token = *target
            .history()
            .last()
            .ok_or_else(|| Error::Sampling("EAGLE requires non-empty prompt".into()))?;

        // 1. Get target's hidden state for the most recent committed token.
        let target_hidden = target.last_hidden_state()?;
        // Reshape to [1, 1, hidden] for draft.forward.
        let hidden_reshaped = target_hidden
            .unsqueeze(0)
            .map_err(Error::Candle)?
            .unsqueeze(0)
            .map_err(Error::Candle)?;

        draft.reset();
        let history_len = target.history_len();

        // 2. Build Cartesian-product tree by running draft `draft_depth` times.
        let mut per_step_top_k: Vec<Vec<u32>> = Vec::with_capacity(config.draft_depth);
        let mut per_step_top_k_log_probs: Vec<Vec<f32>> = Vec::with_capacity(config.draft_depth);
        let mut current_hidden = hidden_reshaped;
        let mut current_token_ids =
            Tensor::from_slice(&[root_token], (1, 1), target_hidden.device())
                .map_err(Error::Candle)?;

        for step in 0..config.draft_depth {
            let draft_hidden =
                draft.forward(&current_hidden, &current_token_ids, history_len + step)?;
            // The target's lm_head may be quantized (F32-only input) while
            // the draft is F16. Promote before applying.
            let draft_hidden_for_head = if draft_hidden.dtype() != DType::F32 {
                draft_hidden.to_dtype(DType::F32).map_err(Error::Candle)?
            } else {
                draft_hidden.clone()
            };
            let logits = target.apply_lm_head(&draft_hidden_for_head)?;
            // Take the last position's logits — for a 1-token forward this
            // is just position 0.
            let last = logits
                .i((0, draft_hidden.dim(1).map_err(Error::Candle)? - 1, ..))
                .map_err(Error::Candle)?
                .to_dtype(DType::F32)
                .map_err(Error::Candle)?
                .to_vec1::<f32>()
                .map_err(Error::Candle)?;
            let top_idx: Vec<usize> = top_k_indices(&last, config.top_k_per_step);
            // Log-softmax over `last` for the kept top-k indices.
            let max_l = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = last.iter().map(|&v| (v - max_l).exp()).sum::<f32>().ln() + max_l;
            let top_log_probs: Vec<f32> = top_idx.iter().map(|&i| last[i] - lse).collect();
            let top: Vec<u32> = top_idx.iter().map(|&i| i as u32).collect();
            per_step_top_k.push(top.clone());
            per_step_top_k_log_probs.push(top_log_probs);

            // For the next step, advance with the top-1 (single greedy
            // chain — Cartesian expansion happens in the tree, not the
            // draft autoregression).
            let next_id = top[0];
            current_token_ids = Tensor::from_slice(&[next_id], (1, 1), target_hidden.device())
                .map_err(Error::Candle)?;
            current_hidden = draft_hidden;
        }

        // 3. Build Cartesian-product tree, optionally pruned to top-N nodes.
        let full_tree = crate::methods::medusa::MedusaHeads::from_config(
            crate::methods::medusa::MedusaConfig {
                n_heads: config.draft_depth,
                hidden_size: draft.config.hidden_size,
                vocab_size: draft.config.vocab_size,
                residual_layers: 1,
            },
        )
        .build_draft_tree(
            root_token,
            &per_step_top_k,
            crate::methods::medusa::TreeTopology::CartesianProduct,
        )?;
        let tree = if let Some(max_n) = config.max_tree_nodes {
            prune_cartesian_tree(&full_tree, &per_step_top_k_log_probs, max_n)?
        } else {
            full_tree
        };

        // 4. Verify via target's tree_logits.
        let per_node_logits = target.tree_logits(&tree)?;

        // 5. Walk paths, greedy acceptance.
        let mut best_path: Vec<usize> = vec![0];
        for path in tree.paths() {
            let mut accepted = 0;
            for w in path.windows(2) {
                let parent = w[0];
                let child = w[1];
                let candidate = tree.token_at(child) as usize;
                let parent_dist = &per_node_logits[parent];
                let argmax = parent_dist
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
                if argmax == candidate {
                    accepted += 1;
                } else {
                    break;
                }
            }
            if accepted + 1 > best_path.len() {
                best_path = path[..=accepted].to_vec();
            }
        }

        // 6. Commit + bonus.
        let mut committed: Vec<u32> = best_path
            .iter()
            .skip(1)
            .map(|&i| tree.token_at(i))
            .collect();
        let deepest_idx = *best_path.last().unwrap();
        if generated.len() + committed.len() < max_new_tokens {
            let bonus_logits = &per_node_logits[deepest_idx];
            let bonus = bonus_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as u32;
            committed.push(bonus);
        }

        if committed.is_empty() {
            return Err(Error::Sampling("EAGLE round committed zero tokens".into()));
        }

        target.observe(&committed)?;
        generated.extend_from_slice(&committed);
    }
    // Suppress unused-rng warning until temperature > 0 lands.
    let _ = (rng, config.temperature, config.top_p);

    generated.truncate(max_new_tokens);
    Ok(generated)
}

/// Prune a Cartesian-product `DraftTree` (built by
/// [`crate::methods::medusa::TreeTopology::CartesianProduct`]) down to at
/// most `max_total_nodes` nodes by keeping the highest-cumulative-log-prob
/// non-root nodes plus every ancestor needed to keep them connected.
///
/// `per_step_log_probs[d][k]` is the log-prob of the `k`-th candidate at
/// depth `d` (depth 0 = first step from the root). The full Cartesian
/// branching factor at depth `d` is `per_step_log_probs[d].len()`.
fn prune_cartesian_tree(
    full: &crate::tree::DraftTree,
    per_step_log_probs: &[Vec<f32>],
    max_total_nodes: usize,
) -> Result<crate::tree::DraftTree> {
    use crate::tree::DraftTree;

    if full.len() <= max_total_nodes {
        // Already small enough — return as-is via reconstruction so the
        // caller always gets an owned tree.
        return clone_tree(full);
    }

    // Score every non-root node by the sum of log-probs along its path
    // from the root. We rebuild the path by walking parent pointers and
    // looking up which candidate index (within its layer) was used.
    // Layer d branches per_step_log_probs[d].len() ways. In the BFS-built
    // Cartesian tree the candidate index of a node at depth d+1 is the
    // node's child position among its parent's children — which we can
    // recover from per-parent child tracking below.

    // Build child -> candidate-index lookup.
    // For each parent, the children appear in the same order as
    // per_step_log_probs[depth_of(parent)]; so child position 0 = candidate 0, etc.
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); full.len()];
    for n in 1..full.len() {
        let p = full.parent_of(n);
        children_of[p].push(n);
    }
    // candidate_index_of[node] = which sibling rank (0..k) this node has.
    let mut candidate_index_of = vec![0usize; full.len()];
    for siblings in &children_of {
        for (rank, &c) in siblings.iter().enumerate() {
            candidate_index_of[c] = rank;
        }
    }

    // Score each non-root by walking up and summing log-probs.
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(full.len() - 1);
    for n in 1..full.len() {
        let depth = full.depth_of(n);
        let mut s = 0f32;
        let mut cur = n;
        for d in (0..depth).rev() {
            let cand = candidate_index_of[cur];
            s += per_step_log_probs[d][cand];
            cur = full.parent_of(cur);
        }
        scores.push((n, s));
    }

    // Pick top (max_total_nodes - 1) nodes (root is always kept) by score.
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let want = max_total_nodes.saturating_sub(1).min(scores.len());
    let mut keep = vec![false; full.len()];
    keep[0] = true;
    for &(n, _) in scores.iter().take(want) {
        keep[n] = true;
    }
    // Ancestor closure.
    for n in (1..full.len()).rev() {
        if keep[n] {
            keep[full.parent_of(n)] = true;
        }
    }

    // Re-emit kept nodes in BFS order to preserve parent < self invariant.
    let mut order: Vec<usize> = (0..full.len()).filter(|&n| keep[n]).collect();
    order.sort_by_key(|&n| full.depth_of(n));
    let new_index: std::collections::HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(new_i, &old_i)| (old_i, new_i))
        .collect();
    let mut entries: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    for &old in &order {
        let parent_old = if old == 0 { 0 } else { full.parent_of(old) };
        let parent_new = *new_index.get(&parent_old).expect("ancestor present");
        entries.push((parent_new, full.token_at(old)));
    }
    DraftTree::from_parent_table(&entries)
        .map_err(|e| Error::Sampling(format!("pruned tree invalid: {e}")))
}

fn clone_tree(t: &crate::tree::DraftTree) -> Result<crate::tree::DraftTree> {
    let entries: Vec<(usize, u32)> = (0..t.len())
        .map(|i| {
            let parent = if i == 0 { 0 } else { t.parent_of(i) };
            (parent, t.token_at(i))
        })
        .collect();
    crate::tree::DraftTree::from_parent_table(&entries)
        .map_err(|e| Error::Sampling(format!("clone tree invalid: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_eagle_llama3_8b() {
        let c = EagleDraftConfig::eagle_llama3_8b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.vocab_size, 128256);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim(), 128);
    }

    #[test]
    fn config_clone() {
        let c = EagleDraftConfig::eagle_llama3_8b();
        let c2 = c.clone();
        assert_eq!(c.hidden_size, c2.hidden_size);
    }

    #[test]
    fn prune_keeps_root_and_top_paths() {
        // Build a Cartesian k=2 depth=2 tree (1 + 2 + 4 = 7 nodes).
        let cart = crate::methods::medusa::MedusaHeads::from_config(
            crate::methods::medusa::MedusaConfig {
                n_heads: 2,
                hidden_size: 4,
                vocab_size: 100,
                residual_layers: 1,
            },
        )
        .build_draft_tree(
            42, // root token
            &[vec![10, 20], vec![30, 40]],
            crate::methods::medusa::TreeTopology::CartesianProduct,
        )
        .expect("build cart");
        assert_eq!(cart.len(), 7);

        // Layer 0: candidate 0 has log-prob -0.1, candidate 1 has -2.0.
        // Layer 1: candidate 0 has -0.2, candidate 1 has -3.0.
        // Best path from root: (10 -> 30) score = -0.1 + -0.2 = -0.3.
        // Worst: (20 -> 40) score = -2.0 + -3.0 = -5.0.
        let log_probs = vec![vec![-0.1f32, -2.0], vec![-0.2, -3.0]];

        // Prune to 4 nodes.
        let pruned = prune_cartesian_tree(&cart, &log_probs, 4).expect("prune");
        assert!(pruned.len() <= 4, "pruned should be ≤ 4 nodes");
        assert!(pruned.len() >= 1);
        assert_eq!(pruned.token_at(0), 42, "root preserved");

        // The single best path (root, 10, 30) must be present.
        let tokens: Vec<u32> = (0..pruned.len()).map(|i| pruned.token_at(i)).collect();
        assert!(tokens.contains(&10), "best layer-0 child kept");
        assert!(tokens.contains(&30), "best layer-1 child kept");
    }

    #[test]
    fn prune_returns_full_tree_when_under_limit() {
        let cart = crate::methods::medusa::MedusaHeads::from_config(
            crate::methods::medusa::MedusaConfig {
                n_heads: 2,
                hidden_size: 4,
                vocab_size: 100,
                residual_layers: 1,
            },
        )
        .build_draft_tree(
            1,
            &[vec![2], vec![3]],
            crate::methods::medusa::TreeTopology::CartesianProduct,
        )
        .expect("build");
        assert_eq!(cart.len(), 3);
        let pruned = prune_cartesian_tree(&cart, &[vec![-0.1], vec![-0.2]], 100).expect("prune");
        assert_eq!(pruned.len(), 3);
    }
}
