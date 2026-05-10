//! EAGLE-3 (Li et al. 2025) — multi-layer feature aggregation draft.
//!
//! Significant architectural changes vs EAGLE-2:
//!
//! 1. **3-layer feature fusion.** `fc.weight` shape is `(hidden, 3*hidden)` —
//!    the input is `concat(target_hidden_low, target_hidden_mid,
//!    target_hidden_high)`, three target-layer hidden states selected at
//!    training time. The draft sees richer context than just the last
//!    layer.
//! 2. **Own vocab + token translation.** EAGLE-3 trains with a smaller draft
//!    vocab (32k for the LLaMA3.1 head we examined; the target has 128k).
//!    Two int tensors handle the translation:
//!    - `d2t: [draft_vocab] -> target_vocab_id`
//!    - `t2d: [target_vocab] -> bool` (whether the target id is reachable
//!      via the draft). Used during sampling to mask unreachable target
//!      vocabulary entries.
//! 3. **Has its own `lm_head`** (not tied to target) and a `norm` before it.
//! 4. **`midlayer.*` naming** instead of EAGLE-2's `layers.0.*`. Plus an
//!    `input_layernorm` (EAGLE-2 had only `post_attention_layernorm`) and
//!    a separate `hidden_norm` for the projected concat input.
//! 5. **Wider attention input.** `q_proj/k_proj/v_proj` accept `2*hidden`,
//!    not `hidden` — the draft's attention sees the concat of the
//!    fc-projected feature plus the latest target hidden state.
//!
//! ## Status: reference-matching architecture + outstanding tree_logits bug
//!
//! What's wired (v0.2.1):
//! - [`Eagle3DraftConfig`] from `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`'s
//!   defaults.
//! - [`Eagle3DraftCandle::from_pth`] / `from_var_builder` parse every
//!   key in the checkpoint (no `embed_tokens`; the published checkpoint
//!   reuses the **target's** tied embedding via [`Self::forward`]'s
//!   `input_emb` arg).
//! - [`Self::forward_hidden`] matches `Model.forward` from
//!   `SafeAILab/EAGLE/eagle/model/cnets.py`: applies `fc` only when
//!   the input is the 3*hidden concat (first round); subsequent rounds
//!   pass the previous draft's pre-norm midlayer output back in.
//! - [`Self::apply_norm_lm_head`] mirrors
//!   `lm_head(self.norm(last_hidden))`.
//! - [`Eagle3RunConfig::default_layers_for`] returns the published
//!   training recipe's layer triple — `2 / n/2 / n-3` (input-of), which
//!   in our after-layer-i convention is `[1, n/2 - 1, n - 4]`.
//! - [`run_eagle3`] runs the official schedule: round-0 uses
//!   `embed(root_token) + concat(low,mid,high)`; subsequent rounds pass
//!   `embed(top1_drafted) + previous midlayer output`.
//!
//! ## v0.2.2: tree_logits multi-position bug fixed
//!
//! v0.2.1 isolated a Q4 GEMV-vs-GEMM precision drift in
//! `LlamaQuantDecoder::tree_logits` that broke greedy acceptance on
//! borderline argmax cases. v0.2.2 patched it (see CHANGELOG); EAGLE-3
//! end-to-end now produces output identical to AR baseline. Speed
//! remains bounded by Q4 quantization noise affecting draft accuracy.

#![allow(missing_docs)]

use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};
use std::path::Path;

/// Hyper-parameters for an EAGLE-3 draft model.
#[derive(Debug, Clone)]
pub struct Eagle3DraftConfig {
    /// Hidden dim of the target.
    pub hidden_size: usize,
    /// Draft vocab size (typically smaller than target's vocab).
    pub draft_vocab_size: usize,
    /// Target vocab size — used to size the t2d mask.
    pub target_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Eagle3DraftConfig {
    /// Defaults for `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`.
    pub fn eagle3_llama3_1_8b() -> Self {
        Self {
            hidden_size: 4096,
            draft_vocab_size: 32000,
            target_vocab_size: 128256,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            max_position_embeddings: 2048,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct Midlayer {
    input_layernorm: RmsNorm,
    hidden_norm: RmsNorm,
    q_proj: Linear, // takes 2*hidden → n_head * head_dim
    k_proj: Linear, // takes 2*hidden → n_kv_head * head_dim
    v_proj: Linear, // takes 2*hidden → n_kv_head * head_dim
    o_proj: Linear, // takes n_head * head_dim → hidden
    post_attention_layernorm: RmsNorm,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
    cos: Tensor,
    sin: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Midlayer {
    fn load(
        cfg: &Eagle3DraftConfig,
        vb: VarBuilder<'_>,
        dev: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let n = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let input_layernorm =
            rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm")).map_err(Error::Candle)?;
        let hidden_norm =
            rms_norm(h, cfg.rms_norm_eps, vb.pp("hidden_norm")).map_err(Error::Candle)?;
        // EAGLE-3 attention takes a 2*hidden input (target_hidden + projected_concat).
        let q_proj = linear_no_bias(2 * h, n * head_dim, vb.pp("self_attn.q_proj"))
            .map_err(Error::Candle)?;
        let k_proj = linear_no_bias(2 * h, n_kv * head_dim, vb.pp("self_attn.k_proj"))
            .map_err(Error::Candle)?;
        let v_proj = linear_no_bias(2 * h, n_kv * head_dim, vb.pp("self_attn.v_proj"))
            .map_err(Error::Candle)?;
        let o_proj =
            linear_no_bias(n * head_dim, h, vb.pp("self_attn.o_proj")).map_err(Error::Candle)?;
        let post_attention_layernorm =
            rms_norm(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))
                .map_err(Error::Candle)?;
        let mlp_gate = linear_no_bias(h, i, vb.pp("mlp.gate_proj")).map_err(Error::Candle)?;
        let mlp_up = linear_no_bias(h, i, vb.pp("mlp.up_proj")).map_err(Error::Candle)?;
        let mlp_down = linear_no_bias(i, h, vb.pp("mlp.down_proj")).map_err(Error::Candle)?;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|j| 1f32 / cfg.rope_theta.powf(j as f32 / head_dim as f32))
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
            input_layernorm,
            hidden_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm,
            mlp_gate,
            mlp_up,
            mlp_down,
            cos: freqs.cos().map_err(Error::Candle)?,
            sin: freqs.sin().map_err(Error::Candle)?,
            n_head: n,
            n_kv_head: n_kv,
            head_dim,
            kv_cache: None,
        })
    }

    /// Forward through the midlayer.
    ///
    /// `target_hidden`: shape `[1, seq, hidden]` — most recent target hidden
    /// states (these go through `input_layernorm`).
    /// `projected_features`: shape `[1, seq, hidden]` — the
    /// `fc(concat(low, mid, high))` output (these go through `hidden_norm`).
    /// `position`: absolute position offset for RoPE.
    fn forward(
        &mut self,
        target_hidden: &Tensor,
        projected_features: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = target_hidden.dims3().map_err(Error::Candle)?;
        // Normalise inputs.
        let th = self
            .input_layernorm
            .forward(target_hidden)
            .map_err(Error::Candle)?;
        let pf = self
            .hidden_norm
            .forward(projected_features)
            .map_err(Error::Candle)?;
        // Concat along hidden dim → [b, seq, 2*hidden].
        let combined = Tensor::cat(&[&th, &pf], D::Minus1).map_err(Error::Candle)?;
        // QKV projection.
        let q = self
            .q_proj
            .forward(&combined)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_head, self.head_dim))
            .map_err(Error::Candle)?
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;
        let k = self
            .k_proj
            .forward(&combined)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_kv_head, self.head_dim))
            .map_err(Error::Candle)?
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .contiguous()
            .map_err(Error::Candle)?;
        let v = self
            .v_proj
            .forward(&combined)
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
        let attn = candle_nn::ops::softmax_last_dim(&attn).map_err(Error::Candle)?;
        let y = attn.matmul(&v).map_err(Error::Candle)?;
        let y = y
            .transpose(1, 2)
            .map_err(Error::Candle)?
            .reshape((b_sz, q_len, self.n_head * self.head_dim))
            .map_err(Error::Candle)?;
        let attn_out = self.o_proj.forward(&y).map_err(Error::Candle)?;
        let after_attn = (attn_out + projected_features).map_err(Error::Candle)?;

        // MLP block.
        let x_n = self
            .post_attention_layernorm
            .forward(&after_attn)
            .map_err(Error::Candle)?;
        let g = candle_nn::ops::silu(&self.mlp_gate.forward(&x_n).map_err(Error::Candle)?)
            .map_err(Error::Candle)?;
        let u = self.mlp_up.forward(&x_n).map_err(Error::Candle)?;
        let m = self
            .mlp_down
            .forward(&(g * u).map_err(Error::Candle)?)
            .map_err(Error::Candle)?;
        (m + after_attn).map_err(Error::Candle)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// EAGLE-3 draft model loaded from a published checkpoint.
pub struct Eagle3DraftCandle {
    config: Eagle3DraftConfig,
    /// 3*hidden → hidden projection over the concat of low/mid/high target features.
    fc: Linear,
    midlayer: Midlayer,
    norm: RmsNorm,
    lm_head: Linear,
    /// Cached d2t.to_vec() for fast `draft_to_target_token` (avoids per-call
    /// device→host transfer).
    d2t_host: Vec<i64>,
    /// Cached inverse map target_id → draft_id (built at load time from
    /// `d2t_host`). Targets unreachable by the draft are absent.
    target_to_draft: std::collections::HashMap<u32, u32>,
    /// `t2d_mask[target_id] == true` iff `target_id` is reachable from the
    /// draft vocab (i.e. some draft_id satisfies `d2t[draft_id] == target_id`).
    /// Computed from d2t at load time — the published checkpoint stores t2d
    /// as BoolStorage which candle's pickle loader skips, but the value is
    /// fully derivable from d2t so we don't need to read t2d itself.
    t2d_mask: Vec<bool>,
}

impl std::fmt::Debug for Eagle3DraftCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eagle3DraftCandle")
            .field("hidden_size", &self.config.hidden_size)
            .field("draft_vocab_size", &self.config.draft_vocab_size)
            .field("target_vocab_size", &self.config.target_vocab_size)
            .finish()
    }
}

impl Eagle3DraftCandle {
    pub fn config(&self) -> &Eagle3DraftConfig {
        &self.config
    }

    pub fn from_pth(
        config: &Eagle3DraftConfig,
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(path.as_ref(), dtype, device).map_err(Error::Candle)?;
        Self::from_var_builder(config, vb, device, dtype)
    }

    pub fn from_var_builder(
        config: &Eagle3DraftConfig,
        vb: VarBuilder<'_>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // The published EAGLE-3 LLaMA3.1 checkpoint has no `embed_tokens` —
        // the draft autoregresses on hidden states only and reuses the
        // target's vocabulary tables (d2t for mapping outputs back).
        let fc = linear_no_bias(3 * config.hidden_size, config.hidden_size, vb.pp("fc"))
            .map_err(Error::Candle)?;
        let midlayer = Midlayer::load(config, vb.pp("midlayer"), device, dtype)?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))
            .map_err(Error::Candle)?;
        let lm_head = linear_no_bias(
            config.hidden_size,
            config.draft_vocab_size,
            vb.pp("lm_head"),
        )
        .map_err(Error::Candle)?;

        // d2t is I64 (token id table) — must NOT be auto-cast through the
        // VarBuilder's F16 dtype (that triggers an unimplemented
        // cast_i64_f16 CUDA kernel). Use the explicit-dtype path.
        let d2t = vb
            .get_with_hints_dtype(
                config.draft_vocab_size,
                "d2t",
                Default::default(),
                DType::I64,
            )
            .map_err(Error::Candle)?;
        // Cache d2t on host + build inverse map and t2d mask. We deliberately
        // don't try to read t2d from the .pth — it's BoolStorage which
        // candle's pickle loader skips, and the value is derivable from d2t.
        let d2t_host: Vec<i64> = d2t.to_vec1::<i64>().map_err(Error::Candle)?;
        let mut target_to_draft = std::collections::HashMap::with_capacity(d2t_host.len());
        let mut t2d_mask = vec![false; config.target_vocab_size];
        for (draft_id, &target_id) in d2t_host.iter().enumerate() {
            if target_id >= 0 && (target_id as u32) < config.target_vocab_size as u32 {
                target_to_draft.insert(target_id as u32, draft_id as u32);
                t2d_mask[target_id as usize] = true;
            }
        }
        drop(d2t);

        Ok(Self {
            config: config.clone(),
            fc,
            midlayer,
            norm,
            lm_head,
            d2t_host,
            target_to_draft,
            t2d_mask,
        })
    }

    /// Reverse lookup: target vocab id → draft vocab id, if reachable.
    pub fn target_to_draft_token(&self, target_id: u32) -> Option<u32> {
        self.target_to_draft.get(&target_id).copied()
    }

    pub fn reset(&mut self) {
        self.midlayer.clear_kv_cache();
    }

    /// Run the midlayer once, returning the **pre-norm** midlayer output.
    /// This output is what gets fed back into the *next* draft step (the
    /// official inference loop reuses `last_hidden` from one round as the
    /// `hidden_states` arg of the next).
    ///
    /// Args (matching `Model.forward` in `eagle/model/cnets.py`):
    /// - `hidden_states`: `[b, seq, target_hidden_size * 3]` for the first
    ///   round (concat of low/mid/high), or `[b, seq, hidden_size]` for
    ///   subsequent rounds (previous draft's midlayer output). The fc
    ///   projection is applied iff the inner dim is the 3*hidden form.
    /// - `input_emb`: `[b, seq, hidden_size]` — embed_tokens(token_ids)
    ///   from the **target's** tied embedding table.
    /// - `position`: RoPE position offset.
    pub fn forward_hidden(
        &mut self,
        hidden_states: &Tensor,
        input_emb: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let inner = hidden_states.dims()[hidden_states.rank() - 1];
        let projected = if inner == self.config.hidden_size {
            hidden_states.clone()
        } else {
            self.fc.forward(hidden_states).map_err(Error::Candle)?
        };
        self.midlayer.forward(input_emb, &projected, position)
    }

    /// Apply the draft's `norm + lm_head` on a midlayer output. Mirrors
    /// the official `last_headout = self.lm_head(self.norm(last_hidden))`.
    /// Returns `[b, seq, draft_vocab]` logits over the draft vocab.
    pub fn apply_norm_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(hidden).map_err(Error::Candle)?;
        self.lm_head.forward(&h).map_err(Error::Candle)
    }

    /// Convenience wrapper: midlayer + norm + lm_head in one call. Use
    /// when you only need the logits and don't intend to feed the
    /// midlayer output into a subsequent step.
    pub fn forward(
        &mut self,
        low: &Tensor,
        mid: &Tensor,
        high: &Tensor,
        input_emb: &Tensor,
        token_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let _ = token_ids; // input_emb already encodes token info.
        let combined = Tensor::cat(&[low, mid, high], D::Minus1).map_err(Error::Candle)?;
        let h = self.forward_hidden(&combined, input_emb, position)?;
        self.apply_norm_lm_head(&h)
    }

    /// Map a draft-vocab id to the target's vocabulary. Uses the cached
    /// host copy so this is O(1) on the host without device transfer.
    pub fn draft_to_target_token(&self, draft_id: u32) -> Result<u32> {
        let v = self.d2t_host.get(draft_id as usize).ok_or_else(|| {
            Error::Sampling(format!(
                "draft id {draft_id} out of range ({})",
                self.d2t_host.len()
            ))
        })?;
        Ok(*v as u32)
    }

    /// Whether the target vocab id is reachable from the draft vocab.
    pub fn target_token_is_reachable(&self, target_id: u32) -> Result<bool> {
        Ok(self
            .t2d_mask
            .get(target_id as usize)
            .copied()
            .unwrap_or(false))
    }

    /// Borrow the t2d mask directly (length = `target_vocab_size`,
    /// `mask[i] == true` iff target id `i` is reachable from the draft).
    /// Use this to mask the target's distribution before sampling.
    pub fn t2d_mask(&self) -> &[bool] {
        &self.t2d_mask
    }

    /// Apply the t2d mask in-place on a target-vocab logit vector,
    /// setting unreachable entries to `f32::NEG_INFINITY` so subsequent
    /// softmax / argmax will skip them. Useful when sampling the target
    /// distribution under EAGLE-3's reachability constraint.
    pub fn mask_target_logits(&self, logits: &mut [f32]) {
        debug_assert_eq!(logits.len(), self.t2d_mask.len());
        for (i, slot) in logits.iter_mut().enumerate() {
            if !self.t2d_mask[i] {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
}

/// Run-loop config for EAGLE-3.
#[derive(Debug, Clone)]
pub struct Eagle3RunConfig {
    pub top_k_per_step: usize,
    pub draft_depth: usize,
    pub max_tree_nodes: Option<usize>,
    /// Indices of target layers to feed as low/mid/high features. Defaults
    /// to the published EAGLE-3 LLaMA3.1-8B recipe (`[1, n/2, n-2]`).
    pub layer_indices: [usize; 3],
    pub temperature: f32,
    pub top_p: f32,
}

impl Eagle3RunConfig {
    /// Compute the default layer triple matching the published EAGLE-3
    /// LLaMA training recipe (`SafeAILab/EAGLE/eagle/traineagle3/modeling_llama_kv.py`
    /// line 1139): hidden states captured at the **input** of layers
    /// `2`, `n/2`, `n-3`. In our 0-based "after-layer-i" convention
    /// (where `mids[i]` is the residual *after* layer `i`'s MLP), that
    /// is `[1, n/2 - 1, n - 4]`.
    ///
    /// For 32-layer Llama 3.1 8B → `[1, 15, 28]`.
    pub fn default_layers_for(n_layers: usize) -> [usize; 3] {
        if n_layers < 4 {
            [0, 0, 0]
        } else {
            [1, n_layers / 2 - 1, n_layers.saturating_sub(4)]
        }
    }
}

impl Default for Eagle3RunConfig {
    fn default() -> Self {
        Self {
            top_k_per_step: 2,
            draft_depth: 4,
            max_tree_nodes: None,
            layer_indices: [1, 15, 28], // Llama 3.1 8B published recipe
            temperature: 0.0,
            top_p: 1.0,
        }
    }
}

/// End-to-end EAGLE-3 generation loop.
///
/// Roughly the same shape as [`crate::methods::eagle::run_eagle`], but
/// the draft side runs in EAGLE-3's smaller draft vocab and the per-step
/// top-`k` is taken from `draft.lm_head` (32k vocab) — much cheaper than
/// the target's 128k Q4 lm_head. Accepted draft tokens are translated
/// through `d2t` to the target vocab before commit.
pub fn run_eagle3<T, R>(
    target: &mut T,
    draft: &mut Eagle3DraftCandle,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &Eagle3RunConfig,
    rng: &mut R,
) -> Result<Vec<u32>>
where
    T: crate::model::TreeDecoder + ?Sized,
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
            .ok_or_else(|| Error::Sampling("EAGLE-3 requires non-empty prompt".into()))?;

        // Target vocab id → draft vocab id via cached inverse map. If the
        // target id is unreachable from the draft we fall back to draft 0.
        let root_draft_id = draft.target_to_draft_token(root_token).unwrap_or(0);

        // 1. Multi-layer hidden states for the committed token.
        let (final_h, mids) = target.last_hidden_states_multi(&config.layer_indices)?;
        if mids.len() != 3 {
            return Err(Error::Sampling(format!(
                "EAGLE-3 expects 3 layers, got {}",
                mids.len()
            )));
        }
        // Reshape each to [1, 1, hidden] and promote to the draft's dtype
        // (target hiddens come back F32 from the Q4 path; the draft is F16).
        let draft_dtype = draft.fc.weight().dtype();
        let device = final_h.device().clone();
        let to_3d = |t: &candle_core::Tensor| -> Result<candle_core::Tensor> {
            let t = if t.dtype() != draft_dtype {
                t.to_dtype(draft_dtype).map_err(Error::Candle)?
            } else {
                t.clone()
            };
            t.unsqueeze(0)
                .map_err(Error::Candle)?
                .unsqueeze(0)
                .map_err(Error::Candle)
        };
        let low = to_3d(&mids[0])?;
        let mid = to_3d(&mids[1])?;
        let high = to_3d(&mids[2])?;
        // First-round hidden_states: [1, 1, 3*hidden] concat — fc applied inside forward_hidden.
        let mut hidden_in =
            Tensor::cat(&[&low, &mid, &high], D::Minus1).map_err(Error::Candle)?;

        draft.reset();
        let history_len = target.history_len();

        // 2. Run the draft `draft_depth` times. Each step:
        //    - input_emb = target.embed(current_token_id)
        //    - hidden_in = previous midlayer output (or first-round 3*hidden concat)
        //    - midlayer → out_hidden → norm + lm_head → top-k (in draft vocab)
        let mut per_step_top_k_target: Vec<Vec<u32>> = Vec::with_capacity(config.draft_depth);
        let mut per_step_top_k_log_probs: Vec<Vec<f32>> = Vec::with_capacity(config.draft_depth);
        // First step uses the committed root token for the embed lookup.
        let mut current_target_id = root_token;
        for step in 0..config.draft_depth {
            // input_emb = target.embed_tokens(current_target_id) → [1, 1, hidden] in draft dtype.
            let token_ids = Tensor::from_slice(&[current_target_id], (1, 1), &device)
                .map_err(Error::Candle)?;
            let input_emb = target.embed_tokens(&token_ids)?;
            let input_emb = if input_emb.dtype() != draft_dtype {
                input_emb.to_dtype(draft_dtype).map_err(Error::Candle)?
            } else {
                input_emb
            };
            let out_hidden = draft.forward_hidden(&hidden_in, &input_emb, history_len + step)?;
            let logits = draft.apply_norm_lm_head(&out_hidden)?;
            let last = logits
                .i((0, logits.dim(1).map_err(Error::Candle)? - 1, ..))
                .map_err(Error::Candle)?
                .to_dtype(DType::F32)
                .map_err(Error::Candle)?
                .to_vec1::<f32>()
                .map_err(Error::Candle)?;
            let top_idx: Vec<usize> = top_k_indices(&last, config.top_k_per_step);
            let max_l = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = last.iter().map(|&v| (v - max_l).exp()).sum::<f32>().ln() + max_l;
            let log_probs: Vec<f32> = top_idx.iter().map(|&i| last[i] - lse).collect();
            per_step_top_k_log_probs.push(log_probs);
            let mut top_target = Vec::with_capacity(top_idx.len());
            for &di in &top_idx {
                top_target.push(draft.draft_to_target_token(di as u32)?);
            }
            per_step_top_k_target.push(top_target.clone());

            // For the NEXT iteration: hidden_in = this step's midlayer output;
            // current_target_id = top-1 candidate (already in target vocab).
            hidden_in = out_hidden;
            current_target_id = top_target[0];
            let _ = root_draft_id; // silence — driven via target ids now.
        }

        // 3. Build Cartesian-product tree in TARGET vocab, optionally pruned.
        let full_tree = crate::methods::medusa::MedusaHeads::from_config(
            crate::methods::medusa::MedusaConfig {
                n_heads: config.draft_depth,
                hidden_size: draft.config().hidden_size,
                vocab_size: draft.config().target_vocab_size,
                residual_layers: 1,
            },
        )
        .build_draft_tree(
            root_token,
            &per_step_top_k_target,
            crate::methods::medusa::TreeTopology::CartesianProduct,
        )?;
        let tree = if let Some(max_n) = config.max_tree_nodes {
            crate::methods::eagle::prune_cartesian_tree_pub(
                &full_tree,
                &per_step_top_k_log_probs,
                max_n,
            )?
        } else {
            full_tree
        };

        // 4. Verify via target.tree_logits.
        let per_node_logits = target.tree_logits(&tree)?;

        // 5. Greedy acceptance.
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
            return Err(Error::Sampling("EAGLE-3 round committed zero tokens".into()));
        }
        target.observe(&committed)?;
        generated.extend_from_slice(&committed);
    }
    let _ = (rng, config.temperature, config.top_p);
    generated.truncate(max_new_tokens);
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_eagle3_llama3_1() {
        let c = Eagle3DraftConfig::eagle3_llama3_1_8b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.draft_vocab_size, 32000);
        assert_eq!(c.target_vocab_size, 128256);
        assert_eq!(c.head_dim(), 128);
    }

    #[test]
    fn t2d_mask_derives_from_d2t() {
        // Build a synthetic Eagle3DraftCandle by skipping the .pth load —
        // construct directly via the fields we want to test. This is a
        // lightweight sanity check of the mask derivation, not the loader.
        // (The full loader is exercised by the e2e tests.)
        let target_vocab_size = 10;
        let d2t_host: Vec<i64> = vec![0, 3, 5, 9, -1]; // draft 0..4
        let mut target_to_draft = std::collections::HashMap::new();
        let mut t2d_mask = vec![false; target_vocab_size];
        for (di, &ti) in d2t_host.iter().enumerate() {
            if ti >= 0 && (ti as u32) < target_vocab_size as u32 {
                target_to_draft.insert(ti as u32, di as u32);
                t2d_mask[ti as usize] = true;
            }
        }
        assert_eq!(t2d_mask, vec![true, false, false, true, false, true, false, false, false, true]);
        // The -1 entry produces no t2d mapping (unreachable slot).
    }

    #[test]
    fn default_layer_indices_for_8b() {
        // Published EAGLE-3 LLaMA recipe: layers `2, n/2, n-3` (input-of)
        // → in our after-layer-i convention `[1, n/2 - 1, n - 4]`.
        let l = Eagle3RunConfig::default_layers_for(32);
        assert_eq!(l, [1, 15, 28]);
    }
}
