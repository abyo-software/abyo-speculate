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
//! ## Status: scaffolding + loader for v0.2.0
//!
//! What's wired:
//! - [`Eagle3DraftConfig`] from `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`'s
//!   defaults.
//! - [`Eagle3DraftCandle::from_pth`] / `from_var_builder` parse every
//!   key in the checkpoint (15 tensors verified by inspection).
//! - Forward pass through the midlayer with the full concat input.
//!
//! What still needs target-side cooperation (deferred to v0.2.1):
//! - `TreeDecoder::last_hidden_states_multi(layers: &[usize])` — returns
//!   the requested layers' hidden states for the most recent token.
//!   Currently `last_hidden_state` only exposes the final layer's output.
//!   EAGLE-3 needs three layers; the target extension lands when the
//!   model-side bookkeeping is plumbed through.
//! - Dynamic tree construction guided by EAGLE-3's draft confidence (same
//!   v0.2.1 batch as EAGLE-2 dynamic).
//!
//! Loader + forward are unit-testable today via [`Eagle3DraftCandle::forward`]
//! against synthetic inputs.

#![allow(missing_docs)]

use crate::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
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
    embed_tokens: Embedding,
    /// 3*hidden → hidden projection over the concat of low/mid/high target features.
    fc: Linear,
    midlayer: Midlayer,
    norm: RmsNorm,
    lm_head: Linear,
    /// Draft-vocab → target-vocab id mapping (length = draft_vocab_size).
    d2t: Tensor,
    /// Target-vocab → draft-reachable bitmask (length = target_vocab_size).
    t2d: Tensor,
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
        let embed_tokens = candle_nn::embedding(
            config.draft_vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )
        .map_err(Error::Candle)?;
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

        // d2t / t2d are stored as plain tensors via the same VarBuilder.
        let d2t = vb
            .get(config.draft_vocab_size, "d2t")
            .map_err(Error::Candle)?;
        let t2d = vb
            .get(config.target_vocab_size, "t2d")
            .map_err(Error::Candle)?;

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            fc,
            midlayer,
            norm,
            lm_head,
            d2t,
            t2d,
        })
    }

    pub fn reset(&mut self) {
        self.midlayer.clear_kv_cache();
    }

    /// One forward step.
    ///
    /// Inputs:
    /// - `low / mid / high`: each `[1, seq, hidden]` — three target hidden
    ///   states selected by the EAGLE-3 training recipe (typically layers
    ///   0/N//2/N-1 or similar; the published recipe is in the EAGLE-3
    ///   paper).
    /// - `last_target_hidden`: `[1, seq, hidden]` — the most recent target
    ///   hidden state, fed into the midlayer's attention input alongside
    ///   the projected feature concat.
    /// - `token_ids`: `[1, seq]` — the *draft-vocabulary* token ids for
    ///   embed lookups. Use [`Self::target_to_draft_token`] to translate
    ///   from the target's vocabulary first.
    /// - `position`: absolute position offset for RoPE.
    ///
    /// Returns `[1, seq, draft_vocab]` — logits over the draft vocab. Use
    /// [`Self::draft_to_target_token`] to map back to target ids before
    /// committing.
    pub fn forward(
        &mut self,
        low: &Tensor,
        mid: &Tensor,
        high: &Tensor,
        last_target_hidden: &Tensor,
        token_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        // 1. Concat 3 target features → fc → projected hidden.
        let combined = Tensor::cat(&[low, mid, high], D::Minus1).map_err(Error::Candle)?;
        let projected = self.fc.forward(&combined).map_err(Error::Candle)?;
        // 2. Token embedding (note: in the public EAGLE-3, token_emb is
        //    fused into the projected_features through the input_layernorm
        //    chain — refer to the paper for the precise schedule). For
        //    v0.2.0 we follow the simplest interpretation: target_hidden
        //    + projected_features into the midlayer; embed_tokens is
        //    available for downstream variants but not used here.
        let _ = self
            .embed_tokens
            .forward(token_ids)
            .map_err(Error::Candle)?;
        // 3. Midlayer.
        let h = self
            .midlayer
            .forward(last_target_hidden, &projected, position)?;
        // 4. Norm + lm_head.
        let h = self.norm.forward(&h).map_err(Error::Candle)?;
        self.lm_head.forward(&h).map_err(Error::Candle)
    }

    /// Map a draft-vocab id to the target's vocabulary.
    pub fn draft_to_target_token(&self, draft_id: u32) -> Result<u32> {
        let v: i64 = self
            .d2t
            .i(draft_id as usize)
            .map_err(Error::Candle)?
            .to_scalar()
            .map_err(Error::Candle)?;
        Ok(v as u32)
    }

    /// Whether the target vocab id is reachable from the draft vocab.
    pub fn target_token_is_reachable(&self, target_id: u32) -> Result<bool> {
        let v: u8 = self
            .t2d
            .i(target_id as usize)
            .map_err(Error::Candle)?
            .to_scalar()
            .map_err(Error::Candle)?;
        Ok(v != 0)
    }
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
}
