//! Vendored Qwen2 / Qwen2.5 model with tree-attention extensions.
//!
//! ## Why we vendor
//!
//! `candle_transformers::models::qwen2` is a fine autoregressive decoder, but
//! its public API does not expose two things abyo-speculate needs:
//!
//! 1. **Per-position positional ids.** `RotaryEmbedding::apply_rotary_emb_qkv`
//!    assumes consecutive positions starting at `seqlen_offset`. Tree
//!    decoding (Medusa / EAGLE) places same-depth siblings at the *same*
//!    absolute position; we need an `index_select`-based variant.
//!
//! 2. **Pre-built 4D attention bias injection.** `Model::forward` accepts an
//!    `attn_mask` argument, but its `prepare_attention_mask` only handles 2D
//!    padding masks of shape `[b, seq]`. Our `DraftTree::full_attention_bias_4d`
//!    produces `[b, 1, n, prefix_len + n]` with `-inf` for non-ancestor
//!    positions — that has to be added directly to the attention logits, no
//!    transformation.
//!
//! ## What's the same as upstream
//!
//! Everything else: `Config`, `MLP`, the autoregressive `Model::forward` path,
//! the GQA `repeat_kv` plumbing, weight loading. Layout matches
//! `candle_transformers::models::qwen2` so a future upstream change can be
//! diffed in without rewriting.
//!
//! Drop only: `with_tracing` wrappers (the regular candle_nn types are fine
//! for our use case) and `ModelForCausalLM` (we drive `Model` directly with
//! our own LM head — see `model::qwen2::Qwen2Decoder`).
//!
//! ## What's new
//!
//! - [`RotaryEmbedding::apply_rotary_emb_qkv_at_positions`] — `&[u32]` of
//!   absolute positions, `index_select`'d into the precomputed cos/sin tables.
//! - [`Attention::forward_with_positions`] — uses the per-position RoPE +
//!   accepts a fully-formed 4D attention bias.
//! - [`DecoderLayer::forward_with_positions`].
//! - [`Model::forward_with_positions`] — the tree-decoding entry point.
//! - [`Model::truncate_kv_cache_to`] — partial KV cache truncation, replaces
//!   the slow clear+replay rollback in `Qwen2Decoder` (Phase 1c).

#![allow(missing_docs)] // vendored from upstream; only the new tree-attention items get docs

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, rms_norm, Activation, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Standard RoPE for autoregressive decoding: contiguous positions starting
    /// at `seqlen_offset`. Identical to upstream.
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Tree-decoding RoPE: explicit per-token positions via `index_select`.
    /// Same-depth siblings legitimately share a position.
    fn apply_rotary_emb_qkv_at_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // positions shape: [seq_len]; cos/sin: [max_seq_len, dim/2]
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// GQA helper: repeat each kv head `n_rep` times along the head dim. Mirrors
/// `candle_transformers::utils::repeat_kv` but local so we don't depend on the
/// upstream `utils` module being public-and-stable.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn project_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn run_attention(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, _, q_len, _) = q.dims4()?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs)?;
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?;
        self.run_attention(q, k, v, attention_mask)
    }

    fn forward_with_positions(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs)?;
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv_at_positions(&q, &k, positions)?;
        self.run_attention(q, k, v, attention_mask)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }

    fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        if let Some((k, v)) = &self.kv_cache {
            let cur_len = k.dim(2)?;
            if len > cur_len {
                candle_core::bail!(
                    "truncate_kv_cache_to({len}) but cache only has {cur_len} positions"
                );
            }
            if len == 0 {
                self.kv_cache = None;
            } else if len < cur_len {
                self.kv_cache = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
            }
        }
        Ok(())
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_cache
            .as_ref()
            .and_then(|(k, _)| k.dim(2).ok())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn forward_with_positions(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_positions(&xs, attention_mask, positions)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }

    fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        self.self_attn.truncate_kv_cache_to(len)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Read-only handle for tied LM-head construction.
    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        // The slice was built as f32; promote to self.dtype *before* the
        // optional cat so we don't try to mix BF16/F16 mask0 with an F32
        // causal block (upstream qwen2 hits this when run with non-F32 dtype).
        let mask =
            Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?.to_dtype(self.dtype)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))
    }

    /// Plain autoregressive forward. Same surface as upstream
    /// `qwen2::Model::forward(input_ids, seqlen_offset, None)`, returning the
    /// hidden states (not logits) for **all** positions.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.apply(&self.norm)
    }

    /// Tree-aware forward.
    ///
    /// - `input_ids`: shape `[1, n_tree]`. Token at each tree node, BFS order.
    /// - `position_ids`: shape `[n_tree]`, dtype `u32`. Absolute positions
    ///   (committed-prefix length + per-node depth). Same-depth siblings
    ///   legitimately share a position.
    /// - `attention_bias_4d`: shape `[1, 1, n_tree, prefix_len + n_tree]`,
    ///   additive (`0.0` on attendable, `-inf` on masked). Built by
    ///   [`crate::tree::DraftTree::full_attention_bias_4d`].
    ///
    /// Updates the per-layer KV cache as if all `n_tree` tokens were
    /// committed. After the caller picks the winning path, call
    /// [`Self::truncate_kv_cache_to`] with the committed prefix length to
    /// drop the unused tree entries.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attention_bias_4d: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward_with_positions(&xs, Some(attention_bias_4d), position_ids)?
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Truncate every layer's KV cache to keep only the first `len` positions.
    ///
    /// `Phase-1c` improvement over `clear_kv_cache + replay`: O(1) work per
    /// layer rather than O(history) re-forward.
    pub fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache_to(len)?;
        }
        Ok(())
    }

    /// Length of the (assumed-uniform) per-layer KV cache.
    pub fn kv_cache_len(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.kv_cache_len())
            .unwrap_or(0)
    }
}
