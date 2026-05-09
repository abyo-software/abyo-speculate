//! Vendored Phi-3 / Phi-3.5 model with tree-attention extensions.
//!
//! Same design as the other `*_local` modules, adapted for Phi-3's
//! architectural quirks:
//! - **Fused QKV projection.** A single `qkv_proj` Linear of shape
//!   `(hidden, num_heads*head_dim + 2*num_kv_heads*head_dim)` whose output
//!   we slice into Q / K / V.
//! - **Fused gate+up MLP.** `gate_up_proj` produces `[..., 2*intermediate]`
//!   which we split into `gate` and `up` halves before SiLU+multiply.
//! - **LM head inside Model upstream** — we drop it, drive `Model` to
//!   hidden states, apply our own LM head externally.
//!
//! Re-exports upstream's stable `Config` so HF config.json deserializes
//! unchanged. The vendored types diverge only on the new tree-attention
//! entrypoints + KV truncation.

#![allow(missing_docs)]

pub use candle_transformers::models::phi3::Config;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
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

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let qe = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let ke = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((qe, ke))
    }

    fn apply_rotary_emb_qkv_at_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        let qe = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let ke = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((qe, ke))
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv_head, seq, head_dim) = xs.dims4()?;
    Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b, n_kv_head * n_rep, seq, head_dim))
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        // Phi-3-mini-4k-instruct stores qkv_proj / o_proj without biases;
        // upstream candle uses `linear` (with bias) which doesn't match.
        let qkv_proj = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn project_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let qkv = self.qkv_proj.forward(xs)?;
        let q_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, q_pos)?;
        let k = qkv.narrow(D::Minus1, q_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            q_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;
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

    fn run(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, _, q_len, _) = q.dims4()?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = match attention_mask {
            None => attn,
            Some(m) => attn.broadcast_add(m)?,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn = attn.matmul(&v)?;
        attn.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
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
        self.run(q, k, v, attention_mask)
    }

    fn forward_with_positions(
        &mut self,
        xs: &Tensor,
        positions: &Tensor,
        attn_bias_4d: &Tensor,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs)?;
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv_at_positions(&q, &k, positions)?;
        self.run(q, k, v, Some(attn_bias_4d))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        if let Some((k, v)) = &self.kv_cache {
            let cur = k.dim(2)?;
            if len > cur {
                candle_core::bail!("truncate_kv_cache_to({len}) exceeds {cur}");
            }
            if len == 0 {
                self.kv_cache = None;
            } else if len < cur {
                self.kv_cache = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate_up_proj: linear_no_bias(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                vb.pp("gate_up_proj"),
            )?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
            i_size: cfg.intermediate_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = xs.apply(&self.gate_up_proj)?;
        let gate = up.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up.narrow(D::Minus1, self.i_size, self.i_size)?;
        (up_states * gate.apply(&self.act_fn))?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let r = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward(&x, attention_mask, seqlen_offset)? + r)?;
        let r = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + r)?;
        Ok(x)
    }

    fn forward_with_positions(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        attn_bias_4d: &Tensor,
    ) -> Result<Tensor> {
        let r = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self
            .self_attn
            .forward_with_positions(&x, positions, attn_bias_4d)?
            + r)?;
        let r = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + r)?;
        Ok(x)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }

    fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        self.self_attn.truncate_kv_cache_to(len)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn prepare_causal_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        // Promote to self.dtype before the optional concatenation so we don't
        // mix F32 / BF16 like upstream qwen2 did.
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

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn forward_with_positions(
        &mut self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attn_bias_4d: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward_with_positions(&xs, position_ids, attn_bias_4d)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    pub fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache_to(len)?;
        }
        Ok(())
    }
}
