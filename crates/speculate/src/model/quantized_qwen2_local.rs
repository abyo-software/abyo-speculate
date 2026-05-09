//! Vendored Qwen2 quantized (GGUF) model with tree-attention extensions.
//!
//! Mirrors [`crate::model::qwen2_local`] but operating on
//! [`candle_core::quantized::QMatMul`] linear layers and loaded from a GGUF
//! file via `candle_core::quantized::gguf_file`.
//!
//! What's the same as upstream `candle_transformers::models::quantized_qwen2`:
//! - Per-layer KV cache (Option<(Tensor, Tensor)>).
//! - Per-layer cos/sin tables built once via `precomput_freqs_cis`.
//! - GGUF metadata layout (`qwen2.attention.head_count` etc).
//!
//! What we drop:
//! - The internal LM-head application + last-position slice in `forward`.
//!   We return all-position hidden states; callers apply the quantized
//!   `output` projection themselves via [`ModelWeights::apply_lm_head`].
//!
//! What we add:
//! - [`ModelWeights::forward_hidden`] — autoregressive hidden states (no slice).
//! - [`ModelWeights::forward_with_positions`] — tree decoding with per-token
//!   positions + 4D additive attention bias.
//! - [`ModelWeights::truncate_kv_cache_to`] — partial KV cache truncation.

#![allow(missing_docs)]

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn project_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let q = self
            .attention_wq
            .forward(xs)?
            .broadcast_add(&self.attention_bq)?;
        let k = self
            .attention_wk
            .forward(xs)?
            .broadcast_add(&self.attention_bk)?;
        let v = self
            .attention_wv
            .forward(xs)?
            .broadcast_add(&self.attention_bv)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    fn rope_at(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn rope_at_positions(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn run_attn(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, _, q_len, _) = q.dims4()?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = match attn_bias {
            None => attn,
            Some(b) => attn.broadcast_add(b)?,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let y = attn.matmul(&v)?;
        let n_embd = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape((b_sz, q_len, n_embd))?;
        self.attention_wo.forward(&y)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        bias: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(x)?;
        let q = self.rope_at(&q, index_pos)?;
        let k = self.rope_at(&k, index_pos)?;
        self.run_attn(q, k, v, bias)
    }

    fn forward_attn_at_positions(
        &mut self,
        x: &Tensor,
        bias: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(x)?;
        let q = self.rope_at_positions(&q, positions)?;
        let k = self.rope_at_positions(&k, positions)?;
        self.run_attn(q, k, v, Some(bias))
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

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// Materialized causal masks keyed by sequence length, lazily built.
    masks: HashMap<usize, Tensor>,
    device: Device,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in GGUF metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("qwen2.context_length")?.to_u32()? as usize;
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            Err(_) => QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?,
        };

        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let prefix = format!("blk.{i}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp = Mlp {
                feed_forward_w1: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?)?,
                feed_forward_w2: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?)?,
                feed_forward_w3: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?)?,
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Apply the (quantized) LM head projection to a hidden state. Caller
    /// supplies the result of [`Self::forward_hidden`] or
    /// [`Self::forward_with_positions`].
    pub fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    /// Causal additive bias of shape `[seq_len, prev_len + seq_len]` with
    /// `0.0` for attendable, `-inf` for masked. Cached per `(prev_len, seq_len)`
    /// pair; we key only by seq_len (prev_len enters via the leading zero
    /// columns and changes per call).
    fn causal_bias(&mut self, prev_len: usize, seq_len: usize) -> Result<Tensor> {
        let total = prev_len + seq_len;
        // Cache key: encode both lengths.
        let key = (prev_len << 20) | seq_len;
        if let Some(m) = self.masks.get(&key) {
            return Ok(m.clone());
        }
        let mut data = vec![0f32; seq_len * total];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    data[i * total + prev_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let m = Tensor::from_slice(&data, (seq_len, total), &self.device)?;
        let m = m.reshape((1, 1, seq_len, total))?;
        self.masks.insert(key, m.clone());
        Ok(m)
    }

    /// Plain autoregressive forward. Returns hidden states for **all**
    /// positions: shape `[batch, seq_len, hidden]`.
    pub fn forward_hidden(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let bias = if seq_len <= 1 {
            None
        } else {
            // For causal autoregressive, prev_len is whatever's already in
            // the first layer's kv cache.
            let prev_len = self.layers[0]
                .kv_cache
                .as_ref()
                .map(|(k, _)| k.dim(2).unwrap_or(0))
                .unwrap_or(0);
            Some(self.causal_bias(prev_len, seq_len)?)
        };
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let xn = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&xn, bias.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            let residual = &x;
            let xn = layer.ffn_norm.forward(&x)?;
            let mlp = layer.mlp.forward(&xn)?;
            layer_in = (mlp + residual)?;
        }
        self.norm.forward(&layer_in)
    }

    /// Tree-decoding forward. See [`crate::model::qwen2_local::Model::forward_with_positions`]
    /// for the contract.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attn_bias_4d: &Tensor,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let xn = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn_at_positions(&xn, attn_bias_4d, position_ids)?;
            let x = (attn + residual)?;
            let residual = &x;
            let xn = layer.ffn_norm.forward(&x)?;
            let mlp = layer.mlp.forward(&xn)?;
            layer_in = (mlp + residual)?;
        }
        self.norm.forward(&layer_in)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
        self.masks.clear();
    }

    pub fn truncate_kv_cache_to(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache_to(len)?;
        }
        Ok(())
    }

    pub fn kv_cache_len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.kv_cache.as_ref())
            .and_then(|(k, _)| k.dim(2).ok())
            .unwrap_or(0)
    }

    /// Helper: take a single row of logits (last position) for a sampled
    /// next-token. Convenience for callers who only care about the next
    /// token from a forward pass.
    pub fn logits_last(&self, hidden: &Tensor) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., seq_len - 1, ..))?;
        self.output.forward(&last_hidden)
    }
}
