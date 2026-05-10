//! Vendored Llama quantized (GGUF) model with tree-attention extensions.
//!
//! Pairs with EAGLE: `LlamaQuantDecoder` lets us load Llama 3 8B Q4
//! (~4.5 GB) plus the EAGLE draft (~1.5 GB) on a 16 GB GPU — the
//! configuration that BF16 alone OOMs on. Architecture matches
//! `candle_transformers::models::quantized_llama` for non-MoE Llamas
//! (Llama 1/2/3.x, Vicuna, code-llama, mistral GGUF), with our standard
//! tree-attention additions.
//!
//! ## Llama-specific notes
//!
//! - **Interleaved RoPE** (`rope_i`), not the rotate-half variant Qwen2 uses.
//! - **No attention biases** (Llama doesn't have them; Qwen does).
//! - **No MoE in this vendor** — Mixtral / DeepSeek-v2 are out of scope
//!   for v0.2.0. Add MoE later only if there's user demand.
//!
//! ## What we drop vs upstream
//!
//! - Internal `output` projection + last-position slice in `forward`. We
//!   return all-position hidden states; callers apply our (still-quantized)
//!   `output` head via [`ModelWeights::apply_lm_head`].
//!
//! ## What we add
//!
//! - [`ModelWeights::forward_hidden`] — autoregressive hidden states.
//! - [`ModelWeights::forward_with_positions`] — tree decoding.
//! - [`ModelWeights::truncate_kv_cache_to`] — fast rollback.

#![allow(missing_docs)]

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.w1.forward(xs)?;
        let u = self.w3.forward(xs)?;
        self.w2.forward(&(candle_nn::ops::silu(&g)? * u)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    attn_norm: RmsNorm,
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
            .wq
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .wk
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .wv
            .forward(xs)?
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    fn rope_at(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn rope_at_positions(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
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

        let n_rep = self.n_head / self.n_kv_head;
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;

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
        self.wo.forward(&y)
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

        // Llama-family GGUFs use the "llama." metadata prefix regardless of
        // the model variant (Llama 2, Llama 3, Vicuna, etc).
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("llama.context_length")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;

        let tok_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        // Tied weights fallback, matches upstream behaviour for some Llama 3 GGUFs.
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => QMatMul::from_qtensor(tok_q)?,
        };

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let prefix = format!("blk.{i}");
            let wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp = Mlp {
                w1: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?)?,
                w2: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?)?,
                w3: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?)?,
            };
            let attn_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                wq: QMatMul::from_qtensor(wq)?,
                wk: QMatMul::from_qtensor(wk)?,
                wv: QMatMul::from_qtensor(wv)?,
                wo: QMatMul::from_qtensor(wo)?,
                attn_norm: RmsNorm::from_qtensor(attn_norm, rms_norm_eps)?,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
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

    /// Apply the (quantized) LM head projection. Same role as
    /// [`crate::model::quantized_qwen2_local::ModelWeights::apply_lm_head`].
    pub fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    /// Number of transformer layers in this model.
    pub fn num_hidden_layers(&self) -> usize {
        self.layers.len()
    }

    /// Embed a single token id (or batch). Used by EAGLE-3's draft loop
    /// where the draft has no embed_tokens of its own and must reuse the
    /// target's tied embedding.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input_ids)
    }

    fn causal_bias(&mut self, prev_len: usize, seq_len: usize) -> Result<Tensor> {
        let total = prev_len + seq_len;
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
        let m = Tensor::from_slice(&data, (seq_len, total), &self.device)?
            .reshape((1, 1, seq_len, total))?;
        self.masks.insert(key, m.clone());
        Ok(m)
    }

    /// Plain autoregressive forward; returns hidden states for all positions
    /// (no last-position slice, no LM head).
    pub fn forward_hidden(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (final_h, _) = self.forward_hidden_with_layers(x, index_pos, &[])?;
        Ok(final_h)
    }

    /// Same as [`Self::forward_hidden`] but additionally collects the hidden
    /// state *after* each layer index in `collect_layers` (each layer index
    /// referring to the residual output of `layers[i]`, before the final
    /// `norm`). Used by EAGLE-3 to get low/mid/high target features.
    pub fn forward_hidden_with_layers(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        collect_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (_b_sz, seq_len) = x.dims2()?;
        let bias = if seq_len <= 1 {
            None
        } else {
            let prev_len = self.layers[0]
                .kv_cache
                .as_ref()
                .map(|(k, _)| k.dim(2).unwrap_or(0))
                .unwrap_or(0);
            Some(self.causal_bias(prev_len, seq_len)?)
        };
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let mut collected: Vec<Option<Tensor>> = vec![None; collect_layers.len()];
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let x = layer_in;
            let residual = &x;
            let xn = layer.attn_norm.forward(&x)?;
            let attn = layer.forward_attn(&xn, bias.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            let residual = &x;
            let xn = layer.ffn_norm.forward(&x)?;
            let m = layer.mlp.forward(&xn)?;
            layer_in = (m + residual)?;
            for (slot, &want) in collect_layers.iter().enumerate() {
                if want == li {
                    collected[slot] = Some(layer_in.clone());
                }
            }
        }
        let final_h = self.norm.forward(&layer_in)?;
        let mut out = Vec::with_capacity(collected.len());
        for (slot, want) in collect_layers.iter().enumerate() {
            out.push(collected[slot].clone().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "collect_layers[{slot}] = {want} out of range (n_layers={})",
                    self.layers.len()
                ))
            })?);
        }
        Ok((final_h, out))
    }

    /// Tree-decoding forward — same contract as
    /// [`crate::model::quantized_qwen2_local::ModelWeights::forward_with_positions`].
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
            let xn = layer.attn_norm.forward(&x)?;
            let attn = layer.forward_attn_at_positions(&xn, attn_bias_4d, position_ids)?;
            let x = (attn + residual)?;
            let residual = &x;
            let xn = layer.ffn_norm.forward(&x)?;
            let m = layer.mlp.forward(&xn)?;
            layer_in = (m + residual)?;
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
}
