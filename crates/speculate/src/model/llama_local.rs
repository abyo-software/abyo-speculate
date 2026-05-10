//! Vendored Llama (1 / 2 / 3.x) model with tree-attention extensions.
//!
//! Same design as [`crate::model::qwen2_local`], adapted for Llama's
//! architecture differences:
//! - Centralised KV cache (one `Cache` struct held outside `Llama`, indexed
//!   per-block) rather than per-attention `Option<(Tensor, Tensor)>`.
//! - Llama3 rope scaling (factor + smoothing on a frequency window).
//! - GQA with `num_attention_heads / num_key_value_heads` repeats.
//!
//! What we drop from upstream:
//! - `flash_attn` path (not needed for SD verification correctness).
//! - The last-position slice + LM-head application in `Llama::forward` (we
//!   drive the model through to all-position hidden states and apply our
//!   own LM head externally).
//!
//! What we add:
//! - [`Cache::truncate_to`] — partial KV cache truncation per layer.
//! - `CausalSelfAttention::forward_with_positions` — per-token absolute
//!   positions for the tree-decoding RoPE.
//! - [`Llama::forward_with_positions`] — tree-decoding forward with a
//!   pre-built 4D attention bias.
//!
//! Re-exports `LlamaConfig` / `Llama3RopeConfig` / `Llama3RopeType` /
//! `LlamaEosToks` from upstream — those types are stable and don't need
//! vendoring.

#![allow(missing_docs)]

pub use candle_transformers::models::llama::{
    Config, Llama3RopeConfig, Llama3RopeType, LlamaConfig, LlamaEosToks,
};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use std::f32::consts::PI;

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

fn rope_inv_freq(cfg: &Config) -> Vec<f32> {
    match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => calculate_default_inv_freq(cfg),
        Some(s) => {
            let low_wavelen = s.original_max_position_embeddings as f32 / s.low_freq_factor;
            let high_wavelen = s.original_max_position_embeddings as f32 / s.high_freq_factor;
            calculate_default_inv_freq(cfg)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_wavelen {
                        freq
                    } else if wavelen > low_wavelen {
                        freq / s.factor
                    } else {
                        let smooth = (s.original_max_position_embeddings as f32 / wavelen
                            - s.low_freq_factor)
                            / (s.high_freq_factor - s.low_freq_factor);
                        (1. - smooth) * freq / s.factor + smooth * freq
                    }
                })
                .collect()
        }
    }
}

/// Llama-style KV cache, centralised across all blocks.
#[derive(Debug, Clone)]
pub struct Cache {
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let theta = rope_inv_freq(config);
        let theta = Tensor::new(theta, device)?;
        let max_seq = config.max_position_embeddings;
        let idx_theta = Tensor::arange(0, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
            device: device.clone(),
        })
    }

    /// Truncate every per-layer KV cache to `len` positions.
    pub fn truncate_to(&mut self, len: usize) -> Result<()> {
        for slot in self.kvs.iter_mut() {
            if let Some((k, v)) = slot.as_ref() {
                let cur = k.dim(2)?;
                if len > cur {
                    candle_core::bail!("truncate_to({len}) exceeds cache length {cur}");
                }
                if len == 0 {
                    *slot = None;
                } else if len < cur {
                    *slot = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
                }
            }
        }
        Ok(())
    }

    /// All layers share a length; return that.
    pub fn kv_len(&self) -> usize {
        self.kvs
            .iter()
            .find_map(|slot| slot.as_ref().and_then(|(k, _)| k.dim(2).ok()))
            .unwrap_or(0)
    }

    /// Reorder every per-layer KV cache so the new sequence is exactly the
    /// rows at `indices` (in the given order). Used by EAGLE/EAGLE-3 to
    /// commit the accepted tree path's KVs without re-running the target
    /// over them.
    ///
    /// `indices` are absolute cache positions (typically
    /// `[0, 1, ..., prefix_len - 1, prefix_len + tree_offset_for_root, ...]`).
    /// The caller is responsible for ensuring the listed positions have
    /// RoPE encodings matching their *new* index in the cache (which
    /// holds for accepted paths because tree positions encode `prefix_len + depth`).
    pub fn keep_kv_indices(&mut self, indices: &[u32]) -> Result<()> {
        if indices.is_empty() {
            self.clear();
            return Ok(());
        }
        let idx_tensor = Tensor::from_slice(indices, (indices.len(),), &self.device)?;
        for slot in self.kvs.iter_mut() {
            if let Some((k, v)) = slot.as_ref() {
                let k_c = k.contiguous()?;
                let v_c = v.contiguous()?;
                let new_k = k_c.index_select(&idx_tensor, 2)?.contiguous()?;
                let new_v = v_c.index_select(&idx_tensor, 2)?.contiguous()?;
                *slot = Some((new_k, new_v));
            }
        }
        Ok(())
    }

    /// Drop everything from every per-layer KV cache.
    pub fn clear(&mut self) {
        for slot in self.kvs.iter_mut() {
            *slot = None;
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
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
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder<'_>, cfg: &Config) -> Result<Self> {
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear_no_bias(cfg.hidden_size, size_q, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    fn project_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn rope_contiguous(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
    }

    fn run(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, _, q_len, _) = q.dims4()?;
        let (k, v) = match &cache.kvs[block_idx] {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[prev_v, &v], 2)?.contiguous()?;
                (k, v)
            }
        };
        cache.kvs[block_idx] = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = match attention_mask {
            None => attn,
            Some(mask) => attn.broadcast_add(mask)?,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn = attn.matmul(&v)?;
        attn.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn forward(
        &self,
        xs: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs)?;
        let seq_len = q.dim(2)?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        let q = self.rope_contiguous(&q, &cos, &sin)?;
        let k = self.rope_contiguous(&k, &cos, &sin)?;
        // Build causal mask for the *new* seq_len positions over the
        // [prev_cache_len + seq_len] keys: prefix is fully attendable, the
        // new positions follow the lower-triangular rule among themselves.
        let prev_len = cache.kvs[block_idx]
            .as_ref()
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0);
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask_with_prefix(
                prev_len,
                seq_len,
                q.dtype(),
                &cache.device,
            )?)
        };
        self.run(q, k, v, mask.as_ref(), block_idx, cache)
    }

    fn forward_with_positions(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        attn_bias_4d: &Tensor,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs)?;
        let cos = cache.cos.index_select(positions, 0)?;
        let sin = cache.sin.index_select(positions, 0)?;
        let q = self.rope_contiguous(&q, &cos, &sin)?;
        let k = self.rope_contiguous(&k, &cos, &sin)?;
        self.run(q, k, v, Some(attn_bias_4d), block_idx, cache)
    }
}

fn causal_mask_with_prefix(
    prev_len: usize,
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Shape: [seq_len, prev_len + seq_len]. Prefix columns are 0; the
    // [seq_len, seq_len] tail is the standard causal mask (NEG_INFINITY
    // strictly above the diagonal).
    let total = prev_len + seq_len;
    let mut data = vec![0f32; seq_len * total];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                data[i * total + prev_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_slice(&data, (seq_len, total), device)?;
    if dtype != DType::F32 {
        t.to_dtype(dtype)?.reshape((1, 1, seq_len, total))
    } else {
        t.reshape((1, 1, seq_len, total))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder<'_>, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            c_fc1: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            c_fc2: linear_no_bias(h, i, vb.pp("up_proj"))?,
            c_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder<'_>, cfg: &Config) -> Result<Self> {
        Ok(Self {
            rms_1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: CausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            rms_2: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let r = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + r)?;
        let r = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + r)?;
        Ok(x)
    }

    fn forward_with_positions(
        &self,
        x: &Tensor,
        positions: &Tensor,
        attn_bias_4d: &Tensor,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let r = x;
        let x = self.rms_1.forward(x)?;
        let x =
            (self
                .attn
                .forward_with_positions(&x, positions, attn_bias_4d, block_idx, cache)?
                + r)?;
        let r = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + r)?;
        Ok(x)
    }
}

/// Llama-family decoder, hidden-state-only (LM head applied externally).
#[derive(Debug, Clone)]
pub struct Llama {
    embed_tokens: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Llama {
    pub fn load(vb: VarBuilder<'_>, cfg: &Config) -> Result<Self> {
        let _ = DEFAULT_MAX_SEQ_LEN; // referenced for parity with upstream; unused
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(Block::load(vb.pp(format!("model.layers.{i}")), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            blocks,
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

    /// Plain autoregressive forward: returns hidden states for **all**
    /// positions (no last-position slice, no LM head). Use the caller's
    /// own LM head to obtain logits.
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (final_h, _) = self.forward_with_layer_hooks(x, index_pos, cache, &[])?;
        Ok(final_h)
    }

    /// As [`Self::forward`], but additionally returns the residual stream
    /// after each requested layer index. Used by EAGLE-3 to fetch
    /// low/mid/high target features in one forward pass.
    pub fn forward_with_layer_hooks(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        collect_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut x = self.embed_tokens.forward(x)?;
        let mut collected: Vec<Option<Tensor>> = vec![None; collect_layers.len()];
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, i, cache)?;
            for (slot, &want) in collect_layers.iter().enumerate() {
                if want == i {
                    collected[slot] = Some(x.clone());
                }
            }
        }
        let final_h = self.norm.forward(&x)?;
        let mut out = Vec::with_capacity(collected.len());
        for (slot, want) in collect_layers.iter().enumerate() {
            out.push(collected[slot].clone().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "collect_layers[{slot}] = {want} out of range (n_layers={})",
                    self.blocks.len()
                ))
            })?);
        }
        Ok((final_h, out))
    }

    /// Number of transformer blocks.
    pub fn num_hidden_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Embed token ids via the model's tied embedding (used by EAGLE-3,
    /// which doesn't ship its own embed_tokens).
    pub fn embed_tokens(&self, x: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(x)
    }

    /// Tree-decoding forward — see [`crate::model::qwen2_local::Model::forward_with_positions`]
    /// for the exact contract. The KV cache is updated as if all input
    /// positions were committed; the caller drops what they don't keep
    /// via [`Cache::truncate_to`].
    pub fn forward_with_positions(
        &self,
        x: &Tensor,
        positions: &Tensor,
        attn_bias_4d: &Tensor,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(x)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_positions(&x, positions, attn_bias_4d, i, cache)?;
        }
        self.norm.forward(&x)
    }
}
