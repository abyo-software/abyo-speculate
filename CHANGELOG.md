# Changelog

All notable changes to abyo-speculate are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is at `0.x`, breaking changes can land in any minor or
patch release; we'll only commit to `1.x`-style stability after the API
shape has been used in anger by at least one external project.

## [Unreleased]

### Coming up

- Real Medusa speedup measurement against Vicuna 7B (currently blocked on
  Vicuna safetensors availability — only PyTorch sharded binaries are
  open-access).
- F16 / Q4 quantisation paths so Qwen 2.5 7B + draft fits on a 16 GB GPU.
- EAGLE-2 / EAGLE-3 implementations.
- Mistral and Phi-3.5 vendored model paths.

## [0.0.1] — 2026-05-10

Initial public release. Alpha / pre-1.0; APIs may change without notice.

### Added

- **Vanilla Speculative Decoding** (Leviathan et al. 2023) with the
  modified rejection rule, statistically validated against analytic
  target distributions (TV distance < 0.025–0.03 across four mismatch
  scenarios).
- **Medusa** (Cai et al. 2024) skeleton + reference loop:
  - `MedusaConfig` / `MedusaHeads` / `build_draft_tree` (greedy and
    Cartesian-product topologies).
  - `run_medusa` reference verifier (mock-validated).
  - `MedusaHeadModule` / `MedusaHeadsCandle` real-model heads with
    `from_random` (synthetic init for plumbing tests) and
    `from_fasterdecoding_pt` (loads `medusa_lm_head.pt` from the
    FasterDecoding repos).
  - `run_medusa_real` end-to-end loop generic over `TreeDecoder`.
- **`DraftTree` primitives**: parent-pointer trees, ancestor mask, RoPE
  position_ids, root-to-leaf paths, plus tensor builders
  (`tree_self_bias`, `full_attention_bias`, `full_attention_bias_4d`)
  for direct injection into model attention layers.
- **Vendored Qwen 2 / 2.5 model** (`model::qwen2_local`) with
  tree-attention extensions (per-token RoPE via `index_select`, 4D
  attention bias injection, partial KV truncation).
- **Vendored Llama 1 / 2 / 3.x model** (`model::llama_local`) with the
  same tree-attention extensions; supports Llama 3 rope scaling.
- **Concrete `Decoder` impls**: `Qwen2Decoder` and `LlamaDecoder` —
  identical public surfaces (`from_paths`, `encode`, `decode`,
  `next_logits`, `batched_logits`, `tree_logits`, `last_hidden_state`,
  `rollback_to`).
- **`SpeculateEngine`**: builder + `with_target` / `with_draft`
  attachment + `generate(text, max_tokens) -> String` /
  `generate_tokens(&[u32], max_tokens) -> Vec<u32>`. Dispatches on
  `Method::Autoregressive`, `Method::Vanilla`. Medusa / EAGLE return
  `UnsupportedMethod`.
- **Sampling primitives**: `softmax_with_temperature`, `top_p_filter`,
  `sample_from_distribution`, all unit-tested.
- **KV `RollbackCache` primitive**: snapshot / append / commit / rollback.
- **`abyo-speculate-bench`**: CLI for autoregressive vs. vanilla SD
  timing on real Qwen2 / Llama checkpoints. Auto-detects model family,
  ships four prompt presets (chat / code / translation / long-context),
  emits a JSON line per run for downstream tables.
- **`abyo-speculate-demo`**: minimal hello-world for the engine API.
- **`hf-hub` download helpers** (`download_qwen2_single_shard`,
  `download_qwen2`, `download_files`) with sharded-checkpoint
  auto-detection.

### Real-GPU measurements

Reproducible on RTX 4070 Ti SUPER, BF16, k = 4, 128 new tokens
(Qwen 2.5 3B target + Qwen 2.5 0.5B draft):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.0 | 48.5 | 1.42× |
| code | 33.9 | 59.8 | 1.76× |
| translation | 33.8 | 47.2 | 1.40× |
| long_context | 31.2 | 48.2 | 1.55× |

See [`BENCHMARKS.md`](./BENCHMARKS.md) for the full table.

### Known limitations

- 7B target + 0.5B draft both in BF16 OOMs on a 16 GB card; needs
  quantisation or a larger GPU.
- Medusa's full speedup measurement is blocked on a safetensors-format
  Vicuna 7B; the loader is verified against the published `.pt` format.
- EAGLE-2 / EAGLE-3 not yet implemented.
