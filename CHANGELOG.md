# Changelog

All notable changes to abyo-speculate are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is at `0.x`, breaking changes can land in any minor or
patch release; we'll only commit to `1.x`-style stability after the API
shape has been used in anger by at least one external project.

## [0.2.1] — 2026-05-10

### EAGLE-3 reference-matching architecture

Refactors `Eagle3DraftCandle` and `run_eagle3` to mirror the published
inference flow (`SafeAILab/EAGLE/eagle/model/cnets.py`):

- The published EAGLE-3 checkpoint has **no `embed_tokens`** — the draft
  reuses the **target's** tied embedding via the new
  `TreeDecoder::embed_tokens` trait method (impl on `LlamaQuantDecoder`
  → `ModelWeights::embed_tokens`).
- New `Eagle3DraftCandle::forward_hidden` returns the pre-norm midlayer
  output (fc applied iff input is the 3*hidden concat). New
  `apply_norm_lm_head` mirrors `lm_head(self.norm(last_hidden))`.
- `run_eagle3` now follows the official schedule:
  - Round 0: `embed(root_token) + concat(low,mid,high)`
  - Round i+1: `embed(top1_drafted) + previous midlayer output`
- `Eagle3RunConfig::default_layers_for(n)` now returns the published
  training recipe — `2 / n/2 / n-3` (input-of), which in our
  after-layer-i convention is `[1, n/2 - 1, n - 4]` =
  **`[1, 15, 28]`** for Llama 3.1 8B (was `[1, 16, 30]`).
- New `last_hidden_states_multi(layers)` on `TreeDecoder` /
  `LlamaQuantDecoder` to fetch the low/mid/high target features in
  one quantized forward.
- New `forward_hidden_with_layers` on `quantized_llama_local::ModelWeights`.

### Known blocker (v0.2.2 critical path)

`LlamaQuantDecoder::tree_logits` returns a different root distribution
than `next_logits` would for the same state when the tree has > 1 node.
Repro on Llama 3.1 8B Q4_K_M, prompt "The capital of France is":

```text
root_token       = 374    (" is")
next_logits  arg = 264    (" a")
tree_logits(1)   = 264    ✓
tree_logits(31)  = 12366  (" Paris")  ✗
```

Captured by `tests/tree_logits_consistency.rs` (gated under `#[ignore]`).
The bug is in `forward_with_positions` / the attention-bias broadcast in
`run_attn` and breaks greedy acceptance correctness whenever the
draft's top-1 disagrees with target argmax. EAGLE-2 + Llama 3.0 happens
to dodge this because the prompt's argmax is stable for both code paths.

Until v0.2.2 lands the fix:
- EAGLE-3 e2e on Llama 3.1 still posts **0.21×** (output is coherent
  but diverges from greedy AR for the same reason).
- EAGLE-2 e2e on Llama 3.0 keeps producing AR-matching output.

### Other

- 73 unit tests pass; 1 new `#[ignore]` regression test for the
  tree_logits invariant.

## [0.2.0] — 2026-05-10

### Methods

- **EAGLE-2** (Li et al. 2024) end-to-end against the published
  `yuhuili/EAGLE-LLaMA3-Instruct-8B` checkpoint with Llama 3 8B
  Instruct **Q4_K_M GGUF** as the target. The full Cartesian-product
  tree path runs greedy-acceptance correctly (output text matches the
  AR baseline).
- **EAGLE-2 dynamic tree pruning** (`EagleRunConfig::max_tree_nodes`):
  builds the full Cartesian tree, then keeps the top-N nodes by
  cumulative log-prob path score plus their ancestor closure. Reduces
  `tree_logits` cost without changing correctness.
- **EAGLE-3** (Li et al. 2025) wired end-to-end against
  `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`: 3-layer feature concat
  (`low / mid / high`), midlayer with `input_layernorm` + `hidden_norm`
  + 2*hidden attention input, own 32k draft `lm_head`, `d2t` translation
  back to the target's 128k vocab.

### Target side

- `LlamaQuantDecoder` for GGUF Llama families (Llama 2 / 3 / 3.1) with
  bundled HF tokenizer.
- New `TreeDecoder::apply_lm_head`, `last_hidden_states_multi`,
  `num_hidden_layers` trait methods so EAGLE/EAGLE-3 don't need to load
  duplicate decoders or hand-roll closure-based plumbing.
- Vendored `quantized_llama_local` with a tree-attention-friendly
  `forward_with_positions` and a `forward_hidden_with_layers` that
  exposes residual outputs at arbitrary depths.

### Honest perf data

End-to-end on Llama 3 8B Instruct **Q4_K_M** + EAGLE-2 LLaMA3 draft
(prompt = "The capital of France is", max_tokens = 32, depth = 4, k = 2):

| variant            | tok/s | speedup |
|--------------------|-------|---------|
| Autoregressive     | 45.84 | 1.00×   |
| EAGLE-2 Cartesian  |  6.62 | 0.14×   |
| EAGLE-2 Dyn-16     |  8.52 | 0.19×   |

EAGLE-2 is **slower than autoregressive** on Q4 8B targets because the
per-step `target.apply_lm_head` (Q4 × 128k vocab ≈ 50 ms each) costs
more than a Q4 target step (~22 ms). EAGLE-2's design assumption
(target step >> draft per-step overhead) breaks under heavy
target-side quantization.

EAGLE-3 against Llama 3.1 8B **Q4_K_M**: 0.21× over AR. Output text is
coherent but diverges from greedy AR — most likely the trained-recipe
layer indices and a midlayer normalisation detail need tuning. **v0.2.1
target**: ≥ 1× on Q4 8B via correct EAGLE-3 wiring.

### Other

- 73 unit + statistical tests pass on CPU; two GPU-gated end-to-end
  tests cover the EAGLE-2 / EAGLE-3 real-checkpoint paths.

## [0.1.0] — 2026-05-10

## [0.1.0] — 2026-05-10

OSS launch. Production-ready alpha — APIs may still change at any 0.x
release, but the algorithmic correctness story is solid (statistical
proofs, real-GPU validation) and the supported model surface covers the
four families most local-LLM users actually run.

### Methods

- **Vanilla Speculative Decoding** (Leviathan et al. 2023) with the
  modified rejection rule, statistically validated against analytic
  target distributions (TV distance < 0.025–0.03 across four
  target/draft mismatch scenarios).
- **Medusa** (Cai et al. 2024):
  - Full structural pipeline: `MedusaConfig`, `MedusaHeads`,
    `build_draft_tree` with greedy + Cartesian-product topologies.
  - Reference loop `run_medusa` validated against the mock harness.
  - candle-backed real heads via `MedusaHeadModule` /
    `MedusaHeadsCandle`. `from_random` (synthetic init for plumbing
    tests), `from_fasterdecoding_pt` (heads-only `.pt`), and
    `from_fasterdecoding_var_builder` (bundled checkpoints loaded from
    a custom `SimpleBackend`).
  - End-to-end loop `run_medusa_real` generic over `TreeDecoder`.

### Models

Each family ships as a vendored copy of the relevant
`candle_transformers` model with the SD-specific extensions
(per-position RoPE via `index_select`, 4D attention bias injection,
`truncate_kv_cache_to` for fast rollback):

- **Qwen 2 / 2.5** — `model::qwen2_local` + `Qwen2Decoder`
- **Llama 1 / 2 / 3.x** — `model::llama_local` + `LlamaDecoder`
- **Phi-3 / 3.5** — `model::phi3_local` + `Phi3Decoder`
  (handles fused QKV + fused gate+up MLP)
- **Mistral** — uses the Llama path; auto-detected by the bench CLI
- **Qwen 2 / 2.5 quantized (GGUF / Q4 / Q5 / Q8)** —
  `model::quantized_qwen2_local` + `Qwen2QuantDecoder`. Lets a 7B
  target fit alongside a draft model on a 16 GB consumer GPU.

### Primitives

- `tree::DraftTree` — parent-pointer trees with ancestor mask, RoPE
  position_ids, root-to-leaf paths, plus tensor builders
  (`tree_self_bias`, `full_attention_bias`, `full_attention_bias_4d`)
  for direct injection into model attention.
- `cache::RollbackCache` — KV cache snapshot / append / commit /
  rollback primitive.
- `sampling::{softmax_with_temperature, top_p_filter,
  sample_from_distribution}` — the unit-tested low-level sampling
  toolkit.

### Engine API

- `SpeculateEngine` builder pattern + `with_target` / `with_draft` to
  attach loaded decoders.
- `generate_tokens(prompt, max)` — autoregressive or vanilla SD
  depending on `Method`.
- `generate_tokens_with(prompt, opts, on_token)` — full control:
  per-call `GenerationOptions { max_new_tokens, stop_tokens }` plus a
  streaming callback that returns `bool` to halt early.
- `generate(text, max)` — text-in / text-out, auto-applies the target's
  EOS tokens.

### Loaders

- `model::hub::download_qwen2` — single-shard / sharded auto-detect for
  any HF safetensors repo.
- `model::hub::download_pth_sharded` + `MultiPthBackend` — load PyTorch
  sharded `.bin` checkpoints (e.g. Vicuna 7B, FasterDecoding bundled
  Medusa) directly without external conversion. Implements
  `candle_nn::var_builder::SimpleBackend`.

### Tooling

- `abyo-speculate-bench` CLI: autoregressive vs SD timing on real Qwen2
  / Llama / Mistral / Phi-3 checkpoints. `--family auto` infers from
  repo id; `--task chat|code|translation|long-context` selects a
  representative prompt; emits one JSON line per run.

### Real-GPU measurements

Reproducible on RTX 4070 Ti SUPER, BF16, k = 4, 128 new tokens
(Qwen 2.5 3B target + Qwen 2.5 0.5B draft):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.0 | 48.5 | 1.42× |
| code | 33.9 | 59.8 | **1.76×** |
| translation | 33.8 | 47.2 | 1.40× |
| long_context | 31.2 | 48.2 | 1.55× |

See [`BENCHMARKS.md`](./BENCHMARKS.md) for the full table including
model-pair sweeps, k-sweeps, multi-family smokes, and the cases where
SD currently does not help.

### Known limitations

- **EAGLE-2 / EAGLE-3 not yet implemented.** v0.2.0 deliverable. Tree
  primitives and verification math are in place; the EAGLE-specific
  draft architecture + dynamic tree construction land next.
- **Real Medusa speedup measurement.** The loader is verified against
  the published FasterDecoding format (`tests/with_real_medusa_heads.rs`
  passes); a full speedup number against Vicuna 7B requires loading the
  PyTorch-sharded base via `MultiPthBackend`, which the integration
  test in `tests/with_real_medusa_e2e.rs` exercises but isn't part of
  the published numbers in BENCHMARKS.md yet.
- **7B + draft pair in BF16** OOMs on a 16 GB GPU. The Q4 path
  (`Qwen2QuantDecoder`) addresses this; integration test pending.
- Multi-batch SD is out of scope. abyo-speculate is single-user
  (batch = 1) by design.

## [0.0.1] — 2026-05-10 (initial pre-release)

Initial workspace scaffold; the algorithmic correctness story (vanilla SD
+ Medusa skeleton) and the first 1.43× speedup on Qwen 2.5 3B + 0.5B
draft. See git log for the granular phase-by-phase history that landed
between 0.0.1 and 0.1.0.
