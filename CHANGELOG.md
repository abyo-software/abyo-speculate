# Changelog

All notable changes to abyo-speculate are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is at `0.x`, breaking changes can land in any minor or
patch release; we'll only commit to `1.x`-style stability after the API
shape has been used in anger by at least one external project.

## [0.4.0] — 2026-05-10

### EAGLE fast path: 4 → 2 target forwards per round

Cuts per-round target overhead by **~50%** by eliminating three of the
four target forwards the v0.3.x EAGLE loop was doing:

1. **No more `last_hidden_state()`**: the new
   `TreeDecoder::observe_returning_last_hidden(ids)` runs `observe`
   AND returns the last position's hidden state from the same forward.
   The deepest-committed token's hidden is chained directly into the
   next round's draft input — no separate forward.
2. **No more tree_logits restoration**: the new
   `TreeDecoder::tree_logits_keep_kv(tree)` returns
   `(per_node_logits, per_node_hidden)` and **leaves** the KV cache
   populated with the tree (no restoration forward).
3. **No more full `observe(committed)`**: the new
   `TreeDecoder::commit_tree_path(tree, accepted_indices)` reorders
   the per-layer KV cache via `index_select` to keep the prefix +
   accepted nodes (and drop unaccepted siblings) — **zero extra
   forwards**. Only the bonus token still goes through a one-token
   `observe`.

The tree forward already populated KV for every tree node; the v0.3.x
implementation threw all of that away and re-forwarded the accepted
path. The KV reorder gives us the same end state for free.

### Measured speedup gain

EAGLE-2 on Llama 2 7B Chat **BF16** + `yuhuili/EAGLE-llama2-chat-7B`
(depth=2 k=2, 64 tokens, RTX 4070 Ti SUPER 16 GB):

| Build | AR tok/s | EAGLE tok/s | Speedup vs AR |
|-------|---------:|------------:|--------------:|
| v0.3.1 (4 forwards/round) | 21.2 | 10.3 | 0.49× |
| **v0.4.0 (2 forwards/round)** | **21.6** | **19.9** | **0.92×** |

Per-prompt: haiku **0.97×**, capital-of-France-style 0.94×, RoPE
explanation 0.91×. Output is byte-for-byte AR-matching for the
acceptance-deep prompts (haiku); on shallow-acceptance prompts the
trajectory may diverge a token or two due to GEMM precision drift on
`per_node_logits[i > 0]` (the v0.2.2 GEMV root-fix is skipped on the
fast path — strict mode is a v0.4.x follow-up).

EAGLE-3 on Llama 3.1 8B Q4_K_M improves from **0.21× → 0.23×** — the
Q4 lm_head per-call cost (~50 ms) still dominates this configuration
and the fast path can't avoid it. BF16 GQA targets (Llama 3.1 8B BF16
on a ≥ 24 GB GPU) are the configuration we'd expect to push past 1×.

### New / changed APIs

- `TreeDecoder::observe_returning_last_hidden(&mut self, &[u32]) -> Result<Tensor>`
- `TreeDecoder::tree_logits_keep_kv(&mut self, &DraftTree) -> Result<(Vec<Vec<f32>>, Vec<Tensor>)>`
- `TreeDecoder::commit_tree_path(&mut self, &DraftTree, &[usize]) -> Result<()>`
- `Cache::keep_kv_indices(&[u32])` (BF16 path) and
  `ModelWeights::keep_kv_indices(&[u32])` (Q4 path) for KV reordering.
- `LlamaQuantDecoder::apply_lm_head` now auto-promotes BF16/F16 input
  to F32 (the dtype the QMatMul kernel expects) — symmetrical with
  `LlamaDecoder::apply_lm_head`'s existing auto-promote.

## [0.3.2] — 2026-05-10

### EAGLE-2 worked example

`crates/speculate/examples/eagle2_bf16.rs` — downloads
`NousResearch/Llama-2-7b-chat-hf` BF16 + `yuhuili/EAGLE-llama2-chat-7B`,
runs greedy generation with EOS detection, prints the result. Use as a
template for production EAGLE setups.

```sh
cargo run --release --features cuda --example eagle2_bf16
```

Caveats baked into the example's docstring: ~15 GB GPU footprint;
EAGLE-2 measures ~0.5× of plain AR on Llama 2 7B BF16 on consumer
16 GB GPUs (architecture is MHA so per-step AR is already cheap).
Documented limitation, not a regression.

## [0.3.1] — 2026-05-10

### EOS support in `run_eagle` and `run_eagle3`

Both EAGLE run loops now check `target.eos_token_ids()` against each
committed batch and break out of the round loop when an EOS token is
emitted (the EOS token itself is included in the output). Previously
the loops kept generating until `max_new_tokens`, producing nonsense
past the natural end of a chat reply.

This fixes the visible regression in the BF16 EAGLE-2 e2e test where
the haiku prompt would correctly produce
`Sure! ... Serenity found` and then keep going into unrelated text.
EAGLE now stops on EOS the same way `Decoder::next_logits` /
`SpeculateEngine::generate_tokens_with` already does.

## [0.3.0] — 2026-05-10

### EAGLE on the architecture it was actually trained for (BF16)

The big v0.2.x finding was that EAGLE-3 + Q4 produces correct output
(post-v0.2.2 `tree_logits` fix) but ~0.21× speed because the draft was
trained on FP16 hidden states. v0.3.0 adds the canonical BF16 path:

- **`LlamaDecoder` (`model::llama`) BF16 safetensors path** gains the
  full TreeDecoder surface: `apply_lm_head`, `last_hidden_states_multi`,
  `num_hidden_layers`, `embed_tokens`. The same v0.2.2 root-replacement
  fix is applied to the BF16 path (single-position GEMV vs
  multi-position GEMM precision drift affects BF16 too, just less
  often).
- **`forward_with_layer_hooks`** on `quantized_llama_local::ModelWeights`
  / `llama_local::Llama` collects residual hidden states at arbitrary
  layer indices in one forward pass — required by EAGLE-3's
  low/mid/high feature concat.
- **`EagleDraftConfig::eagle_llama2_chat_7b()`** preset for the
  `yuhuili/EAGLE-llama2-chat-7B` checkpoint (Llama 2 7B is MHA, not GQA;
  RoPE base 10 000; 32k vocab via SentencePiece).
- **`tests/with_eagle_bf16_e2e.rs`**: end-to-end multi-prompt benchmark
  for `NousResearch/Llama-2-7b-chat-hf` BF16 (mirrored to dodge the
  meta-llama gating) + `yuhuili/EAGLE-llama2-chat-7B`. ~15 GB total →
  fits a 16 GB GPU.

### t2d sampling mask

`Eagle3DraftCandle::t2d_mask` is now derived from `d2t` at load time
(the published checkpoint stores `t2d` as BoolStorage which candle's
pickle loader skips, but the value is fully derivable from `d2t`).
`mask_target_logits()` applies the mask in-place, useful when sampling
the target distribution under EAGLE-3's reachability constraint.

### Vanilla SD speedups re-validated

Re-measured at v0.3.0 on Qwen 2.5 3B + 0.5B BF16 (RTX 4070 Ti SUPER,
128 tokens, k=4, mean of 3 runs):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.4 | 49.5 | **1.44×** |
| code | 35.3 | 61.4 | **1.74×** |
| translation | 34.1 | 48.0 | **1.41×** |
| long_context | 31.5 | 49.6 | **1.57×** |

### Quality

- 74 unit + statistical tests pass on CPU.
- `cargo clippy --all-targets -- -D warnings` clean.
- `cargo audit` clean (only 2 unmaintained-transitive-dep warnings;
  `number_prefix` from `indicatif`, `paste` from `gemm`; both passive).
- New `cargo-audit` step in CI.
- `CONTRIBUTING.md` added.
- `scripts/run_benchmarks.sh` for one-line repro of the README table.

## [0.2.2] — 2026-05-10

### tree_logits multi-position correctness fix

Root cause for the v0.2.1 EAGLE-3 divergence: candle's QMatMul takes a
GEMV path for `seq_len == 1` but a GEMM path for `seq_len > 1`, and the
two have different FP accumulation orders. On Q4 × 128k vocab, the
per-token logit values drift by ~0.01–0.05 across tree sizes — enough
to flip a borderline argmax (e.g. ` a` vs ` Paris` differ by 0.02 at
the end of "The capital of France is").

Fix in `LlamaQuantDecoder::tree_logits`: overwrite `per_node_logits[0]`
with the GEMV-path logits captured at the restoration
`forward_advance_logits([last_committed])` step that already runs after
the tree forward. The root row is now bit-for-bit identical to
`next_logits`; deeper rows still go through the GEMM path but are only
consulted after the corresponding draft token is accepted (i.e. matches
root's argmax, which is what the bonus-or-accept logic uses).

### EAGLE-3 e2e (Llama 3.1 8B Q4_K_M, depth=4 k=2 dyn=16, 32 tokens)

```text
AR baseline : 45.82 tok/s
EAGLE-3     :  9.53 tok/s   (0.21× — output now matches AR exactly)
```

Output is now byte-for-byte the AR trajectory (greedy acceptance over
the corrected tree). Speed is bounded by Q4 quantization: EAGLE-3 was
trained on FP16 target hidden states, but we feed Q4 hiddens, so the
draft's predictions diverge from target's argmax often enough that
acceptance amortises poorly. Closing the speed gap further requires
either a BF16 target (≥ 24 GB GPU) or an EAGLE-3 retrained on the Q4
feature distribution.

### Tests

`tests/tree_logits_consistency.rs` (gated under `#[ignore]`) now
**passes** — sweeps 1, 2, 3, 4, 5, 8, 16, 32-node linear chains plus
the 31-node Cartesian and asserts `tree_logits[0] argmax == next_logits
argmax` on Llama 3.1 8B Q4_K_M.

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
