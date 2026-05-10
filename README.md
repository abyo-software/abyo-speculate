# abyo-speculate

Pure Rust [Speculative Decoding](https://arxiv.org/abs/2211.17192) library for
**local LLMs**, optimized for **batch size 1** single-user inference.

> **Status: 0.3.0** — APIs may change at any 0.x release. Algorithmic
> correctness is solid (statistical proofs + greedy-acceptance match
> against AR baseline on real Llama / Qwen / EAGLE checkpoints).

## What

abyo-speculate provides multiple Speculative Decoding (SD) algorithms behind
a unified Rust API:

| Method | Status (v0.4.1) | Measured speedup* |
|--------|------------------|-------------------|
| Vanilla SD (Leviathan 2023) | ✅ shipped | **0.78–1.13×** vs optimized AR |
| Medusa (Cai 2024) | ✅ shipped (loader + reference loop) | full speedup needs ≥24 GB GPU |
| EAGLE-2 (Li 2024) | ✅ shipped (KV-reorder fast path) | 0.46× on consumer 16 GB |
| EAGLE-3 (Li 2025) | ✅ shipped (multi-layer + d2t/t2d) | 0.62× on EC2 L4 24 GB |

> **v0.4.1 honesty pivot.** Earlier releases reported a 1.4–1.7× speedup
> for vanilla SD that was inflated by an inefficient AR baseline (the
> `next_logits` impl was doing a redundant forward per token). v0.4.1
> caches the next-token logits as a side effect of `observe`, making AR
> ~2× faster — and revealing that SD methods on candle's current BF16
> inference path are roughly break-even with AR for most task / model
> combinations on a consumer 16 GB GPU. The crate's value is "correct,
> reference SD implementations + honest measurement infrastructure",
> not "drop-in speedup." See [`BENCHMARKS.md`](./BENCHMARKS.md).

\* Qwen 2.5 3B target + Qwen 2.5 0.5B draft, k = 4, RTX 4070 Ti SUPER, BF16.
See [`BENCHMARKS.md`](./BENCHMARKS.md).

### Supported model families

| Family | Module | Notes |
|--------|--------|-------|
| Qwen 2 / 2.5 (BF16 / F32) | `model::qwen2_local` + `Qwen2Decoder` | |
| Qwen 2 / 2.5 (Q4 / Q5 / Q8 GGUF) | `model::quantized_qwen2_local` + `Qwen2QuantDecoder` | Lets 7B fit on 16 GB |
| Llama 1 / 2 / 3.x (BF16 safetensors) | `model::llama_local` + `LlamaDecoder` | EAGLE-friendly; tree-attention extensions |
| Llama 2 / 3 / 3.1 (Q4_K_M GGUF) | `model::quantized_llama_local` + `LlamaQuantDecoder` | Lets 8B fit on 16 GB; bundled HF tokenizer |
| Mistral | uses Llama path | bench `--family auto` detects |
| Phi-3 / 3.5 | `model::phi3_local` + `Phi3Decoder` | fused QKV + gate-up |

## Why

`vLLM` / `SGLang` / `TensorRT-LLM` target the data-center, high-batch case. `llama.cpp` is C++. **The Rust ecosystem has no integrated SD library.** abyo-speculate fills that gap, with explicit focus on the single-user, local-inference workload that powers Ollama-style apps and Rust agents.

### Scope

**In:** Llama 3.x, Qwen 2.5, Mistral 7B, Phi-3.5, batch size 1, candle backend.
**Out (for now):** large-batch serving, MoE acceleration, speculative streaming, non-Hugging-Face checkpoints.

## Quick start

```rust
use abyo_speculate::{SpeculateEngine, Method};
use abyo_speculate::model::qwen2::Qwen2Decoder;

// 1. Configure (no I/O yet).
let mut engine = SpeculateEngine::builder()
    .target_model("Qwen/Qwen2.5-7B-Instruct")
    .draft_model("Qwen/Qwen2.5-0.5B-Instruct")
    .method(Method::Vanilla)
    .draft_lookahead(4)
    .build()?;

// 2. Load decoders (CPU example shown; CUDA/Metal via crate features).
let target = Qwen2Decoder::from_paths(/* config, weights, tokenizer */)?;
let draft  = Qwen2Decoder::from_paths(/* ... */)?;
engine = engine.with_target(target).with_draft(draft);

// 3. Generate. Tokenize via the underlying decoder; keep this crate
//    tokenizer-agnostic for now.
let prompt_ids = vec![/* tokenized prompt */];
let out_ids = engine.generate_tokens(&prompt_ids, 200)?;
```

Preset shortcut for known model families:

```rust
let engine = SpeculateEngine::preset_for("qwen-2.5-7b")?;
// then attach decoders as above.
```

> The current builder is *config-only*; you attach loaded decoders with
> `with_target` / `with_draft`. A higher-level "load everything in one line"
> helper lands in Phase 1c — see [`ARCHITECTURE.md`](./ARCHITECTURE.md#what-is-intentionally-left-for-follow-up-sessions).

## Honest benchmarks

### Vanilla SD on Qwen 2.5 BF16 (RTX 4070 Ti SUPER, 128 tokens, k = 4)

Qwen 2.5 3B target + Qwen 2.5 0.5B draft, re-measured at v0.4.1
against an **optimized** AR baseline (no redundant per-token forward):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 67.0 | 64.6 | 0.96× |
| **code** | **67.4** | **76.1** | **1.13×** |
| translation | 65.3 | 59.6 | 0.91× |
| long_context | 62.4 | 48.5 | 0.78× |

Only code generation crosses 1× — its tokens (whitespace, brackets,
common keywords) are the most predictable, so per-round acceptance is
high enough to amortize the SD overhead. Chat / translation /
long-context lose against this hardware's per-AR-token cost.

### EAGLE on consumer 16 GB GPU + EC2 L4 24 GB

| Config | AR tok/s | EAGLE tok/s | Speedup |
|--------|---------:|------------:|--------:|
| Llama 2 7B BF16 + EAGLE-llama2-chat-7B (depth=2 k=2, RTX 4070 Ti SUPER) | 43.7 | 20.2 | **0.46×** |
| Llama 3 8B Q4_K_M + EAGLE-LLaMA3-8B (RTX 4070 Ti SUPER) | 47.0 | 10.6 | 0.23× |
| Llama 3.1 8B Q4_K_M + EAGLE3-LLaMA3.1-8B (RTX 4070 Ti SUPER) | 47.6 | 10.9 | 0.23× |
| Llama 3.1 8B BF16 + EAGLE3-LLaMA3.1-8B (EC2 L4 24 GB) | 8.2 | 5.1 | 0.62× |

**EAGLE on candle's BF16 inference path doesn't beat AR on any tested
config.** Per-round overhead (tree forward + bonus forward + draft) is
larger than the per-AR-token cost candle achieves on these GPUs. The
v0.4.0 KV-reorder fast path is in place (eliminates 2 of the 4
per-round target forwards a naive impl needs); the remaining gap is
candle's lack of Flash Attention / kernel fusion — features
production frameworks like vLLM use to make per-step AR much more
expensive, which is what amortizes EAGLE's per-round work in the
published paper.

Greedy-acceptance correctness is preserved by the v0.2.2 GEMV
root-fix in the strict path; the fast path may diverge from AR by a
token on borderline-argmax prompts.

EAGLE's value here is reference correctness + the infrastructure to
re-evaluate on faster inference frameworks as candle adds them, not a
production speedup today.

We do **not** quote a single headline ratio — see
[`BENCHMARKS.md`](./BENCHMARKS.md) for the full table including
model-pair sweeps, `draft_lookahead` sweeps, Llama numbers, Medusa
loader-compat notes, and the cases where SD currently does not help.

## Building

```bash
# CPU-only (default; slow but correct)
cargo build --release

# With CUDA (Linux / Windows + NVIDIA)
cargo build --release --features cuda

# With Metal (macOS)
cargo build --release --features metal
```

Minimum Rust version: **1.82** (stable).

## Roadmap

- [`abyo_speculate_plan.md`](./abyo_speculate_plan.md) — strategy, risks, competitive analysis, multi-phase plan.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — code layout, design decisions, what's deferred to follow-up sessions.
- [`CHANGELOG.md`](./CHANGELOG.md) — release notes.

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

## Contributing

Pre-alpha — issues and design discussions welcome. Please open a GitHub issue before sending large PRs so we can align on direction.
