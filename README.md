# abyo-speculate

Pure Rust [Speculative Decoding](https://arxiv.org/abs/2211.17192) library for
**local LLMs**, optimized for **batch size 1** single-user inference.

> **Status: 0.3.0** — APIs may change at any 0.x release. Algorithmic
> correctness is solid (statistical proofs + greedy-acceptance match
> against AR baseline on real Llama / Qwen / EAGLE checkpoints).

## What

abyo-speculate provides multiple Speculative Decoding (SD) algorithms behind
a unified Rust API:

| Method | Status (v0.3.0) | Measured speedup* |
|--------|------------------|-------------------|
| Vanilla SD (Leviathan 2023) | ✅ shipped | **1.41–1.74×** (see below) |
| Medusa (Cai 2024) | ✅ shipped (loader + reference loop) | TBD on real heads |
| EAGLE-2 (Li 2024) | ✅ shipped (loader + Cartesian + dynamic tree) | depends on target dtype — see EAGLE notes |
| EAGLE-3 (Li 2025) | ✅ shipped (multi-layer feature fusion + d2t/t2d) | depends on target dtype — see EAGLE notes |

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

Qwen 2.5 3B target + Qwen 2.5 0.5B draft (re-measured at v0.3.0):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.4 | 49.5 | **1.44×** |
| **code** | **35.3** | **61.4** | **1.74×** |
| translation | 34.1 | 48.0 | **1.41×** |
| long_context | 31.5 | 49.6 | **1.57×** |

Code generation wins — its tokens are the most predictable.

### EAGLE on consumer 16 GB hardware: honest negatives

We carry working EAGLE-2 / EAGLE-3 implementations against published
checkpoints (`yuhuili/EAGLE-llama2-chat-7B`,
`yuhuili/EAGLE-LLaMA3-Instruct-8B`, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`).
Greedy-acceptance correctness is verified by the v0.2.2
`tree_logits[0] == next_logits` invariant.

**On a single 16 GB GPU, EAGLE does not currently beat AR** in our
measurements:

| Config | AR tok/s | EAGLE tok/s | Speedup |
|--------|---------:|------------:|--------:|
| Llama 3 8B Q4_K_M  + EAGLE-LLaMA3-8B  (depth=4 k=2 dyn=16) | 45.0 | 8.5 | 0.19× |
| Llama 3.1 8B Q4_K_M + EAGLE3-LLaMA3.1-8B (depth=4 k=2 dyn=16) | 45.8 | 9.5 | 0.21× |
| Llama 2 7B BF16 + EAGLE-llama2-chat-7B (depth=4 k=2 dyn=16) | 21.2 | 10.3 | 0.49× |

Two compounding effects on the Q4 paths: (1) EAGLE drafts are trained on
FP16 hiddens, so Q4 quantization noise lowers acceptance rate; (2) the
Q4 lm_head is a 50 ms per-call bottleneck. The BF16 path is closer to
parity but still loses on Llama 2 7B's MHA architecture (per-token AR
is already cheap because KV reads are smaller, leaving less headroom
for SD overhead to amortize).

These tests ship in `tests/with_eagle*_e2e.rs` so the negatives are
fully reproducible. Closing the gap is a v0.3.x deliverable —
candidates: BF16 LLaMA 3 8B + EAGLE-3 on a 24 GB GPU (ours is 16 GB),
EAGLE acceptance-rate tuning per architecture, batched lm_head_apply.

For a **demonstrably-faster** SD path, **use Vanilla SD with a Qwen 2.5
target+draft pair** as documented above.

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
