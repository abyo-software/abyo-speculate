# abyo-speculate

Pure Rust [Speculative Decoding](https://arxiv.org/abs/2211.17192) library for **local LLMs**, optimized for **batch size 1** single-user inference.

> **Status: pre-alpha (0.0.1)** — under active development. APIs will change.

## What

abyo-speculate provides multiple Speculative Decoding (SD) algorithms behind a unified Rust API:

| Method | Status | Speedup target |
|--------|--------|----------------|
| Vanilla SD (Leviathan 2023) | 🚧 Phase 1 | 1.5–2× |
| Medusa (Cai 2024) | 🚧 Phase 1 | 1.5–2× |
| EAGLE-2 (Li 2024) | 📋 Phase 2 | 2.5–3× |
| EAGLE-3 (Li 2025) | 📋 Phase 2 | 3–3.5× |
| SAGUARO (2026) | 📋 Phase 3 | TBD |

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

Measured on RTX 4070 Ti SUPER (BF16, 128 tokens, k = 4, Qwen 2.5 3B target +
Qwen 2.5 0.5B draft):

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.0 | 48.5 | **1.42×** |
| **code** | **33.9** | **59.8** | **1.76×** |
| translation | 33.8 | 47.2 | **1.40×** |
| long_context | 31.2 | 48.2 | **1.55×** |

Code generation has the highest speedup — its tokens are the most
predictable. We do **not** quote a single headline ratio — see
[`BENCHMARKS.md`](./BENCHMARKS.md) for the full table including model-pair
sweeps, `draft_lookahead` sweeps, Llama numbers, Medusa loader-compat
notes, and the cases where SD currently does not help.

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
