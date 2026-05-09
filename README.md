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

let engine = SpeculateEngine::builder()
    .target_model("llama-3.1-8b-instruct")
    .method(Method::Medusa)
    .draft_path("path/to/medusa-llama-3.1-8b")
    .build()?;

let output = engine.generate("Hello, world!", 200)?;
println!("{}", output);
```

Or use a one-liner preset:

```rust
let engine = SpeculateEngine::preset_for("llama-3.1-8b")?;
```

## Honest benchmarks

Numbers are filled in once a real-GPU run is complete. We will publish:

- per-model × per-task speedups (chat / coding / translation / long-context)
- absolute `tok/s` (baseline vs each SD method)
- the cases where SD does **not** help (MoE models, high-temperature sampling, very short outputs)

We won't quote a single "3× faster" headline number. SD is workload-dependent and we'll say so.

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

See [`abyo_speculate_plan.md`](./abyo_speculate_plan.md) for the full plan, risks, and competitive analysis.

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

## Contributing

Pre-alpha — issues and design discussions welcome. Please open a GitHub issue before sending large PRs so we can align on direction.
