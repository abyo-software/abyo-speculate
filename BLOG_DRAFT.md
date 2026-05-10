# abyo-speculate v0.2.0 — EAGLE-2 / EAGLE-3 wired against published checkpoints

*v0.2.0 release notes draft. Edit before posting.*

## What changed since v0.1.0

- **EAGLE-2** (Li et al. 2024) end-to-end against
  `yuhuili/EAGLE-LLaMA3-Instruct-8B` with Llama 3 8B Instruct
  **Q4_K_M GGUF** as the target. Output text matches the AR baseline
  byte-for-byte (greedy acceptance over the verified target tree).
- **EAGLE-2 dynamic tree pruning** via `EagleRunConfig::max_tree_nodes` —
  builds the full Cartesian tree, scores every node by cumulative
  log-prob, keeps the top-N plus the ancestor closure.
- **EAGLE-3** (Li et al. 2025) wired against
  `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`: 3-layer feature concat
  (`low / mid / high`), midlayer with `input_layernorm` + `hidden_norm`
  + 2*hidden attention input, own 32k draft `lm_head`, `d2t` translation
  back to the target's 128k vocab.
- New `TreeDecoder::apply_lm_head`, `last_hidden_states_multi`,
  `num_hidden_layers` trait methods so the EAGLE / EAGLE-3 run loops
  share the target without duplicate decoders or closure-based
  plumbing.
- New `LlamaQuantDecoder` for GGUF Llama families (Llama 2 / 3 / 3.1)
  with a bundled HF tokenizer and a tree-attention-friendly
  `forward_with_positions` + `forward_hidden_with_layers`.

## Honest perf data on Q4 8B targets

End-to-end on Llama 3 8B Instruct **Q4_K_M** + EAGLE-2 LLaMA3 draft
(prompt = "The capital of France is", max_tokens = 32, depth = 4, k = 2):

| variant            | tok/s  | speedup |
|--------------------|-------:|--------:|
| Autoregressive     | 45.84  | 1.00×   |
| EAGLE-2 Cartesian  |  6.62  | 0.14×   |
| EAGLE-2 Dyn-16     |  8.52  | 0.19×   |

EAGLE-2 is **slower than autoregressive** on Q4 8B targets in this
configuration. The bottleneck is the per-step
`target.apply_lm_head`: a Q4 × 128k vocab QMatMul that costs ~50 ms
per call on this GPU, vs ~22 ms for an entire Q4 target step.
EAGLE-2's design assumption (target step >> draft per-step overhead)
breaks under heavy target-side quantization.

EAGLE-3 against Llama 3.1 8B **Q4_K_M** posts 0.21× over AR. Output
text is coherent ("Paris, which is located in the northern part of the
country…") but diverges from greedy AR — the trained-recipe layer
indices and a midlayer normalisation detail need tuning. **v0.2.1
target**: ≥ 1× on Q4 8B via correct EAGLE-3 wiring (smaller draft
vocab dodges the per-step Q4 lm_head bottleneck).

## What's still v0.2.1 work

- EAGLE-3 layer-index / midlayer tuning to land a real ≥ 1× on Q4 8B.
- `t2d` BoolStorage workaround so the sampling-mask path uses the
  reachable-targets bitmap rather than falling back to "all reachable".
- Fp16 Llama 3 / 3.1 target on a 24 GB GPU as the canonical benchmark
  (Q4 is the practical case but not the easiest perf surface for SD).

---

# abyo-speculate v0.1.0 — Pure Rust Speculative Decoding for local LLMs

*A Day-1 launch draft. Edit before posting to HN / r/LocalLLaMA / r/rust.*

---

## TL;DR

[abyo-speculate](https://github.com/abyo-software/abyo-speculate) is a Pure
Rust library that accelerates local language-model inference using
Speculative Decoding. It supports four model families (Qwen 2/2.5,
Llama 1/2/3.x, Mistral, Phi-3/3.5) plus quantized GGUF variants, and
ships with a unified `cargo run --bin abyo-speculate-bench` for honest
tok/s numbers.

On an RTX 4070 Ti SUPER, BF16, with Qwen 2.5 3B as the target and Qwen
2.5 0.5B as the draft, we measure **1.42–1.76× speedup over plain
autoregressive** depending on the task — code generation hits 1.76×
because draft tokens are highly predictable (whitespace, brackets,
keywords).

The library is 0.1.0 alpha; the algorithmic correctness story is
solid (statistical proofs against analytic distributions plus
real-GPU validation against published checkpoints).

## Why Pure Rust

vLLM / SGLang / TensorRT-LLM dominate the data-center batch-throughput
case. llama.cpp dominates the single-user CPU + small-GPU case in C++.
The Rust ecosystem has had no integrated SD library — abyo-speculate
fills that gap.

Concrete reasons to care:

- **Single-binary deployment.** Ships with the model loader + bench in
  one cargo workspace. No Python runtime, no PyTorch.
- **Embeddable in Rust applications.** Tauri desktop apps,
  command-line tools, agent runtimes — anything that already wants to
  stay in Rust.
- **Honest benchmarks.** No marketing speedup quoted in headline form;
  every number in `BENCHMARKS.md` is reproducible by `cargo run --bin
  abyo-speculate-bench`.

## What's in 0.1.0

### Methods
- **Vanilla SD** (Leviathan et al. 2023) — modified rejection rule
  with the `max(0, p_target - q_draft)` resample on rejection.
  Statistically validated: TV distance < 0.025 across four mismatch
  scenarios.
- **Medusa** (Cai et al. 2024) — multi-head architecture, both greedy
  and Cartesian-product tree topologies, real heads loaded from the
  published `FasterDecoding/medusa-*` checkpoints.

### Models
Qwen 2/2.5, Llama 1/2/3.x, Mistral (via Llama path), Phi-3/3.5, plus
quantized Qwen 2/2.5 (GGUF Q4/Q5/Q8). Adding a family is mechanical:
each model gets a vendored `*_local.rs` that mirrors the upstream
candle file with three additions — per-position RoPE via `index_select`,
4D attention bias injection, partial KV truncation.

### API

```rust
use abyo_speculate::{Method, SpeculateEngine};
use abyo_speculate::model::qwen2::Qwen2Decoder;

let target = Qwen2Decoder::from_paths(...)?;  // BF16 / F32 / Q4 GGUF
let draft  = Qwen2Decoder::from_paths(...)?;
let mut engine = SpeculateEngine::builder()
    .target_model("Qwen/Qwen2.5-3B-Instruct")
    .draft_model("Qwen/Qwen2.5-0.5B-Instruct")
    .method(Method::Vanilla)
    .draft_lookahead(4)
    .build()?
    .with_target(target)
    .with_draft(draft);

let out = engine.generate("Why does speculative decoding work?", 128)?;
```

`generate_tokens_with(prompt, opts, on_token)` adds streaming + stop
tokens for chat / agent UIs.

## Honest numbers (RTX 4070 Ti SUPER, BF16)

`cargo run --bin abyo-speculate-bench` produces these. Every row is one
`--target X --draft Y --task Z` invocation; they're a single point on a
much larger product surface.

### Per-task (Qwen 2.5 3B + 0.5B draft, k = 4)

| Task | AR tok/s | SD tok/s | Speedup |
|------|---------:|---------:|--------:|
| chat | 34.0 | 48.5 | 1.42× |
| code | 33.9 | 59.8 | **1.76×** |
| translation | 33.8 | 47.2 | 1.40× |
| long_context | 31.2 | 48.2 | 1.55× |

Code wins by a wide margin — its tokens (whitespace, brackets, common
keywords) are the most predictable, so the draft's acceptance rate is
highest.

### Where SD does NOT help

- **1.5B + 0.5B** (3× param ratio): 0.99× — target ≈ draft cost, the
  per-round overhead eats the parallelisation.
- **High temperature + diverse tasks** (translation): 1.40× rather
  than 1.55× because more rejections waste draft work.
- **MoE models** (not benchmarked, but Leviathan §3.4 explains why): SD
  gives little to no benefit because expert activation overhead
  dominates.

We don't quote a single ratio. That would be a marketing lie.

## What's NOT in 0.1.0

- **EAGLE-2 / EAGLE-3** (Li et al. 2024-2025). The tree primitives and
  rejection-sampling math are landed; the EAGLE-specific draft
  architecture is the v0.2.0 deliverable.
- **Real Medusa speedup numbers.** The loader is verified against
  published format; full speedup measurement against Vicuna 7B is
  pending due to PyTorch-only checkpoints (we have a
  `MultiPthBackend` for loading; just hasn't run on the bench yet).
- **GGUF Q4 speedup measurement.** Plumbing in
  `quantized_qwen2_local`; bench integration follows.

## Why we built this

[abyo software](https://github.com/abyo-software) is shipping a series
of Rust LLM utilities — `abyo-llm-probe` (encoder), `abyo-speculate`
(decoder accelerator). The thesis is that the local-LLM ecosystem in
Rust needs primitives, not just bindings.

The competitive frame: vLLM / SGLang / TensorRT-LLM target the
data-center high-batch case. llama.cpp targets single-user
single-binary on CPU + small GPU in C++. The Rust ecosystem has been
covered by candle as a tensor library but has lacked the SD integration
layer. abyo-speculate fills that.

## Get started

```sh
cargo add abyo-speculate
```

or for the bench CLI:

```sh
cargo install --git https://github.com/abyo-software/abyo-speculate \
  abyo-speculate-cli
abyo-speculate-bench --target Qwen/Qwen2.5-3B-Instruct \
                     --draft  Qwen/Qwen2.5-0.5B-Instruct \
                     --method both --task code
```

## Links

- [GitHub](https://github.com/abyo-software/abyo-speculate)
- [Architecture document](https://github.com/abyo-software/abyo-speculate/blob/main/ARCHITECTURE.md)
- [Benchmarks](https://github.com/abyo-software/abyo-speculate/blob/main/BENCHMARKS.md)
- [Project plan & honest risk register](https://github.com/abyo-software/abyo-speculate/blob/main/abyo_speculate_plan.md)

Issues + PRs welcome. The roadmap is in `CHANGELOG.md` under
"Unreleased — v0.2.0".
