# abyo-speculate Usage Guide

Short, practical walkthrough for the most common scenarios. For the full
API surface see [docs.rs](https://docs.rs/abyo-speculate); for benchmark
numbers see [`BENCHMARKS.md`](./BENCHMARKS.md); for the design rationale
see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Install

```sh
# Library:
cargo add abyo-speculate

# CLI bench / demo (from the workspace):
cargo install --git https://github.com/abyo-software/abyo-speculate \
  abyo-speculate-cli
```

GPU support: enable `--features cuda` for NVIDIA, `--features metal` for
Apple Silicon. Default builds run on CPU (slow but correct, useful for
unit tests).

## Scenario 1: minimal text-in / text-out

```rust
use abyo_speculate::{Method, SpeculateEngine};
use abyo_speculate::model::hub::download_qwen2;
use abyo_speculate::model::qwen2::Qwen2Decoder;
use abyo_speculate::model::qwen2_local::Config;
use candle_core::{DType, Device};

let repo = "Qwen/Qwen2.5-0.5B-Instruct";
let (config_path, tokenizer_path, weight_paths) = download_qwen2(repo)?;
let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

let device = Device::new_cuda(0)?;            // or Device::Cpu
let target = Qwen2Decoder::from_paths(
    &config, &weight_paths, &tokenizer_path, device, DType::BF16,
)?;

let mut engine = SpeculateEngine::builder()
    .target_model(repo)
    .method(Method::Autoregressive)
    .build()?
    .with_target(target);

println!("{}", engine.generate("The capital of France is", 32)?);
```

This is the same as `cargo run --release --features cuda --example simple_generate`.

## Scenario 2: vanilla SD with target + draft

```rust
let target = Qwen2Decoder::from_paths(/* Qwen 2.5 3B */ ..)?;
let draft  = Qwen2Decoder::from_paths(/* Qwen 2.5 0.5B */ ..)?;

let mut engine = SpeculateEngine::builder()
    .target_model("Qwen/Qwen2.5-3B-Instruct")
    .draft_model("Qwen/Qwen2.5-0.5B-Instruct")
    .method(Method::Vanilla)
    .draft_lookahead(4)
    .build()?
    .with_target(target)
    .with_draft(draft);

println!("{}", engine.generate("Why does SD help?", 200)?);
```

EOS is auto-applied based on `target.eos_token_ids()`.

## Scenario 3: streaming + custom stop tokens

```rust
use abyo_speculate::GenerationOptions;

let opts = GenerationOptions::new(256)
    .with_stops(target.eos_token_ids())   // stop on natural EOS
    .with_stop(some_custom_tok_id);       // ...or a custom marker

let prompt_ids = target.encode("…", true)?;
let _generated_ids = engine.generate_tokens_with(&prompt_ids, &opts, |tok| {
    print!(" [{tok}]");
    std::io::stdout().flush().ok();
    true   // return false to halt
})?;
```

## Scenario 4: Llama / Mistral / Phi-3

Identical surface — swap the decoder type:

```rust
// Llama 3.x:
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
let hf_config: LlamaConfig = serde_json::from_str(...)?;
let target = LlamaDecoder::from_paths(&hf_config.into_config(false), ..)?;

// Phi-3:
use abyo_speculate::model::phi3::Phi3Decoder;
use abyo_speculate::model::phi3_local::Config;
let target = Phi3Decoder::from_paths(&config, ..)?;

// Mistral: use LlamaDecoder + LlamaConfig (architectures are compatible).
```

## Scenario 5: Q4 (GGUF) target on a 16 GB GPU

```rust
use abyo_speculate::model::quantized_qwen2::Qwen2QuantDecoder;

// Pre-download:
//   - any Qwen 2.5 7B Q4_K_M GGUF file (e.g. qwen2.5-7b-instruct-q4_k_m.gguf)
//   - the matching upstream tokenizer.json (the GGUF embedded vocab is
//     not directly compatible with the `tokenizers` crate)
let target = Qwen2QuantDecoder::from_gguf(
    gguf_path,
    tokenizer_path,
    Device::new_cuda(0)?,
    /* eos */ vec![151645, 151643],     // Qwen 2.5 chat-end + endoftext
)?;

let draft = Qwen2Decoder::from_paths(/* Qwen 2.5 0.5B BF16 */ ..)?;

// Engine + run as in Scenario 2.
```

This pairing fits 7B Q4 (~4 GB) + 0.5B BF16 (~1 GB) on a 16 GB card with
plenty of room left for activations. (Note: dispatch through
`SpeculateEngine` requires both target and draft to use the same
[`Decoder`] trait — works because `Qwen2QuantDecoder` implements both
`Decoder` and `TreeDecoder` like the rest.)

## Scenario 6: real Medusa heads

The Medusa run loop is invoked directly (not through `SpeculateEngine`):

```rust
use abyo_speculate::methods::medusa::{
    run_medusa_real, Acceptance, MedusaConfig, MedusaHeads, MedusaHeadsCandle,
    MedusaRunConfig, TreeTopology,
};
use abyo_speculate::model::hub::download_files;

// Heads-only checkpoint (FasterDecoding/medusa-vicuna-7b-v1.3):
let head_path = download_files(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    &["medusa_lm_head.pt"],
)?[0].clone();

let cfg = MedusaConfig {
    n_heads: 5, hidden_size: 4096, vocab_size: 32000, residual_layers: 1,
};
let heads = MedusaHeadsCandle::from_fasterdecoding_pt(
    &cfg, &head_path, target.device(), target.dtype(),
)?;
let skeleton = MedusaHeads::from_config(cfg.clone());

let prompt_ids = target.encode("Hi", true)?;
let run_cfg = MedusaRunConfig {
    topology: TreeTopology::CartesianProduct,
    top_k_per_head: 2,
    acceptance: Acceptance::Greedy,
};
let mut rng = rand::thread_rng();

let out = run_medusa_real(
    &mut target, &heads, &skeleton, &prompt_ids, 128, &run_cfg, &mut rng,
)?;
println!("{}", target.decode(&out, true)?);
```

For checkpoints that bundle base + heads (`FasterDecoding/medusa-1.0-vicuna-7b-v1.5`)
use `MultiPthBackend` + `from_fasterdecoding_var_builder` — see
`crates/speculate/tests/with_real_medusa_e2e.rs` for a worked example.

## Bench CLI

```sh
abyo-speculate-bench \
    --target Qwen/Qwen2.5-3B-Instruct \
    --draft  Qwen/Qwen2.5-0.5B-Instruct \
    --method both \
    --task code \
    --max-tokens 128 --warmup 1 --runs 3 --draft-lookahead 4
```

Outputs human-readable progress on stderr plus a single JSON line on
stdout suitable for jq / awk pipelines.

## When SD does NOT help

- **Small target / draft ratio.** When the draft costs >= ~30% of the
  target's per-token cost, the per-round overhead eats the speedup. Our
  Qwen 2.5 1.5B + 0.5B pair lands at 0.99×.
- **High temperature + diverse tasks.** Acceptance rates fall, more
  draft work wasted.
- **MoE models.** Expert-activation overhead dominates, draft proposals
  rarely line up with the routing decision.
- **Very short outputs (≤ 8 tokens).** Per-round overhead doesn't
  amortize.

See `BENCHMARKS.md` for the measured numbers on a 4070 Ti SUPER.
