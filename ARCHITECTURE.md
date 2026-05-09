# abyo-speculate — Architecture

This document captures the design decisions made during the initial
build-out so future sessions can pick up cold. Treat it as a partner to
[`abyo_speculate_plan.md`](./abyo_speculate_plan.md) — the plan describes
*why* we are building this, this document describes *how* the code is
organised.

## Crate layout (v0.1.0)

```
abyo-speculate/
├── crates/
│   ├── speculate/                       # the library
│   │   ├── src/
│   │   │   ├── lib.rs                   # public re-exports + module roster
│   │   │   ├── error.rs                 # crate Error / Result
│   │   │   ├── device.rs                # CPU / CUDA / Metal selection
│   │   │   ├── engine.rs                # SpeculateEngine + builder +
│   │   │   │                            #   GenerationOptions + dispatch
│   │   │   ├── presets.rs               # Llama / Qwen / Mistral / Phi configs
│   │   │   ├── sampling/{mod,tokens}.rs # softmax / top-p / categorical sampler
│   │   │   ├── cache/{mod,rollback}.rs  # KV snapshot / append / commit / rollback
│   │   │   ├── tree.rs                  # DraftTree + tensor mask builders
│   │   │   ├── methods/
│   │   │   │   ├── mod.rs               # Method enum
│   │   │   │   ├── vanilla.rs           # Leviathan 2023 reference impl
│   │   │   │   ├── medusa.rs            # Cai 2024 reference + real-head loaders
│   │   │   │   └── eagle.rs             # v0.2.0 skeleton + UnsupportedMethod
│   │   │   ├── model/
│   │   │   │   ├── mod.rs               # Decoder + TreeDecoder traits
│   │   │   │   ├── loader.rs            # ModelSource (HF id / local path)
│   │   │   │   ├── hub.rs               # download helpers + MultiPthBackend
│   │   │   │   ├── mock.rs              # MockDecoder for tests
│   │   │   │   ├── qwen2{,_local}.rs    # Qwen 2 / 2.5 BF16/F32 path
│   │   │   │   ├── quantized_qwen2{,_local}.rs # Q4 / Q5 / Q8 GGUF path
│   │   │   │   ├── llama{,_local}.rs    # Llama 1/2/3.x; also serves Mistral
│   │   │   │   └── phi3{,_local}.rs     # Phi-3 / 3.5 (fused QKV + gate-up)
│   │   │   └── examples/                # simple_generate, vanilla_sd_streaming
│   │   ├── tests/                       # integration tests, all #[ignore]'d
│   │   │   ├── with_qwen2_05b.rs        # Qwen 2.5 0.5B end-to-end
│   │   │   ├── with_tinyllama.rs        # TinyLlama 1.1B (Llama 2 arch)
│   │   │   ├── with_phi3_mini.rs        # Phi-3 mini 4k Instruct
│   │   │   ├── with_real_medusa_heads.rs        # FasterDecoding head .pt
│   │   │   └── with_real_medusa_e2e.rs          # Vicuna + Medusa E2E
│   │   └── Cargo.toml
│   └── speculate-cli/                   # bench binary
│       └── src/bin/bench.rs
├── scripts/                             # release helpers
│   └── convert_pth_to_safetensors.py    # uvx torch + safetensors converter
├── Cargo.toml                           # workspace
├── abyo_speculate_plan.md               # original strategy + risk register
├── ARCHITECTURE.md                      # this file
├── BENCHMARKS.md                        # reproducible measurements
├── BLOG_DRAFT.md                        # OSS launch post draft
└── CHANGELOG.md                         # release notes
```

## Layers, top to bottom

```
                    ┌────────────────────────────────────────┐
  Public API   →    │ SpeculateEngine builder / generate*()  │
                    └────────────────────────────────────────┘
                                       │
                    ┌────────────────────────────────────────┐
  Methods      →    │ vanilla.rs · medusa.rs* · eagle*.rs*   │
                    └────────────────────────────────────────┘
                       │              │             │
                       ▼              ▼             ▼
                    ┌────────┐  ┌────────────┐  ┌────────────┐
  Primitives   →    │tree.rs │  │ sampling/  │  │  cache/    │
                    │        │  │ tokens.rs  │  │ rollback.rs│
                    └────────┘  └────────────┘  └────────────┘
                                       │
                    ┌────────────────────────────────────────┐
  Decoder      →    │  Decoder trait (model/mod.rs)          │
                    └────────────────────────────────────────┘
                       ▲              ▲             ▲
                       │              │             │
                  MockDecoder   Qwen2Decoder   (Llama / etc — TODO)

* not yet implemented
```

## The `Decoder` trait

Everything above the Decoder layer is **tensor-free**: it talks in
`Vec<f32>` logit slabs and `&[u32]` token slices. Concrete decoders own
candle tensors and only materialize them at the trait boundary. This
gives us:

- A trivially-cheap mock for the SD-correctness test harness.
- A clean swap from `MockDecoder` to `Qwen2Decoder` without touching the
  generation loop.
- Unit-testable SD math that does not require any model weights.

Trait surface is intentionally narrow:

```rust
pub trait Decoder {
    fn vocab_size(&self) -> usize;
    fn history(&self) -> &[u32];
    fn reset(&mut self);
    fn observe(&mut self, ids: &[u32]) -> Result<()>;
    fn next_logits(&mut self) -> Result<Vec<f32>>;
    fn batched_logits(&mut self, drafts: &[u32]) -> Result<Vec<Vec<f32>>>;
    fn rollback_to(&mut self, len: usize) -> Result<()>;
}
```

Critical contract: `batched_logits(drafts)` advances the decoder's history
by `drafts` (matching real-model semantics where the parallel forward
mutates the KV cache). The SD loop uses `rollback_to` after each
verification round to discard whatever wasn't committed.

## Vanilla SD loop (the reference algorithm)

`methods::vanilla::run_vanilla_sd` is the **load-bearing correctness
implementation**. Every other SD method (Medusa, EAGLE) will be unit-tested
against it for distribution-matching, much as it itself is unit-tested
against analytic distributions via `MockDecoder`.

The loop, per round:

1. Snapshot `pre_target_len = target.history_len()` and the same for draft.
2. Draft `k` tokens from the draft model, one at a time.
3. Call `target.batched_logits(drafts)` — this returns `k+1` logit
   distributions (one per prefix length) from one parallel forward pass.
4. Walk the draft positions; for each apply Leviathan's modified rejection
   rule. On rejection at index `i`, sample a replacement from the
   adjusted distribution `norm(max(0, p_target - q_draft))` and stop.
5. If all `k` were accepted, sample one bonus token from
   `target_batched[k]`.
6. Roll **both** decoders back to their pre-round lengths and observe the
   committed prefix on top. After this both decoders again hold identical
   histories.

Steps 5 and 6 are subtle:
- The bonus token is the whole speedup story — without it, SD with
  always-accept-everything still costs the same as plain autoregressive.
- Re-anchoring both decoders is what keeps `next_logits` / `batched_logits`
  consistent between rounds.

### Statistical correctness tests

`methods::vanilla::tests` runs the loop 10–20 k times with controlled
target / draft mismatches and checks the empirical first-token
distribution against the analytic target by total-variation distance. The
tests cover:

- Uniform draft vs skewed target (worst acceptance rate, math still holds).
- Opposite-skewed draft vs target (strongest mismatch).
- Identical draft and target (sanity / always-accept).
- Target with zero-mass tokens (verifies the modified rejection rule
  never emits an unsupported token).

These are the *only* places in the crate where the SD math is asserted
end-to-end. If you change the loop, run them.

## KV-cache rollback

`cache::rollback::RollbackCache` is a per-layer snapshot/restore primitive
that tracks `committed_len` and `total_len`. Snapshot is O(1); rollback
truncates `total_len`. The intended usage pattern:

```text
snap = cache.snapshot();         // committed = C
cache.append(draft_kv);          // total = C + k
match outcome {
    AllAccepted => cache.commit(),
    Rejected(i) => { cache.rollback(snap); cache.append(prefix_through_i); cache.commit(); }
}
```

**Currently this primitive is not yet wired into `Qwen2Decoder`** —
Phase-1a uses `clear_kv_cache + replay` for correctness, which is
`O(history)` per rollback rather than `O(rollback_distance)`. Phase 1c
will route Qwen2's per-layer KV through `RollbackCache`. Until then,
real-model SD is correct but does not yet beat plain autoregressive on
wall-clock.

## Tree primitives (Phase 2a)

`tree::DraftTree` is a parent-pointer representation used by Medusa and
EAGLE. It exposes:

- `linear(root, tail)` — equivalent to vanilla SD's `k` consecutive drafts.
- `from_parent_table(nodes)` — branching trees with shared prefixes.
- `attention_mask_bool()` — the boolean ancestor-attendance matrix.
- `position_ids(prefix_len)` — per-node depth offsets for RoPE.
- `paths()` / `path_to(idx)` — root-to-leaf chains, used to commit a
  selected branch.

The tensor-side glue (Phase 2a tensor builders) is also in this module:

- `tree_self_bias(device, dtype)` — `[n, n]` additive bias (`0.0` /
  `-inf`) for self-attention among tree nodes.
- `full_attention_bias(prefix_len, device, dtype)` — `[n, prefix_len + n]`
  bias covering the committed prefix (all-attend) plus the tree.
- `full_attention_bias_4d(prefix_len, batch, head_dim_size, ...)` — the
  same expanded to `[b, h, n, prefix_len + n]` ready to drop into a
  candle attention layer.

> **Wiring into Qwen2 still requires a vendored model.** candle's
> `qwen2::Model::forward` accepts an `attn_mask` argument but its
> `prepare_attention_mask` only handles padding masks shaped `[b, seq]`,
> not the full `[b, 1, n, prefix_len + n]` bias we need. Phase 1c will
> vendor `qwen2.rs` into `model/qwen2_local.rs` so we can inject our mask
> directly. Until then, the tensor builders are tested standalone.

## Medusa primitives (Phase 1b)

`methods::medusa` is the multi-head SD path. Phase 1b ships:

- `MedusaConfig` / `MedusaHead` / `MedusaHeads` — structural metadata.
  The released vicuna-7b heads have `n_heads=4, hidden_size=4096`; that
  preset lives on `MedusaConfig::vicuna_7b_defaults`.
- `MedusaHeads::build_draft_tree(committed_root, head_top_k, topology)`
  with two topologies:
  - `Greedy` — each head's top-1 forms a linear chain (== vanilla SD).
  - `CartesianProduct` — every (cand_0, cand_1, ..., cand_{N-1})
    combination becomes a path.
- `top_k_indices(logits, k)` — stable-tie-break top-k helper.
- `run_medusa(target, heads, head_draft_fn, prompt, max_tokens, config)`
  — the reference loop (mirrors `vanilla::run_vanilla_sd`):
  1. Ask `head_draft_fn` for per-head top-`k` candidates.
  2. Build a `DraftTree`.
  3. For each tree node, fetch the target's next-token distribution
     given the path root→node (single forward in real Medusa; one
     observe+rollback per path in the mock reference).
  4. Walk every root-to-leaf path; greedily accept tokens while the
     `Acceptance` rule passes.
  5. Commit the longest accepted prefix + a bonus token from the
     deepest accepted node's distribution.
- Two acceptance rules supported:
  - `Acceptance::Greedy` — `argmax(p_target) == draft_token`.
  - `Acceptance::Typical { epsilon, delta }` — Cai §3.2:
    `accept iff p_target(x) >= max(epsilon, delta * exp(-H(p_target)))`.

What is **not** here yet (deliberately):
- The `Decoder` impl that runs a real target through Medusa heads (residual
  MLP + projection). Needs the head weights from a published checkpoint.
- HF download / loader for those checkpoints.
- Engine wiring — `engine.generate_tokens(... Method::Medusa)` still
  returns `UnsupportedMethod`. The plumbing change is mechanical once the
  loader lands.

## The `SpeculateEngine` façade

Two-stage construction:

1. `SpeculateEngine::builder()...build()` produces a config-only engine.
   Every parameter (target / draft model id, method, sampling config,
   seed) is captured here. No I/O happens.
2. `engine.with_target(decoder).with_draft(decoder)` attaches loaded
   decoders. Until both required decoders are attached, `is_ready()`
   returns `false` and `generate_tokens` errors with `MissingField`.

`generate_tokens` dispatches on `Method`:
- `Autoregressive` → `run_autoregressive` (plain sample loop).
- `Vanilla` → `run_vanilla_sd` against attached target+draft.
- `Medusa` / `Eagle*` → `UnsupportedMethod` error (placeholder until those
  methods land).

`generate(text, ...)` currently returns an explicit "use the lower-level
path" error — tokenization belongs to the concrete decoder
(`Qwen2Decoder::encode`), and we have not yet decided whether to expose
it through a `Backend` trait or to leave it to the caller. **Decision
deferred to Phase 1c.**

## What is intentionally left for follow-up sessions

## Status snapshot (v0.1.0)

| Item | Status | Notes |
|------|--------|-------|
| Vanilla SD (Leviathan 2023) | ✅ shipped | TV distance < 0.025 across 4 mismatch scenarios |
| Medusa skeleton + reference loop | ✅ shipped | Mock-validated against analytic distributions |
| Medusa real heads | ✅ shipped | Loader verified against `FasterDecoding/medusa-vicuna-7b-v1.3` |
| Vendored qwen2_local + Qwen2Decoder | ✅ shipped | Tree decoding + fast KV truncate |
| Vendored llama_local + LlamaDecoder | ✅ shipped | Llama 1/2/3.x; serves Mistral too |
| Vendored phi3_local + Phi3Decoder | ✅ shipped | Fused QKV + gate+up MLP |
| Vendored quantized_qwen2_local + Qwen2QuantDecoder | ✅ shipped | GGUF Q4/Q5/Q8 |
| `MultiPthBackend` for sharded `.bin` | ✅ shipped | Vicuna + bundled Medusa loadable |
| `engine.generate(text)` with EOS + streaming | ✅ shipped | `GenerationOptions` + callback |
| Bench CLI per-task / per-family | ✅ shipped | `--family auto|qwen2|llama|mistral|phi3` |
| Real Medusa speedup numbers | 🚧 in flight | `tests/with_real_medusa_e2e.rs` runs E2E; bench numbers next |
| Q4 speedup numbers | 📋 v0.2.0 | Plumbing complete; just needs published GGUF + integration test |
| EAGLE-2 (Li 2024) | 📋 v0.2.0 | Skeleton in `methods::eagle`; `run_eagle_real` returns `UnsupportedMethod` |
| EAGLE-3 (Li 2025) | 📋 v0.2.0 | Builds on EAGLE-2 |
| SAGUARO | 📋 v0.3.0 | Paper still pending verification (see plan §14) |

## Testing strategy

- `cargo test --workspace` runs **all** unit tests (43 at last commit);
  none require model weights or network. CI runs this on every push.
- Real-model integration tests will live in `crates/speculate/tests/`
  marked `#[ignore]`, runnable manually with `cargo test -- --ignored`.
  These require the user to pre-fetch a model into `models/` (path
  configurable via env var). None exist yet.
- Statistical correctness tests in `methods/vanilla.rs` use 10–20 k
  iterations and complete in ~30 ms in release mode. Do not lower the
  iteration counts — the TV-distance bounds were chosen to allow ~2σ
  margin against true convergence rates.

## Code style

- `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets
  -- -D warnings` must pass on every commit.
- No `unwrap()` outside of tests and `lib.rs` doctest examples.
- Module-level rustdoc explains *why* the module exists; per-item rustdoc
  explains *what* it does. Avoid repeating signatures.
- Comments inside fn bodies are reserved for non-obvious logic
  (commit-with-bonus-vs-rejection, rollback semantics, etc).
