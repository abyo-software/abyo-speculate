# abyo-speculate — Architecture

This document captures the design decisions made during the initial
build-out so future sessions can pick up cold. Treat it as a partner to
[`abyo_speculate_plan.md`](./abyo_speculate_plan.md) — the plan describes
*why* we are building this, this document describes *how* the code is
organised.

## Crate layout

```
abyo-speculate/
├── crates/
│   ├── speculate/                 # the library
│   │   ├── src/
│   │   │   ├── lib.rs             # public re-exports + module roster
│   │   │   ├── error.rs           # crate Error / Result
│   │   │   ├── device.rs          # CPU / CUDA / Metal selection
│   │   │   ├── engine.rs          # SpeculateEngine + builder + dispatch
│   │   │   ├── presets.rs         # Llama / Qwen / Mistral / Phi configs
│   │   │   ├── sampling/
│   │   │   │   ├── mod.rs
│   │   │   │   └── tokens.rs      # softmax / top-p / categorical sampler
│   │   │   ├── cache/
│   │   │   │   ├── mod.rs
│   │   │   │   └── rollback.rs    # snapshot / append / commit / rollback
│   │   │   ├── tree.rs            # DraftTree (Medusa / EAGLE shared)
│   │   │   ├── methods/
│   │   │   │   ├── mod.rs         # Method enum
│   │   │   │   └── vanilla.rs     # Leviathan SD reference impl
│   │   │   └── model/
│   │   │       ├── mod.rs         # Decoder trait
│   │   │       ├── loader.rs      # ModelSource (HF id / local path)
│   │   │       ├── mock.rs        # MockDecoder for tests
│   │   │       └── qwen2.rs       # Qwen 2 / 2.5 real-model impl
│   │   └── Cargo.toml
│   └── speculate-cli/             # bench + demo binaries
│       └── src/bin/{bench,demo}.rs
├── Cargo.toml                     # workspace
├── abyo_speculate_plan.md
└── ARCHITECTURE.md  (this file)
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

## Tree primitives (Phase 2a foundation)

`tree::DraftTree` is a parent-pointer representation used by Medusa and
EAGLE. It exposes:

- `linear(root, tail)` — equivalent to vanilla SD's `k` consecutive drafts.
- `from_parent_table(nodes)` — branching trees with shared prefixes.
- `attention_mask_bool()` — the boolean ancestor-attendance matrix.
- `position_ids(prefix_len)` — per-node depth offsets for RoPE.
- `paths()` / `path_to(idx)` — root-to-leaf chains, used to commit a
  selected branch.

This module is **tensor-free on purpose**. The Phase 2 model-side glue
turns `attention_mask_bool` into a `[1, 1, n, n]` `f32` bias added to
attention logits inside the model's forward pass. That glue lives in the
concrete decoder (Qwen2 first, then Llama).

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

| Phase | Item | Why deferred |
|-------|------|--------------|
| 1c | Wire `RollbackCache` into `Qwen2Decoder` | Need the full SD verify path before optimisation matters |
| 1c | `Backend` trait wrapping decoder + tokenizer | Wait until we have ≥2 model families to see the right shape |
| 1b | Medusa multi-head | Requires loading published Medusa heads; needs HF download helper |
| 2a | Tree-attention tensor glue | Wire `tree::DraftTree::attention_mask_bool` into a Qwen2 attention bias |
| 2b | EAGLE-2 dynamic tree construction | Research-grade implementation; needs careful study of the paper |
| 2c | EAGLE-3 multi-layer features | Builds on 2b |
| 3 | SAGUARO | Paper not yet verified (see plan §14 checklist) |
| n/a | HF-hub model download helper | Trivial to add when first needed |
| n/a | Real-GPU benchmarks | Needs the GPU restored + EAGLE-3 head HF availability |

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
