# abyo-speculate v0.4 — what we shipped, and the day we cut our own headline number in half

*Pure Rust Speculative Decoding library. v0.4.x release notes /
post-mortem. Edit before posting to HN / r/LocalLLaMA / r/rust.*

---

## The TL;DR you'd want from any honest crate

[abyo-speculate](https://github.com/abyo-software/abyo-speculate) is a
Pure Rust library for [Speculative
Decoding](https://arxiv.org/abs/2211.17192) — the technique where a
small "draft" model proposes several tokens and a big "target" model
verifies them in one forward pass.

We started v0.1.0 quoting **1.42–1.76× speedup** for Vanilla SD on
Qwen 2.5 3B + 0.5B BF16. Through v0.2 / v0.3 we added Medusa, EAGLE-2,
EAGLE-3, EOS, BF16 Llama, KV-reorder fast paths.

In v0.4.1, while chasing a different optimization, we discovered that
**our autoregressive baseline was running 2× slower than it should
have been** — every `next_logits` call was redundantly re-forwarding
the last committed token whose logits had already been computed in the
preceding `observe`. Fixing it doubled AR throughput everywhere.

That fix also exposed the truth: **most of the SD speedup we'd been
quoting was an artifact of a slow AR baseline.** Re-measuring against
the now-honest AR:

| Task | Old AR | Old SD | "Speedup" | **New AR** | **New SD** | **Real ratio** |
|------|-------:|-------:|----------:|-----------:|-----------:|---------------:|
| chat | 34.4 | 49.5 | 1.44× | **67.0** | 64.6 | **0.96×** |
| **code** | 35.3 | 61.4 | 1.74× | **67.4** | 76.1 | **1.13×** |
| translation | 34.1 | 48.0 | 1.41× | **65.3** | 59.6 | **0.91×** |
| long_context | 31.5 | 49.6 | 1.57× | **62.4** | 48.5 | **0.78×** |

So the only task that beats AR on a properly-optimized baseline is
code generation. The rest break even or lose.

We could have buried this. Instead we wrote a release note titled
**"honest pivot,"** put the new numbers in `README.md`,
`BENCHMARKS.md`, and `CHANGELOG.md`, and shipped v0.4.1 the same day.

## What changed

- `Decoder::next_logits` now returns logits cached as a side effect of
  the prior `observe([tok])` instead of truncating the KV cache and
  re-forwarding the last token. Bit-identical output, ~2× the
  throughput. Applied to every decoder (Llama BF16, LlamaQuant,
  Qwen2, Qwen2Quant, Phi3).
- Tried the analogous trick for `batched_logits` (vanilla SD's
  verification path). It **lowered** SD acceptance by 25-35% because
  the GEMV-cached `last` KV doesn't match the GEMM-batched re-compute
  the drafts attend against. Reverted with explanatory comments —
  failed-experiment trail kept in git history because the surface
  temptation is obvious.
- `EagleRunConfig::strict_root_gemv` toggle (default off) restores the
  v0.2.2 GEMV root-fix: guarantees the EAGLE trajectory matches AR
  exactly under greedy acceptance, at the cost of fewer accepted
  draft tokens per round.
- `tests/with_qwen2_q4_cross_dtype.rs` measures the Q4 7B target — it
  hits 99 tok/s AR (faster than BF16 3B at 67 tok/s) because Q4
  weights are 4× smaller and inference is memory-bandwidth-bound.
  SD's structural overhead doesn't fit under that ceiling.

## Why we shipped the bad numbers anyway

The crate's value proposition pivoted: **"correct, reference
implementations of SD methods + honest measurement infrastructure"**
is what we deliver, not "drop-in 1.7× speedup." That's a smaller
claim, but it's a true one. The implementations are:

- Vanilla SD with the Leviathan modified-rejection rule,
  statistically validated (TV distance < 0.025 across four mismatch
  scenarios).
- Medusa with both Greedy and CartesianProduct topology, real
  `FasterDecoding/medusa-*` checkpoint loader.
- EAGLE-2 with KV-reorder fast path (4 → 2 target forwards/round)
  against `yuhuili/EAGLE-llama2-chat-7B`.
- EAGLE-3 with multi-layer feature fusion + d2t/t2d translation
  against `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`.

For users who want a known-correct SD baseline to layer their own
optimizations on top of — or to measure a new acceleration framework
against — the crate has clear value. For users who want a turnkey
"3× faster locally" experience, candle 0.8 doesn't have the kernel
fusion and Flash Attention support the published EAGLE paper relies
on. We'd need to either contribute that to candle or migrate to a
different inference backend.

## The structural ceiling

We measured EAGLE-3 BF16 on a 24 GB EC2 L4 with the published
`Llama 3.1 8B Instruct` + `EAGLE3-LLaMA3.1-Instruct-8B` checkpoints
— the configuration the EAGLE-2 paper claims 2-3× on. We measured
**0.62×.** The implementation is reference-correct (output matches
greedy AR byte-for-byte under strict mode); candle's per-step
forward is just slow enough that EAGLE's per-round overhead can't be
amortized.

Where SD currently wins:

- Code generation (high acceptance rate amortizes the per-round
  overhead).
- Long-context inference where prompt processing dominates (not
  measured in this release).
- BF16 inference on specialty kernels candle 0.8 doesn't yet have.

Where SD currently doesn't help:

- Chat / translation / general dialogue on consumer GPUs.
- Q4 quantized targets: the per-step is too cheap to leave SD any
  budget.
- 7B MHA models (Llama 2): MHA per-step is already cheap, no headroom.

## Why we built this anyway

The Rust ecosystem has no integrated SD library; that gap is real
even if the speedup isn't always there. abyo-speculate is the place
the speedup *will* exist when candle adds Flash Attention or when
someone plugs us into a faster inference framework. The
infrastructure — `DraftTree`, `TreeDecoder` trait,
`commit_tree_path`, `observe_returning_last_hidden`, the Qwen / Llama
/ Phi-3 vendored adapters with tree-attention extensions — is the
actually-hard part. Layering a faster forward underneath it is the
mechanical part.

## Get started

```sh
cargo add abyo-speculate
```

```sh
cargo install --git https://github.com/abyo-software/abyo-speculate \
  abyo-speculate-cli
abyo-speculate-bench --target Qwen/Qwen2.5-3B-Instruct \
                     --draft  Qwen/Qwen2.5-0.5B-Instruct \
                     --method both --task code
```

Worked example for EAGLE-2 BF16:
`cargo run --release --features cuda --example eagle2_bf16`

## Links

- [GitHub](https://github.com/abyo-software/abyo-speculate)
- [Architecture](https://github.com/abyo-software/abyo-speculate/blob/main/ARCHITECTURE.md)
- [Benchmarks (with the unfortunate honesty pivot)](https://github.com/abyo-software/abyo-speculate/blob/main/BENCHMARKS.md)
- [CHANGELOG](https://github.com/abyo-software/abyo-speculate/blob/main/CHANGELOG.md)
