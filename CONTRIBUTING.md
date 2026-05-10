# Contributing to abyo-speculate

Thanks for your interest! This is a single-maintainer project at the
moment, so please open an issue *before* sending a non-trivial PR — it
saves both of us the round-trip if the change overlaps with in-flight
work or doesn't fit the scope.

## Scope reminder

abyo-speculate is intentionally **batch-size-1, single-user, local-LLM
focused.** The competitive frame is `llama.cpp` / `ollama` / single-user
inference loops embedded in Rust apps. Out of scope: multi-batch
serving (vLLM / SGLang territory), continuous batching, MoE expert
acceleration, non-Hugging-Face checkpoints, GGUF inference outside the
Llama / Qwen 2 families.

## Local development

```sh
git clone https://github.com/abyo-software/abyo-speculate
cd abyo-speculate
cargo build --release            # CPU-only (default)
cargo test --release --lib       # 74 unit + statistical tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

GPU paths require:

- `--features cuda` on Linux + NVIDIA (CUDA toolkit 12.x).
- `--features metal` on Apple Silicon.

Most integration tests are gated under `#[ignore]` because they
download multi-GB checkpoints and run on real GPUs. Run a specific one
with:

```sh
cargo test --release --features cuda \
    -p abyo-speculate --test with_eagle_bf16_e2e -- --ignored --nocapture
```

## What to know before opening an issue

- **Numbers without a reproduction script are not actionable.** The
  bench CLI emits a single JSON line on stdout suitable for jq /
  comparison. Please include it for any "X is slower than expected" /
  "X.x× speedup is wrong" report.
- **EAGLE on Q4 targets is a known low-acceptance regime** — see
  `BENCHMARKS.md`. Bug reports asking why EAGLE-3 + Q4 doesn't beat AR
  will get pointed at that section.
- **Greedy acceptance is the correctness reference**, not absolute
  output text. Two SD methods can both be "correct" yet emit different
  text under any temperature > 0 — only the long-run distribution must
  match the target's.

## Code layout

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the per-module design and
why each model family has both a `*.rs` (`Decoder` impl) and a
`*_local.rs` (vendored model with tree-attention extensions).

## Commit message convention

- One change per commit; group related changes into a PR.
- Subject ≤ 72 chars, imperative mood ("Fix X" not "Fixed X").
- Body: explain *why*, not *what* (the diff already shows what).

## Releases

Releases follow SemVer at the workspace level (`Cargo.toml`'s
`workspace.package.version`). For a release:

1. Run the bench CLI on Qwen 2.5 3B + 0.5B across all 4 tasks; update
   `BENCHMARKS.md` if numbers materially shifted.
2. Update `CHANGELOG.md` with the released version section.
3. Bump version, commit, tag (`v0.x.y`), `cargo publish -p abyo-speculate`,
   `git push origin main && git push origin v0.x.y`.

## License

By contributing you agree your work will be licensed under the
project's dual MIT / Apache-2.0 license.
