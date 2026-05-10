#!/usr/bin/env bash
# Run the standard v0.3.0 benchmark suite. Reproduces the numbers in
# README.md / BENCHMARKS.md.
#
# Requirements:
#   - NVIDIA GPU with ≥ 16 GB VRAM and CUDA 12.x.
#   - Cargo with `cuda` feature working.
#   - First run downloads ~7 GB of Qwen 2.5 weights and (optionally) ~15 GB
#     for the BF16 EAGLE test.
#
# Usage:
#   scripts/run_benchmarks.sh                   # vanilla SD across 4 tasks
#   scripts/run_benchmarks.sh --with-eagle-bf16 # also run EAGLE-2 BF16 e2e
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

with_eagle_bf16=0
for arg in "$@"; do
  case "$arg" in
    --with-eagle-bf16) with_eagle_bf16=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=== Vanilla SD (Qwen 2.5 3B + 0.5B, k=4, BF16, 128 tokens, 1 warmup + 3 runs) ==="
for task in chat code translation long-context; do
  echo
  echo "--- task: $task ---"
  cargo run --release --features cuda --quiet --bin abyo-speculate-bench -- \
    --target Qwen/Qwen2.5-3B-Instruct \
    --draft  Qwen/Qwen2.5-0.5B-Instruct \
    --method both --task "$task" \
    --max-tokens 128 --warmup 1 --runs 3 --draft-lookahead 4
done

if [[ "$with_eagle_bf16" -eq 1 ]]; then
  echo
  echo "=== EAGLE-2 BF16 e2e (Llama-2-7B-Chat + EAGLE-llama2-chat-7B, ~15 GB) ==="
  cargo test --release --features cuda \
    -p abyo-speculate --test with_eagle_bf16_e2e -- \
    --ignored --nocapture --test-threads=1
fi

echo
echo "Done. See BENCHMARKS.md for context on each row."
