#!/usr/bin/env python3
"""Convert a sharded PyTorch checkpoint to a single safetensors file.

Reads `{repo_dir}/pytorch_model.bin.index.json`, loads every referenced
shard, dedupes shared tensors, writes one `model.safetensors` next to the
input shards. Intended use: bridge HF repos that only ship `.bin` shards
(e.g. lmsys/vicuna-7b-v1.5, FasterDecoding/medusa-1.0-vicuna-7b-v1.5)
into safetensors so candle's `VarBuilder::from_mmaped_safetensors` can
load them.

Usage:
    uvx --with torch --with safetensors python convert_pth_to_safetensors.py \\
        ~/.cache/huggingface/hub/models--FasterDecoding--medusa-1.0-vicuna-7b-v1.5/snapshots/<commit>

Or via the standard HF cache snapshot directory; the script will find the
shards next to the index.

Output: writes `model.safetensors` (single file, may be large) and
`model.safetensors.index.json` (so candle's sharded-detector still works
for downstream tooling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch  # noqa: E402  (uvx-injected)
from safetensors.torch import save_file  # noqa: E402


def find_index(snapshot_dir: Path) -> Path:
    candidates = [
        snapshot_dir / "pytorch_model.bin.index.json",
        snapshot_dir / "model.bin.index.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"no PyTorch index json found under {snapshot_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot_dir", type=Path, help="HF cache snapshot dir")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output safetensors path (default: <snapshot_dir>/model.safetensors)",
    )
    args = p.parse_args()

    snapshot = args.snapshot_dir.expanduser().resolve()
    if not snapshot.is_dir():
        sys.exit(f"{snapshot} is not a directory")

    output = args.output or (snapshot / "model.safetensors")
    if output.exists():
        print(f"output already exists: {output} (skipping)")
        return 0

    index_path = find_index(snapshot)
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"loading {len(shard_files)} shard(s) from {snapshot}")

    state: dict[str, torch.Tensor] = {}
    for fn in shard_files:
        shard_path = snapshot / fn
        # Resolve symlink (HF cache uses blobs/.lock targets)
        shard_path = shard_path.resolve()
        print(f"  loading {fn} ...")
        sd = torch.load(shard_path, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k in state:
                # PyTorch checkpoints sometimes ship duplicates for tied weights;
                # safetensors disallows aliasing — drop the second copy.
                continue
            # safetensors requires contiguous tensors.
            if not v.is_contiguous():
                v = v.contiguous()
            state[k] = v

    print(f"  total tensors: {len(state)}")
    print(f"  writing {output} ...")
    save_file(state, str(output), metadata={"format": "pt"})

    # Drop a single-file index.json so downstream tooling that expects
    # sharded layout still resolves (each tensor → model.safetensors).
    out_index = output.with_suffix(".safetensors.index.json")
    new_weight_map = {k: output.name for k in state.keys()}
    total_size = sum(v.numel() * v.element_size() for v in state.values())
    out_index.write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": new_weight_map})
    )
    print(f"  wrote {out_index}")
    print(f"done. {output.stat().st_size / 1024**3:.2f} GiB on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
