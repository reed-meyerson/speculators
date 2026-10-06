#!/usr/bin/env python
"""Estimate per-layer RMS norms of verifier hidden states from a captured pool.

CPU-only (lazy slice reads; never loads a whole file, never touches CUDA).
Draws one seeded, fixed set of (file, contiguous-position-block) tokens and
computes, for every captured layer id, the RMS norm of the hidden-state
vector over that SAME token set — so layer-to-layer comparisons use
identical tokens.

Pool file layout (see pool_manifest.json + train/data.py):
  hidden_states: [seq_len, num_slots, hidden_size]  (bf16, slots in
                 manifest layer_ids order)
  token_ids:     [seq_len]

Usage:
  python estimate_layer_rms.py [--pool-dir DIR] [--num-tokens 1000]
      [--tokens-per-file 20] [--seed 42] [--output out.json]

Reports, per layer id: RMS = sqrt(mean x^2) over all sampled (token, dim)
elements, the mean per-token L2 norm (RMS * sqrt(hidden) for uniform data,
but reported independently), and the ratio vs layer 0.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open

MANIFEST = "pool_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool-dir",
        type=Path,
        default=Path(
            "/data/playground/reed-meyerson/layer-id-search/hidden_states"
        ),
        help="hidden-states pool directory (contains pool_manifest.json)",
    )
    p.add_argument(
        "--num-tokens", type=int, default=1000,
        help="approximate total tokens to sample (default 1000)",
    )
    p.add_argument(
        "--tokens-per-file", type=int, default=20,
        help="contiguous tokens read per sampled file (default 20)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON output path for the results table",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    manifest = json.loads((args.pool_dir / MANIFEST).read_text())
    layer_ids = [int(x) for x in manifest["layer_ids"]]
    num_slots = int(manifest["num_slots"])
    if len(layer_ids) != num_slots:
        raise ValueError(
            f"manifest layer_ids ({len(layer_ids)}) != num_slots ({num_slots})"
        )

    files = sorted(args.pool_dir.glob("hs_*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no hs_*.safetensors under {args.pool_dir}")
    rng.shuffle(files)

    target = args.num_tokens
    per_file = args.tokens_per_file

    # Per-slot accumulators (float64 for a stable sum of squares).
    sumsq = torch.zeros(num_slots, dtype=torch.float64)
    per_token_l2_sum = torch.zeros(num_slots, dtype=torch.float64)
    n_tokens = 0
    n_files = 0

    for path in files:
        if n_tokens >= target:
            break
        with safe_open(str(path), framework="torch") as f:
            sl = f.get_slice("hidden_states")
            seq_len = sl.get_shape()[0]
            k = min(per_file, seq_len, target - n_tokens)
            if k <= 0:
                continue
            start = rng.randrange(seq_len - k + 1)
            block = sl[start : start + k]  # [k, num_slots, hidden] bf16
        block = block.to(torch.float32)  # [k, slots, hidden]
        sumsq += (block.double() ** 2).sum(dim=(0, 2))
        per_token_l2_sum += block.float().norm(dim=2).double().sum(dim=0)
        n_tokens += k
        n_files += 1

    if n_tokens == 0:
        raise RuntimeError("sampled zero tokens")

    hidden = block.shape[2]
    rms = (sumsq / (n_tokens * hidden)).sqrt()
    mean_l2 = per_token_l2_sum / n_tokens

    # Report in layer-id order (slots are already manifest-ordered, which is
    # ascending here, but sort defensively).
    order = sorted(range(num_slots), key=lambda i: layer_ids[i])
    rms0 = rms[order[0]].item()  # reference: lowest layer id (0)

    width = 44
    max_rms = max(rms[i].item() for i in order)
    lines = []
    lines.append(
        f"# pool={args.pool_dir}  tokens={n_tokens}  files={n_files}  "
        f"seed={args.seed}  hidden={hidden}"
    )
    lines.append(f"{'layer':>5}  {'RMS':>10}  {'mean L2':>10}  {'vs L0':>7}  norm")
    for i in order:
        r = rms[i].item()
        bar = "#" * max(1, round(width * r / max_rms))
        lines.append(
            f"{layer_ids[i]:>5}  {r:>10.4f}  {mean_l2[i].item():>10.2f}  "
            f"{r / rms0:>7.2f}x  {bar}"
        )
    report = "\n".join(lines)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "pool_dir": str(args.pool_dir),
                    "seed": args.seed,
                    "num_tokens": n_tokens,
                    "num_files": n_files,
                    "hidden_size": hidden,
                    "layer_ids": layer_ids,
                    "rms": {layer_ids[i]: rms[i].item() for i in order},
                    "mean_token_l2": {
                        layer_ids[i]: mean_l2[i].item() for i in order
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
