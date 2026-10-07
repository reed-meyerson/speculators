#!/usr/bin/env python
"""Expand a DFlash-family draft checkpoint's fc input to new target layers.

Between token-only pretraining (``target_layer_ids: [0]``, hard-label CE, the
fc consumes the verifier's layer-0/embedding hidden state) and multi-layer
distillation (fc consumes the concatenation of several verifier layers), the
draft's ``fc.weight`` changes shape from ``[hidden, hidden]`` to
``[hidden, num_layers * hidden]``. This script zero-expands a stage-1
checkpoint so it can warm-start stage-2:

* every column block of the old fc lands at the position of its layer id in
  the NEW ``--new-target-layer-ids`` list (order matters: it must match the
  training config AND the verifier server's capture order);
* column blocks for layer ids not in the old checkpoint are **zeros**.

Because the forward is ``fc(hidden_states)`` with
``hidden_states = concat(h_id for id in target_layer_ids)``, the converted
model's fc output equals the old model's whenever the new input places the
old layers' hidden states in their blocks — the zero blocks contribute
nothing. So the converted model behaves **identically** to the stage-1
checkpoint on the same tokens, and stage-2 finetuning can grow weights into
the zeroed columns from a warm trunk.

The output directory keeps the checkpoint layout (``config.json`` +
``model.safetensors`` + ``config.py`` if present) and is loadable via
``--from-pretrained``. Optimizer/training state files are deliberately NOT
copied: the output is a fresh model, not a resumable run.

Usage:
    python scripts/expand_target_layers.py CHECKPOINT_DIR \
        --new-target-layer-ids 0 4 12 20 28 36 44 52 60 \
        --output OUT_DIR

Optionally set per-slot pre-fc input scales (entry 21) on the output config:
    ... --slot-scales 1.0 0.012 0.010 0.007 0.0038
aligned index-for-index with --new-target-layer-ids. When omitted, the
source config's aux_hidden_state_scales (if any) are propagated to their
slots' new positions and NEW slots default to 1.0. Function preservation
of the warm start holds when every OLD (nonzero) slot's scale is unchanged
(the typical case: layer 0 keeps 1.0, the reference RMS); the script warns
when an old slot's effective scale changes.

Note for the stage-2 run itself: include layer 0 in the new ids (that is
where the pretrained weights live) and launch the verifier hidden-states
server with the same id list (``launch_vllm.py --target-layer-ids ...``).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

FC_KEY = "fc.weight"


def expand_fc(
    old_weight: "torch.Tensor",  # noqa: F821 (torch imported lazily below)
    old_ids: list[int],
    new_ids: list[int],
    hidden_size: int,
) -> "torch.Tensor":
    """Zero-expand ``old_weight`` from ``old_ids`` columns to ``new_ids``."""
    import torch  # noqa: PLC0415

    if len(set(new_ids)) != len(new_ids):
        raise ValueError(f"Duplicate ids in new target layer ids: {new_ids}.")
    missing = [i for i in old_ids if i not in new_ids]
    if missing:
        raise ValueError(
            f"Old target layer ids {missing} are absent from the new ids "
            f"{new_ids}; dropping a pretrained layer would change model "
            f"behavior. New ids must be a superset of {old_ids}."
        )
    if new_ids == old_ids:
        print("New ids identical to old ids; copying unchanged.", file=sys.stderr)

    new_weight = torch.zeros(
        (old_weight.shape[0], len(new_ids) * hidden_size),
        dtype=old_weight.dtype,
    )
    for k, old_id in enumerate(old_ids):
        pos = new_ids.index(old_id)
        new_weight[:, pos * hidden_size : (pos + 1) * hidden_size] = old_weight[
            :, k * hidden_size : (k + 1) * hidden_size
        ]
    return new_weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="stage-1 checkpoint dir")
    parser.add_argument(
        "--new-target-layer-ids",
        type=int,
        nargs="+",
        required=True,
        help="new target layer ids, in concat order (must include every old id)",
    )
    parser.add_argument(
        "--slot-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "per-slot pre-fc input scales for the OUTPUT config, aligned "
            "index-for-index with --new-target-layer-ids (entry 21). "
            "Omitted: propagate the source config's scales (new slots 1.0)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="output dir")
    args = parser.parse_args()

    ckpt = args.checkpoint
    config_path = ckpt / "config.json"
    weights_path = ckpt / "model.safetensors"
    for path, what in ((config_path, "config"), (weights_path, "model weights")):
        if not path.is_file():
            if what == "model weights" and (ckpt / "model.safetensors.index.json").is_file():
                raise SystemExit(
                    f"{ckpt} contains a sharded checkpoint; this script only "
                    "handles single-file model.safetensors (the draft is "
                    "~3GB bf16, well under the 5GB shard threshold)."
                )
            raise SystemExit(f"No {what} found at {path}.")

    config = json.loads(config_path.read_text())
    old_ids = config["aux_hidden_state_layer_ids"]
    hidden_size = config["transformer_layer_config"]["hidden_size"]
    state_dict = load_file(weights_path)

    if FC_KEY not in state_dict:
        raise SystemExit(f"'{FC_KEY}' not found in {weights_path}.")
    old_weight = state_dict[FC_KEY]
    expected_in = len(old_ids) * hidden_size
    if old_weight.shape != (hidden_size, expected_in):
        raise SystemExit(
            f"{FC_KEY} has shape {tuple(old_weight.shape)}, expected "
            f"({hidden_size}, {expected_in}) from aux_hidden_state_layer_ids="
            f"{old_ids} and hidden_size={hidden_size}."
        )

    state_dict[FC_KEY] = expand_fc(old_weight, old_ids, args.new_target_layer_ids, hidden_size)

    # Resolve output scales (entry 21): explicit --slot-scales win; else
    # propagate the source's scales with new slots defaulting to 1.0.
    old_scales = config.get("aux_hidden_state_scales")
    if args.slot_scales is not None:
        if len(args.slot_scales) != len(args.new_target_layer_ids):
            raise SystemExit(
                f"--slot-scales has {len(args.slot_scales)} values but "
                f"--new-target-layer-ids has {len(args.new_target_layer_ids)}; "
                "they must align index-for-index."
            )
        new_scales = list(args.slot_scales)
    elif old_scales is not None:
        if len(old_scales) != len(old_ids):
            raise SystemExit(
                f"source config aux_hidden_state_scales has {len(old_scales)} "
                f"entries but aux_hidden_state_layer_ids has {len(old_ids)}."
            )
        new_scales = [
            old_scales[old_ids.index(i)] if i in old_ids else 1.0
            for i in args.new_target_layer_ids
        ]
    else:
        new_scales = None

    if new_scales is not None:
        # Warm-start function preservation: an old slot's fc block was
        # trained under its source effective scale (1.0 when the source had
        # no scales). If the output rescales that slot, the warm start is
        # NOT function-preserving on it (fine for zero-expansion slots,
        # questionable for pretrained ones — see entry 21's caveat).
        for old_id in old_ids:
            src_eff = old_scales[old_ids.index(old_id)] if old_scales else 1.0
            out_eff = new_scales[args.new_target_layer_ids.index(old_id)]
            if out_eff != src_eff:
                print(
                    f"WARNING: old slot layer {old_id} effective scale changes "
                    f"{src_eff} -> {out_eff}; the warm start is NOT "
                    "function-preserving on this slot's pretrained fc block.",
                    file=sys.stderr,
                )

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, args.output / "model.safetensors")
    config["aux_hidden_state_layer_ids"] = list(args.new_target_layer_ids)
    if new_scales is not None:
        config["aux_hidden_state_scales"] = new_scales
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    if (ckpt / "config.py").is_file():
        shutil.copy(ckpt / "config.py", args.output / "config.py")

    n_new = len(args.new_target_layer_ids)
    print(
        f"Expanded {FC_KEY}: {tuple(old_weight.shape)} -> "
        f"({hidden_size}, {n_new * hidden_size}) "
        f"(old ids {old_ids} -> new ids {list(args.new_target_layer_ids)}); "
        f"scales={new_scales if new_scales is not None else 'none'}; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
