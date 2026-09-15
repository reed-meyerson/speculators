"""
Widen a DFlash-family checkpoint to consume more auxiliary hidden-state layers.

Pretraining has only the verifier's input embedding to learn from, so it runs
with a single auxiliary layer (``--target-layer-ids 0``) and produces an ``fc``
projection sized for one. Distillation consumes several. This command widens
the projection to the requested selection, leaving the new slots at zero.

Zeroed slots contribute nothing on the first step, so distillation resumes from
exactly the projection pretraining learned and grows the auxiliary contributions
from there. ``fc`` is the only weight whose shape depends on the layer count;
everything else is copied through untouched.

Because the pretrained weights depend on the embedding alone, one pretrained
base can be widened to any selection and any count -- which is what makes a
layer-selection sweep share a single pretraining run.

Usage::

    speculators expand-aux-layers ./pretrained 0 18 33 --output ./widened

    # overwrite the checkpoint instead of writing a copy:
    speculators expand-aux-layers ./pretrained 0 18 33 --in-place
"""

import json
import shutil
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from safetensors.torch import load_file, save_file

console = Console()

__all__ = ["expand_aux_layers"]

_WEIGHTS = "model.safetensors"
_CONFIG = "config.json"
_FC = "fc.weight"


def _widen_fc(
    fc: torch.Tensor, source_ids: list[int], target_ids: list[int]
) -> torch.Tensor:
    """Place each source layer's columns at its slot in ``target_ids``.

    Columns for layers the source does not carry are left at zero.
    """
    hidden_size = fc.shape[0]
    if fc.shape[1] != len(source_ids) * hidden_size:
        raise typer.BadParameter(
            f"{_FC} is {tuple(fc.shape)}, which is not "
            f"{len(source_ids)} blocks of {hidden_size} columns as its config's "
            f"aux_hidden_state_layer_ids={source_ids} implies."
        )
    widened = torch.zeros(
        (hidden_size, len(target_ids) * hidden_size), dtype=fc.dtype, device=fc.device
    )
    for source_slot, layer in enumerate(source_ids):
        target_slot = target_ids.index(layer)
        widened[:, target_slot * hidden_size : (target_slot + 1) * hidden_size] = fc[
            :, source_slot * hidden_size : (source_slot + 1) * hidden_size
        ]
    return widened


def expand_aux_layers(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="DFlash-family checkpoint to widen (usually pretrained)."),
    ],
    target_layer_ids: Annotated[
        list[int],
        typer.Argument(
            help="Verifier layer ids the widened checkpoint should consume. Must "
            "include every layer the source already carries."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the widened checkpoint here."),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option("--in-place", help="Overwrite the checkpoint instead."),
    ] = False,
):
    """Widen a checkpoint's ``fc`` projection to more auxiliary layers."""
    if (output is None) == (not in_place):
        raise typer.BadParameter("Pass exactly one of --output or --in-place.")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise typer.BadParameter(f"Duplicate layer ids: {target_layer_ids}")

    config_path = checkpoint / _CONFIG
    if not config_path.is_file():
        raise typer.BadParameter(f"No {_CONFIG} in {checkpoint}")
    config = json.loads(config_path.read_text())
    source_ids = config.get("aux_hidden_state_layer_ids")
    if source_ids is None:
        raise typer.BadParameter(
            f"{checkpoint} has no aux_hidden_state_layer_ids; it is not a "
            "DFlash-family checkpoint."
        )

    missing = sorted(set(source_ids) - set(target_layer_ids))
    if missing:
        raise typer.BadParameter(
            f"The checkpoint carries trained weights for verifier layer(s) "
            f"{missing}, which the requested selection {target_layer_ids} drops. "
            "Those weights would be discarded."
        )

    weights = load_file(checkpoint / _WEIGHTS)
    if _FC not in weights:
        raise typer.BadParameter(f"No {_FC} in {checkpoint / _WEIGHTS}")
    before = tuple(weights[_FC].shape)
    weights[_FC] = _widen_fc(weights[_FC], list(source_ids), list(target_layer_ids))
    config["aux_hidden_state_layer_ids"] = list(target_layer_ids)

    destination = checkpoint if in_place else output
    assert destination is not None  # noqa: S101 -- guarded above
    if not in_place:
        shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
    save_file(weights, destination / _WEIGHTS, metadata={"format": "pt"})
    (destination / _CONFIG).write_text(json.dumps(config, indent=2) + "\n")

    console.print(
        f"Widened [bold]{_FC}[/bold] {before} -> {tuple(weights[_FC].shape)}\n"
        f"  aux layers: {list(source_ids)} -> {list(target_layer_ids)}\n"
        f"  written to: {destination}"
    )
