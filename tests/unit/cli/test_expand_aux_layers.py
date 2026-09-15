"""Widening a DFlash checkpoint's auxiliary-layer projection.

Pretraining learns from the embedding alone, so it runs with one auxiliary
layer and the checkpoint has to be widened before distillation can use it. The
widening has to be exact -- the learned columns land on the slot their layer
occupies in the new selection, and the slots the source never trained stay at
zero -- or distillation does not resume where pretraining stopped.
"""

from __future__ import annotations

import json

import pytest
import torch
import typer
from safetensors.torch import load_file, save_file

from speculators.cli.expand_aux_layers import expand_aux_layers

HIDDEN = 8


def _checkpoint(tmp_path, layer_ids, name="ckpt"):
    """A minimal checkpoint whose fc block for layer L is filled with L + 1."""
    path = tmp_path / name
    path.mkdir()
    fc = torch.cat(
        [torch.full((HIDDEN, HIDDEN), float(layer + 1)) for layer in layer_ids], dim=1
    )
    save_file(
        {"fc.weight": fc, "norm.weight": torch.ones(HIDDEN)},
        path / "model.safetensors",
        metadata={"format": "pt"},
    )
    (path / "config.json").write_text(
        json.dumps({"aux_hidden_state_layer_ids": list(layer_ids), "unrelated": 7})
    )
    return path


def _blocks(path):
    """Per-slot constant value of each fc block, and the saved layer ids."""
    fc = load_file(path / "model.safetensors")["fc.weight"]
    n = fc.shape[1] // HIDDEN
    values = [
        float(fc[:, i * HIDDEN : (i + 1) * HIDDEN].unique().item()) for i in range(n)
    ]
    ids = json.loads((path / "config.json").read_text())["aux_hidden_state_layer_ids"]
    return values, ids


def test_widening_puts_each_trained_block_on_its_new_slot(tmp_path):
    source = _checkpoint(tmp_path, [0])
    expand_aux_layers(source, [0, 18, 33], output=tmp_path / "wide", in_place=False)

    values, ids = _blocks(tmp_path / "wide")
    assert ids == [0, 18, 33]
    assert values == [1.0, 0.0, 0.0]  # layer 0 trained; new slots zero


def test_widening_is_not_limited_to_the_pretraining_case(tmp_path):
    """Any selection widens to a superset, so a sweep can share one base."""
    source = _checkpoint(tmp_path, [0, 18])
    expand_aux_layers(source, [0, 8, 18, 24], output=tmp_path / "wide", in_place=False)

    values, ids = _blocks(tmp_path / "wide")
    assert ids == [0, 8, 18, 24]
    assert values == [1.0, 0.0, 19.0, 0.0]


def test_in_place_rewrites_the_checkpoint(tmp_path):
    source = _checkpoint(tmp_path, [0])
    expand_aux_layers(source, [0, 5], output=None, in_place=True)

    values, ids = _blocks(source)
    assert ids == [0, 5]
    assert values == [1.0, 0.0]


def test_unrelated_weights_and_config_survive(tmp_path):
    source = _checkpoint(tmp_path, [0])
    expand_aux_layers(source, [0, 5], output=tmp_path / "wide", in_place=False)

    weights = load_file(tmp_path / "wide" / "model.safetensors")
    torch.testing.assert_close(weights["norm.weight"], torch.ones(HIDDEN))
    assert json.loads((tmp_path / "wide" / "config.json").read_text())["unrelated"] == 7


def test_dropping_a_trained_layer_is_refused(tmp_path):
    """Narrowing would silently discard learned columns."""
    source = _checkpoint(tmp_path, [0, 18])
    with pytest.raises(typer.BadParameter, match="18"):
        expand_aux_layers(source, [0, 33], output=tmp_path / "wide", in_place=False)


@pytest.mark.parametrize(
    ("output", "in_place"), [(None, False), ("wide", True)], ids=["neither", "both"]
)
def test_exactly_one_destination_is_required(tmp_path, output, in_place):
    source = _checkpoint(tmp_path, [0])
    with pytest.raises(typer.BadParameter, match="exactly one"):
        expand_aux_layers(
            source,
            [0, 5],
            output=(tmp_path / output) if output else None,
            in_place=in_place,
        )


def test_duplicate_layer_ids_are_refused(tmp_path):
    source = _checkpoint(tmp_path, [0])
    with pytest.raises(typer.BadParameter, match="Duplicate"):
        expand_aux_layers(source, [0, 5, 5], output=tmp_path / "wide", in_place=False)


def test_a_non_dflash_checkpoint_is_refused(tmp_path):
    path = tmp_path / "other"
    path.mkdir()
    save_file({"fc.weight": torch.zeros(HIDDEN, HIDDEN)}, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"speculators_model_type": "eagle3"}))
    with pytest.raises(typer.BadParameter, match="aux_hidden_state_layer_ids"):
        expand_aux_layers(path, [0, 5], output=tmp_path / "wide", in_place=False)
