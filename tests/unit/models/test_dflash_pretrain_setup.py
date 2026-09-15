"""Model setup for pretraining (DFlashDraftModel.prepare_for_pretraining).

Pretraining substitutes the draft's frozen embedding for the verifier's
layer-0 hidden state. Two things have to hold for that substitution to be
sound: the verifier must actually feed its first layer the unscaled
embedding, and the ``fc`` slots pretraining cannot reach must arrive at the
distillation run as exact zeros rather than random init -- that is what lets a
pretrained checkpoint be loaded without any weight surgery.
"""

from __future__ import annotations

import json

import pytest
import torch

from speculators.losses import eager
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dspark.core import DSparkDraftModel

from .test_checkpoint_key_ownership import VERIFIER_VOCAB, _make_model

FAMILY = [DFlashDraftModel, DSparkDraftModel, DFlash2DraftModel]


def _point_at_fake_verifier(model, tmp_path, model_type: str):
    verifier_dir = tmp_path / model_type
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    model.config.speculators_config.verifier.name_or_path = str(verifier_dir)
    return model


@pytest.mark.parametrize("model_cls", FAMILY, ids=lambda c: c.__name__)
def test_prepare_for_pretraining_zeroes_only_the_non_embedding_slots(
    model_cls, tmp_path
):
    model = _point_at_fake_verifier(
        _make_model(model_cls, VERIFIER_VOCAB), tmp_path, "qwen3"
    )
    hidden = model.hidden_size
    torch.nn.init.normal_(model.fc.weight)
    embedding_slot = model.fc.weight[:, :hidden].clone()

    model.prepare_for_pretraining()

    torch.testing.assert_close(model.fc.weight[:, :hidden], embedding_slot)
    assert torch.all(model.fc.weight[:, hidden:] == 0)


@pytest.mark.parametrize(
    ("model_type", "supported"),
    [
        ("qwen3", True),
        ("llama", True),
        ("deepseek_v3", True),
        ("gemma3_text", False),
        ("gemma4", False),
        ("granite", False),
    ],
)
def test_prepare_for_pretraining_rejects_verifiers_that_scale_embeddings(
    model_type, supported, tmp_path
):
    """A verifier that scales its embeddings before layer 0 would silently
    train the draft against mis-scaled features, so it is refused outright."""
    model = _point_at_fake_verifier(
        _make_model(DFlashDraftModel, VERIFIER_VOCAB), tmp_path, model_type
    )
    if supported:
        model.prepare_for_pretraining()
    else:
        with pytest.raises(ValueError, match="scales its"):
            model.prepare_for_pretraining()


def test_pretraining_requires_layer_zero_among_the_target_layers(tmp_path):
    model = _point_at_fake_verifier(
        _make_model(DFlashDraftModel, VERIFIER_VOCAB), tmp_path, "qwen3"
    )
    model.config.aux_hidden_state_layer_ids = [1, 2]
    with pytest.raises(ValueError, match="layer 0 must be"):
        model.prepare_for_pretraining()


def test_zeroed_slots_take_no_gradient_so_the_checkpoint_stays_an_identity(tmp_path):
    """The invariant the whole warm-start story rests on: pretraining feeds the
    non-embedding slots nothing, so they cannot drift off zero."""
    model = _point_at_fake_verifier(
        _make_model(DFlashDraftModel, VERIFIER_VOCAB), tmp_path, "qwen3"
    )
    model.prepare_for_pretraining()
    hidden, seq_len = model.hidden_size, 32

    _, loss, _ = model(
        input_ids=torch.randint(0, VERIFIER_VOCAB, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        max_anchors=4,
        training_mode="pretrain",
        loss_config={"ce": (eager.ce_loss, 1.0)},
    )
    loss.backward()

    assert model.fc.weight.grad[:, hidden:].abs().max() == 0
    assert model.fc.weight.grad[:, :hidden].abs().max() > 0
