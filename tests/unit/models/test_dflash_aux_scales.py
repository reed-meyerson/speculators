"""Entry-21 per-slot pre-fc input scaling (``aux_hidden_state_scales``).

The distillation branch of ``_backbone_forward`` scales each aux slot's
hidden state before the fc; the token-only (embedding) branch and the
verifier-side target construction are untouched. Scales align
index-for-index with ``aux_hidden_state_layer_ids`` in CONFIG order.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel

HIDDEN = 16
REPO_ROOT = Path(__file__).parents[3]
EXPAND_SCRIPT = REPO_ROOT / "scripts" / "expand_target_layers.py"

SPECULATORS_CONFIG = {
    "algorithm": "dflash",
    "default_proposal_method": "greedy",
    "proposal_methods": [
        {
            "accept_tolerance": 0.0,
            "proposal_type": "greedy",
            "speculative_tokens": 8,
            "verifier_accept_k": 1,
        }
    ],
    "verifier": {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "name_or_path": "Qwen/Qwen3.8-27B",
    },
}


def _tiny_config(aux_ids, scales=None) -> DFlashSpeculatorConfig:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    return DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=aux_ids,
        aux_hidden_state_scales=scales,
        mask_token_id=0,
        speculators_config=SPECULATORS_CONFIG,
    )


def _tiny_model(aux_ids, scales=None) -> DFlashDraftModel:
    model = DFlashDraftModel(_tiny_config(aux_ids, scales))
    # Verifier-owned weights are left uninitialized by __init__ (reconstructed
    # from the verifier on real loads); init them so the draft path computes.
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


def _run_backbone(model, hidden_states, hard_targets=False):
    seq_len = 32
    torch.manual_seed(0)  # identical anchors/input_ids/verifier states per call
    input_ids = torch.randint(0, 64, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        return model._backbone_forward(
            input_ids=input_ids,
            loss_mask=loss_mask,
            document_ids=document_ids,
            hidden_states=hidden_states,
            verifier_last_hidden_states=torch.randn(1, seq_len, HIDDEN),
            max_anchors=5,
            hard_targets=hard_targets,
        )


def test_distill_path_applies_scales_in_config_order():
    """Scaled model == unscaled model fed manually pre-scaled inputs.

    Unsorted aux ids [4, 0] pin the config-order (not sorted) alignment:
    scales [0.5, 1.0] multiply slot blocks 0 (layer 4) and 1 (layer 0)
    respectively.
    """
    torch.manual_seed(1)
    scaled = _tiny_model([4, 0], scales=[0.5, 1.0])
    plain = _tiny_model([4, 0])
    plain.load_state_dict(scaled.state_dict())

    torch.manual_seed(2)
    hs = torch.randn(1, 32, 2 * HIDDEN)

    out_scaled = _run_backbone(scaled, hs)
    out_plain = _run_backbone(plain, hs.clone())
    # sanity: without scaling the outputs differ
    assert not torch.allclose(out_scaled[1], out_plain[1])

    manual = hs.clone()
    manual[:, :, :HIDDEN] *= 0.5  # slot 0 (layer 4) in config order
    out_manual = _run_backbone(plain, manual)
    torch.testing.assert_close(out_scaled[1], out_manual[1])  # logits
    torch.testing.assert_close(out_scaled[2], out_manual[2])  # targets


def test_token_only_path_unaffected_by_scales():
    """The token-only branch feeds embeddings straight into the fc; even a
    non-unity scale on the single slot must not change it. (Token-only is
    structurally single-slot: the fc in-features match the embedding dim.)"""
    torch.manual_seed(1)
    scaled = _tiny_model([0], scales=[2.0])
    plain = _tiny_model([0])
    plain.load_state_dict(scaled.state_dict())

    out_scaled = _run_backbone(scaled, None, hard_targets=True)
    out_plain = _run_backbone(plain, None, hard_targets=True)
    torch.testing.assert_close(out_scaled[1], out_plain[1])  # logits
    torch.testing.assert_close(out_scaled[2], out_plain[2])  # targets


def test_verifier_targets_untouched():
    """Targets come from verifier_last_hidden_states only — identical across
    models with different slot orders/scales given the same RNG stream
    (anchor sampling is stochastic; seed per call)."""
    torch.manual_seed(1)
    scaled = _tiny_model([4, 0], scales=[0.5, 1.0])
    plain = _tiny_model([0, 4])  # different slot order entirely
    plain.load_state_dict(scaled.state_dict())

    torch.manual_seed(2)
    hs = torch.randn(1, 32, 2 * HIDDEN)

    seq_len = 32
    kw = dict(
        input_ids=torch.randint(0, 64, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        verifier_last_hidden_states=torch.randn(1, seq_len, HIDDEN),
        max_anchors=5,
    )
    with torch.no_grad():
        torch.manual_seed(123)
        _, _, t_scaled, _, idx = scaled._backbone_forward(hidden_states=hs, **kw)
        torch.manual_seed(123)
        _, _, t_plain, _, idx2 = plain._backbone_forward(
            hidden_states=hs.flip(-1), **kw
        )
    assert torch.equal(idx, idx2)
    torch.testing.assert_close(t_scaled, t_plain)


def test_config_validator_rejects_misaligned_scales():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="aligned index-for-index"):
        _tiny_config([0, 4], scales=[1.0])
    with pytest.raises(ValidationError, match="strictly positive"):
        _tiny_config([0, 4], scales=[1.0, -0.5])
    base = _tiny_config([0, 4]).model_dump()
    base.pop("speculators_config")  # avoid None round-trip of the sub-config
    with pytest.raises(ValidationError, match="requires aux_hidden_state_layer_ids"):
        DFlashSpeculatorConfig(
            **{**base, "aux_hidden_state_layer_ids": None,
               "aux_hidden_state_scales": [1.0]}
        )


def test_scale_tensor_cache_and_meta_init_survival():
    """The scale tensor is config-derived and cached, NOT a registered buffer:
    transformers 5.x from_pretrained constructs models under a
    torch.device("meta") context; tensors created in __init__ land on meta
    and are later materialized as UNINITIALIZED memory for anything absent
    from the checkpoint (non-persistent buffers are, by design). Reading
    from the config at forward time must survive that path."""
    scaled = _tiny_model([0, 4], scales=[1.0, 0.5])
    assert "aux_input_scales" not in scaled.state_dict()
    t = scaled._aux_scale_tensor(torch.device("cpu"), torch.float32)
    assert t.shape == (1, 1, 2, 1)
    torch.testing.assert_close(t.flatten(), torch.tensor([1.0, 0.5]))
    assert scaled._aux_scale_tensor(torch.device("cpu"), torch.float32) is t
    plain = _tiny_model([0, 4])
    assert plain._aux_scale_tensor(torch.device("cpu"), torch.float32) is None

    # The from_pretrained failure mode: meta-context init + to_empty garbage.
    with torch.device("meta"):
        meta_model = DFlashDraftModel(_tiny_config([0, 4], scales=[1.0, 0.5]))
    meta_model.to_empty(device="cpu")
    t2 = meta_model._aux_scale_tensor(torch.device("cpu"), torch.float32)
    torch.testing.assert_close(t2.flatten(), torch.tensor([1.0, 0.5]))


def test_config_round_trip_via_pretrained(tmp_path):
    config = _tiny_config([0, 36, 44], scales=[1.0, 0.012, 0.01])
    config.save_pretrained(tmp_path)
    loaded = DFlashSpeculatorConfig.from_pretrained(tmp_path)
    assert loaded.aux_hidden_state_scales == [1.0, 0.012, 0.01]
    assert loaded.aux_hidden_state_layer_ids == [0, 36, 44]


def _write_tiny_checkpoint(tmp_path, ids, fc_weight):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "speculators_model_type": "dflash",
                "architectures": ["DFlashSpeculator"],
                "transformer_layer_config": {
                    "model_type": "qwen3",
                    "hidden_size": HIDDEN,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "vocab_size": 64,
                },
                "draft_vocab_size": 64,
                "block_size": 4,
                "aux_hidden_state_layer_ids": ids,
                "mask_token_id": 0,
            }
        )
    )
    save_file({"fc.weight": fc_weight}, tmp_path / "model.safetensors")


def _run_expand(src, out, new_ids, scales=None):
    cmd = [
        sys.executable, str(EXPAND_SCRIPT), str(src),
        "--new-target-layer-ids", *map(str, new_ids),
    ]
    if scales is not None:
        cmd += ["--slot-scales", *map(str, scales)]
    cmd += ["--output", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_expand_target_layers_slot_scales(tmp_path):
    """--slot-scales writes the aligned field; old slot-0 block preserved."""
    torch.manual_seed(3)
    old_fc = torch.randn(HIDDEN, HIDDEN)
    src = tmp_path / "src"
    src.mkdir()
    _write_tiny_checkpoint(src, [0], old_fc)

    out = tmp_path / "out"
    scales = [1.0, 0.012, 0.0038]
    res = _run_expand(src, out, [0, 36, 60], scales)
    assert "WARNING" not in res.stderr  # slot 0 keeps effective scale 1.0

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["aux_hidden_state_layer_ids"] == [0, 36, 60]
    assert cfg["aux_hidden_state_scales"] == scales

    new_fc = load_file(out / "model.safetensors")["fc.weight"]
    assert new_fc.shape == (HIDDEN, 3 * HIDDEN)
    torch.testing.assert_close(new_fc[:, :HIDDEN], old_fc)  # bit-exact slot 0
    assert torch.count_nonzero(new_fc[:, HIDDEN:]) == 0  # zero blocks


def test_expand_target_layers_propagates_source_scales(tmp_path):
    """Without --slot-scales, source scales propagate; new slots get 1.0."""
    torch.manual_seed(3)
    old_fc = torch.randn(HIDDEN, HIDDEN)
    src = tmp_path / "src"
    src.mkdir()
    _write_tiny_checkpoint(src, [0], old_fc)
    cfg = json.loads((src / "config.json").read_text())
    cfg["aux_hidden_state_scales"] = [0.25]
    (src / "config.json").write_text(json.dumps(cfg))

    out = tmp_path / "out"
    _run_expand(src, out, [0, 36])
    cfg2 = json.loads((out / "config.json").read_text())
    assert cfg2["aux_hidden_state_scales"] == [0.25, 1.0]


def test_expand_target_layers_warns_on_rescaled_old_slot(tmp_path):
    """Rescaling a slot that carries pretrained weights is not
    function-preserving — the script must warn."""
    torch.manual_seed(3)
    old_fc = torch.randn(HIDDEN, HIDDEN)
    src = tmp_path / "src"
    src.mkdir()
    _write_tiny_checkpoint(src, [0], old_fc)

    out = tmp_path / "out"
    res = _run_expand(src, out, [0, 36], [0.5, 1.0])
    assert "WARNING" in res.stderr
    assert "NOT function-preserving" in res.stderr
