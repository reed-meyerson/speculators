import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


def _tiny_model(sample_from_anchor: bool) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


@pytest.mark.parametrize("max_anchors", [5, 16])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_targets_match_full_sequence_roll(sample_from_anchor, max_anchors):
    torch.manual_seed(0)
    model = _tiny_model(sample_from_anchor)
    seq_len = 32
    hidden_states = torch.randn(1, seq_len, 2 * 16)
    verifier_last_hidden_states = torch.randn(1, seq_len, 16)
    input_ids = torch.randint(0, 64, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        _, _, targets, _, anchored_block_indices = model._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            max_anchors=max_anchors,
        )

        full_logits = model.verifier_lm_head(
            model.verifier_norm(verifier_last_hidden_states)
        )
        if not sample_from_anchor:
            full_logits = torch.roll(full_logits, 1, dims=1)
        expected = full_logits[:, anchored_block_indices]

    torch.testing.assert_close(targets, expected, atol=1e-5, rtol=0)


def _oracle_model(sample_from_anchor: bool) -> DFlashDraftModel:
    """A model whose verifier head is the identity, so a one-hot verifier
    hidden state decodes back to exactly the token it encodes."""
    size = 16
    tl_config = Qwen3Config(
        hidden_size=size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=size,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=size,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.eye_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    torch.nn.init.normal_(model.embed_tokens.weight)
    return model.eval()


@pytest.mark.parametrize("max_anchors", [5, 16])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_pretrain_hard_targets_match_distilled_argmax(sample_from_anchor, max_anchors):
    """Pretraining's hard labels must name the same tokens distillation's soft
    targets put their mass on -- the two paths differ in representation, not in
    which position of the sequence each block slot is asked to predict."""
    torch.manual_seed(0)
    model = _oracle_model(sample_from_anchor)
    size, seq_len = 16, 32
    input_ids = torch.randint(0, size, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    # Verifier hidden at position p one-hot encodes the token at p + 1, so the
    # reconstructed verifier distribution is a perfect next-token oracle.
    next_ids = torch.roll(input_ids, -1, dims=1)
    verifier_last_hidden_states = torch.nn.functional.one_hot(
        next_ids, num_classes=size
    ).float()
    hidden_states = torch.randn(1, seq_len, 2 * size)

    # Anchors are sampled, so both passes must draw the same ones to compare.
    with torch.no_grad():
        torch.manual_seed(1)
        _, _, soft_targets, soft_mask, soft_blocks = model._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            max_anchors=max_anchors,
        )
        torch.manual_seed(1)
        _, _, hard_targets, hard_mask, hard_blocks = model._backbone_forward(
            None,
            input_ids,
            loss_mask,
            None,
            document_ids,
            max_anchors=max_anchors,
            training_mode="pretrain",
        )
    assert torch.equal(soft_blocks, hard_blocks)

    assert hard_targets.shape == soft_targets.shape[:-1]
    torch.testing.assert_close(hard_mask, soft_mask)
    scored = soft_mask.to(torch.bool)
    assert torch.equal(hard_targets[scored], soft_targets.argmax(dim=-1)[scored])
