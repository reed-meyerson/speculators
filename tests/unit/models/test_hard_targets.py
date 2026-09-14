"""Losses against hard token-id targets (speculators.losses.targets).

Pretraining scores the draft against the corpus's own next tokens rather than
the verifier's distribution. A hard label is the point-mass limit of a soft
target, so every loss that accepts both must agree with its own soft form when
that soft form concentrates on one token -- these tests pin that equivalence
rather than the hard branch's implementation.
"""

import pytest
import torch

from speculators.losses import eager
from speculators.losses.targets import IGNORE_INDEX, as_target_ids, is_hard

VOCAB = 32
SEQ = 12
# Large enough that softmax(one_hot * SATURATION) is a point mass in fp32.
SATURATION = 40.0


def _hard_and_equivalent_soft():
    torch.manual_seed(0)
    logits = torch.randn(1, SEQ, VOCAB, dtype=torch.float32)
    hard = torch.randint(0, VOCAB, (1, SEQ))
    soft = torch.nn.functional.one_hot(hard, VOCAB).float() * SATURATION
    return logits, hard, soft


def test_is_hard_and_as_target_ids_round_trip():
    _, hard, soft = _hard_and_equivalent_soft()
    assert is_hard(hard)
    assert not is_hard(soft)
    assert torch.equal(as_target_ids(hard), hard)
    assert torch.equal(as_target_ids(soft), hard)


def test_ce_loss_hard_matches_point_mass_soft():
    logits, hard, soft = _hard_and_equivalent_soft()
    torch.testing.assert_close(eager.ce_loss(logits, hard), eager.ce_loss(logits, soft))


def test_tv_loss_hard_matches_point_mass_soft():
    """Against a point mass the overlap collapses to the draft's probability of
    the true token, so TV is ``1 - p_t``. DSpark's confidence head depends on
    this holding, since it scores acceptance through TV."""
    logits, hard, soft = _hard_and_equivalent_soft()
    torch.testing.assert_close(
        eager.tv_loss(logits, hard), eager.tv_loss(logits, soft), atol=1e-6, rtol=0
    )

    expected = 1.0 - torch.softmax(logits.float(), dim=-1).gather(
        -1, hard.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(eager.tv_loss(logits, hard), expected)


@pytest.mark.parametrize("loss_fn", [eager.ce_loss, eager.tv_loss])
def test_ignored_labels_contribute_no_signal(loss_fn):
    """Tokens outside a pruned draft vocabulary are labelled IGNORE_INDEX; they
    must neither contribute loss nor index out of bounds."""
    logits, hard, _ = _hard_and_equivalent_soft()
    logits = logits.requires_grad_(True)
    ignored = hard.clone()
    ignored[:, ::2] = IGNORE_INDEX

    per_position = loss_fn(logits, ignored)
    assert torch.all(per_position[:, ::2] == 0)
    torch.testing.assert_close(per_position[:, 1::2], loss_fn(logits, hard)[:, 1::2])

    per_position.sum().backward()
    assert torch.all(logits.grad[:, ::2] == 0)
