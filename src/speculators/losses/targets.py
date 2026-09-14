"""The two representations a training target can take.

Distillation trains against the verifier's full output distribution, so its
targets are soft logits ``[1, T, V]``. Pretraining has no verifier forward to
distill from and trains against the corpus itself, so its targets are hard
token ids ``[1, T]``. Everything downstream of the target -- the losses, the
accuracy metrics, the accepted-length counters -- is otherwise identical, and
reduces the target to ids anyway.
"""

import torch

IGNORE_INDEX = -100
_SOFT_TARGET_DIMS = 3


def is_hard(targets: torch.Tensor) -> bool:
    """Whether ``targets`` are token ids rather than a distribution over them."""
    return targets.dim() != _SOFT_TARGET_DIMS


def as_target_ids(targets: torch.Tensor) -> torch.Tensor:
    """Token ids from either representation of ``targets``.

    Hard targets pass through; soft targets collapse to their argmax. Ids may
    be :data:`IGNORE_INDEX`, which the cross-entropy losses skip.
    """
    return targets if is_hard(targets) else targets.argmax(dim=-1)
