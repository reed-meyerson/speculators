"""Checkpoint schedule for warmup-stable-merge.

WSM replaces the decay phase of a warmup-stable-decay run by averaging several
checkpoints from the stable phase. Averaging is only worth anything if those
checkpoints are spread over a meaningful stretch of training, and the stretch
that matters is a *fraction* of what has been trained so far -- the trailing
10% at step 100k is a different span than the trailing 10% at step 1M. Holding
that invariant at every step is what lets a run be merged whenever a finetune
wants a base, rather than only at a planned end.

A window of `[(1-f)*t, t]` has constant width in log space, so spreading
checkpoints evenly across it means spacing them by a constant *ratio* rather
than a constant number of steps. Saves therefore get further apart as training
proceeds, and checkpoints thin themselves out: the total is logarithmic in run
length, and only ``num_checkpoints`` are ever on disk at once.

Saving on a fixed step interval instead would either fall out of the window
(too sparse later) or bury the run in checkpoints (too dense later). Saving
reactively -- whenever the oldest leaves the window -- collapses even faster,
because the first checkpoints are written on consecutive steps and then age out
on consecutive steps forever.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "WSMSchedule",
    "WSMStore",
    "merge_checkpoints",
    "wsm_checkpoint_dirs",
]


class WSMSchedule:
    """Which steps to checkpoint so the trailing window stays covered.

    Args:
        total_steps: Length of the run, used to place the first save and to
            force a save on the final step.
        num_checkpoints: How many checkpoints to hold at once.
        window_fraction: Fraction of elapsed training the held checkpoints
            should span.
        start_fraction: Point in the run at which to begin saving, as a
            fraction of ``total_steps``. Merging checkpoints from the first
            moments of training is not useful, and saves there are densest.
    """

    def __init__(
        self,
        total_steps: int,
        num_checkpoints: int,
        window_fraction: float = 0.10,
        start_fraction: float = 0.01,
    ):
        if num_checkpoints < 1:
            raise ValueError(f"num_checkpoints must be >= 1, got {num_checkpoints}")
        if not 0.0 < window_fraction < 1.0:
            raise ValueError(
                f"window_fraction must be in (0, 1), got {window_fraction}"
            )
        if not 0.0 <= start_fraction < 1.0:
            raise ValueError(f"start_fraction must be in [0, 1), got {start_fraction}")

        self.total_steps = total_steps
        self.num_checkpoints = num_checkpoints
        self.window_fraction = window_fraction
        # Each save advances the log-position by one n-th of the window, so the
        # oldest of n checkpoints sits exactly at the window's trailing edge.
        self.ratio = (1.0 - window_fraction) ** (-1.0 / num_checkpoints)
        self.start_step = max(1, int(total_steps * start_fraction))
        self._kept: list[int] = []
        self._next_save = self.start_step

    @property
    def kept(self) -> tuple[int, ...]:
        """Steps currently held, oldest first."""
        return tuple(self._kept)

    def window_start(self, step: int) -> float:
        """Oldest step the window admits at ``step``."""
        return (1.0 - self.window_fraction) * step

    def should_save(self, step: int) -> bool:
        if step < self.start_step:
            return False
        return step >= self._next_save or step >= self.total_steps

    def record_saved(self, step: int) -> tuple[int, ...]:
        """Register a save and report the steps it evicted."""
        if self._kept and self._kept[-1] == step:
            return ()
        self._kept.append(step)
        # Truncate rather than round up: rounding each gap up compounds across
        # the window and walks the oldest checkpoint off its trailing edge.
        self._next_save = max(step + 1, int(step * self.ratio))
        evicted = []
        while len(self._kept) > self.num_checkpoints:
            evicted.append(self._kept.pop(0))
        return tuple(evicted)

    def estimated_saves(self) -> int:
        """Roughly how many saves the whole run will perform."""
        if self.start_step >= self.total_steps:
            return 0
        span = math.log(self.total_steps / self.start_step)
        return max(1, round(span / math.log(self.ratio)))


_STEP_PREFIX = "step_"


def wsm_checkpoint_dirs(root: Path) -> list[Path]:
    """Held WSM checkpoints under ``root``, oldest first."""
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(_STEP_PREFIX)]
    return sorted(dirs, key=lambda d: int(d.name.removeprefix(_STEP_PREFIX)))


class WSMStore:
    """Keeps the window on disk: saves on schedule, deletes what ages out."""

    def __init__(self, root: Path, schedule: WSMSchedule, is_primary: bool = True):
        self.root = Path(root)
        self.schedule = schedule
        self.is_primary = is_primary

    def path_for(self, step: int) -> Path:
        return self.root / f"{_STEP_PREFIX}{step}"

    def maybe_save(self, step: int, save: Callable[[Path], None]) -> Path | None:
        """Save at ``step`` if the schedule calls for it, evicting what expired.

        ``save`` writes a checkpoint to the path it is given. It runs on every
        rank, since gathering a sharded model is collective; only the primary
        rank prunes.
        """
        if not self.schedule.should_save(step):
            return None
        destination = self.path_for(step)
        destination.mkdir(parents=True, exist_ok=True)
        save(destination)
        for expired in self.schedule.record_saved(step):
            if self.is_primary:
                shutil.rmtree(self.path_for(expired), ignore_errors=True)
        return destination


def merge_checkpoints(sources: Sequence[Path], destination: Path) -> dict[str, int]:
    """Average the weights of several checkpoints into one.

    Accumulates in float32 one checkpoint at a time, so peak memory is the size
    of a single model rather than the whole window. Integer and boolean tensors
    -- vocabulary maps and masks -- are identical across the window and are
    copied rather than averaged, which would be meaningless for them.
    """
    if not sources:
        raise ValueError("No checkpoints to merge.")

    totals: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    verbatim: dict[str, torch.Tensor] = {}
    for source in sources:
        tensors = load_file(source / "model.safetensors")
        for name, tensor in tensors.items():
            if not tensor.is_floating_point():
                verbatim[name] = tensor
                continue
            dtypes.setdefault(name, tensor.dtype)
            promoted = tensor.to(torch.float32)
            if name in totals:
                totals[name] += promoted
            else:
                totals[name] = promoted

    merged = {
        name: total.div_(len(sources)).to(dtypes[name])
        for name, total in totals.items()
    }
    merged.update(verbatim)

    destination.mkdir(parents=True, exist_ok=True)
    save_file(merged, destination / "model.safetensors", metadata={"format": "pt"})
    for extra in ("config.json", "config.py"):
        candidate = sources[-1] / extra
        if candidate.is_file():
            shutil.copy2(candidate, destination / extra)
    return {"merged": len(sources), "averaged_tensors": len(totals)}
