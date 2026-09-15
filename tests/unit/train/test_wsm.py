"""Properties of the WSM checkpoint schedule.

The schedule is only useful if, at any step, the held checkpoints span the
trailing window and are spread across it. Both are asserted by stepping the
schedule through whole runs rather than by checking individual decisions,
since the failure modes are emergent: a schedule can satisfy "every checkpoint
is inside the window" while bunching them all at one end.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators.train.wsm import (
    WSMSchedule,
    WSMStore,
    merge_checkpoints,
    wsm_checkpoint_dirs,
)


def run(total, n, window=0.10, start=0.01):
    """Step a schedule through a full run, yielding (step, kept) after saves."""
    schedule = WSMSchedule(total, n, window_fraction=window, start_fraction=start)
    evicted_total: list[int] = []
    for step in range(total + 1):
        if schedule.should_save(step):
            evicted_total.extend(schedule.record_saved(step))
        yield step, schedule, evicted_total


def settled(total, n, **kw):
    """Only the portion of the run where the schedule holds a full set."""
    for step, schedule, _ in run(total, n, **kw):
        if len(schedule.kept) == n:
            yield step, schedule


@pytest.mark.parametrize(("total", "n"), [(100_000, 8), (1_000_000, 8), (200_000, 16)])
def test_every_held_checkpoint_stays_inside_the_window(total, n):
    for step, schedule in settled(total, n):
        # One step of slack: the boundary is fractional, the steps are integers.
        assert schedule.kept[0] >= schedule.window_start(step) - 1, (
            f"step {step}: oldest {schedule.kept[0]} left the window"
        )


@pytest.mark.parametrize(("total", "n"), [(100_000, 8), (1_000_000, 8), (200_000, 16)])
def test_checkpoints_stay_spread_across_the_window(total, n):
    """The point of the schedule: no bunching at either end."""
    worst = 1.0
    for step, schedule in settled(total, n):
        lattice = schedule.kept[:-1] if step == total else schedule.kept
        gaps = [b - a for a, b in zip(lattice, lattice[1:], strict=False)]
        if len(gaps) > 1:
            worst = max(worst, max(gaps) / min(gaps))
    # Geometric spacing makes the widest gap ratio (1-f)^(-(n-1)/n); the bound
    # is loose enough for integer rounding but far below any bunching.
    assert worst < 1.5, f"gaps varied by {worst:.2f}x"


def test_the_window_is_a_fraction_of_elapsed_training_not_a_fixed_width():
    """The distinguishing property: a fixed-width window would cover a shrinking
    share of training as the run goes on."""
    spans = {}
    for step, schedule in settled(1_000_000, 8):
        if step in (200_000, 500_000, 1_000_000):
            spans[step] = (step - schedule.kept[0]) / step
    assert len(spans) == 3
    for step, share in spans.items():
        assert 0.07 < share <= 0.10, f"step {step}: covered {share:.3f} of training"


def test_saves_are_logarithmic_in_run_length():
    """Ten times the steps must not mean ten times the checkpoints."""
    counts = {}
    for total in (100_000, 1_000_000, 10_000_000):
        schedule = WSMSchedule(total, 8)
        saves = 0
        for step in range(total + 1):
            if schedule.should_save(step):
                schedule.record_saved(step)
                saves += 1
        counts[total] = saves
    assert counts[1_000_000] < 1.2 * counts[100_000]
    assert counts[10_000_000] < 1.2 * counts[1_000_000]


def test_evicted_steps_are_reported_exactly_once():
    total, n = 50_000, 8
    seen: list[int] = []
    for _step, _schedule, evicted in run(total, n):
        seen = evicted
    schedule_final = WSMSchedule(total, n)
    saved = []
    for step in range(total + 1):
        if schedule_final.should_save(step):
            schedule_final.record_saved(step)
            saved.append(step)
    assert len(seen) == len(set(seen)), "a step was evicted twice"
    assert set(seen) | set(schedule_final.kept) == set(saved)


def test_the_final_step_is_always_held():
    for total, n in [(10_000, 8), (99_999, 8), (12_345, 5)]:
        last = None
        for _step, schedule, _e in run(total, n):
            last = schedule
        assert last is not None
        assert last.kept[-1] == total


def test_rejects_nonsense_configuration():
    with pytest.raises(ValueError, match="num_checkpoints"):
        WSMSchedule(1000, 0)
    with pytest.raises(ValueError, match="window_fraction"):
        WSMSchedule(1000, 8, window_fraction=1.0)
    with pytest.raises(ValueError, match="start_fraction"):
        WSMSchedule(1000, 8, start_fraction=1.0)


# --------------------------------------------------------------------------- #
# Store and merge: the parts that touch disk
# --------------------------------------------------------------------------- #


def _write(path, value, extra=None):
    """A stand-in checkpoint whose weights are a known constant."""
    path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "fc.weight": torch.full((4, 4), float(value)),
        "t2d": torch.tensor([True, False, True]),  # non-float: must be copied
        "d2t": torch.tensor([7, 8, 9]),
    }
    if extra:
        tensors.update(extra)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text('{"speculators_model_type": "dflash"}')


def test_store_keeps_only_the_window_on_disk(tmp_path):
    schedule = WSMSchedule(20_000, 4, start_fraction=0.05)
    store = WSMStore(tmp_path, schedule)
    for step in range(20_001):
        store.maybe_save(step, lambda dest, s=step: _write(dest, s))

    held = wsm_checkpoint_dirs(tmp_path)
    assert len(held) == 4, "evicted checkpoints were left behind"
    steps = [int(d.name.removeprefix("step_")) for d in held]
    assert steps == list(schedule.kept)
    assert steps[-1] == 20_000


def test_store_does_not_prune_on_non_primary_ranks(tmp_path):
    """Every rank must save (gathering is collective); only one may delete."""
    schedule = WSMSchedule(5_000, 3, start_fraction=0.05)
    store = WSMStore(tmp_path, schedule, is_primary=False)
    for step in range(5_001):
        store.maybe_save(step, lambda dest, s=step: _write(dest, s))
    assert len(wsm_checkpoint_dirs(tmp_path)) > 3


def test_merge_averages_float_weights_and_copies_the_rest(tmp_path):
    for i, value in enumerate([1.0, 2.0, 6.0]):
        _write(tmp_path / f"step_{i}", value)
    sources = wsm_checkpoint_dirs(tmp_path)

    stats = merge_checkpoints(sources, tmp_path / "merged")
    merged = load_file(tmp_path / "merged" / "model.safetensors")

    assert stats == {"merged": 3, "averaged_tensors": 1}
    torch.testing.assert_close(merged["fc.weight"], torch.full((4, 4), 3.0))
    assert torch.equal(merged["t2d"], torch.tensor([True, False, True]))
    assert torch.equal(merged["d2t"], torch.tensor([7, 8, 9]))
    assert (tmp_path / "merged" / "config.json").is_file()


def test_merge_preserves_dtype(tmp_path):
    for i, value in enumerate([1.0, 2.0]):
        _write(
            tmp_path / f"step_{i}",
            value,
            extra={"half": torch.tensor([value], dtype=torch.bfloat16)},
        )
    merge_checkpoints(wsm_checkpoint_dirs(tmp_path), tmp_path / "merged")
    merged = load_file(tmp_path / "merged" / "model.safetensors")
    assert merged["half"].dtype == torch.bfloat16
    assert merged["fc.weight"].dtype == torch.float32


def test_merging_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No checkpoints"):
        merge_checkpoints([], tmp_path / "merged")
