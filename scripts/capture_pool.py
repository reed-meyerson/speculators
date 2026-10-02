#!/usr/bin/env python
"""Offline hidden-states pool capture for verifier layer-ID search mini-experiments.

Two phases, run separately:

``compute``
    Simulate the exact training stream (``MultipackDistributedBatchSamplerV2``
    semantics: sampler seed 0, epoch 0, DP replicas, ``total_seq_len`` packing)
    over a ``prepare-data`` arrow dataset and write the row sets consumed by the
    first N train steps and first M val steps (plus a small prefetch margin).
    The fast prefix simulation replicates ``_assign_to_packed_batches``'s outer
    loop with an early break; it is cross-validated against the real sampler
    (full packing, union over all ranks) unless ``--no-validate``.

    Rows are emitted as *file indices* (global row ids: train-local ids are
    identical to file ids; val file id = split_idx + val-local id), matching
    what ``FileTransfer`` reads (``hs_{file_idx}.safetensors``).

``capture``
    Given ``rows.json`` and a running hidden-states vLLM server, prefill each
    row and persist ``hs_{file_idx}.safetensors`` in the pool directory.
    Resumable (existing files are skipped); writes ``pool_manifest.json``
    recording the capture order of layer ids. The server must be launched with
    the SAME layer-id list, e.g.::

        python scripts/launch_vllm.py train \\
            Qwen/Qwen3.8-27B --target-layer-ids 0 4 8 12 16 20 24 28 32 36 40 \\
            44 48 52 56 60 --hidden-states-path <staging-dir> -- \\
            --port 8400 --tensor-parallel-size 8 --max-model-len 8200

    (``launch_vllm.py`` auto-appends ``num_hidden_layers`` as the last captured
    slot — the verifier hidden state.)

The captured pool is consumed by training with ``hidden_states_backend: file``,
``hidden_states_path: <pool dir>``, ``on_missing: raise``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# compute phase
# ---------------------------------------------------------------------------


def _simulate_windows(
    plengths: np.ndarray,
    max_len: int,
    replicas: int,
    max_steps: int,
) -> tuple[list[int], int]:
    """Replicate ``_assign_to_packed_batches``'s outer loop, breaking early.

    Returns ``(window_sizes, ind)``: the LPT window size per step and the total
    number of permuted rows consumed after ``max_steps`` steps. The rows
    consumed by steps 1..max_steps (union over all ranks) are exactly
    ``plengths[:ind]`` — the packing window per step is rank-independent; the
    rank only selects samples within a window.
    """
    from speculators.train.distributed_batch_sampler import _lpt_packed_batch

    lengths_so_far = 0
    ind = 0
    windows: list[int] = []
    lengths_cumsum = np.cumsum(plengths)
    n = len(plengths)

    while len(windows) < max_steps:
        if n - ind < replicas:
            break  # tail drop, same as the original loop

        left = 1
        right = 1 + int(
            np.searchsorted(
                lengths_cumsum[ind:], lengths_so_far + max_len * replicas, "right"
            )
        )
        rotation = len(windows)

        batch = None
        while right - left > 1 and right > replicas:
            mid = (left + right) // 2
            batch = _lpt_packed_batch(
                plengths[ind : ind + mid], max_len, replicas, ind, 0, rotation
            )
            if batch is None:
                right = mid
            else:
                left = mid

        if batch is None:
            batch = _lpt_packed_batch(
                plengths[ind : ind + left], max_len, replicas, ind, 0, rotation
            )

        ind += left
        lengths_so_far = lengths_cumsum[ind - 1]
        windows.append(left)

    return windows, ind


def _real_sampler_union(
    lengths: np.ndarray, max_len: int, replicas: int, steps: int
) -> set[int]:
    """Union of dataset rows consumed by the first ``steps`` batches across all
    ranks, using the real sampler (full epoch packing). Local row ids."""
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )

    union: set[int] = set()
    counts = []
    for r in range(replicas):
        sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=max_len,
            lengths=lengths,
            num_replicas=replicas,
            rank=r,
        )
        batches = sampler._generate_batches(0)  # noqa: SLF001
        counts.append(len(batches))
        if len(batches) < steps:
            raise AssertionError(
                f"rank {r}: only {len(batches)} batches available, need {steps}"
            )
        for b in batches[:steps]:
            union.update(int(x) for x in b)
    if len(set(counts)) != 1:
        raise AssertionError(f"ranks disagree on batch counts: {counts}")
    return union


def _split_lengths(seq_len: np.ndarray, max_len: int, name: str):
    """Replicate the sampler's truncate_long_samples=True length handling."""
    lengths = seq_len.copy()
    over = int((lengths > max_len).sum())
    if over:
        logger.warning(
            "%s split: %d rows longer than max_len %d; clipping for packing "
            "(truncate_long_samples=True semantics)",
            name,
            over,
            max_len,
        )
        lengths = np.clip(lengths, 0, max_len)
    return lengths


def cmd_compute(args: argparse.Namespace) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(args.data)
    seq_len = np.asarray(ds.data.column("seq_len"))
    n_rows = len(seq_len)
    split_idx = int(n_rows * args.train_ratio)
    logger.info(
        "dataset %s: %d rows, split_idx=%d (train %d / val %d)",
        args.data,
        n_rows,
        split_idx,
        split_idx,
        n_rows - split_idx,
    )

    out: dict[str, Any] = {
        "version": 1,
        "data": str(Path(args.data).resolve()),
        "split_idx": split_idx,
        "geometry": {
            "dp": args.dp,
            "seq_len": args.seq_len,
            "train_ratio": args.train_ratio,
            "sampler_seed": args.seed,
            "epoch": args.epoch,
            "margin_steps": args.margin,
        },
    }

    for split_name, lengths, offset in (
        ("train", _split_lengths(seq_len[:split_idx], args.seq_len, "train"), 0),
        ("val", _split_lengths(seq_len[split_idx:], args.seq_len, "val"), split_idx),
    ):
        steps = args.train_steps if split_name == "train" else args.val_steps
        if steps is None or steps <= 0:
            continue

        perm = np.random.default_rng(args.seed + args.epoch).permutation(len(lengths))
        plen = lengths[perm]
        cum = np.cumsum(plen)

        _, ind_core = _simulate_windows(plen, args.seq_len, args.dp, steps)
        total_steps = steps + args.margin
        windows, ind_margin = _simulate_windows(plen, args.seq_len, args.dp, total_steps)
        if len(windows) != total_steps:
            raise AssertionError(
                f"{split_name}: dataset exhausted after {len(windows)} steps, "
                f"need {total_steps}"
            )

        rows = (perm[:ind_margin] + offset).tolist()
        entry = {
            "steps": steps,
            "margin_steps": args.margin,
            "rows": rows,
            "num_rows": len(rows),
            "tokens": int(cum[ind_core - 1]),
            "tokens_with_margin": int(cum[ind_margin - 1]),
        }
        out[split_name] = entry
        logger.info(
            "%s: %d steps -> %d rows / %d tokens (with +%d margin: %d rows / "
            "%d tokens)",
            split_name,
            steps,
            ind_core,
            entry["tokens"],
            args.margin,
            ind_margin,
            entry["tokens_with_margin"],
        )

        if args.validate:
            t0 = time.perf_counter()
            real_union = _real_sampler_union(
                lengths, args.seq_len, args.dp, total_steps
            )
            expected = {int(r) - offset for r in rows}
            if real_union != expected:
                raise AssertionError(
                    f"{split_name}: simulated row set does not match the real "
                    f"sampler (sim {len(expected)} rows vs real {len(real_union)}; "
                    f"sym-diff {len(expected ^ real_union)})"
                )
            logger.info(
                "%s: validated against real sampler (%d rows, %.1fs)",
                split_name,
                len(real_union),
                time.perf_counter() - t0,
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    logger.info("wrote %s", args.output)

    total_tokens = sum(
        e["tokens_with_margin"] for e in (out.get("train"), out.get("val")) if e
    )
    logger.info(
        "total capture: %d rows / %.2fB tokens (incl. margin)",
        sum(e["num_rows"] for e in (out.get("train"), out.get("val")) if e),
        total_tokens / 1e9,
    )


# ---------------------------------------------------------------------------
# capture phase
# ---------------------------------------------------------------------------


class _FailureTracker:
    """Abort after N consecutive failures with no success in between."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> bool:
        self._consecutive += 1
        return self._consecutive >= self.threshold


async def _capture_worker(  # noqa: C901, PLR0913
    client,
    model: str,
    queue: "asyncio.Queue[tuple[int, list[int]]]",
    pbar,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    pool_dir: Path,
    num_slots: int,
    validate_outputs: bool,
    request_timeout: float | None,
    max_retries: int,
    fail_on_error: bool,
    skipped_indices: list[int],
    cancel_event: asyncio.Event,
    failure_tracker: _FailureTracker | None,
    stats: dict[str, Any],
) -> None:
    from safetensors.torch import load_file
    from speculators.data_generation.offline import check_hidden_states
    from speculators.data_generation.vllm_client import (
        generate_hidden_states_async,
        wait_for_lock_async,
    )

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        file_idx, tokens = item
        if cancel_event.is_set():
            queue.task_done()
            continue

        target = pool_dir / f"hs_{file_idx}.safetensors"
        try:
            async with vllm_semaphore:
                t_vllm = time.perf_counter()
                handle = await generate_hidden_states_async(
                    client,
                    model,
                    {"input_ids": tokens},
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
                vllm_s = time.perf_counter() - t_vllm
            lock_path = handle + ".lock"
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(lock_path)
            async with write_semaphore:
                t_write = time.perf_counter()
                await asyncio.to_thread(shutil.move, handle, target)
                write_s = time.perf_counter() - t_write
                if validate_outputs:

                    def _load_and_check(path=target, toks=tokens):
                        loaded = load_file(path)
                        hs = loaded["hidden_states"]
                        if hs.shape[1] != num_slots:
                            raise ValueError(
                                f"slot count mismatch: got {hs.shape[1]}, "
                                f"expected {num_slots}"
                            )
                        check_hidden_states(loaded, toks)

                    await asyncio.to_thread(_load_and_check)
        except Exception as e:
            if fail_on_error:
                logger.exception(
                    "Fatal: sample %d aborted with --fail-on-error: %s", file_idx, e
                )
                logging.shutdown()
                os._exit(1)
            logger.warning("Skipping sample %d due to error: %s", file_idx, e)
            skipped_indices.append(file_idx)
            stats["errors"] += 1
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from e
        else:
            stats["ok"] += 1
            stats["total_vllm_s"] += vllm_s
            stats["total_write_s"] += write_s
            if failure_tracker is not None:
                failure_tracker.record_success()
        finally:
            elapsed = time.perf_counter() - stats["start_time"]
            postfix = {"ok": stats["ok"], "err": stats["errors"]}
            if elapsed > 0 and stats["ok"] > 0:
                postfix["rps"] = f"{stats['ok'] / elapsed:.1f}"
                postfix["vllm"] = f"{stats['total_vllm_s'] / stats['ok'] * 1000:.0f}ms"
                postfix["write"] = f"{stats['total_write_s'] / stats['ok'] * 1000:.0f}ms"
            pbar.set_postfix(postfix, refresh=False)
            pbar.update(1)
            queue.task_done()


def _resolve_layer_ids(args: argparse.Namespace) -> list[int]:
    ids = sorted(set(int(x) for x in args.layer_ids))
    n_layers = int(args.num_hidden_layers)
    if n_layers not in ids:
        ids = ids + [n_layers]
    if ids != sorted(set(ids)):
        raise ValueError(f"layer ids not ascending/unique after append: {ids}")
    if ids[-1] != n_layers:
        raise ValueError(
            f"last captured slot must be the verifier layer {n_layers}, got {ids[-1]}"
        )
    return ids


def _write_or_check_manifest(pool_dir: Path, manifest: dict[str, Any]) -> None:
    path = pool_dir / "pool_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        core_keys = ("layer_ids", "verifier_layer_id", "model")
        for k in core_keys:
            if existing.get(k) != manifest[k]:
                raise ValueError(
                    f"pool_manifest.json disagrees on {k}: "
                    f"{existing.get(k)!r} != {manifest[k]!r}. Refusing to mix "
                    "captures with different layer layouts."
                )
        return
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", path)


def _staging_report(staging_dir: Path | None) -> None:
    if staging_dir is None or not staging_dir.exists():
        return
    files = list(staging_dir.glob("*.safetensors"))
    locks = list(staging_dir.glob("*.lock"))
    total = sum(f.stat().st_size for f in files)
    logger.info(
        "staging dir %s: %d orphaned files (%.1f GB), %d lock files",
        staging_dir,
        len(files),
        total / 1e9,
        len(locks),
    )


async def _run_capture(args: argparse.Namespace) -> None:
    from datasets import load_from_disk
    from speculators.data_generation.offline import (
        get_existing_hidden_state_indices,
    )
    from speculators.train.data import build_client_item
    from tqdm import tqdm

    with open(args.rows) as f:
        rows_spec = json.load(f)

    pool_dir = Path(args.pool)
    pool_dir.mkdir(parents=True, exist_ok=True)

    layer_ids = _resolve_layer_ids(args)
    num_slots = len(layer_ids)

    manifest = {
        "layer_ids": layer_ids,
        "num_slots": num_slots,
        "verifier_layer_id": int(args.num_hidden_layers),
        "model": args.model,
        "data": rows_spec["data"],
        "rows_file": str(Path(args.rows).resolve()),
        "geometry": rows_spec.get("geometry", {}),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_rows": rows_spec.get("train", {}).get("num_rows", 0),
        "val_rows": rows_spec.get("val", {}).get("num_rows", 0),
    }
    _write_or_check_manifest(pool_dir, manifest)

    # Disk pre-flight: bf16 per-token bytes = num_slots * hidden * 2 (+ token ids).
    hidden = int(args.hidden_size)
    per_token = num_slots * hidden * 2 + 8
    need_tokens = sum(
        e.get("tokens_with_margin", 0)
        for e in (rows_spec.get("train"), rows_spec.get("val"))
        if e
    )
    need_bytes = need_tokens * per_token
    free = shutil.disk_usage(pool_dir).free
    logger.info(
        "pool needs ~%.1f TB (%d tokens x %d B); %.1f TB free at %s",
        need_bytes / 1e12,
        need_tokens,
        per_token,
        free / 1e12,
        pool_dir,
    )
    if free < need_bytes * 1.15:
        raise RuntimeError(
            f"insufficient space at {pool_dir}: need ~{need_bytes / 1e12:.1f} TB "
            f"(with headroom), free {free / 1e12:.1f} TB"
        )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in ("train", "val")]
    if unknown:
        raise ValueError(f"unknown splits {unknown}; use subsets of train,val")
    file_indices: list[int] = []
    for s in splits:
        entry = rows_spec.get(s)
        if entry:
            file_indices.extend(entry["rows"])
    if args.limit is not None:
        file_indices = file_indices[: args.limit]

    existing = set(get_existing_hidden_state_indices(pool_dir))
    todo = [i for i in file_indices if i not in existing]
    logger.info(
        "capture %d rows (%d skipped as already present) into %s",
        len(todo),
        len(file_indices) - len(todo),
        pool_dir,
    )
    if not todo:
        _staging_report(Path(args.staging) if args.staging else None)
        return

    ds = load_from_disk(rows_spec["data"])

    import openai

    queue: "asyncio.Queue[tuple[int, list[int]]]" = asyncio.Queue(
        maxsize=args.concurrency * 4
    )
    vllm_semaphore = asyncio.Semaphore(args.concurrency)
    write_semaphore = asyncio.Semaphore(args.concurrency)
    skipped_indices: list[int] = []
    cancel_event = asyncio.Event()
    stats: dict[str, Any] = {
        "ok": 0,
        "errors": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": time.perf_counter(),
    }
    failure_tracker = (
        None if args.fail_on_error else _FailureTracker(args.max_consecutive_errors)
    )

    async with openai.AsyncOpenAI(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        list_models = await client.models.list()
        if not list_models.data:
            raise RuntimeError("No models on the vLLM server; is it up?")
        model_id = list_models.data[0].id
        if args.model and args.model != model_id:
            raise ValueError(
                f"--model {args.model} does not match server model {model_id}"
            )

        with tqdm(total=len(todo)) as pbar:
            workers = [
                asyncio.create_task(
                    _capture_worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        pool_dir,
                        num_slots,
                        args.validate_outputs,
                        args.request_timeout,
                        args.max_retries,
                        args.fail_on_error,
                        skipped_indices,
                        cancel_event,
                        failure_tracker,
                        stats,
                    )
                )
                for _ in range(args.concurrency * 2)
            ]

            for i in todo:
                if cancel_event.is_set():
                    break
                dataset_item = await asyncio.to_thread(ds.__getitem__, i)
                client_item = build_client_item(dataset_item)
                while not cancel_event.is_set():
                    try:
                        queue.put_nowait((i, client_item["input_ids"]))
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.05)

            if not cancel_event.is_set():
                for _ in range(len(workers)):
                    await queue.put(None)
            else:
                for w in workers:
                    w.cancel()
            results = await asyncio.gather(*workers, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    raise r

    elapsed = time.perf_counter() - stats["start_time"]
    if stats["ok"] > 0:
        logger.info(
            "Timing: %.1fs elapsed, %.1f rows/s, avg vLLM %.0f ms, avg write "
            "%.0f ms",
            elapsed,
            stats["ok"] / elapsed,
            stats["total_vllm_s"] / stats["ok"] * 1000,
            stats["total_write_s"] / stats["ok"] * 1000,
        )
    logger.info("captured %d rows into %s", stats["ok"], pool_dir)
    if skipped_indices:
        logger.warning("skipped %d rows: %s", len(skipped_indices), skipped_indices[:50])
    _staging_report(Path(args.staging) if args.staging else None)


def cmd_capture(args: argparse.Namespace) -> None:
    asyncio.run(_run_capture(args))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compute", help="compute stream row sets (CPU-only)")
    p.add_argument("--data", required=True, help="prepare-data arrow dataset root")
    p.add_argument("--output", required=True, help="output rows.json path")
    p.add_argument("--dp", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--train-ratio", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0, help="sampler seed")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=1526)
    p.add_argument("--val-steps", type=int, default=153)
    p.add_argument("--margin", type=int, default=8, help="extra steps of rows")
    p.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cross-check vs the real sampler (slow: full packing x dp ranks)",
    )
    p.set_defaults(func=cmd_compute)

    p = sub.add_parser("capture", help="capture hidden states into the pool")
    p.add_argument("--rows", required=True, help="rows.json from compute")
    p.add_argument("--pool", required=True, help="hidden-states pool directory")
    p.add_argument("--endpoint", default="http://127.0.0.1:8400/v1")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        required=True,
        help="fc layer ids passed to launch_vllm.py --target-layer-ids (the "
        "verifier layer is appended automatically)",
    )
    p.add_argument("--num-hidden-layers", type=int, required=True)
    p.add_argument("--hidden-size", type=int, default=5120)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--validate-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-load each file and check token ids, finiteness, slot count",
    )
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--fail-on-error", action="store_true")
    p.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=None,
        help="abort after this many consecutive failures (default: concurrency)",
    )
    p.add_argument("--limit", type=int, default=None, help="capture only first N rows")
    p.add_argument(
        "--splits", default="train,val", help="comma-separated: train,val (order)"
    )
    p.add_argument("--staging", default=None, help="server staging dir (report only)")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args()
    if getattr(args, "max_consecutive_errors", None) is None and args.cmd == "capture":
        args.max_consecutive_errors = args.concurrency

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
