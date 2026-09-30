from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import os

from functools import partial

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from hs_connectors.transfer import MooncakeTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int, bind_device: bool = True) -> None:  # noqa: ARG001
    torch.set_num_threads(1)

    # Device binding exists for hidden-states backends whose workers touch
    # CUDA (mooncake's transfer engine allocates its local segment on the
    # rank's device — upstream #1168). For file-backed/token-only paths the
    # workers are pure CPU: pinning happens in the main process and H2D in
    # the trainer. Creating a CUDA context per worker there costs ~616MB of
    # GPU memory each (12 train + 12 val workers per rank) and context
    # creation has been observed to fail with cudaErrorMemoryAllocation
    # when a fresh val-worker pool spawns while training holds the device —
    # so it is opt-in via ``bind_device``.
    if bind_device and torch.accelerator.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.accelerator.set_device_index(local_rank)


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    max_batches: int | None = None,
    worker_bind_device: bool = False,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        max_batches=max_batches,
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=CollateFn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=(
            partial(_worker_init_fn, bind_device=worker_bind_device)
            if use_workers
            else None
        ),
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    generation_validation_retries: int,
    max_consecutive_generation_failures: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
    max_train_batches: int | None = None,
    require_hidden_states: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    _limit_worker_threads()
    # The noise transform indexes hidden-state keys unconditionally; skip it
    # for token-only datasets whose batches carry none.
    noise_transform = AddUniformNoise(std=noise_std) if require_hidden_states else None
    # Workers only touch CUDA for hidden-states backends that move tensors
    # on-device in-process (mooncake). File-backed/token-only workers are
    # pure CPU (see _worker_init_fn).
    worker_bind_device = isinstance(transfer, MooncakeTransfer)

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        transform=noise_transform,
        train_ratio=train_data_ratio,
        split="train",
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        generation_validation_retries=generation_validation_retries,
        max_consecutive_generation_failures=max_consecutive_generation_failures,
        require_hidden_states=require_hidden_states,
    )
    val_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        train_ratio=train_data_ratio,
        split="val",
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        generation_validation_retries=generation_validation_retries,
        max_consecutive_generation_failures=max_consecutive_generation_failures,
        require_hidden_states=require_hidden_states,
    )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        max_batches=max_train_batches,
        worker_bind_device=worker_bind_device,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        worker_bind_device=worker_bind_device,
    )

    return train_loader, val_loader
