"""Streaming corpus input for pretraining.

Distillation reads a finite directory of verifier hidden states, so its
dataset is map-style and its length is whatever was generated ahead of time.
Pretraining reads raw text instead and is bounded by a token budget rather
than a corpus, so the stream is consumed lazily and packed into fixed-length
sequences on the way through.

The budget makes the sequence count known up front, which is what lets this
stay an ordinary dataset from the trainer's point of view: ``__len__`` is
defined, so the epoch ends on its own and the LR schedule resolves from it
without a token-aware special case anywhere downstream.

Packing concatenates whole documents up to ``total_seq_len`` and records where
each one started. ``position_ids`` restart per document and ``CollateFn``
turns the per-document lengths into ``document_ids``, so neither RoPE nor
attention ever runs across a document boundary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from speculators.train.data import BatchType, CollateFn

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "PackedCorpusStream",
    "create_pretrain_loaders",
    "sequences_for_token_budget",
]


def sequences_for_token_budget(
    token_budget: int, total_seq_len: int, world_size: int
) -> int:
    """Packed sequences each rank must yield to spend ``token_budget`` in total."""
    if token_budget <= 0:
        raise ValueError(f"token budget must be positive, got {token_budget}")
    per_rank = token_budget / (total_seq_len * world_size)
    return max(1, round(per_rank))


class PackedCorpusStream(IterableDataset):
    """Tokenize a streaming text corpus and pack it into fixed-length sequences.

    Yields ``sequences_per_rank`` samples of exactly ``total_seq_len`` tokens.
    Documents are sharded across ranks and then across that rank's dataloader
    workers, so no sequence is produced twice.
    """

    def __init__(
        self,
        corpus,
        tokenizer,
        total_seq_len: int,
        sequences_per_rank: int,
        text_column: str = "text",
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.total_seq_len = total_seq_len
        self.sequences_per_rank = sequences_per_rank
        self.text_column = text_column
        self.rank = rank
        self.world_size = world_size
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError(
                "The pretraining corpus is packed with the verifier's EOS token "
                "as a document separator, but this tokenizer defines none."
            )
        self.eos_token_id = eos

    def __len__(self) -> int:
        return self.sequences_per_rank

    def _worker_share(self) -> tuple[int, int]:
        info = get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def _documents(self) -> Iterator[list[int]]:
        """Tokenized documents belonging to this rank's share of this worker.

        Ranks and workers form one flat set of shards, so a document is read by
        exactly one reader anywhere in the job. Sharding per rank is not
        optional: the token budget is already divided by ``world_size``, so
        without it every rank would train on the same documents.
        """
        worker_id, num_workers = self._worker_share()
        shard = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        for index, record in enumerate(self.corpus):
            if index % num_shards != shard:
                continue
            text = record.get(self.text_column)
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                yield [*ids, self.eos_token_id]

    def __iter__(self) -> Iterator[BatchType]:
        worker_id, num_workers = self._worker_share()
        # Each worker owns a whole number of this rank's sequences; the
        # remainder goes to the low-numbered workers.
        quota, remainder = divmod(self.sequences_per_rank, num_workers)
        quota += int(worker_id < remainder)
        if quota == 0:
            return

        produced = 0
        tokens: list[int] = []
        lengths: list[int] = []
        positions: list[int] = []

        for document in self._documents():
            offset = 0
            while offset < len(document):
                room = self.total_seq_len - len(tokens)
                chunk = document[offset : offset + room]
                tokens.extend(chunk)
                positions.extend(range(offset, offset + len(chunk)))
                lengths.append(len(chunk))
                offset += len(chunk)

                if len(tokens) == self.total_seq_len:
                    yield self._sample(tokens, lengths, positions)
                    produced += 1
                    if produced == quota:
                        return
                    tokens, lengths, positions = [], [], []

        logger.warning(
            "Pretraining corpus was exhausted after %d of %d sequences on worker "
            "%d; the token budget will not be met.",
            produced,
            quota,
            worker_id,
        )

    def _sample(
        self, tokens: list[int], lengths: list[int], positions: list[int]
    ) -> BatchType:
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "loss_mask": torch.ones(len(tokens), dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }


def _loader(dataset: PackedCorpusStream, num_workers: int, prefetch_factor: int):
    # The stream already packs to exactly total_seq_len, so each sample is a
    # full batch; CollateFn is reused only for its document_ids bookkeeping.
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        collate_fn=CollateFn(max_len=dataset.total_seq_len, hidden_size=0),
    )


def create_pretrain_loaders(  # noqa: PLR0917
    corpus,
    val_corpus,
    tokenizer,
    total_seq_len: int,
    train_sequences: int,
    val_sequences: int,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    text_column: str = "text",
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Training and validation loaders over a streaming text corpus."""
    train = PackedCorpusStream(
        corpus,
        tokenizer,
        total_seq_len,
        train_sequences,
        text_column,
        rank=rank,
        world_size=world_size,
    )
    # Validation is the same few sequences on every rank, so it is not sharded:
    # the ranks agree on the number, and the metric is reduced across them.
    val = PackedCorpusStream(
        val_corpus, tokenizer, total_seq_len, val_sequences, text_column
    )
    return (
        _loader(train, num_workers, prefetch_factor),
        _loader(val, 0, prefetch_factor),
    )
