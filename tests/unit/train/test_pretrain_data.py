"""Packing invariants for the streaming pretraining corpus.

The draft must never attend or apply RoPE across a document boundary, and the
run must stop when the token budget is spent. Both are properties of how the
stream packs, so they are asserted on the collated batch the trainer actually
receives rather than on the dataset's intermediate state.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from speculators.train.data import CollateFn
from speculators.train.pretrain_data import (
    PackedCorpusStream,
    sequences_for_token_budget,
)

SEQ = 32


class _FakeTokenizer:
    """Maps each word to a token id, so documents have controllable lengths."""

    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(tok) for tok in text.split()]}


def _corpus(doc_lengths):
    # Token ids start at 2 so they never collide with EOS.
    return [
        {"text": " ".join(str(i + 2) for i in range(length))} for length in doc_lengths
    ]


def _stream(doc_lengths, sequences, seq_len=SEQ):
    return PackedCorpusStream(
        _corpus(doc_lengths), _FakeTokenizer(), seq_len, sequences
    )


def test_sequences_for_token_budget_splits_across_ranks():
    assert sequences_for_token_budget(8192, 1024, world_size=1) == 8
    assert sequences_for_token_budget(8192, 1024, world_size=4) == 2
    with pytest.raises(ValueError, match="must be positive"):
        sequences_for_token_budget(0, 1024, 1)


def test_every_emitted_sequence_is_exactly_total_seq_len():
    stream = _stream([10] * 40, sequences=3)
    samples = list(stream)
    assert len(samples) == 3
    for sample in samples:
        assert sample["input_ids"].shape == (SEQ,)
        assert int(sample["lengths"].sum()) == SEQ


def test_stops_at_the_budget_rather_than_the_corpus():
    """A corpus far larger than the budget must not be walked to the end."""
    stream = _stream([10] * 10_000, sequences=2)
    assert len(list(stream)) == 2
    assert len(stream) == 2


def test_positions_restart_at_each_document_boundary():
    stream = _stream([7] * 40, sequences=2)
    for sample in stream:
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long), sample["lengths"][:-1]]), 0
        )
        # Every packed document begins a fresh position sequence.
        assert torch.all(sample["position_ids"][starts] == 0)
        for start, length in zip(
            starts.tolist(), sample["lengths"].tolist(), strict=True
        ):
            segment = sample["position_ids"][start : start + length]
            torch.testing.assert_close(segment, torch.arange(length))


def test_documents_longer_than_a_sequence_continue_across_the_split():
    """A document that overruns the window keeps counting positions, so the
    second half is not mistaken for a new document starting at position 0."""
    stream = _stream([SEQ * 2 + 5], sequences=2)
    first, second = list(stream)
    assert int(first["position_ids"][0]) == 0
    assert int(second["position_ids"][0]) == SEQ


def test_collated_document_ids_mark_each_document():
    stream = _stream([8] * 40, sequences=2)
    loader = DataLoader(
        stream, batch_size=1, collate_fn=CollateFn(max_len=SEQ, hidden_size=0)
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (1, SEQ)
    assert "hidden_states" not in batch
    assert "verifier_last_hidden_states" not in batch
    assert batch["error_records"] == 0

    document_ids = batch["document_ids"][0]
    assert torch.all(document_ids >= 0)  # fully packed, no padding
    # Ids must be contiguous runs, one per packed document.
    boundaries = int((document_ids[1:] != document_ids[:-1]).sum()) + 1
    assert boundaries == len(torch.unique(document_ids)) == SEQ // 8
