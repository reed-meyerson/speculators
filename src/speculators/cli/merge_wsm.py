"""
Average a warmup-stable-merge checkpoint window into one model.

Training holds a rolling set of checkpoints spanning the trailing fraction of
elapsed steps. Averaging them approximates what a decay phase would have
produced, without committing to a decay horizon -- so a long run can hand a
usable model to a finetune at any point, not only at a planned end.

Training merges the window automatically when it finishes. Use this to merge
mid-run, or to re-merge a subset.

Usage::

    speculators merge-wsm ./checkpoints/wsm --output ./merged

    # merge only the most recent few:
    speculators merge-wsm ./checkpoints/wsm --output ./merged --last 4
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from speculators.train.wsm import merge_checkpoints, wsm_checkpoint_dirs

console = Console()

__all__ = ["merge_wsm"]


def merge_wsm(
    window: Annotated[
        Path,
        typer.Argument(help="The `wsm` directory written during training."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the merged model."),
    ],
    last: Annotated[
        int | None,
        typer.Option("--last", help="Merge only the most recent N checkpoints."),
    ] = None,
):
    """Average a WSM checkpoint window into a single model."""
    sources = wsm_checkpoint_dirs(window)
    if not sources:
        raise typer.BadParameter(f"No WSM checkpoints found under {window}")
    if last is not None:
        if last < 1:
            raise typer.BadParameter(f"--last must be >= 1, got {last}")
        sources = sources[-last:]

    stats = merge_checkpoints(sources, output)
    console.print(
        f"Merged [bold]{stats['merged']}[/bold] checkpoints "
        f"({', '.join(s.name for s in sources)})\n"
        f"  averaged tensors: {stats['averaged_tensors']}\n"
        f"  written to:       {output}"
    )
