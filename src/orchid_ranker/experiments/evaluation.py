"""Evaluation utilities, loggers, and data classes for Orchid Ranker experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


def _fmt_scalar_or_mean(x, ndigits=3) -> str:
    """Format a scalar or mean of a sequence with specified precision.

    Parameters
    ----------
    x : scalar or array-like
        The value(s) to format.
    ndigits : int, optional
        Number of decimal digits to display, by default 3.

    Returns
    -------
    str
        Formatted string representation of the value or mean.
    """
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            return f"{float(np.mean(x)):.{ndigits}f}"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "nan"


class _CompositeLogger:
    """Logger that writes records to both memory and disk (JSONL format).

    Parameters
    ----------
    path : Path
        Path to write the JSONL log file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.records: List[dict] = []

    def log(self, obj: dict) -> None:
        """Log a record to both memory and disk.

        Parameters
        ----------
        obj : dict
            The record to log.
        """
        self.records.append(obj)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


class _MemoryLogger:
    """Logger that stores records in memory only.
    """

    def __init__(self) -> None:
        self.records: List[dict] = []

    def log(self, obj: dict) -> None:
        """Log a record to memory.

        Parameters
        ----------
        obj : dict
            The record to log.
        """
        self.records.append(obj)


@dataclass
class SummaryRow:
    """Summary statistics for an experiment mode/run.

    Parameters
    ----------
    mode : str
        Mode label (e.g., "baseline", "adaptive").
    accuracy : float
        Accuracy metric value.
    accept_rate : float
        Fraction of items accepted by users.
    novelty_rate : float
        Fraction of novel/unseen items recommended.
    serendipity : float
        Serendipity metric value.
    mean_knowledge : float
        Average user knowledge across rounds.
    epsilon_cum : float
        Cumulative privacy budget consumed.
    mean_engagement : float, optional
        Average user engagement (default: NaN).
    """
    mode: str
    accuracy: float
    accept_rate: float
    novelty_rate: float
    serendipity: float
    mean_knowledge: float
    epsilon_cum: float
    mean_engagement: float = float("nan")


__all__ = [
    "_fmt_scalar_or_mean",
    "_CompositeLogger",
    "_MemoryLogger",
    "SummaryRow",
]
