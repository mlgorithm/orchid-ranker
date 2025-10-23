"""Common data visualisations used in Orchid Ranker experiments."""
from __future__ import annotations

from typing import Optional, Tuple

import math
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "adaptive": "#d62728",
    "fixed": "#1f77b4",
    "linucb": "#2ca02c",
    "als": "#9467bd",
    "random": "#8c564b",
    "popularity": "#7f7f7f",
}


def _ensure_matplotlib_style():
    plt.rcParams.setdefault("axes.grid", True)
    plt.rcParams.setdefault("grid.alpha", 0.2)


def plot_user_activity(
    interactions: pd.DataFrame,
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of the busiest users by number of interactions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    _ensure_matplotlib_style()

    counts = interactions.groupby("u").size().sort_values(ascending=False).head(top_n)
    counts.plot(kind="barh", ax=ax, color="#4F6D7A")
    ax.set_xlabel("Interactions")
    ax.set_ylabel("User ID")
    ax.invert_yaxis()
    ax.set_title(f"Top {len(counts)} active users")
    return ax


def plot_item_difficulty(
    item_side: pd.DataFrame,
    difficulty_col: str = "difficulty",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram of item difficulties (assumes column scaled to [0,1])."""
    if difficulty_col not in item_side.columns:
        raise KeyError(f"Column '{difficulty_col}' not found in item side info")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    _ensure_matplotlib_style()

    ax.hist(item_side[difficulty_col].dropna(), bins=20, color="#FF8C42", alpha=0.85)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Count")
    ax.set_title("Item difficulty distribution")
    return ax


def plot_learning_curve(
    round_summary: pd.DataFrame,
    metric: str = "mean_accuracy",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Line plot of a round-level metric for adaptive vs fixed policies."""
    if {"round", "policy", metric}.difference(round_summary.columns):
        missing = {"round", "policy", metric}.difference(round_summary.columns)
        raise KeyError(f"round_summary missing columns: {missing}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    _ensure_matplotlib_style()

    for policy, grp in round_summary.groupby("policy"):
        ax.plot(grp["round"], grp[metric], marker="o", label=policy.title())
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} across rounds")
    ax.legend()
    return ax


def plot_acceptance_heatmap(
    interactions: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    bins: int = 50,
) -> plt.Axes:
    """2D heatmap of acceptance vs difficulty (requires both columns)."""
    for col in ("accept", "difficulty"):
        if col not in interactions.columns:
            raise KeyError(f"Column '{col}' required for heatmap")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    _ensure_matplotlib_style()

    accept = interactions["accept"].astype(float)
    diff = interactions["difficulty"].astype(float)
    h = ax.hist2d(diff, accept, bins=bins, range=[[0, 1], [0, 1]], cmap="Blues")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Acceptance")
    ax.set_title("Acceptance vs difficulty density")
    plt.colorbar(h[3], ax=ax, label="Counts")
    return ax


def plot_round_comparison(df_round: pd.DataFrame, metrics: Iterable[str], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    _ensure_matplotlib_style()
    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x"]
    df_round = df_round.sort_values("round")
    metrics = [m for m in metrics if m in df_round.columns]
    grouped = df_round.groupby("mode", sort=False)
    fallback_colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for m_idx, metric in enumerate(metrics):
        style = styles[m_idx % len(styles)]
        marker = markers[m_idx % len(markers)]
        metric_label = metric.replace("_", " ").title()
        for mode, grp in grouped:
            color = PALETTE.get(mode.lower(), next(fallback_colors))
            ax.plot(
                grp["round"],
                grp[metric],
                label=f"{mode.title()} – {metric_label}",
                color=color,
                linestyle=style,
                linewidth=2.0 if mode.lower() == "adaptive" else 1.4,
                marker=marker if mode.lower() == "adaptive" else None,
                alpha=0.95 if mode.lower() == "adaptive" else 0.75,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.set_title("Metric comparison across rounds")
    return ax


def plot_knowledge_trajectory(round_summary: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if {"round", "mean_knowledge", "mode"}.difference(round_summary.columns):
        raise KeyError("round_summary must contain round, mean_knowledge, mode")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    _ensure_matplotlib_style()
    for policy, grp in round_summary.groupby("mode"):
        ax.plot(grp["round"], grp["mean_knowledge"], marker="o", label=policy.title())
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean knowledge")
    ax.set_title("Knowledge trajectory")
    ax.legend()
    return ax


def plot_metric_trajectory(
    round_summary: pd.DataFrame,
    metric: str,
    *,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    required = {"round", "mode", metric}
    missing = required.difference(round_summary.columns)
    if missing:
        raise KeyError(f"round_summary missing columns: {missing}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    _ensure_matplotlib_style()

    grouped = round_summary.groupby("mode", sort=False)
    # deterministically order modes
    order = sorted(grouped.groups.keys(), key=lambda m: (0 if m == "adaptive" else 1, m))
    fallback_colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for idx, policy in enumerate(order):
        grp = grouped.get_group(policy).sort_values("round")
        color = PALETTE.get(policy.lower(), next(fallback_colors))
        is_adaptive = policy.lower() == "adaptive"
        styles = ["-", "--", "-.", ":"]
        style = styles[idx % len(styles)]
        width = 2.4 if is_adaptive else 1.6
        marker = "o" if is_adaptive else ["s", "^", "D", "x"][idx % 4]
        alpha = 0.95 if is_adaptive else 0.8
        ax.plot(
            grp["round"],
            grp[metric],
            label=policy.title(),
            color=color,
            linestyle=style,
            linewidth=width,
            marker=marker,
            markersize=5 if is_adaptive else 4,
            alpha=alpha,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} trajectory")
    ax.legend(frameon=True)
    return ax


def plot_metric_grid(
    round_summary: pd.DataFrame,
    metrics: list[str],
    *,
    ncols: int = 3,
    figsize: Tuple[int, int] = (14, 8),
    sharex: bool = True,
    sharey: bool = False,
) -> plt.Figure:
    if not metrics:
        raise ValueError("metrics list cannot be empty")
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes, metrics):
        plot_metric_trajectory(
            round_summary,
            metric,
            ylabel=metric.replace("_", " ").title(),
            title=None,
            ax=ax,
        )
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
