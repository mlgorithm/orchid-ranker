"""
Plot metrics (engagement and knowledge) over rounds for each CSV in a directory.

For each *_user_rounds.csv file, this script will:
 - group rows by student_method and round (sround)
 - compute mean value per round (optionally show std shading)
 - plot a line per student_method
 - save a PNG per selected metric next to the CSV (or in an output folder if provided)
 - also save a per-metric grid PNG combining all CSVs into subplots (one panel per model)
 - also save, per CSV and metric, a figure of profiles averaged within each student method (subplots by student_method)
 - also save, per CSV and metric, a figure where each line is a student method averaged over profiles (single panel)

Usage:
  # Default metrics: post_engagement and post_knowledge
  python experiments/plot_engagement.py --input-dir runs/oulad/open_nodp \
      --output-dir runs/oulad/open_nodp/plots

    # Custom metrics
  python experiments/plot_engagement.py --input-dir runs/oulad/open_nodp \
      --metrics post_engagement post_knowledge pre_engagement \
      --output-dir runs/oulad/open_nodp/plots

    # Recurse into subdirectories under runs and mirror outputs
    python experiments/plot_engagement.py --input-dir runs \
            --recursive \
            --output-dir runs/plots

        # Save plots within each folder (next to CSVs) regardless of output-dir
        python experiments/plot_engagement.py --input-dir runs \
                        --recursive \
                        --inplace

        # Choose label format for lines: model-method (default), method-profile, or method
        python experiments/plot_engagement.py --input-dir runs/oulad/open_nodp \
            --label-format method-profile

        # Generate additional per-CSV figures: profiles averaged within each student method
        python experiments/plot_engagement.py --input-dir runs/oulad/open_nodp \
            --inplace

        # Generate per-CSV figures: each line is a method averaged over profiles
        python experiments/plot_engagement.py --input-dir runs/oulad/open_nodp \
            --inplace

Notes:
 - CSVs are expected to contain: 'sround' (or 'round'), 'student_method', and the chosen metric columns.
 - '--engagement-column' is kept for backward compatibility; if '--metrics' is not provided,
   the script will plot [<engagement-column>, 'post_knowledge'] by default (deduplicated).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:  # ImportError or runtime issues
    sns = None  # type: ignore
    HAS_SEABORN = False


def _prettify_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def _find_round_column(columns: Sequence[str]) -> Optional[str]:
    """Return the column name used for rounds, supporting 'sround' or 'round'."""
    candidates = ["sround", "round", "Sround", "Round"]
    for c in candidates:
        if c in columns:
            return c
    return None


def _prepare_df_for_metric(
    csv_path: Path,
    metric_col: str,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load and validate data for a metric. Returns cleaned df and round column name."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}", file=sys.stderr)
        return None

    round_col = _find_round_column(df.columns)
    if round_col is None:
        print(
            f"[WARN] Skipping {csv_path.name}: could not find round column (expected one of 'sround' or 'round')",
            file=sys.stderr,
        )
        return None

    required_cols = {round_col, "student_method", metric_col}
    if not required_cols.issubset(df.columns):
        print(
            f"[WARN] Skipping {csv_path.name}: missing columns {required_cols - set(df.columns)}",
            file=sys.stderr,
        )
        return None

    # Ensure numeric types for plotting
    df = df.copy()
    df[round_col] = pd.to_numeric(df[round_col], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[round_col, metric_col, "student_method"])  # type: ignore[arg-type]

    if df.empty:
        print(f"[WARN] No valid data after cleaning for {csv_path.name}", file=sys.stderr)
        return None

    return df, round_col


def _plot_df_on_axes(
    df: pd.DataFrame,
    round_col: str,
    metric_col: str,
    ax: Optional[plt.Axes] = None,
    show_std: bool = True,
    label_format: str = "model-method",
):
    if HAS_SEABORN:
        sns.set(style="whitegrid")  # type: ignore[attr-defined]
    else:
        plt.style.use("ggplot")
    if ax is None:
        ax = plt.gca()

    # Build series according to label format
    model_name = None
    if "mode" in df.columns and not df["mode"].empty:
        try:
            model_name = str(df["mode"].iloc[0])
        except Exception:
            model_name = None

    series: List[Tuple[str, pd.DataFrame]] = []
    if label_format == "method-profile" and "profile" in df.columns:
        # Group by both to split lines if profiles differ
        for (method, profile), sub in df.groupby(["student_method", "profile"], dropna=False):
            profile_txt = "" if pd.isna(profile) or str(profile) == "" else f" - {profile}"
            label = f"{method}{profile_txt}"
            series.append((label, sub))
    else:
        # Default: group per method, optionally prefix model name
        for method in sorted(df["student_method"].unique()):
            sub = df[df["student_method"] == method]
            if label_format == "model-method" and model_name:
                label = f"{model_name} - {method}"
            else:
                label = f"{method}"
            series.append((label, sub))

    # Plot
    for label, sub in series:
        grouped = (
            sub.groupby(round_col, as_index=True)[metric_col]
            .agg(["mean", "std", "count"])
            .sort_index()
        )
        # Plot mean line
        ax.plot(grouped.index, grouped["mean"], label=label, linewidth=2.0)
        # Optional std shading where count > 1
        if show_std and (grouped["count"] > 1).any():
            upper = grouped["mean"] + grouped["std"].fillna(0)
            lower = grouped["mean"] - grouped["std"].fillna(0)
            ax.fill_between(grouped.index, lower, upper, alpha=0.15)

    ax.set_xlabel("Rounds")
    ax.set_ylabel(_prettify_metric(metric_col))
    legend_title = (
        "Learning model - Student method" if label_format == "model-method"
        else ("Student method - Profile" if label_format == "method-profile" else "Student method")
    )
    ax.legend(title=legend_title, loc="best")


def plot_csv_metric(
    csv_path: Path,
    metric_col: str,
    output_dir: Optional[Path] = None,
    show_std: bool = True,
    label_format: str = "model-method",
) -> Optional[Path]:
    """Plot one metric over rounds for a single CSV and save PNG."""
    prep = _prepare_df_for_metric(csv_path, metric_col)
    if prep is None:
        return None
    df, round_col = prep

    plt.figure(figsize=(9, 5))
    _plot_df_on_axes(df, round_col, metric_col, ax=plt.gca(), show_std=show_std, label_format=label_format)

    # Titles and labels
    pretty_metric = _prettify_metric(metric_col)
    title = f"{csv_path.stem}: {pretty_metric} by Student Method"
    plt.title(title)
    plt.tight_layout()

    # Save
    out_dir = output_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_metric = metric_col.lower()
    out_path = out_dir / f"{csv_path.stem}_{safe_metric}_by_method.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def find_csvs(input_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(input_dir.glob("**/*_user_rounds.csv"))
    return sorted(input_dir.glob("*_user_rounds.csv"))


def plot_csv_profile_averages(
    csv_path: Path,
    metric_col: str,
    output_dir: Optional[Path] = None,
    show_std: bool = True,
) -> Optional[Path]:
    """For a single CSV and metric, plot average per profile within each student_method.

    Layout: one subplot per student_method; each subplot shows lines for profiles,
    where each line is the mean of the metric per round for that profile.
    """
    prep = _prepare_df_for_metric(csv_path, metric_col)
    if prep is None:
        return None
    df, round_col = prep

    if "profile" not in df.columns:
        print(f"[WARN] {csv_path.name} has no 'profile' column; skipping profile averages.", file=sys.stderr)
        return None

    methods = sorted(df["student_method"].dropna().unique())
    if len(methods) == 0:
        print(f"[WARN] {csv_path.name} has no student_method values; skipping profile averages.", file=sys.stderr)
        return None

    # Figure layout: up to 3 columns for readability
    import math
    cols = min(3, max(1, int(math.ceil(math.sqrt(len(methods))))))
    rows = math.ceil(len(methods) / cols)

    if HAS_SEABORN:
        sns.set(style="whitegrid")  # type: ignore[attr-defined]
    else:
        plt.style.use("ggplot")
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, method in enumerate(methods):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        mdf = df[df["student_method"] == method]
        profiles = sorted(mdf["profile"].dropna().unique())
        if len(profiles) == 0:
            # Nothing to plot in this panel
            ax.text(0.5, 0.5, f"{method}: no profiles", ha="center", va="center")
            ax.axis("off")
            continue

        for profile in profiles:
            sub = mdf[mdf["profile"] == profile]
            grouped = (
                sub.groupby(round_col, as_index=True)[metric_col]
                .agg(["mean", "std", "count"]).sort_index()
            )
            ax.plot(grouped.index, grouped["mean"], label=str(profile), linewidth=2.0)
            if show_std and (grouped["count"] > 1).any():
                upper = grouped["mean"] + grouped["std"].fillna(0)
                lower = grouped["mean"] - grouped["std"].fillna(0)
                ax.fill_between(grouped.index, lower, upper, alpha=0.15)

        ax.set_title(str(method))
        ax.set_xlabel("Rounds")
        ax.set_ylabel(_prettify_metric(metric_col))
        ax.legend(title="Profile", loc="best")

    # Hide unused axes
    for idx in range(len(methods), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle(f"{csv_path.stem}: Profile averages per student method — {_prettify_metric(metric_col)}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_dir = output_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_{metric_col.lower()}_profiles_by_method.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_csv_method_avg_over_profiles(
    csv_path: Path,
    metric_col: str,
    output_dir: Optional[Path] = None,
    show_std: bool = True,
) -> Optional[Path]:
    """For a single CSV and metric, plot one panel where each line is a student_method
    averaged over profiles (mean across profiles at each round).
    """
    prep = _prepare_df_for_metric(csv_path, metric_col)
    if prep is None:
        return None
    df, round_col = prep

    if "profile" not in df.columns:
        # Fall back to regular per-method plot if no profiles exist
        print(f"[WARN] {csv_path.name} has no 'profile' column; skipping method avg over profiles.", file=sys.stderr)
        return None

    if HAS_SEABORN:
        sns.set(style="whitegrid")  # type: ignore[attr-defined]
    else:
        plt.style.use("ggplot")

    plt.figure(figsize=(9, 5))

    methods = sorted(df["student_method"].dropna().unique())
    if len(methods) == 0:
        print(f"[WARN] {csv_path.name} has no student_method values; skipping method avg over profiles.", file=sys.stderr)
        return None

    for method in methods:
        sub = df[df["student_method"] == method]
        # First compute mean per profile per round, then aggregate across profiles
        per_prof = (
            sub.dropna(subset=["profile"])  # exclude NaN profiles for profile-based averaging
            .groupby([round_col, "profile"], as_index=False)[metric_col]
            .mean()
        )
        grouped = per_prof.groupby(round_col)[metric_col].agg(["mean", "std", "count"]).sort_index()
        plt.plot(grouped.index, grouped["mean"], label=str(method), linewidth=2.0)
        if show_std and (grouped["count"] > 1).any():
            upper = grouped["mean"] + grouped["std"].fillna(0)
            lower = grouped["mean"] - grouped["std"].fillna(0)
            plt.fill_between(grouped.index, lower, upper, alpha=0.15)

    pretty_metric = _prettify_metric(metric_col)
    plt.title(f"{csv_path.stem}: {pretty_metric} — method averages over profiles")
    plt.xlabel("Rounds")
    plt.ylabel(pretty_metric)
    plt.legend(title="Student method", loc="best")
    plt.tight_layout()

    out_dir = output_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_{metric_col.lower()}_method_avg_over_profiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_grid_for_metric(
    csvs: List[Path],
    metric_col: str,
    output_dir: Optional[Path],
    show_std: bool = True,
    label_format: str = "model-method",
) -> Optional[Path]:
    """Create a grid of subplots (one per CSV/model) for a given metric."""
    # Prepare data frames that can actually be plotted
    prepared: List[Tuple[Path, pd.DataFrame, str]] = []
    for csv in csvs:
        prep = _prepare_df_for_metric(csv, metric_col)
        if prep is None:
            continue
        df, round_col = prep
        prepared.append((csv, df, round_col))

    if not prepared:
        print(f"[WARN] No valid CSVs for metric {metric_col}; skipping grid.", file=sys.stderr)
        return None

    n = len(prepared)
    # Layout: aim for a rectangular grid close to square
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    if HAS_SEABORN:
        sns.set(style="whitegrid")  # type: ignore[attr-defined]
    else:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, (csv, df, round_col) in enumerate(prepared):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        _plot_df_on_axes(df, round_col, metric_col, ax=ax, show_std=show_std, label_format=label_format)
        ax.set_title(csv.stem)

    # Hide unused axes if any
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle(f"{_prettify_metric(metric_col)} by Student Method (All Models)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    first_dir = output_dir or prepared[0][0].parent
    first_dir.mkdir(parents=True, exist_ok=True)
    out_path = first_dir / f"all_models_{metric_col.lower()}_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot metrics over rounds per student method.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_user_rounds.csv files",
    )
    parser.add_argument(
        "--engagement-column",
        type=str,
        default="post_engagement",
        help="Backward-compatible engagement column; ignored if --metrics is provided.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="List of metric columns to plot (e.g., post_engagement post_knowledge)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: alongside each CSV)",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable standard deviation shading",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable generating per-metric grid figures",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for *_user_rounds.csv under --input-dir",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Save plots in each CSV's folder and grids in each directory, ignoring --output-dir",
    )
    parser.add_argument(
        "--label-format",
        type=str,
        choices=["model-method", "method-profile", "method"],
        default="model-method",
        help="How to label lines: 'model-method' (default), 'method-profile', or 'method'",
    )
    parser.add_argument(
        "--no-profile-avg",
        action="store_true",
        help="Disable generating per-CSV profile-average figures",
    )
    parser.add_argument(
        "--no-method-avg",
        action="store_true",
        help="Disable generating per-CSV method-average-over-profiles figures",
    )

    args = parser.parse_args(argv)
    input_dir: Path = args.input_dir
    engagement_col: str = args.engagement_column
    metrics: Optional[Sequence[str]] = args.metrics
    output_dir: Optional[Path] = args.output_dir
    show_std: bool = not args.no_std
    no_grid: bool = args.no_grid
    recursive: bool = args.recursive
    inplace: bool = args.inplace
    label_format: str = args.label_format
    no_profile_avg: bool = args.no_profile_avg
    no_method_avg: bool = args.no_method_avg

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input dir not found or not a directory: {input_dir}", file=sys.stderr)
        return 2

    csvs = find_csvs(input_dir, recursive=recursive)
    if not csvs:
        print(f"[WARN] No *_user_rounds.csv files found in {input_dir}", file=sys.stderr)
        return 1

    # Determine metrics to plot
    if metrics is None:
        # Default to two plots: engagement and knowledge
        default_metrics = [engagement_col, "post_knowledge"]
        # Deduplicate while preserving order
        seen = set()
        metrics_to_plot = [m for m in default_metrics if not (m in seen or seen.add(m))]
    else:
        metrics_to_plot = list(metrics)

    # Group CSVs by their parent directory to build per-directory grids
    from collections import defaultdict
    groups: dict[Path, List[Path]] = defaultdict(list)
    for csv in csvs:
        groups[csv.parent].append(csv)

    saved: List[Path] = []
    for csv in csvs:
        # Determine output directory for this csv (mirror structure if recursive and --output-dir provided)
        per_csv_out_dir: Optional[Path]
        if inplace:
            per_csv_out_dir = csv.parent
        elif output_dir is not None:
            try:
                rel = csv.parent.relative_to(input_dir)
            except Exception:
                rel = Path("")
            per_csv_out_dir = output_dir / rel
        else:
            per_csv_out_dir = None

        for metric_col in metrics_to_plot:
            out_path = plot_csv_metric(csv, metric_col=metric_col, output_dir=per_csv_out_dir, show_std=show_std, label_format=label_format)
            if out_path is not None:
                saved.append(out_path)
                print(f"[OK] Saved {out_path}")
            if not no_profile_avg:
                avg_out = plot_csv_profile_averages(csv, metric_col=metric_col, output_dir=per_csv_out_dir, show_std=show_std)
                if avg_out is not None:
                    saved.append(avg_out)
                    print(f"[OK] Saved {avg_out}")
            if not no_method_avg:
                meth_out = plot_csv_method_avg_over_profiles(csv, metric_col=metric_col, output_dir=per_csv_out_dir, show_std=show_std)
                if meth_out is not None:
                    saved.append(meth_out)
                    print(f"[OK] Saved {meth_out}")

    # Per metric grid plots
    if not no_grid:
        for group_dir, group_csvs in groups.items():
            # Determine output directory for this group
            per_group_out_dir: Optional[Path]
            if inplace:
                per_group_out_dir = group_dir
            elif output_dir is not None:
                try:
                    rel = group_dir.relative_to(input_dir)
                except Exception:
                    rel = Path("")
                per_group_out_dir = output_dir / rel
            else:
                per_group_out_dir = None

            for metric_col in metrics_to_plot:
                grid_out = plot_grid_for_metric(group_csvs, metric_col=metric_col, output_dir=per_group_out_dir, show_std=show_std, label_format=label_format)
                if grid_out is not None:
                    saved.append(grid_out)
                    print(f"[OK] Saved {grid_out}")

    print(f"Done. Generated {len(saved)} plot(s).")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
