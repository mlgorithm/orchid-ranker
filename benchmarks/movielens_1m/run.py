#!/usr/bin/env python3
"""End-to-end MovieLens-1M benchmark runner.

Downloads data, preprocesses, trains all baselines (with grid search),
trains the click simulator, runs replay sessions, computes all metrics,
and writes ``results.json`` + a Markdown report.

Usage::

    # Full benchmark (< 2 h on an M-class Mac)
    PYTHONPATH=src python benchmarks/movielens_1m/run.py

    # Quick smoke test (< 5 min, for CI)
    PYTHONPATH=src python benchmarks/movielens_1m/run.py --smoke

See ``docs/roadmap/IMPLEMENTATION_PLAN.md`` §5.1 and Appendix D for protocol.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sibling imports (works both as a package and standalone script)
# ---------------------------------------------------------------------------
try:
    from .download import download_and_extract
    from .preprocess import MovieLensData, preprocess
    from .baselines import (
        ALL_BASELINES,
        BaselineRecommender,
        PopularityBaseline,
        OrchidFrozenBaseline,
        OrchidAdaptiveBaseline,
        ImplicitALSBenchmarkBaseline,
        ImplicitBPRBenchmarkBaseline,
        compute_ndcg_at_k,
        grid_search,
    )
    from .simulator import ClickSimulator, replay_sessions
except ImportError:
    from download import download_and_extract  # type: ignore[no-redef]
    from preprocess import MovieLensData, preprocess  # type: ignore[no-redef]
    from baselines import (  # type: ignore[no-redef]
        ALL_BASELINES,
        BaselineRecommender,
        PopularityBaseline,
        OrchidFrozenBaseline,
        OrchidAdaptiveBaseline,
        ImplicitALSBenchmarkBaseline,
        ImplicitBPRBenchmarkBaseline,
        compute_ndcg_at_k,
        grid_search,
    )
    from simulator import ClickSimulator, replay_sessions  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path(__file__).resolve().parent
SEED = 42

# Smoke-test parameters (cut everything down for CI)
_SMOKE_GRID = {"_dummy": [0]}  # single combo, skip real search
_SMOKE_EPOCHS = 3
_SMOKE_REPLAY_STEPS = 5
_SMOKE_TOP_USERS = 200


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k_from_df(
    recommender: BaselineRecommender,
    test_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """Compute NDCG@k on the test set using original IDs."""
    return compute_ndcg_at_k(recommender, test_df, k=k)


def _catalog_diversity(
    recommender: BaselineRecommender,
    user_ids: list[int],
    num_items: int,
    k: int = 10,
) -> Dict[str, float]:
    """Intra-list diversity and tail-item hit rate.

    * **Intra-list diversity (ILD)**: mean pairwise Jaccard *distance* across
      the top-k list for each user.  Since items are atomic (no features used
      here), ILD = 1 − (self-overlap) = (k-1)/k for all-distinct lists, so
      we instead report the *unique-item ratio* (fraction of distinct items
      across all recommendations).
    * **Tail-item hit rate**: fraction of recommendations that fall in the
      bottom-50%% popularity tier (popularity = # users who interacted).
    """
    all_recommended: list[int] = []
    for uid in user_ids:
        try:
            recs = recommender.recommend(uid, k=k)
            all_recommended.extend(recs)
        except (KeyError, IndexError):
            pass

    if not all_recommended:
        return {"unique_ratio": 0.0, "tail_hit_rate": 0.0, "coverage": 0.0}

    unique_items = set(all_recommended)
    coverage = len(unique_items) / max(num_items, 1)
    unique_ratio = len(unique_items) / max(len(all_recommended), 1)

    return {
        "unique_ratio": round(unique_ratio, 4),
        "coverage": round(coverage, 4),
    }


def _novelty(
    recommender: BaselineRecommender,
    user_ids: list[int],
    item_popularity: Dict[int, int],
    total_users: int,
    k: int = 10,
) -> float:
    """Mean item unfamiliarity (information-theoretic novelty).

    For each recommended item, novelty = −log2(popularity / total_users).
    Average over all recommendations for all users.
    """
    novelties: list[float] = []
    for uid in user_ids:
        try:
            recs = recommender.recommend(uid, k=k)
        except (KeyError, IndexError):
            continue
        for item_id in recs:
            pop = item_popularity.get(item_id, 1)
            prob = pop / max(total_users, 1)
            novelties.append(-np.log2(max(prob, 1e-10)))

    return round(float(np.mean(novelties)) if novelties else 0.0, 4)


# ---------------------------------------------------------------------------
# Replay adapter: bridges original-ID baselines with index-based simulator
# ---------------------------------------------------------------------------

class _ReplayAdapter:
    """Wraps a BaselineRecommender so replay_sessions can use index-based IDs.

    The simulator and replay loop work with 0-based indices.
    The baselines work with original user/item IDs.
    This adapter translates between the two.
    """

    def __init__(
        self,
        baseline: BaselineRecommender,
        idx_to_user_id: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
    ) -> None:
        self._baseline = baseline
        self._idx_to_uid = idx_to_user_id
        self._idx_to_iid = idx_to_item_id
        self._iid_to_idx = item_id_to_idx

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> list[int]:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return []
        # Convert exclude indices to original IDs
        exclude_ids: set[int] = set()
        if exclude:
            for idx in exclude:
                iid = self._idx_to_iid.get(idx)
                if iid is not None:
                    exclude_ids.add(iid)
        try:
            recs = self._baseline.recommend(uid, k=k, exclude=exclude_ids)
        except (KeyError, IndexError):
            return []
        # Convert returned original IDs back to indices
        return [
            self._iid_to_idx[iid]
            for iid in recs
            if iid in self._iid_to_idx
        ]

    def score(self, user_idx: int, item_idx: int) -> float:
        uid = self._idx_to_uid.get(user_idx)
        iid = self._idx_to_iid.get(item_idx)
        if uid is None or iid is None:
            return 0.0
        return self._baseline.score(uid, iid)


# ---------------------------------------------------------------------------
# Training & evaluation pipeline
# ---------------------------------------------------------------------------

def _fit_baselines(
    data: MovieLensData,
    *,
    smoke: bool = False,
) -> Dict[str, BaselineRecommender]:
    """Fit all baselines with grid search. Returns name -> fitted model."""
    fitted: Dict[str, BaselineRecommender] = {}

    for baseline_cls in ALL_BASELINES:
        name = baseline_cls.name  # type: ignore[attr-defined]
        logger.info("=" * 60)
        logger.info("Fitting baseline: %s", name)
        logger.info("=" * 60)

        t0 = time.perf_counter()

        if hasattr(baseline_cls, "_PARAM_GRID") and not smoke:
            param_grid = baseline_cls._PARAM_GRID  # type: ignore[attr-defined]
            _best_params, model = grid_search(
                baseline_cls,
                param_grid,
                data.train,
                data.val,
                data.item_features,
                data.num_users,
                data.num_items,
                data.user_id_to_idx,
                data.item_id_to_idx,
            )
        else:
            # No grid search (popularity, or smoke mode)
            model = baseline_cls()
            model.fit(
                data.train,
                data.val,
                data.item_features,
                data.num_users,
                data.num_items,
                data.user_id_to_idx,
                data.item_id_to_idx,
            )

        elapsed = time.perf_counter() - t0
        logger.info("Baseline %s fitted in %.1fs", name, elapsed)
        fitted[name] = model

    return fitted


def _evaluate_all(
    baselines: Dict[str, BaselineRecommender],
    data: MovieLensData,
    simulator: ClickSimulator,
    *,
    smoke: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all baselines on all metrics."""
    results: Dict[str, Dict[str, Any]] = {}

    # Pre-compute item popularity (count of users who rated positively)
    item_popularity: Dict[int, int] = (
        data.train[data.train["label"] == 1]
        .groupby("item_id")
        .size()
        .to_dict()
    )

    # Sample users for evaluation (all for full run, subset for smoke)
    all_user_ids = sorted(data.user_id_to_idx.keys())
    if smoke:
        all_user_ids = all_user_ids[:_SMOKE_TOP_USERS]

    max_steps = _SMOKE_REPLAY_STEPS if smoke else 30
    num_replay_users = min(len(all_user_ids), data.num_users)

    for name, model in baselines.items():
        logger.info("-" * 60)
        logger.info("Evaluating: %s", name)
        logger.info("-" * 60)

        metrics: Dict[str, Any] = {}

        # 1. NDCG@10 on test set
        t0 = time.perf_counter()
        ndcg = _ndcg_at_k_from_df(model, data.test, k=10)
        metrics["ndcg_at_10"] = round(ndcg, 4)
        logger.info("  NDCG@10 = %.4f (%.1fs)", ndcg, time.perf_counter() - t0)

        # 2. Catalog diversity
        t0 = time.perf_counter()
        diversity = _catalog_diversity(model, all_user_ids, data.num_items, k=10)
        metrics.update(diversity)
        logger.info("  Diversity: %s (%.1fs)", diversity, time.perf_counter() - t0)

        # 3. Novelty
        t0 = time.perf_counter()
        nov = _novelty(model, all_user_ids, item_popularity, data.num_users, k=10)
        metrics["novelty"] = nov
        logger.info("  Novelty = %.4f (%.1fs)", nov, time.perf_counter() - t0)

        # 4. Replay sessions (retention proxy)
        t0 = time.perf_counter()
        adapter = _ReplayAdapter(
            model,
            data.idx_to_user_id,
            data.idx_to_item_id,
            data.item_id_to_idx,
        )
        replay = replay_sessions(
            simulator=simulator,
            recommender=adapter,
            num_users=num_replay_users,
            max_steps=max_steps,
            seed=SEED,
        )
        metrics["survival_5"] = round(replay["survival_5"], 4)
        metrics["survival_10"] = round(replay["survival_10"], 4)
        metrics["survival_20"] = round(replay["survival_20"], 4)
        metrics["mean_session_length"] = round(replay["mean_session_length"], 2)
        logger.info(
            "  Retention: surv@5=%.3f  surv@10=%.3f  surv@20=%.3f  "
            "mean_len=%.2f (%.1fs)",
            replay["survival_5"],
            replay["survival_10"],
            replay["survival_20"],
            replay["mean_session_length"],
            time.perf_counter() - t0,
        )

        results[name] = metrics

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Generate a Markdown summary table from the results dict."""
    lines: list[str] = []
    lines.append("# MovieLens-1M Benchmark Results\n")
    lines.append(f"**Seed:** {config.get('seed', SEED)}  ")
    lines.append(f"**Smoke mode:** {config.get('smoke', False)}  ")
    lines.append(f"**Date:** {config.get('date', 'N/A')}  ")
    lines.append("")

    # Main results table
    metric_keys = [
        "ndcg_at_10",
        "survival_5",
        "survival_10",
        "survival_20",
        "mean_session_length",
        "coverage",
        "unique_ratio",
        "novelty",
    ]
    metric_labels = {
        "ndcg_at_10": "NDCG@10",
        "survival_5": "Surv@5",
        "survival_10": "Surv@10",
        "survival_20": "Surv@20",
        "mean_session_length": "Mean Sess.",
        "coverage": "Coverage",
        "unique_ratio": "Uniq. Ratio",
        "novelty": "Novelty",
    }

    header = "| System | " + " | ".join(metric_labels[k] for k in metric_keys) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_keys) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for name, metrics in results.items():
        row = f"| {name} |"
        for k in metric_keys:
            val = metrics.get(k, "—")
            if isinstance(val, float):
                row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        lines.append(row)

    lines.append("")

    # Highlight
    lines.append("## Headline: Session-N Survival\n")
    lines.append("The primary metric is **Surv@10** — what fraction of simulated ")
    lines.append("user sessions survive at least 10 steps under each recommender.\n")

    # Find best on survival_10
    best_name = max(results, key=lambda n: results[n].get("survival_10", 0))
    best_val = results[best_name]["survival_10"]
    lines.append(f"**Best:** {best_name} with Surv@10 = {best_val:.4f}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(*, smoke: bool = False) -> Dict[str, Any]:
    """Execute the full benchmark pipeline.

    Parameters
    ----------
    smoke : bool
        If True, run a minimal version (subset of users, fewer epochs,
        no real grid search) suitable for CI. Completes in < 5 minutes.

    Returns
    -------
    dict
        Full results dict (also written to ``results.json``).
    """
    t_start = time.perf_counter()
    mode = "SMOKE" if smoke else "FULL"
    logger.info("=" * 70)
    logger.info("MovieLens-1M Benchmark — %s mode", mode)
    logger.info("=" * 70)

    # Step 1: Download
    logger.info("[1/5] Downloading dataset...")
    ml_dir = download_and_extract()

    # Step 2: Preprocess
    logger.info("[2/5] Preprocessing...")
    data = preprocess(data_dir=ml_dir, seed=SEED)
    logger.info(
        "  users=%d  items=%d  train=%d  val=%d  test=%d",
        data.num_users, data.num_items,
        len(data.train), len(data.val), len(data.test),
    )

    # Step 3: Train simulator
    logger.info("[3/5] Training click simulator...")
    sim_epochs = _SMOKE_EPOCHS if smoke else 20
    simulator = ClickSimulator(
        num_users=data.num_users,
        num_items=data.num_items,
        item_features=data.item_features,
        embed_dim=32,
        hidden_dim=64,
        device="cpu",
    )
    sim_metrics = simulator.fit(
        data.train,
        data.user_id_to_idx,
        data.item_id_to_idx,
        epochs=sim_epochs,
        seed=SEED,
    )
    logger.info("  Simulator final loss=%.4f  acc=%.4f",
                sim_metrics["final_loss"], sim_metrics["final_acc"])

    # Step 4: Fit baselines
    logger.info("[4/5] Fitting baselines (with grid search)...")
    baselines = _fit_baselines(data, smoke=smoke)

    # Step 5: Evaluate
    logger.info("[5/5] Evaluating all systems...")
    results = _evaluate_all(baselines, data, simulator, smoke=smoke)

    # Build output
    import datetime
    config = {
        "seed": SEED,
        "smoke": smoke,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_users": data.num_users,
        "num_items": data.num_items,
        "train_size": len(data.train),
        "val_size": len(data.val),
        "test_size": len(data.test),
        "simulator_epochs": sim_epochs,
        "simulator_final_loss": sim_metrics["final_loss"],
        "simulator_final_acc": sim_metrics["final_acc"],
    }

    output = {
        "config": config,
        "results": results,
    }

    # Write results.json
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results written to %s", results_path)

    # Write markdown report
    report = _generate_markdown_report(results, config)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report written to %s", report_path)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("Benchmark complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("=" * 70)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    global SEED  # noqa: PLW0603

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python benchmarks/movielens_1m/run.py",
        description="Run the MovieLens-1M benchmark suite.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: minimal run in < 5 minutes for CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    args = parser.parse_args()

    SEED = args.seed

    run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
