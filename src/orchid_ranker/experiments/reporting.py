"""Batch runner for Orchid Ranker experiments on EdNet and OULAD.

Layout (per run):
  runs/<dataset>/<experiment>/
    logs/{mode}.jsonl
    tables/
      per_round/{mode}_rounds.csv
      per_user_round/{mode}_user_rounds.csv
      per_student/{mode}/{method}/{user_name}.csv
      aggregates/adaptive_model_means.csv
      aggregates/mode_means.csv
    figures/
      adaptive/<metric>_by_model.{png,svg}
      compare_modes/<metric>_compare_modes.{png,svg}
    summary.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orchid_ranker.dp import get_dp_config
from orchid_ranker.experiments import RankingExperiment

# -----------------------------
# Cohort + profiles
# -----------------------------
COHORT_SIZE = 2
STUDENT_METHODS = ["irt", "zpd"]
INITIAL_PROFILES = [
    {"name": "struggling", "knowledge": 0.2, "fatigue": 0.7, "engagement": 0.4, "trust": 0.3},
]

# -----------------------------
# Adaptive policy presets
# -----------------------------
ADAPTIVE_BASE_PARAMS = {
    "hidden": 96,
    "emb_dim": 48,
    "adapter_slots": 512,
    "kl_beta": 0.02,
    "blend_increment": 0.16,
    "teacher_ema": 0.9,
    "entropy_lambda": 0.08,
    "info_gain_lambda": 0.12,
    "linucb_alpha": 1.6,
    "use_linucb": False,
    "use_bootts": False,
    "ts_heads": 16,
    "ts_alpha": 0.95,
    "zpd_margin": 0.14,
}
DATASET_ADJUSTMENTS = {
    "ednet": {"blend_increment": 0.18, "entropy_lambda": 0.09, "info_gain_lambda": 0.14},
    "oulad": {"blend_increment": 0.16},
}
DP_STRENGTH_BASE = {
    "none": {},
    "standard": {"entropy_lambda": 0.11, "info_gain_lambda": 0.20},
    "locked": {"entropy_lambda": 0.11, "info_gain_lambda": 0.20},
    "strong": {"hidden": 128, "emb_dim": 64, "kl_beta": 0.015, "entropy_lambda": 0.12, "info_gain_lambda": 0.24},
}
DP_POLICY_SCALE = {
    "none": {"linucb_alpha": 1.6, "ts_alpha": 0.95},
    "standard": {"linucb_alpha": 1.85, "ts_alpha": 1.05},
    "locked": {"linucb_alpha": 1.85, "ts_alpha": 1.05},
    "strong": {"linucb_alpha": 2.2, "ts_alpha": 1.15},
}

# -----------------------------
# Config overrides
# -----------------------------
WARM_START_CFG = {"enabled": True, "epochs": 3, "batch_size": 256, "max_batches": 320}

BASE_CONFIG_OVERRIDES = {
    "policy_gain": 1.6,
    "alpha_bounds": (0.05, 0.9),
    "k_bounds": (2, 8),
    "zpd_bounds": (0.06, 0.22),
    "zpd_margin": 0.1,
}
DP_CONFIG_OVERRIDES = {
    "policy_gain": 1.85,
    "alpha_bounds": (0.05, 0.9),
    "k_bounds": (2, 9),
    "zpd_bounds": (0.05, 0.24),
}
DP_CONFIG_OVERRIDES_STRONG = {
    **DP_CONFIG_OVERRIDES,
    "policy_gain": 2.1,
    "zpd_bounds": (0.05, 0.26),
}

# -----------------------------
# Plans (save_dir derived automatically)
# -----------------------------
EDNET_RUNS = [
    {
        "name": "open_nodp",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "linucb",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
]
OULAD_RUNS = [
    {
        "name": "open_nodp",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
]

PLANS: List[Dict] = [
    {"name": "ednet", "config": "configs/ednet.yaml", "dataset": "ednet", "cohort_size": COHORT_SIZE, "runs": EDNET_RUNS},
    {"name": "oulad", "config": "configs/oulad.yaml", "dataset": "oulad", "cohort_size": COHORT_SIZE, "runs": OULAD_RUNS},
]

# =============================
# Directory structure helpers
# =============================
def build_paths(dataset: str, experiment: str) -> Dict[str, Path]:
    base = Path("runs") / dataset / experiment
    return {
        "base":                 base,
        "logs":                 base / "logs",
        "per_round":            base / "tables" / "per_round",
        "per_user_round":       base / "tables" / "per_user_round",
        "per_student_root":     base / "tables" / "per_student",
        "aggregates":           base / "tables" / "aggregates",
        "fig_adaptive":         base / "figures" / "adaptive",
        "fig_compare_modes":    base / "figures" / "compare_modes",
    }

def ensure_dirs(d: Dict[str, Path]) -> None:
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)

# =============================
# User-name mapping helpers
# =============================
USER_NAME_CANDIDATES = ["name", "user_name", "username", "display_name", "full_name", "profile_name"]

def _guess_user_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("u", "user_id", "id", "uid"):
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            return c
    return None

def build_user_name_map(runner: RankingExperiment) -> Dict[int, str]:
    name_map: Dict[int, str] = {}
    try:
        su = getattr(runner, "side_users", pd.DataFrame())
        if su is None or su.empty:
            return {}
        id_col = getattr(runner, "user_col", None) or _guess_user_id_col(su)
        if id_col is None or id_col not in su.columns:
            return {}
        name_col = None
        for cand in USER_NAME_CANDIDATES:
            if cand in su.columns:
                name_col = cand
                break
        if name_col is None:
            return {}
        sub = su[[id_col, name_col]].dropna()
        sub[id_col] = pd.to_numeric(sub[id_col], errors="coerce").dropna().astype(int)
        sub[name_col] = sub[name_col].astype(str)
        name_map = dict(zip(sub[id_col].tolist(), sub[name_col].tolist()))
    except Exception:
        return {}
    return name_map

def sanitize_filename(s: str) -> str:
    s = s.strip().replace("/", "-").replace("\\", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-", "_", ".", "+"))
    s = s.replace(" ", "_")
    return s[:128] if len(s) > 128 else s

# =============================
# Policy helpers
# =============================
def _infer_dp_strength(run: Dict) -> str:
    name = str(run.get("name", "")).lower()
    if "strong" in name:
        return "strong"
    if "locked" in name:
        return "locked"
    dp_cfg = run.get("dp", {})
    if isinstance(dp_cfg, dict) and dp_cfg.get("enabled"):
        return "standard"
    return "none"

def _resolve_adaptive_policy(dataset: str, run: Dict) -> str:
    policy = str(run.get("adaptive_policy", "auto")).lower()
    if policy == "auto":
        return "linucb" if dataset.lower() == "ednet" else "bootts"
    if policy in {"linucb", "bootts", "hybrid"}:
        return policy
    raise ValueError(f"Unknown adaptive_policy '{policy}'")

def _build_adaptive_overrides(dataset: str, run: Dict, dp_cfg: Dict) -> Dict[str, object]:
    dataset_key = dataset.lower()
    dp_strength = run.get("dp_strength") or _infer_dp_strength(run)
    if (dp_strength in (None, "none")) and isinstance(dp_cfg, dict) and dp_cfg.get("enabled"):
        dp_strength = "standard"
    dp_strength = str(dp_strength or "none").lower()
    policy = _resolve_adaptive_policy(dataset_key, run)

    cfg: Dict[str, object] = dict(ADAPTIVE_BASE_PARAMS)
    cfg.update(DATASET_ADJUSTMENTS.get(dataset_key, {}))
    cfg.update(DP_STRENGTH_BASE.get(dp_strength, {}))

    scale = DP_POLICY_SCALE.get(dp_strength, DP_POLICY_SCALE["none"])
    use_linucb = policy in {"linucb", "hybrid"}
    use_bootts = policy in {"bootts", "hybrid"}
    cfg["use_linucb"] = use_linucb
    cfg["use_bootts"] = use_bootts
    if use_linucb:
        cfg["linucb_alpha"] = scale["linucb_alpha"]
    if use_bootts:
        cfg["ts_alpha"] = scale["ts_alpha"]
        cfg["ts_heads"] = max(12, int(cfg.get("ts_heads", 16)))

    extra = run.get("adaptive_overrides")
    if extra:
        cfg.update(extra)

    co = run.get("config_overrides") or {}
    if "zpd_bounds" in co:
        zb = co["zpd_bounds"]
        if isinstance(zb, (list, tuple)) and len(zb) == 2:
            cfg["zpd_bounds"] = (float(zb[0]), float(zb[1]))
    if "zpd_margin" in co:
        cfg["zpd_margin"] = float(co["zpd_margin"])
    if "zpd_margin" not in cfg and "zpd_bounds" in cfg:
        lo, hi = cfg["zpd_bounds"]
        cfg["zpd_margin"] = max(1e-6, 0.5 * (hi - lo))
    return cfg

# =============================
# Export helpers (schema-flexible)
# =============================
def _coalesce_col(df: pd.DataFrame, *cands: str, default=None, new_name: str | None = None) -> pd.DataFrame:
    for c in cands:
        if c in df.columns:
            if new_name and new_name != c:
                df[new_name] = df[c]
            elif new_name is None:
                new_name = c
            return df
    if new_name:
        df[new_name] = default
    return df

def _normalize_user_df(df_user: pd.DataFrame) -> pd.DataFrame:
    df = df_user.copy()
    df = _coalesce_col(df, "method", "student_method", new_name="method", default="unknown")
    df = _coalesce_col(df, "profile", new_name="profile", default="")
    df = _coalesce_col(df, "telemetry_accepted", "tel_accepted", new_name="accepted", default=np.nan)
    df = _coalesce_col(df, "telemetry_correct", "tel_correct", new_name="correct", default=np.nan)
    df = _coalesce_col(df, "telemetry_accept_rate", "tel_accept_rate", new_name="accept_rate", default=np.nan)
    df = _coalesce_col(df, "post_knowledge", "state_knowledge", new_name="knowledge", default=np.nan)
    df = _coalesce_col(df, "post_engagement", "state_engagement", new_name="engagement", default=np.nan)
    df = _coalesce_col(df, "telemetry_novelty_rate", new_name="novelty_rate", default=np.nan)
    df = _coalesce_col(df, "telemetry_serendipity", new_name="serendipity", default=np.nan)
    if "accuracy" not in df.columns:
        if "correct" in df.columns and "accepted" in df.columns:
            df["accuracy"] = df.apply(
                lambda r: (float(r["correct"]) / float(r["accepted"]))
                if pd.notna(r["correct"]) and pd.notna(r["accepted"]) and float(r["accepted"]) > 0
                else np.nan,
                axis=1,
            )
        else:
            df["accuracy"] = np.nan
    return df

def export_per_student_csvs(
    df_user: pd.DataFrame,
    *,
    per_student_root: Path,
    mode: str,
    name_map: Dict[int, str],
) -> None:
    df = _normalize_user_df(df_user)
    df["user_name"] = df["user_id"].map(name_map)
    df["user_name"] = df.apply(
        lambda r: r["user_name"] if isinstance(r["user_name"], str) and r["user_name"].strip()
        else f"user_{int(r['user_id'])}", axis=1
    )
    name_counts = df.groupby("user_name")["user_id"].nunique()
    ambiguous = set(name_counts[name_counts > 1].index.tolist())

    for (m, uid), g in df.groupby(["method", "user_id"], dropna=False):
        name = g["user_name"].iloc[0]
        if name in ambiguous:
            name = f"{name}_{int(uid)}"
        fname = sanitize_filename(name) + ".csv"
        out_dir = per_student_root / mode / str(m)
        out_dir.mkdir(parents=True, exist_ok=True)
        g.sort_values("round").to_csv(out_dir / fname, index=False)

# =============================
# Plot helpers (clean + consistent)
# =============================
# y-limits by metric
METRIC_BOUNDS = {
    "engagement": (0.0, 1.0),
    "knowledge": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "accept_rate": (0.0, 1.0),
    "novelty_rate": (0.0, 1.0),
    "serendipity": (0.0, 1.0),
    "mean_engagement": (0.0, 1.0),
    "mean_knowledge": (0.0, 1.0),
}

def _clean_title(s: str) -> str:
    return s.replace("_", " ").title()

def _save_fig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)

def _plot_lines(
    *,
    series: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, g in series.items():
        gg = g[[x_col, y_col]].dropna().sort_values(x_col)
        if gg.empty:
            continue
        ax.plot(gg[x_col], gg[y_col], marker="o", linewidth=2.0, markersize=4, label=str(label))
    if y_col in METRIC_BOUNDS:
        ax.set_ylim(*METRIC_BOUNDS[y_col])
    ax.set_xlabel(_clean_title(x_col))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # legend outside to avoid overlap
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.subplots_adjust(right=0.78)
    _save_fig(fig, out_path)

def export_adaptive_model_means(df_user: pd.DataFrame, out_csv: Path, out_fig_dir: Path):
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # mean over students for each learning model and round
    grp = (
        df_user.groupby(["student_method", "round"], as_index=False)[
            ["post_engagement", "post_knowledge", "pre_engagement", "pre_knowledge",
             "tel_accept_rate", "tel_correct"]
        ].mean()
    )
    grp.rename(columns={
        "post_engagement": "engagement",
        "post_knowledge": "knowledge",
        "tel_accept_rate": "accept_rate",
        "tel_correct": "correct"
    }, inplace=True)

    grp.to_csv(out_csv, index=False)

    metrics = ["engagement", "knowledge", "accept_rate", "correct"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, sub in grp.groupby("student_method"):
            ax.plot(sub["round"], sub[metric], marker="o", linewidth=2, label=str(method))
        ax.set_title(f"{metric.replace('_', ' ').title()} by Learning Model (Adaptive)")  # <-- fixed
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend(title="Learning model", frameon=False)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"adaptive_{metric}_by_model.png", dpi=150)
        plt.close(fig)

def export_mode_means_and_plots(
    mode_to_rounds: Dict[str, pd.DataFrame],
    *,
    out_csv: Path,
    out_fig_dir: Path,
) -> None:
    if not mode_to_rounds:
        return
    # collect available round-level metrics
    all_cols = set().union(*(df.columns for df in mode_to_rounds.values() if df is not None))
    metrics = [m for m in ("mean_engagement", "mean_knowledge", "accuracy", "accept_rate", "novelty_rate", "serendipity") if m in all_cols]
    if not metrics:
        return

    frames = []
    for mode, df_round in mode_to_rounds.items():
        if df_round is None or df_round.empty:
            continue
        cols = ["round"] + [m for m in metrics if m in df_round.columns]
        g = df_round[cols].copy()
        g["mode"] = mode
        frames.append(g)
    if not frames:
        return

    comp = pd.concat(frames, ignore_index=True)
    comp.sort_values(["round", "mode"], inplace=True)
    comp.to_csv(out_csv, index=False)

    # one plot per metric, lines by mode
    for metric in metrics:
        lines: Dict[str, pd.DataFrame] = {}
        for mode in comp["mode"].unique():
            g = comp[comp["mode"] == mode][["round", metric]].dropna().sort_values("round")
            if not g.empty:
                lines[mode] = g
        if not lines:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for mode, g in lines.items():
            ax.plot(g["round"], g[metric], marker="o", linewidth=2.0, markersize=4, label=mode)
        if metric in METRIC_BOUNDS:
            ax.set_ylim(*METRIC_BOUNDS[metric])
        ax.set_xlabel("Round")
        ax.set_ylabel(_clean_title(metric))
        ax.set_title(f"{_clean_title(metric)} — Compare Modes")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.subplots_adjust(right=0.78)
        _save_fig(fig, out_fig_dir / f"{metric}_compare_modes")

# =============================
# Runner
# =============================
def run_plan(plan: Dict) -> None:
    runner = RankingExperiment(
        plan["config"],
        dataset=plan["dataset"],
        cohort_size=plan.get("cohort_size", COHORT_SIZE),
        seed=42,
        student_methods=STUDENT_METHODS,
        initial_profiles=INITIAL_PROFILES,
    )

    dataset = plan["dataset"]
    user_name_map = build_user_name_map(runner)

    for run in plan["runs"]:
        experiment = run["name"]
        paths = build_paths(dataset, experiment)
        ensure_dirs(paths)

        dp_cfg_in = run["dp"]
        dp_cfg = get_dp_config(dp_cfg_in) if isinstance(dp_cfg_in, str) else dict(dp_cfg_in)

        mode_to_user_rounds: Dict[str, pd.DataFrame] = {}
        mode_to_rounds: Dict[str, pd.DataFrame] = {}
        summary_rows: List[Dict] = []

        for mode in run["modes"]:
            print(f"Running {dataset} / {experiment} / {mode}")
            adaptive_kwargs = None
            if mode == "adaptive":
                adaptive_kwargs = _build_adaptive_overrides(dataset, run, dp_cfg)
            print(f"adaptive_kwargs={adaptive_kwargs}")

            res = runner.run(
                mode,
                dp_enabled=dp_cfg.get("enabled", False),
                dp_params=dict(dp_cfg),
                config_overrides=run.get("config_overrides"),
                log_path=str(paths["logs"] / f"{mode}.jsonl"),
                adaptive_kwargs=adaptive_kwargs,
                warm_start=run.get("warm_start", WARM_START_CFG),
            )

            df_round = res.get("round_metrics", pd.DataFrame())
            df_user  = res.get("user_rounds", pd.DataFrame())

            # attach user_name to df_user before writing/plots
            if not df_user.empty and "user_id" in df_user.columns:
                df_user = df_user.copy()
                df_user["user_name"] = df_user["user_id"].map(user_name_map)
                df_user["user_name"] = df_user.apply(
                    lambda r: r["user_name"] if isinstance(r["user_name"], str) and r["user_name"].strip()
                    else f"user_{int(r['user_id'])}",
                    axis=1,
                )

            # write tables
            df_round.sort_values("round").to_csv(paths["per_round"] / f"{mode}_rounds.csv", index=False)
            if not df_user.empty:
                df_user.sort_values(["user_id", "round"]).to_csv(paths["per_user_round"] / f"{mode}_user_rounds.csv", index=False)
                export_per_student_csvs(df_user, per_student_root=paths["per_student_root"], mode=mode, name_map=user_name_map)
            else:
                (paths["per_user_round"] / f"{mode}_user_rounds.csv").write_text("", encoding="utf-8")

            mode_to_user_rounds[mode] = df_user.copy()
            mode_to_rounds[mode] = df_round.copy()

            # add short summary row (final round snapshot)
            last = df_round.sort_values("round").iloc[-1].to_dict() if not df_round.empty else {}
            last.update({"dataset": dataset, "experiment": experiment, "mode": mode})
            summary_rows.append(last)

        # summary table at experiment root
        pd.DataFrame(summary_rows).to_csv(paths["base"] / "summary.csv", index=False)

        # (2) adaptive per-model means & plots (user-level)
        if "adaptive" in mode_to_user_rounds and not mode_to_user_rounds["adaptive"].empty:
            export_adaptive_model_means(
                mode_to_user_rounds["adaptive"],
                out_csv=paths["aggregates"] / "adaptive_model_means.csv",
                out_fig_dir=paths["fig_adaptive"],
            )

        # (3) compare modes (round-level)
        export_mode_means_and_plots(
            mode_to_rounds,
            out_csv=paths["aggregates"] / "mode_means.csv",
            out_fig_dir=paths["fig_compare_modes"],
        )

def main() -> None:
    for plan in PLANS:
        run_plan(plan)

if __name__ == "__main__":
    main()
