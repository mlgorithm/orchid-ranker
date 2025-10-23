#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

from orchid_ranker.preprocessing.base import BasePreprocessor, PreprocessorConfig, register_preprocessor

# Optional parallel reading using joblib (threading avoids pickling issues on 3.13)
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


def _load_user_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # embed user id from filename "uXXXX.csv"
    try:
        user_id = int(os.path.basename(path).split("u")[-1].split(".")[0])
    except Exception:
        user_id = -1
    df["user_id"] = user_id
    return df


def _mode_or_none(series: pd.Series):
    try:
        mode = series.mode(dropna=True)
        if not mode.empty:
            return mode.iloc[0]
    except Exception:
        pass
    series = series.dropna()
    return series.iloc[0] if not series.empty else None


def _tag_stats(series: pd.Series) -> pd.Series:
    tokens = []
    for val in series.dropna():
        tokens.extend([tok.strip() for tok in str(val).split(";") if tok.strip()])
    if not tokens:
        return pd.Series({"tag_top": None, "tag_count": 0, "tag_diversity": 0.0})
    counts = pd.Series(tokens).value_counts()
    diversity = float(counts.size)
    return pd.Series({"tag_top": counts.index[0], "tag_count": int(diversity), "tag_diversity": diversity})


def _exp_decay_weights(ts: np.ndarray, half_life: float) -> np.ndarray:
    if ts.size == 0:
        return np.array([])
    ts = np.asarray(ts, dtype=float)
    max_ts = np.nanmax(ts)
    decay = math.log(2) / max(half_life, 1e-6)
    weights = np.exp(-(max_ts - ts) * decay)
    weights[~np.isfinite(weights)] = 0.0
    return weights


def _decayed_mean(df: pd.DataFrame, key: str, ts_col: str, value_col: str, half_life: float, out_name: str) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame({key: [], out_name: []})

    def _agg(group: pd.DataFrame) -> float:
        vals = group[value_col].to_numpy(dtype=float)
        ts = group[ts_col].to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(ts)
        if not mask.any():
            return np.nan
        vals = vals[mask]
        ts = ts[mask]
        w = _exp_decay_weights(ts, half_life)
        if w.sum() <= 0:
            return float(np.nanmean(vals)) if vals.size else np.nan
        return float(np.sum(w * vals) / np.sum(w))

    res = df.groupby(key).apply(_agg).reset_index(name=out_name)
    return res


def _avg_time_gaps(df: pd.DataFrame, key: str, ts_col: str, out_name: str) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return pd.DataFrame({key: [], out_name: []})

    def _gap(series: pd.Series) -> float:
        ts = np.sort(series.dropna().to_numpy(dtype=float))
        if ts.size <= 1:
            return 0.0
        gaps = np.diff(ts)
        return float(np.mean(gaps))

    res = df.groupby(key)[ts_col].apply(_gap).reset_index(name=out_name)
    return res


def preprocess_ednet(base_path, content_path, output_path, n_users=None, seed=42, n_jobs=-1):
    """
    Preprocess EdNet KT2/KT4 into train/val/test interactions and side info.

    Outputs (CSV files in output_path):
      - train.csv, val.csv, test.csv with: u, i, label, timestamp, [elapsed_time?], correct, accept
      - side_information_items.csv with: i, part, tags, source, platform, [elapsed_time mean?], num_tags, difficulty
      - side_information_users.csv with: u, platform_pref, source_pref, avg_accuracy, start_time, end_time, active_days

    Notes:
      - We retain `elapsed_time` (if present) so the runner maps it to dwell_s during normalization.
      - `difficulty` ∈ [0,1] from 1 - item accuracy (winsorized 5–95% then min–max).
      - Uses joblib threading backend by default to avoid pickling issues in Python 3.13.
    """
    rng = np.random.RandomState(seed)

    # ---------- Collect user CSVs ----------
    files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith("u") and f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No user files like 'u*.csv' found under {base_path}")

    # ---------- Load ----------
    if HAS_JOBLIB and n_jobs != 1:
        dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_load_user_file)(f) for f in files)
    else:
        dfs = [_load_user_file(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)

    # ---------- Load content metadata ----------
    questions = pd.read_csv(os.path.join(content_path, "questions.csv"))

    # ---------- Keep only respond actions ----------
    if "action_type" in data.columns:
        data = data[data["action_type"] == "respond"].copy()

    # ---------- Map users/items ----------
    user2idx = {u: i for i, u in enumerate(data["user_id"].unique())}
    item2idx = {q: j for j, q in enumerate(data["item_id"].unique())}
    data["u"] = data["user_id"].map(user2idx)
    data["i"] = data["item_id"].map(item2idx)

    # ---------- Merge with correct answers ----------
    merged = pd.merge(
        data,
        questions,
        left_on="item_id",
        right_on="question_id",
        how="left"
    )

    # ---------- Labels & timestamp ----------
    merged["label"] = (merged["user_answer"] == merged["correct_answer"]).astype(int)
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    if "elapsed_time" in merged.columns:
        merged["elapsed_time"] = pd.to_numeric(merged["elapsed_time"], errors="coerce")

    # ---------- Optional user sampling (before the split) ----------
    if n_users is not None:
        unique_u = merged["u"].dropna().unique()
        take = min(n_users, unique_u.size)
        sampled_users = rng.choice(unique_u, size=take, replace=False)
        merged = merged[merged["u"].isin(sampled_users)]

    # ---------- Interactions (keep elapsed_time if present) ----------
    cols = ["u", "i", "label", "timestamp"]
    if "elapsed_time" in merged.columns:
        cols.append("elapsed_time")
    interactions = merged[cols].dropna().copy()

    # seeding-friendly fields
    interactions["correct"] = interactions["label"]
    interactions["accept"]  = interactions["label"]  # reasonable proxy

    # ---------- Item side info ----------
    agg_dict = {}
    if "part" in merged.columns:
        agg_dict["part"] = "first"
    if "source" in merged.columns:
        agg_dict["source"] = _mode_or_none
    if "platform" in merged.columns:
        agg_dict["platform"] = _mode_or_none
    if "elapsed_time" in merged.columns:
        agg_dict["elapsed_time"] = "mean"

    item_side = merged.groupby("i").agg(agg_dict).reset_index()
    if "elapsed_time" in item_side.columns:
        item_side = item_side.rename(columns={"elapsed_time": "mean_elapsed_time"})
    else:
        item_side["mean_elapsed_time"] = np.nan

    if "tags" in merged.columns:
        tag_features = merged.groupby("i")["tags"].apply(_tag_stats).reset_index()
        item_side = item_side.merge(tag_features, on="i", how="left")
    else:
        item_side["tag_top"] = None
        item_side["tag_count"] = 0
        item_side["tag_diversity"] = 0.0
    if "tag_count" in item_side.columns:
        item_side["tag_count"] = item_side["tag_count"].fillna(0).astype(int)
    if "tag_diversity" in item_side.columns:
        item_side["tag_diversity"] = item_side["tag_diversity"].fillna(0.0)
    if "tag_top" not in item_side.columns:
        item_side["tag_top"] = None

    # ---------- Difficulty from historical accuracy ----------
    if len(interactions) > 0:
        acc_by_item = interactions.groupby("i")["label"].mean().rename("acc").reset_index()
        acc_by_item["difficulty_raw"] = 1.0 - acc_by_item["acc"]
        lo = acc_by_item["difficulty_raw"].quantile(0.05)
        hi = acc_by_item["difficulty_raw"].quantile(0.95)
        acc_by_item["difficulty"] = ((acc_by_item["difficulty_raw"] - lo) / (hi - lo + 1e-9)).clip(0, 1)
        item_side = item_side.merge(acc_by_item[["i", "difficulty"]], on="i", how="left")
        item_side["difficulty"] = item_side["difficulty"].fillna(item_side["difficulty"].median())
    else:
        item_side["difficulty"] = 0.5

    # Decayed accuracy & recency counts
    half_life_days = 45.0
    half_life_ms = half_life_days * 24 * 60 * 60 * 1000.0
    decayed_item = _decayed_mean(interactions, "i", "timestamp", "label", half_life_ms, "decayed_accuracy")
    item_side = item_side.merge(decayed_item, on="i", how="left")
    item_side["decayed_accuracy"] = item_side["decayed_accuracy"].fillna(item_side["decayed_accuracy"].median())
    item_side["difficulty_recency"] = 1.0 - item_side["decayed_accuracy"]

    if interactions["timestamp"].notna().any():
        max_ts = interactions["timestamp"].max()
        recent_window = max_ts - 15 * 24 * 60 * 60 * 1000.0
        recent_counts = (interactions[interactions["timestamp"] >= recent_window]
                         .groupby("i").size().rename("recent_15d_interactions").reset_index())
        item_side = item_side.merge(recent_counts, on="i", how="left")
    else:
        item_side["recent_15d_interactions"] = np.nan
    item_side["recent_15d_interactions"] = item_side["recent_15d_interactions"].fillna(0).astype(int)

    # Ensure expected columns exist
    required_item_cols = [
        "part", "tag_top", "source", "platform", "tag_count", "tag_diversity",
        "mean_elapsed_time", "difficulty", "decayed_accuracy", "difficulty_recency",
        "recent_15d_interactions"
    ]
    for col in required_item_cols:
        if col not in item_side.columns:
            if col in ("tag_count", "recent_15d_interactions"):
                item_side[col] = 0
            elif col in ("mean_elapsed_time", "difficulty", "decayed_accuracy", "difficulty_recency", "tag_diversity"):
                item_side[col] = np.nan
            else:
                item_side[col] = None
    item_side = item_side[["i"] + required_item_cols]

    # ---------- User side info ----------
    g = merged.groupby("u")
    total_interactions = g.size().rename("total_interactions")
    mean_accuracy = g["label"].mean().rename("mean_accuracy")
    span_days = ((g["timestamp"].max() - g["timestamp"].min()) / (1000 * 60 * 60 * 24)).rename("activity_span_days")

    user_side = total_interactions.reset_index()
    user_side = user_side.merge(mean_accuracy.reset_index(), on="u", how="left")
    user_side = user_side.merge(span_days.reset_index(), on="u", how="left")
    user_side["activity_span_days"] = user_side["activity_span_days"].fillna(0.0)
    user_side["total_interactions"] = user_side["total_interactions"].astype(int)
    user_side["activity_velocity"] = user_side["total_interactions"] / np.clip(user_side["activity_span_days"], 1.0, None)

    if "platform" in merged.columns:
        platform_pref = g["platform"].agg(_mode_or_none).rename("platform_pref").reset_index()
        user_side = user_side.merge(platform_pref, on="u", how="left")
    else:
        user_side["platform_pref"] = None
    if "source" in merged.columns:
        source_pref = g["source"].agg(_mode_or_none).rename("source_pref").reset_index()
        user_side = user_side.merge(source_pref, on="u", how="left")
    else:
        user_side["source_pref"] = None

    # Decayed accuracy per user
    decayed_user = _decayed_mean(interactions, "u", "timestamp", "label", half_life_ms, "decayed_accuracy")
    user_side = user_side.merge(decayed_user, on="u", how="left")

    # Recent performance window
    if interactions["timestamp"].notna().any():
        max_ts = interactions["timestamp"].max()
        recent_window = max_ts - 15 * 24 * 60 * 60 * 1000.0
        recent_stats = (interactions[interactions["timestamp"] >= recent_window]
                        .groupby("u")["label"].agg(["mean", "size"]).reset_index()
                        .rename(columns={"mean": "recent_accuracy_15d", "size": "recent_interactions_15d"}))
        user_side = user_side.merge(recent_stats, on="u", how="left")
    else:
        user_side["recent_accuracy_15d"] = np.nan
        user_side["recent_interactions_15d"] = 0
    user_side["recent_accuracy_15d"] = user_side["recent_accuracy_15d"].fillna(user_side["mean_accuracy"])
    user_side["recent_interactions_15d"] = user_side["recent_interactions_15d"].fillna(0).astype(int)

    # Response-time stats
    if "elapsed_time" in interactions.columns:
        rt_stats = interactions.groupby("u")["elapsed_time"].agg(["mean", "std"]).reset_index().rename(
            columns={"mean": "mean_elapsed_time", "std": "std_elapsed_time"})
        user_side = user_side.merge(rt_stats, on="u", how="left")
    else:
        user_side["mean_elapsed_time"] = np.nan
        user_side["std_elapsed_time"] = np.nan

    gap_stats = _avg_time_gaps(interactions, "u", "timestamp", "avg_inter_event_ms")
    if not gap_stats.empty:
        user_side = user_side.merge(gap_stats, on="u", how="left")
    else:
        user_side["avg_inter_event_ms"] = 0.0
    user_side["avg_inter_event_ms"] = user_side["avg_inter_event_ms"].fillna(0.0)

    # ensure column order
    user_cols = ["u", "platform_pref", "source_pref", "mean_accuracy", "decayed_accuracy",
                 "recent_accuracy_15d", "activity_span_days", "activity_velocity",
                 "total_interactions", "recent_interactions_15d",
                 "mean_elapsed_time", "std_elapsed_time", "avg_inter_event_ms"]
    for col in user_cols:
        if col not in user_side.columns:
            user_side[col] = np.nan if col not in ("total_interactions", "recent_interactions_15d") else 0
    user_side = user_side[user_cols]

    # ---------- Train/val/test split (stratified with fallback) ----------
    try:
        train, test = train_test_split(interactions, test_size=0.2, random_state=seed, stratify=interactions["u"])
        train, val  = train_test_split(train, test_size=0.1, random_state=seed, stratify=train["u"])
    except ValueError:
        train, test = train_test_split(interactions, test_size=0.2, random_state=seed)
        train, val  = train_test_split(train, test_size=0.1, random_state=seed)

    # ---------- Save ----------
    os.makedirs(output_path, exist_ok=True)
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    val.to_csv(os.path.join(output_path, "val.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)
    item_side.to_csv(os.path.join(output_path, "side_information_items.csv"), index=False)
    user_side.to_csv(os.path.join(output_path, "side_information_users.csv"), index=False)

    # ---------- Report ----------
    print(f"✅ Preprocessed EdNet saved to {output_path}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Item side info: {len(item_side)} | User side info: {len(user_side)}")
    print("Interactions columns:", list(interactions.columns))
    print("Item side columns:", list(item_side.columns))
    print("User side columns:", list(user_side.columns))



from orchid_ranker.preprocessing.base import BasePreprocessor, PreprocessorConfig, register_preprocessor


@register_preprocessor("ednet")
class EdNetPreprocessor(BasePreprocessor):
    name = "ednet"

    def run(self, cfg: PreprocessorConfig) -> None:
        if cfg.extra is None or "content_path" not in cfg.extra:
            raise ValueError("EdNet preprocessing requires 'content_path' in extra config")
        preprocess_ednet(
            base_path=cfg.base_path,
            content_path=cfg.extra["content_path"],
            output_path=cfg.output_path,
            n_users=cfg.n_users,
            seed=cfg.seed,
            n_jobs=cfg.extra.get("n_jobs", -1),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, required=True, help="Path to raw EdNet KT2/KT4 data (u*.csv)")
    ap.add_argument("--content_path", type=str, required=True, help="Path to EdNet content folder (questions.csv)")
    ap.add_argument("--output_path", type=str, required=True, help="Where to save processed data")
    ap.add_argument("--n_users", type=int, default=None, help="Limit number of users")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1, help="Joblib threads (-1=all cores, 1=no parallel)")
    args = ap.parse_args()

    preprocess_ednet(args.base_path, args.content_path, args.output_path, args.n_users, args.seed, args.n_jobs)


if __name__ == "__main__":
    main()
