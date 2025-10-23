#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

from orchid_ranker.preprocessing.base import BasePreprocessor, PreprocessorConfig, register_preprocessor


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

    return df.groupby(key).apply(_agg).reset_index(name=out_name)



@register_preprocessor("oulad")
class OuladPreprocessor(BasePreprocessor):
    name = "oulad"

    def run(self, cfg: PreprocessorConfig) -> None:
        preprocess_oulad(
            base_path=cfg.base_path,
            output_path=cfg.output_path,
            n_users=cfg.n_users,
            seed=cfg.seed,
        )

def _avg_time_gaps(df: pd.DataFrame, key: str, ts_col: str, out_name: str) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return pd.DataFrame({key: [], out_name: []})

    def _gap(series: pd.Series) -> float:
        ts = np.sort(series.dropna().to_numpy(dtype=float))
        if ts.size <= 1:
            return 0.0
        gaps = np.diff(ts)
        return float(np.mean(gaps))

    return df.groupby(key)[ts_col].apply(_gap).reset_index(name=out_name)


def preprocess_oulad(base_path, output_path, n_users=None, seed=42):
    """
    Preprocess OULAD dataset into train/val/test interactions and richer side info.

    Outputs (CSV files in output_path):
      - train.csv, val.csv, test.csv with: u, i, label, timestamp, correct, accept
      - side_information_items.csv with: i, activity_type, week_from, week_to, code_module, code_presentation,
                                         avg_clicks, duration, difficulty
      - side_information_users.csv with: u, gender, region, highest_education, imd_band, age_band,
                                         num_of_prev_attempts, studied_credits, disability, final_result
    Notes:
      - OULAD has no per-event dwell/latency; the runner’s seeder will fall back for those.
      - `difficulty` ∈ [0,1] is derived from 1 - item historical accuracy (winsorized 5–95% then min–max).
    """
    rng = np.random.RandomState(seed)

    # ---------- Load raw OULAD files ----------
    student_vle = pd.read_csv(os.path.join(base_path, "studentVLE.csv"))
    vle         = pd.read_csv(os.path.join(base_path, "vle.csv"))
    student_info= pd.read_csv(os.path.join(base_path, "studentInfo.csv"))

    # ---------- Merge ----------
    merged = pd.merge(student_vle, vle,
                      on=["id_site", "code_module", "code_presentation"],
                      how="left")
    merged = pd.merge(merged, student_info,
                      on=["id_student", "code_module", "code_presentation"],
                      how="left")

    # ---------- Encode users/items ----------
    user2idx = {u: i for i, u in enumerate(merged["id_student"].unique())}
    item2idx = {i: j for j, i in enumerate(merged["id_site"].unique())}
    merged["u"] = merged["id_student"].map(user2idx)
    merged["i"] = merged["id_site"].map(item2idx)

    # ---------- Interaction label and timestamp ----------
    merged["label"] = (merged["sum_click"] > 0).astype(int)
    merged["timestamp"] = merged["date"]  # day offset within presentation

    # ---------- Optional user sampling (before the split) ----------
    if n_users is not None:
        unique_u = merged["u"].dropna().unique()
        take = min(n_users, unique_u.size)
        sampled_users = rng.choice(unique_u, size=take, replace=False)
        merged = merged[merged["u"].isin(sampled_users)]

    # ---------- Final interactions (+ seeding-friendly columns) ----------
    cols = ["u", "i", "label", "timestamp"]
    if "sum_click" in merged.columns:
        cols.append("sum_click")
    interactions = merged[cols].dropna(subset=["u", "i", "label", "timestamp"]).copy()
    interactions["correct"] = interactions["label"]
    interactions["accept"]  = interactions["label"]  # reasonable proxy in OULAD
    if "sum_click" not in interactions.columns:
        interactions["sum_click"] = 0.0

    # ---------- Item side info ----------
    item_side = merged.groupby("i").agg({
        "activity_type": "first",
        "week_from": "first",
        "week_to": "first",
        "code_module": "first",
        "code_presentation": "first",
        "sum_click": "mean"
    }).reset_index().rename(columns={"sum_click": "avg_clicks"})
    item_side["week_from"] = item_side["week_from"].fillna(0)
    item_side["week_to"] = item_side["week_to"].fillna(item_side["week_from"])
    item_side["duration"] = (item_side["week_to"] - item_side["week_from"]).clip(lower=0)
    item_side["click_velocity"] = item_side["avg_clicks"] / np.clip(item_side["duration"], 1.0, None)

    # ---------- Difficulty from historical accuracy ----------
    if len(interactions) > 0:
        acc_by_item = interactions.groupby("i")["label"].mean().rename("acc").reset_index()
        acc_by_item["difficulty_raw"] = 1.0 - acc_by_item["acc"]
        lo = acc_by_item["difficulty_raw"].quantile(0.05)
        hi = acc_by_item["difficulty_raw"].quantile(0.95)
        acc_by_item["difficulty"] = ((acc_by_item["difficulty_raw"] - lo) / (hi - lo + 1e-9)).clip(0, 1)
        item_side = item_side.merge(acc_by_item[["i", "difficulty"]], on="i", how="left")
        # backfill missing with median difficulty
        item_side["difficulty"] = item_side["difficulty"].fillna(item_side["difficulty"].median())
    else:
        item_side["difficulty"] = 0.5

    # Decayed engagement / success and recent activity
    half_life_days = 45.0
    decayed_clicks = _decayed_mean(interactions, "i", "timestamp", "sum_click", half_life_days, "decayed_clicks")
    decayed_success = _decayed_mean(interactions, "i", "timestamp", "label", half_life_days, "decayed_success")
    item_side = item_side.merge(decayed_clicks, on="i", how="left")
    item_side = item_side.merge(decayed_success, on="i", how="left")
    item_side["decayed_clicks"] = item_side["decayed_clicks"].fillna(item_side["decayed_clicks"].median())
    item_side["decayed_success"] = item_side["decayed_success"].fillna(item_side["decayed_success"].median())
    if interactions["timestamp"].notna().any():
        max_ts = interactions["timestamp"].max()
        recent_window = max_ts - 28
        recent = (interactions[interactions["timestamp"] >= recent_window]
                  .groupby("i")[["sum_click", "label"]]
                  .agg({"sum_click": "sum", "label": "mean"})
                  .reset_index()
                  .rename(columns={"sum_click": "recent_clicks_4w", "label": "recent_success_4w"}))
        item_side = item_side.merge(recent, on="i", how="left")
    else:
        item_side["recent_clicks_4w"] = np.nan
        item_side["recent_success_4w"] = np.nan
    item_side["recent_clicks_4w"] = item_side["recent_clicks_4w"].fillna(0.0)
    item_side["recent_success_4w"] = item_side["recent_success_4w"].fillna(item_side.get("decayed_success", 0.5))

    item_side_cols = [
        "i", "activity_type", "week_from", "week_to", "code_module", "code_presentation",
        "avg_clicks", "duration", "click_velocity", "difficulty",
        "decayed_clicks", "decayed_success", "recent_clicks_4w", "recent_success_4w"
    ]
    item_side = item_side[item_side_cols]

    # ---------- User side info ----------
    user_side = merged.groupby("u").agg({
        "gender": "first",
        "region": "first",
        "highest_education": "first",
        "imd_band": "first",
        "age_band": "first",
        "num_of_prev_attempts": "first",
        "studied_credits": "first",
        "disability": "first",
        "final_result": "first"
    }).reset_index()

    # Add learning analytics features
    total_clicks = merged.groupby("u")["sum_click"].sum().rename("total_clicks").reset_index()
    user_side = user_side.merge(total_clicks, on="u", how="left")
    user_side["total_clicks"] = user_side["total_clicks"].fillna(0.0)

    span_days = merged.groupby("u")["timestamp"].agg(lambda s: s.max() - s.min()).reset_index().rename(columns={"timestamp": "activity_span_days"})
    user_side = user_side.merge(span_days, on="u", how="left")
    user_side["activity_span_days"] = user_side["activity_span_days"].fillna(0.0)
    user_side["activity_velocity"] = user_side["total_clicks"] / np.clip(user_side["activity_span_days"], 1.0, None)

    mean_success = interactions.groupby("u")["label"].mean().rename("mean_success").reset_index()
    user_side = user_side.merge(mean_success, on="u", how="left")

    decayed_success_user = _decayed_mean(interactions, "u", "timestamp", "label", half_life_days, "decayed_success")
    user_side = user_side.merge(decayed_success_user, on="u", how="left")

    if interactions["timestamp"].notna().any():
        max_ts_u = interactions["timestamp"].max()
        recent_window_u = max_ts_u - 28
        recent_user = (interactions[interactions["timestamp"] >= recent_window_u]
                       .groupby("u")[["label", "sum_click"]]
                       .agg({"label": "mean", "sum_click": "sum"})
                       .reset_index()
                       .rename(columns={"label": "recent_success_4w", "sum_click": "recent_clicks_4w"}))
        user_side = user_side.merge(recent_user, on="u", how="left")
    else:
        user_side["recent_success_4w"] = np.nan
        user_side["recent_clicks_4w"] = np.nan
    user_side["recent_success_4w"] = user_side["recent_success_4w"].fillna(user_side["mean_success"])
    user_side["recent_clicks_4w"] = user_side["recent_clicks_4w"].fillna(0.0)

    user_side["decayed_success"] = user_side["decayed_success"].fillna(user_side["mean_success"])
    user_side["mean_success"] = user_side["mean_success"].fillna(0.0)

    if "sum_click" in interactions.columns:
        clicks_stats = (interactions.groupby("u")["sum_click"]
                        .agg(["mean", "std"])  # noqa
                        .reset_index()
                        .rename(columns={"mean": "mean_clicks_per_event", "std": "std_clicks_per_event"}))
        user_side = user_side.merge(clicks_stats, on="u", how="left")
    else:
        user_side["mean_clicks_per_event"] = np.nan
        user_side["std_clicks_per_event"] = np.nan
    user_side["mean_clicks_per_event"] = user_side["mean_clicks_per_event"].fillna(0.0)
    user_side["std_clicks_per_event"] = user_side["std_clicks_per_event"].fillna(0.0)

    gap_stats = _avg_time_gaps(interactions, "u", "timestamp", "avg_inter_event_days")
    user_side = user_side.merge(gap_stats, on="u", how="left") if not gap_stats.empty else user_side.assign(avg_inter_event_days=0.0)
    user_side["avg_inter_event_days"] = user_side["avg_inter_event_days"].fillna(0.0)

    user_side_cols = [
        "u", "gender", "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability", "final_result",
        "total_clicks", "activity_span_days", "activity_velocity",
        "mean_success", "decayed_success", "recent_success_4w", "recent_clicks_4w",
        "mean_clicks_per_event", "std_clicks_per_event", "avg_inter_event_days"
    ]
    user_side = user_side[user_side_cols]

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
    print(f"✅ Preprocessed OULAD saved to {output_path}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Item side info: {len(item_side)} | User side info: {len(user_side)}")
    print("Interactions columns:", list(interactions.columns))
    print("Item side columns:", list(item_side.columns))
    print("User side columns:", list(user_side.columns))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, required=True, help="Path to raw OULAD data folder")
    ap.add_argument("--output_path", type=str, required=True, help="Where to save processed data")
    ap.add_argument("--n_users", type=int, default=None, help="Limit number of users")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    preprocess_oulad(args.base_path, args.output_path, args.n_users, args.seed)


if __name__ == "__main__":
    main()
