"""Agentic benchmark on the MovieLens 100K dataset.

The script compares a fixed recommender against an adaptive one using the
MultiUserOrchestrator. User/item feature matrices are derived from a quick
ExplicitMF (FunkSVD-style) fit on a filtered subset of ML-100K so the
simulation starts with meaningful structure.

Example usage (CPU safe defaults):

```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py \
  --rounds 80 --top-users 400 --top-items 800 --top-k 6 --dim 16
```

You can toggle Funk-guided candidate generation or distillation via
`--funk-candidates` / `--funk-distill` flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from surprise import Dataset

from orchid_ranker import MultiConfig, MultiUserOrchestrator, UserCtx
from orchid_ranker.agents.recommender_agent import DualRecommender, TwoTowerRecommender
from orchid_ranker.agents.student_agent import StudentAgent
from orchid_ranker.baselines import ExplicitMFBaseline


def load_ml100k(top_users: int, top_items: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml = Dataset.load_builtin("ml-100k", prompt=False)
    raw = ml.raw_ratings
    frame = pd.DataFrame(raw, columns=["user_id", "item_id", "rating", "timestamp"])
    frame["user_id"] = frame["user_id"].astype(int)
    frame["item_id"] = frame["item_id"].astype(int)
    frame["rating"] = frame["rating"].astype(float)

    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < 0.8
    train_df = frame[mask].copy()
    test_df = frame[~mask].copy()

    uids = train_df["user_id"].value_counts().head(top_users).index
    iids = train_df["item_id"].value_counts().head(top_items).index
    train_df = train_df[train_df.user_id.isin(uids) & train_df.item_id.isin(iids)].reset_index(drop=True)
    test_df = test_df[test_df.user_id.isin(train_df.user_id.unique()) & test_df.item_id.isin(train_df.item_id.unique())].reset_index(drop=True)
    return train_df, test_df


def build_embeddings(train_df: pd.DataFrame, dim: int) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    user_ids = sorted(train_df["user_id"].unique())
    item_ids = sorted(train_df["item_id"].unique())
    uid2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    iid2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    mf = ExplicitMFBaseline(
        num_users=len(user_ids),
        num_items=len(item_ids),
        device=torch.device("cpu"),
        emb_dim=dim,
        epochs=15,
        lr=5e-3,
        weight_decay=1e-4,
    )
    mf.fit(train_df["user_id"].map(uid2idx), train_df["item_id"].map(iid2idx), train_df["rating"])

    user_emb = mf.model.user_emb.weight.detach().cpu().numpy()
    item_emb = mf.model.item_emb.weight.detach().cpu().numpy()
    return user_emb, item_emb, uid2idx, iid2idx


def build_simulation_matrices(
    train_df: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    uid2idx: dict[int, int],
    iid2idx: dict[int, int],
    dim: int,
    device: torch.device,
):
    users = sorted(uid2idx.keys())
    items = sorted(iid2idx.keys())

    U = torch.tensor(user_emb, dtype=torch.float32, device=device)
    W = torch.tensor(item_emb, dtype=torch.float32, device=device)

    # Normalize to avoid exploding dot products
    U = torch.nn.functional.normalize(U, dim=1)
    W = torch.nn.functional.normalize(W, dim=1)

    # Difficulty meta derived from item mean rating (lower rating => harder)
    item_means = (
        train_df.groupby("item_id")["rating"].mean().reindex(items).fillna(train_df["rating"].mean()).astype(float)
    )
    difficulty = 1.0 - (item_means - item_means.min()) / (item_means.max() - item_means.min() + 1e-6)
    meta = {int(iid): {"difficulty": float(difficulty.loc[iid])} for iid in items}

    pos2id = items
    id2pos = {iid: idx for idx, iid in enumerate(items)}
    item_ids_pos = torch.arange(len(items), device=device)
    return U, W, pos2id, id2pos, item_ids_pos, meta


def run_once(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _ = load_ml100k(args.top_users, args.top_items, args.seed)
    train_df["label"] = (train_df["rating"] >= 4.0).astype(int)

    user_emb, item_emb, uid2idx, iid2idx = build_embeddings(train_df, args.dim)
    U, W, pos2id, id2pos, item_ids_pos, meta = build_simulation_matrices(train_df, user_emb, item_emb, uid2idx, iid2idx, args.dim, device)

    users = []
    for uid, idx in uid2idx.items():
        sa = StudentAgent(user_id=uid, seed=args.seed + uid)
        users.append(UserCtx(user_id=uid, user_idx=idx, student=sa, user_vec=U[idx : idx + 1]))

    num_users = len(uid2idx)
    num_items = len(iid2idx)

    fixed = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False})
    fixed.user_matrix = U.clone().to(device)

    teacher = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student.blend_increment = 0.3
    student.teacher_ema = 0.85
    adaptive = DualRecommender(teacher=teacher, student=student, device=str(device), warm_start=True, replay_size=512, replay_steps=1)
    for rec in (teacher, student):
        rec.user_matrix = U.clone().to(device)

    base_cfg = dict(
        rounds=args.rounds,
        top_k_base=args.top_k,
        min_candidates=num_items,
        console=False,
        console_user=False,
        deterministic_pool=True,
        persistent_pool=True,
        train_on_all_shown=True,
        train_steps_per_round=2,
        warmup_rounds=args.warmup_rounds,
        warmup_steps=args.warmup_steps,
        warmup_top_k_boost=args.warmup_top_k_boost,
        warmup_diversity_scale=args.warmup_diversity_scale,
        warmup_preloop=True,
        use_funk_candidates=args.funk_candidates,
        funk_pool_size=(args.funk_pool or num_items // 2),
        funk_distill=args.funk_distill,
        funk_lambda=args.funk_lambda,
        log_path=str(args.log_dir / "tmp.jsonl"),
    )

    cfg_fixed = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "fixed.jsonl")})
    orch_fixed = MultiUserOrchestrator(
        rec=fixed,
        users=users,
        item_matrix_normal=W,
        item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos,
        pos2id=pos2id,
        id2pos=id2pos,
        item_meta_by_id=meta,
        cfg=cfg_fixed,
        device=device,
        mode_label="fixed",
    )
    orch_fixed.run()

    cfg_adapt = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "adaptive.jsonl")})
    orch_adapt = MultiUserOrchestrator(
        rec=adaptive,
        users=users,
        item_matrix_normal=W,
        item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos,
        pos2id=pos2id,
        id2pos=id2pos,
        item_meta_by_id=meta,
        cfg=cfg_adapt,
        device=device,
        mode_label="adaptive",
    )
    orch_adapt.run()

    import pandas as pd

    def load(path: Path) -> pd.DataFrame:
        rows = []
        for line in path.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "round_summary":
                m = obj.get("metrics", {})
                rows.append({k: m.get(k) for k in ["accept_rate", "accuracy", "novelty_rate", "serendipity", "mean_knowledge"]})
        return pd.DataFrame(rows)

    fx = load(args.log_dir / "fixed.jsonl").mean(numeric_only=True).to_dict()
    ad = load(args.log_dir / "adaptive.jsonl").mean(numeric_only=True).to_dict()
    return {"fixed": fx, "adaptive": ad}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic fixed vs adaptive benchmark on MovieLens 100K")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--top-users", type=int, default=400)
    p.add_argument("--top-items", type=int, default=800)
    p.add_argument("--top-k", dest="top_k", type=int, default=6)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=Path, default=Path("runs/agentic-ml100k"))
    # Warmup scheduling knobs
    p.add_argument("--warmup-rounds", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--warmup-top-k-boost", type=int, default=2)
    p.add_argument("--warmup-diversity-scale", type=float, default=0.3)
    # Funk options
    p.add_argument("--funk-candidates", action="store_true")
    p.add_argument("--funk-pool", type=int, default=0)
    p.add_argument("--funk-distill", action="store_true")
    p.add_argument("--funk-lambda", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out = run_once(args)
    print("Fixed means:", out["fixed"])
    print("Adaptive means:", out["adaptive"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

