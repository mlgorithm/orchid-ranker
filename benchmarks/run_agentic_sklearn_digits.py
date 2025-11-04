"""Agentic fixed vs adaptive benchmark using an sklearn dataset (digits).

Users are samples from sklearn.datasets.load_digits(), reduced to `dim` via PCA.
Items are pixels (64 items); item features are a deterministic random projection
to `dim`. This keeps the loop CPU-safe and reproducible.

You can enable Funk candidates/distillation through MultiConfig flags, same as
other agentic benches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from orchid_ranker import MultiConfig, MultiUserOrchestrator, UserCtx
from orchid_ranker.agents.student_agent import StudentAgent
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender, DualRecommender


def build_from_digits(dim: int, users: int, device: torch.device):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0  # scale to [0,1]
    X = X[:users] if users < X.shape[0] else X
    pca = PCA(n_components=dim, random_state=42)
    U = pca.fit_transform(X).astype(np.float32)  # [users, dim]

    # Items = pixels (64). Build item matrix as deterministic random projection to dim
    rng = np.random.default_rng(42)
    M = 64
    W = rng.standard_normal((M, dim)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-6)

    # Item metadata: difficulty derived from global pixel intensity (harder if darker)
    pix_mean = X.mean(axis=0)  # [64]
    difficulty = (1.0 - (pix_mean - pix_mean.min()) / (np.ptp(pix_mean) + 1e-6)).astype(float)
    item_meta = {int(i): {"difficulty": float(difficulty[i])} for i in range(M)}

    U_t = torch.tensor(U, dtype=torch.float32, device=device)
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    pos2id = list(range(M))
    id2pos = {i: i for i in pos2id}
    item_ids_pos = torch.arange(M, device=device)
    return U_t, W_t, pos2id, id2pos, item_ids_pos, item_meta


def run_once(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U, W, pos2id, id2pos, item_ids_pos, meta = build_from_digits(args.dim, args.users, device)

    users = []
    for idx in range(args.users):
        sa = StudentAgent(user_id=idx, seed=args.seed + idx)
        users.append(UserCtx(user_id=idx, user_idx=idx, student=sa, user_vec=U[idx : idx + 1]))

    # Fixed
    fixed = TwoTowerRecommender(args.users, W.shape[0], args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False})
    fixed.user_matrix = U.clone().to(device)

    # Adaptive
    teacher = TwoTowerRecommender(args.users, W.shape[0], args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student = TwoTowerRecommender(args.users, W.shape[0], args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student.blend_increment = 0.3
    student.teacher_ema = 0.85
    adaptive = DualRecommender(teacher=teacher, student=student, device=str(device), warm_start=True, replay_size=256, replay_steps=1)
    for rec in (teacher, student):
        rec.user_matrix = U.clone().to(device)

    # Shared config
    base_cfg = dict(
        rounds=args.rounds,
        top_k_base=args.top_k,
        min_candidates=W.shape[0],
        console=False,
        console_user=False,
        deterministic_pool=True,
        persistent_pool=True,
        train_on_all_shown=True,
        train_steps_per_round=2,
        # Warmup with pseudo-labels to fit Funk + student
        warmup_rounds=8,
        warmup_steps=3,
        warmup_top_k_boost=2,
        warmup_diversity_scale=0.3,
        warmup_preloop=True,
        # Optional Funk flags (candidates / distill)
        use_funk_candidates=args.funk_candidates,
        funk_pool_size=(args.funk_pool or W.shape[0] // 2),
        funk_distill=args.funk_distill,
        funk_lambda=args.funk_lambda,
        log_path=str(args.log_dir / "tmp.jsonl"),
    )

    # Run fixed
    cfg_fixed = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "fixed.jsonl")})
    orch_fixed = MultiUserOrchestrator(
        rec=fixed, users=users, item_matrix_normal=W, item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos, pos2id=pos2id, id2pos=id2pos, item_meta_by_id=meta,
        cfg=cfg_fixed, device=device, mode_label="fixed",
    )
    orch_fixed.run()

    # Run adaptive
    cfg_adapt = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "adaptive.jsonl")})
    orch_adapt = MultiUserOrchestrator(
        rec=adaptive, users=users, item_matrix_normal=W, item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos, pos2id=pos2id, id2pos=id2pos, item_meta_by_id=meta,
        cfg=cfg_adapt, device=device, mode_label="adaptive",
    )
    orch_adapt.run()

    import pandas as pd
    def load(path: Path) -> pd.DataFrame:
        rows=[]
        for line in path.read_text().splitlines():
            try:
                obj=json.loads(line)
            except Exception:
                continue
            if obj.get("type")=="round_summary":
                m=obj.get("metrics",{})
                rows.append({k:m.get(k) for k in ["accept_rate","accuracy","novelty_rate","serendipity","mean_knowledge"]})
        return pd.DataFrame(rows)

    fx = load(args.log_dir / "fixed.jsonl").mean(numeric_only=True).to_dict()
    ad = load(args.log_dir / "adaptive.jsonl").mean(numeric_only=True).to_dict()
    return {"fixed": fx, "adaptive": ad}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic benchmark on sklearn digits (fixed vs adaptive)")
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--users", type=int, default=64)
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=Path, default=Path("runs/agentic-digits"))
    p.add_argument("--funk-candidates", action="store_true")
    p.add_argument("--funk-pool", type=int, default=0)
    p.add_argument("--funk-distill", action="store_true")
    p.add_argument("--funk-lambda", type=float, default=0.3)
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
