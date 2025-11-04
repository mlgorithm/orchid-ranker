"""Compare fixed vs adaptive (DualRecommender) under identical conditions.

- Shared candidate pools (deterministic)
- Optional train-on-all-shown negatives and extra steps per round
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from orchid_ranker import MultiConfig, MultiUserOrchestrator, UserCtx
from orchid_ranker.agents.student_agent import StudentAgent
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender, DualRecommender


def build_users(n: int, d: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    U = torch.rand(n, d, device=device)
    users = []
    for idx in range(n):
        sa = StudentAgent(user_id=idx, seed=seed + idx)
        users.append(UserCtx(user_id=idx, user_idx=idx, student=sa, user_vec=U[idx : idx + 1]))
    return users, U


def build_items(m: int, d: int, device: torch.device):
    X = torch.rand(m, d, device=device)
    pos2id = list(range(m))
    id2pos = {i: i for i in pos2id}
    item_ids_pos = torch.arange(m, device=device)
    meta = {i: {"difficulty": float(np.random.rand())} for i in pos2id}
    return X, pos2id, id2pos, item_ids_pos, meta


def run_once(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    users, U = build_users(args.users, args.dim, args.seed, device)
    X, pos2id, id2pos, item_ids_pos, meta = build_items(args.items, args.dim, device)

    # Fixed
    fixed = TwoTowerRecommender(args.users, args.items, args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False})
    fixed.user_matrix = U.clone().to(device)

    # Adaptive with warm start + replay + exploration
    teacher = TwoTowerRecommender(args.users, args.items, args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student = TwoTowerRecommender(args.users, args.items, args.dim, args.dim, hidden=32, emb_dim=16, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16)
    student.blend_increment = 0.3
    student.teacher_ema = 0.85
    adaptive = DualRecommender(teacher=teacher, student=student, device=str(device), warm_start=True, replay_size=256, replay_steps=1)
    for rec in (teacher, student):
        rec.user_matrix = U.clone().to(device)

    # Shared config
    base_cfg = dict(
        rounds=args.rounds,
        top_k_base=args.top_k,
        min_candidates=args.items,
        console=False,
        console_user=False,
        deterministic_pool=True,
        persistent_pool=True,
        train_on_all_shown=True,
        train_steps_per_round=2,
        warmup_rounds=8,
        warmup_steps=3,
        warmup_top_k_boost=2,
        warmup_diversity_scale=0.3,
        warmup_preloop=True,
        log_path=str(args.log_dir / "tmp.jsonl"),  # overwritten per run below
    )

    # Run fixed
    cfg_fixed = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "fixed.jsonl")})
    orch_fixed = MultiUserOrchestrator(
        rec=fixed, users=users, item_matrix_normal=X, item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos, pos2id=pos2id, id2pos=id2pos, item_meta_by_id=meta,
        cfg=cfg_fixed, device=device, mode_label="fixed",
    )
    orch_fixed.run()

    # Run adaptive
    cfg_adapt = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "adaptive.jsonl")})
    # Optional Funk-based candidate generation
    if hasattr(args, 'funk_candidates') and args.funk_candidates:
        cfg_adapt.use_funk_candidates = True
        cfg_adapt.funk_pool_size = int(getattr(args, 'funk_pool', 0))
        cfg_adapt.warmup_preloop = True
    orch_adapt = MultiUserOrchestrator(
        rec=adaptive, users=users, item_matrix_normal=X, item_matrix_sanitized=None,
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
    p = argparse.ArgumentParser(description="Compare fixed vs adaptive under identical pools")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--users", type=int, default=12)
    p.add_argument("--items", type=int, default=48)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=Path, default=Path("runs/agentic-adaptive"))
    p.add_argument("--funk-candidates", action="store_true", help="Use Funk top-N candidate generation in adaptive run")
    p.add_argument("--funk-pool", type=int, default=0, help="Candidate pool size for Funk; 0 => use min_candidates")
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
