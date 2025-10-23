"""Run a lightweight agentic simulation with synthetic data.

This script wires together the primary components exposed by Orchid Ranker so
both engineers and researchers can validate that the orchestration loop runs on
fresh installs. It keeps dimensions tiny and disables DP to finish quickly.

Usage
-----
    python benchmarks/run_agentic_smoke.py --rounds 2 --users 3 --items 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from orchid_ranker import MultiConfig, MultiUserOrchestrator, StudentAgentFactory, UserCtx
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender


def _build_students(num_users: int, knowledge_dim: int, device: torch.device) -> tuple[list[UserCtx], torch.Tensor]:
    user_vecs = torch.rand(num_users, knowledge_dim, device=device)
    users: list[UserCtx] = []
    for idx in range(num_users):
        agent = StudentAgentFactory.create("zpd", user_id=idx, seed=idx + 7, verbose=False)
        users.append(UserCtx(user_id=idx, user_idx=idx, student=agent, user_vec=user_vecs[idx : idx + 1]))
    return users, user_vecs


def _build_item_matrix(num_items: int, item_dim: int, device: torch.device) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    matrix = torch.rand(num_items, item_dim, device=device)
    ids = list(range(num_items))
    mapping = {i: i for i in ids}
    return matrix, ids, mapping


def _build_recommender(num_users: int, num_items: int, dim: int, device: torch.device) -> TwoTowerRecommender:
    rec = TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=dim,
        item_dim=dim,
        hidden=16,
        emb_dim=8,
        device=str(device),
        dp_cfg={"enabled": False},
        mmr_lambda=0.1,
        novelty_bonus=0.05,
    )
    rec.eval()
    return rec


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic agentic simulation loop.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds to simulate")
    parser.add_argument("--users", type=int, default=3, help="Number of synthetic users")
    parser.add_argument("--items", type=int, default=12, help="Number of synthetic items")
    parser.add_argument("--log-path", type=Path, default=Path("runs/agentic-smoke.jsonl"), help="Where to write the JSONL log")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    users, user_matrix = _build_students(args.users, knowledge_dim=4, device=device)
    item_matrix, pos2id, id2pos = _build_item_matrix(args.items, item_dim=4, device=device)
    item_ids_pos = torch.arange(args.items, device=device)
    item_meta = {iid: {"difficulty": float(np.random.rand())} for iid in pos2id}

    recommender = _build_recommender(args.users, args.items, dim=4, device=device)
    recommender.user_matrix = user_matrix.clone().to(device)

    cfg = MultiConfig(
        rounds=args.rounds,
        top_k_base=3,
        min_candidates=args.items,
        novelty_bonus=0.05,
        mmr_lambda=0.1,
        log_path=str(args.log_path),
        console=False,
        console_user=False,
    )

    orchestrator = MultiUserOrchestrator(
        rec=recommender,
        users=users,
        item_matrix_normal=item_matrix,
        item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos,
        pos2id=pos2id,
        id2pos=id2pos,
        item_meta_by_id=item_meta,
        cfg=cfg,
        device=device,
        mode_label="smoke",
    )

    # ensure log directory exists
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    orchestrator.run()

    if Path(args.log_path).exists():
        rows = [json.loads(line) for line in Path(args.log_path).read_text().splitlines() if line.strip()]
        summary = [row for row in rows if row.get("type") == "round_summary"]
        print(f"Completed smoke run with {len(summary)} round summaries written to {args.log_path}")
    else:
        print("Smoke run completed without log output (likely because logging was disabled).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
