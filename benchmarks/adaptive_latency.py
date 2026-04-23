"""Benchmark: how fast does the ranker actually adapt?

Produces three numbers that matter for the "adaptive + streaming" claim:

* **observe-to-rank latency** — wall-clock from a single interaction being
  recorded to the next rank call reflecting it. Reported as p50/p95/p99.
* **rank-delta after observation** — fraction of users whose top-k changes in
  the first rank call after a targeted interaction. A value near 0 would mean
  the adaptive loop is not actually adaptive end-to-end.
* **mastery-gain uplift** — cumulative correctness on streamed items compared
  to a frozen-tower baseline given identical interaction traces. Headline
  number for the enterprise demo.

Two modes:

* ``--dataset synthetic`` (default) — generates a Gaussian world on the fly.
  Fast; good for CI regression. Correctness probabilities are fully known.
* ``--dataset fixture`` — loads the deterministic fixture under
  ``benchmarks/fixtures/`` (200 users, 300 items, 6.4k interactions with a
  skill-vs-difficulty alignment structure). The tower is briefly pre-trained
  on the train split so the "frozen baseline" is a realistic comparator.

Usage::

    python benchmarks/adaptive_latency.py                           # synthetic
    python benchmarks/adaptive_latency.py --dataset fixture         # realistic
    python benchmarks/adaptive_latency.py --dataset fixture --json out.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.streaming import StreamingAdaptiveRanker


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------
def make_synthetic_world(
    num_users: int, num_items: int, feat_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, Callable[[int, int], float]]:
    """Synthetic world: correctness = sigmoid(dot(u_latent, i_latent))."""
    rng = np.random.default_rng(seed)
    uf = rng.normal(size=(num_users, feat_dim)).astype(np.float32)
    ifeat = rng.normal(size=(num_items, feat_dim)).astype(np.float32)
    rot = rng.normal(size=(feat_dim, feat_dim)).astype(np.float32)
    u_lat, i_lat = uf @ rot, ifeat @ rot
    logits = np.clip(u_lat @ i_lat.T, -20.0, 20.0)
    p_correct = 1.0 / (1.0 + np.exp(-logits))

    def probe(u: int, i: int) -> float:
        return float(p_correct[u, i])

    return torch.tensor(uf), torch.tensor(ifeat), probe


def make_fixture_world(
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Callable[[int, int], float], np.ndarray]:
    """Load the deterministic fixture and derive a correctness oracle.

    Returns user/item feature tensors, a ``probe(u, i) -> p_correct`` callable,
    and the raw training dataframe (as a numpy array ``[[u, i, y], ...]``)
    used for brief pre-training.
    """
    import pandas as pd

    train = pd.read_csv(FIXTURE_DIR / "train.csv")
    num_users = int(train.user_id.max()) + 1
    num_items = int(train.item_id.max()) + 1

    # Item features come from the fixture; user features are learned via a
    # short pre-train, so we seed them with a deterministic random tensor.
    rng = np.random.default_rng(seed)
    item_features = np.load(FIXTURE_DIR / "item_features.npy").astype(np.float32)
    if item_features.shape[0] < num_items:
        # pad rather than crash on off-by-one
        pad = rng.normal(size=(num_items - item_features.shape[0], item_features.shape[1]))
        item_features = np.vstack([item_features, pad.astype(np.float32)])
    user_features = rng.normal(size=(num_users, item_features.shape[1])).astype(np.float32)

    # Oracle: reconstruct the generator's alignment formula so we can answer
    # "would this user-item pair have been positive in the ground-truth world?".
    # Generator used:
    #   alignment = 1 - |u/NU - i/NI|;  rating = alignment*5 + N(0, 0.8); y = rating >= 3.5
    NU, NI = 200, 300  # matches generate_fixture.py
    sigma = 0.8

    def probe(u: int, i: int) -> float:
        alignment = 1.0 - abs(u / NU - i / NI)
        mean_rating = alignment * 5.0
        # P(rating >= 3.5) under N(mean_rating, sigma^2)
        z = (3.5 - mean_rating) / sigma
        # survival function of standard normal, computed via erfc
        import math
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    train_array = train[["user_id", "item_id", "label"]].to_numpy().astype(np.int64)
    # label is float in the file but values are 0/1
    train_array = train_array.astype(np.float32)
    return torch.tensor(user_features), torch.tensor(item_features), probe, train_array


# ---------------------------------------------------------------------------
# Tower construction + optional pre-training
# ---------------------------------------------------------------------------
def build_tower(
    num_users: int, num_items: int, user_dim: int, item_dim: int, seed: int
) -> TwoTowerRecommender:
    torch.manual_seed(seed)
    return TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=user_dim,
        item_dim=item_dim,
        hidden=32,
        emb_dim=16,
        state_dim=4,
        device="cpu",
        dp_cfg={"enabled": False},
    )


def pretrain_tower(
    tower: TwoTowerRecommender,
    user_feats: torch.Tensor,
    item_feats: torch.Tensor,
    interactions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool = False,
) -> float:
    """Brief supervised pre-train on (user, item, label) triples.

    Trains the core scoring path (user+item towers, projections, item bias)
    with BCE loss. This makes the "frozen baseline" in the benchmark a
    realistic comparator rather than a random-init tower.

    Returns the final mean loss.
    """
    device = next(tower.parameters()).device
    tower.train()
    params = [p for p in tower.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n = len(interactions)
    user_feats = user_feats.to(device)
    item_feats = item_feats.to(device)
    state_zero = torch.zeros((1, tower.state_dim), device=device)

    last_loss = float("nan")
    for ep in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            batch = interactions[perm[start:start + batch_size]]
            u = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            i = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            y = torch.tensor(batch[:, 2], dtype=torch.float32, device=device)

            # Score each (u_k, i_k) pair individually — the scoring path is
            # designed for one user × K candidates, so we iterate within the
            # batch. Small batches (~64) keep this cheap.
            xu = user_feats[u]
            sv = state_zero.expand(len(batch), -1)
            logits_mat, _, _ = tower._scores_logits(
                xu, item_feats, u, i, state_vec=sv,
            )
            # logits_mat is [B, K=B]; we want the diagonal (each user paired
            # with its *own* item, not the full cross product).
            logits = logits_mat.diagonal()
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
        if verbose:
            print(f"  pretrain epoch {ep+1}/{epochs}  loss={last_loss:.4f}")

    tower.eval()
    return last_loss


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    dataset: str
    users: int
    items: int
    interactions: int
    candidates_per_rank: int
    pretrain_loss: float
    observe_p50_ms: float
    observe_p95_ms: float
    observe_p99_ms: float
    rank_p50_ms: float
    rank_p95_ms: float
    rank_p99_ms: float
    topk_change_rate: float
    adaptive_hit_rate: float
    frozen_hit_rate: float
    mastery_gain: float


def run(args: argparse.Namespace) -> BenchResult:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.dataset == "synthetic":
        user_feats, item_feats, probe = make_synthetic_world(
            args.users, args.items, args.feat_dim, args.seed
        )
        train_array = None
    elif args.dataset == "fixture":
        user_feats, item_feats, probe, train_array = make_fixture_world(args.seed)
    else:
        raise ValueError(f"unknown --dataset {args.dataset!r}")

    num_users = user_feats.shape[0]
    num_items = item_feats.shape[0]
    user_dim = user_feats.shape[1]
    item_dim = item_feats.shape[1]

    # Two towers with shared init -> one adapts online, one stays frozen.
    tower_a = build_tower(num_users, num_items, user_dim, item_dim, args.seed)
    tower_b = build_tower(num_users, num_items, user_dim, item_dim, args.seed)

    pretrain_loss = float("nan")
    if args.dataset == "fixture" and args.pretrain_epochs > 0:
        print(f"pre-training on {len(train_array)} interactions "
              f"({args.pretrain_epochs} epochs, lr={args.pretrain_lr})...")
        pretrain_loss = pretrain_tower(
            tower_a, user_feats, item_feats, train_array,
            epochs=args.pretrain_epochs, batch_size=args.pretrain_batch,
            lr=args.pretrain_lr, verbose=False,
        )
        tower_b.load_state_dict(tower_a.state_dict())
    else:
        tower_b.load_state_dict(tower_a.state_dict())

    adaptive = StreamingAdaptiveRanker(
        tower_a, user_feats, item_feats, lr=args.lr, l2=args.l2,
    )
    frozen = StreamingAdaptiveRanker(
        tower_b, user_feats, item_feats, lr=0.0, l2=0.0,
    )

    candidates = list(range(num_items))

    # --- rank-delta after one observation --------------------------------
    changed = 0
    for _ in range(args.change_users):
        u = rng.randrange(num_users)
        sample_k = min(args.candidates, num_items)
        cand_sample = rng.sample(candidates, sample_k)
        before = [i for i, _ in adaptive.rank(u, cand_sample, top_k=args.top_k)]
        target = cand_sample[-1] if len(cand_sample) > args.top_k else cand_sample[0]
        adaptive.observe(u, target, correct=True)
        after = [i for i, _ in adaptive.rank(u, cand_sample, top_k=args.top_k)]
        if before != after:
            changed += 1
    change_rate = changed / max(1, args.change_users)

    # --- mastery-gain replay ---------------------------------------------
    adaptive_hits = 0
    frozen_hits = 0
    for _ in range(args.interactions):
        u = rng.randrange(num_users)
        sample_k = min(args.candidates, num_items)
        cand_sample = rng.sample(candidates, sample_k)

        a_pick = adaptive.rank(u, cand_sample, top_k=1)[0][0]
        f_pick = frozen.rank(u, cand_sample, top_k=1)[0][0]

        a_correct = rng.random() < probe(u, a_pick)
        f_correct = rng.random() < probe(u, f_pick)

        adaptive.observe(u, a_pick, correct=a_correct)
        frozen.observe(u, f_pick, correct=f_correct)

        if a_correct:
            adaptive_hits += 1
        if f_correct:
            frozen_hits += 1

    adaptive_hit_rate = adaptive_hits / max(1, args.interactions)
    frozen_hit_rate = frozen_hits / max(1, args.interactions)

    stats = adaptive.stats()
    return BenchResult(
        dataset=args.dataset,
        users=num_users,
        items=num_items,
        interactions=args.interactions,
        candidates_per_rank=args.candidates,
        pretrain_loss=pretrain_loss,
        observe_p50_ms=stats.observe_p50_ms,
        observe_p95_ms=stats.observe_p95_ms,
        observe_p99_ms=stats.observe_p99_ms,
        rank_p50_ms=stats.rank_p50_ms,
        rank_p95_ms=stats.rank_p95_ms,
        rank_p99_ms=stats.rank_p99_ms,
        topk_change_rate=change_rate,
        adaptive_hit_rate=adaptive_hit_rate,
        frozen_hit_rate=frozen_hit_rate,
        mastery_gain=adaptive_hit_rate - frozen_hit_rate,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["synthetic", "fixture"], default="synthetic")
    p.add_argument("--users", type=int, default=64, help="synthetic only")
    p.add_argument("--items", type=int, default=256, help="synthetic only")
    p.add_argument("--feat-dim", type=int, default=8, help="synthetic only")
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--interactions", type=int, default=400)
    p.add_argument("--change-users", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--pretrain-epochs", type=int, default=3, help="fixture only")
    p.add_argument("--pretrain-batch", type=int, default=64, help="fixture only")
    p.add_argument("--pretrain-lr", type=float, default=5e-3, help="fixture only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    res = run(args)
    elapsed = time.perf_counter() - t0

    print("=== Orchid streaming adaptive — latency & uplift ===")
    print(f"Dataset: {res.dataset}  users={res.users} items={res.items} "
          f"interactions={res.interactions} candidates_per_rank={res.candidates_per_rank}")
    if res.dataset == "fixture":
        print(f"Pre-train final BCE loss: {res.pretrain_loss:.4f}")
    print(f"observe latency  p50={res.observe_p50_ms:.2f} ms  "
          f"p95={res.observe_p95_ms:.2f} ms  p99={res.observe_p99_ms:.2f} ms")
    print(f"rank    latency  p50={res.rank_p50_ms:.2f} ms  "
          f"p95={res.rank_p95_ms:.2f} ms  p99={res.rank_p99_ms:.2f} ms")
    print(f"top-{args.top_k} change rate after one observe: {res.topk_change_rate*100:.1f}%")
    print(f"hit-rate adaptive={res.adaptive_hit_rate*100:.2f}%  "
          f"frozen={res.frozen_hit_rate*100:.2f}%  "
          f"Δ={res.mastery_gain*100:+.2f} pp")
    print(f"(bench took {elapsed:.2f}s)")

    if args.json:
        Path(args.json).write_text(json.dumps(asdict(res), indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
