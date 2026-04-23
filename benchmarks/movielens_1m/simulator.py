"""Click-probability simulator for replay-based retention scoring.

Trains a small MLP on actual user-item interaction data from MovieLens-1M,
then serves as a "ground truth" oracle to simulate whether a user would
click/enjoy a recommended item.  Used by the benchmark replay loop to
evaluate recommenders on session-level retention metrics.

The MLP is trained on binary labels (rating >= 4 => positive) and predicts
``P(click | user, item)`` for arbitrary (user, item) pairs.

Standalone smoke-test::

    python benchmarks/movielens_1m/simulator.py --help

See Appendix D of the technical report for the full protocol.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

__all__ = [
    "ClickSimulator",
    "replay_sessions",
]


# ---------------------------------------------------------------------------
# Recommender protocol (duck-typed interface expected by replay_sessions)
# ---------------------------------------------------------------------------
@runtime_checkable
class Recommender(Protocol):
    """Minimal interface a recommender must expose for replay."""

    def recommend(
        self, user_idx: int, k: int, exclude: set[int]
    ) -> list[int]:
        """Return top-k item indices, excluding items in *exclude*."""
        ...

    def score(self, user_idx: int, item_idx: int) -> float:
        """Return a scalar relevance score for (user, item)."""
        ...


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class _ClickMLP(nn.Module):
    """Two-layer MLP for click prediction.

    Architecture
    ------------
    * User path:  ``Embedding(num_users, embed_dim)``
    * Item path:  ``Linear(item_feat_dim, embed_dim)``  (projects
      pre-computed item features into the same space)
    * Concat ``[user_emb, item_emb]`` (2 * embed_dim)
    * Hidden: ``Linear(2*embed_dim, hidden_dim) -> ReLU -> Dropout(0.2)``
    * Output: ``Linear(hidden_dim, 1) -> Sigmoid``
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_feat_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_proj = nn.Linear(item_feat_dim, embed_dim)
        self.hidden = nn.Linear(2 * embed_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self, user_ids: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        """Return click probabilities in [0, 1].

        Parameters
        ----------
        user_ids : Tensor[B]  (long)
        item_features : Tensor[B, item_feat_dim]  (float)

        Returns
        -------
        Tensor[B] — predicted P(click).
        """
        u = self.user_emb(user_ids)               # [B, embed_dim]
        i = self.item_proj(item_features)          # [B, embed_dim]
        x = torch.cat([u, i], dim=-1)              # [B, 2*embed_dim]
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x)).squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# ClickSimulator
# ---------------------------------------------------------------------------
class ClickSimulator:
    """MLP-based click probability model for replay evaluation.

    Trained on historical interactions.  Given a ``(user, item)`` pair,
    predicts the probability that the user would "click" (i.e., rate >= 4).

    Parameters
    ----------
    num_users : int
        Total number of distinct users (determines embedding table size).
    num_items : int
        Total number of distinct items.
    item_features : np.ndarray
        Pre-computed item feature matrix of shape ``(num_items, feat_dim)``.
        Expected dimensionality: 24 (18 genre + 5 year-bucket + 1 avg_rating)
        from ``preprocess.py``, but any positive integer works.
    embed_dim : int
        Embedding dimensionality for both user and item paths.
    hidden_dim : int
        Width of the hidden layer.
    device : str
        ``"cpu"`` or a CUDA device string.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_features: np.ndarray,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users and num_items must be positive")
        if item_features.shape[0] != num_items:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows but "
                f"num_items={num_items}"
            )

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.device = str(device)
        self._item_feat_dim = int(item_features.shape[1])

        # Store item features as a pinned tensor for fast lookup.
        self._item_features = torch.tensor(
            item_features, dtype=torch.float32, device=self.device
        )

        self._model = _ClickMLP(
            num_users=self.num_users,
            num_items=self.num_items,
            item_feat_dim=self._item_feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        user_id_to_idx: Dict[str | int, int],
        item_id_to_idx: Dict[str | int, int],
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 1024,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Train the MLP on historical interactions.

        Parameters
        ----------
        train_df : DataFrame
            Must contain columns ``user_id``, ``item_id``, ``rating``,
            ``label``, ``timestamp``.  ``label`` should be 1 if
            ``rating >= 4``, else 0.
        user_id_to_idx : dict
            Maps raw user IDs to contiguous indices in ``[0, num_users)``.
        item_id_to_idx : dict
            Maps raw item IDs to contiguous indices in ``[0, num_items)``.
        epochs : int
            Number of training epochs.
        lr : float
            Adam learning rate.
        batch_size : int
            Mini-batch size.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Training metrics: ``epoch_losses`` (list of mean loss per epoch),
            ``final_loss``, ``final_acc``.
        """
        # Seed everything
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Map raw IDs to contiguous indices
        user_idxs = train_df["user_id"].map(user_id_to_idx).to_numpy(dtype=np.int64)
        item_idxs = train_df["item_id"].map(item_id_to_idx).to_numpy(dtype=np.int64)
        labels = train_df["label"].to_numpy(dtype=np.float32)

        # Validate — drop any rows whose IDs were not in the mapping
        valid = ~(np.isnan(user_idxs.astype(float)) | np.isnan(item_idxs.astype(float)))
        if not valid.all():
            n_drop = int((~valid).sum())
            logger.warning(
                "Dropping %d rows with unmapped user/item IDs", n_drop
            )
            user_idxs = user_idxs[valid]
            item_idxs = item_idxs[valid]
            labels = labels[valid]

        n = len(labels)
        logger.info(
            "Training ClickSimulator: %d samples, %d epochs, lr=%.4f, "
            "batch_size=%d",
            n, epochs, lr, batch_size,
        )

        # Build item feature batch from stored tensor
        item_feat_batch = self._item_features[
            torch.tensor(item_idxs, dtype=torch.long, device=self.device)
        ]  # [N, feat_dim]

        dataset = TensorDataset(
            torch.tensor(user_idxs, dtype=torch.long, device=self.device),
            item_feat_batch,
            torch.tensor(labels, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self._model.train()
        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            for u_batch, feat_batch, y_batch in loader:
                preds = self._model(u_batch, feat_batch)
                loss = loss_fn(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach()) * len(y_batch)
                correct += int(
                    ((preds.detach() >= 0.5).float() == y_batch).sum()
                )
                total += len(y_batch)

            mean_loss = running_loss / max(total, 1)
            acc = correct / max(total, 1)
            epoch_losses.append(mean_loss)
            logger.info(
                "  epoch %d/%d  loss=%.4f  acc=%.4f", epoch, epochs, mean_loss, acc
            )

        self._model.eval()
        self._fitted = True

        return {
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "final_acc": acc,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def click_prob(self, user_idx: int, item_idx: int) -> float:
        """Predict click probability for a single (user, item) pair.

        Parameters
        ----------
        user_idx : int
            Contiguous user index in ``[0, num_users)``.
        item_idx : int
            Contiguous item index in ``[0, num_items)``.

        Returns
        -------
        float
            Predicted P(click) in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError(
                "ClickSimulator has not been fitted yet. Call .fit() first."
            )
        u = torch.tensor([int(user_idx)], dtype=torch.long, device=self.device)
        feat = self._item_features[int(item_idx)].unsqueeze(0)  # [1, feat_dim]
        p = self._model(u, feat)
        return float(p.item())

    @torch.no_grad()
    def click_probs_batch(
        self, user_idxs: np.ndarray, item_idxs: np.ndarray
    ) -> np.ndarray:
        """Vectorised click probabilities.

        Parameters
        ----------
        user_idxs : np.ndarray[int]
            Array of user indices.
        item_idxs : np.ndarray[int]
            Array of item indices (same length as *user_idxs*).

        Returns
        -------
        np.ndarray[float]
            Predicted P(click) for each pair.
        """
        if not self._fitted:
            raise RuntimeError(
                "ClickSimulator has not been fitted yet. Call .fit() first."
            )
        if len(user_idxs) != len(item_idxs):
            raise ValueError("user_idxs and item_idxs must have the same length")

        u = torch.tensor(
            np.asarray(user_idxs, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        feats = self._item_features[
            torch.tensor(
                np.asarray(item_idxs, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )
        ]  # [B, feat_dim]
        preds = self._model(u, feats)
        return preds.cpu().numpy()


# ---------------------------------------------------------------------------
# Replay session simulation
# ---------------------------------------------------------------------------
def replay_sessions(
    simulator: ClickSimulator,
    recommender: Any,
    num_users: int,
    max_steps: int = 30,
    top_k: int = 1,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run replay-based session simulation.

    For each user:

    1. Start a session with an empty history (seen set).
    2. At each step, the recommender produces a top-1 recommendation
       (excluding previously seen items).
    3. The simulator samples a click via ``Bernoulli(click_prob(user, item))``.
    4. If clicked, the session continues and the item is added to the seen set.
    5. If not clicked, the session ends.

    The *threshold* parameter is **not** used for deterministic override by
    default — all clicks are stochastic (Bernoulli).  The threshold is
    available if callers want a deterministic variant, but the seeded RNG
    ensures reproducibility either way.

    Parameters
    ----------
    simulator : ClickSimulator
        Trained click oracle.
    recommender
        Must expose ``recommend(user_idx, k, exclude) -> list[int]`` and
        ``score(user_idx, item_idx) -> float``.
    num_users : int
        Number of users to simulate (indices ``0 .. num_users-1``).
    max_steps : int
        Maximum session length (hard cap).
    top_k : int
        Passed to the recommender; only the first recommendation is used
        per step.
    threshold : float
        Currently unused in the default stochastic mode.  Reserved for a
        deterministic variant (``click_prob >= threshold`` => click).
    seed : int
        RNG seed for Bernoulli sampling.

    Returns
    -------
    dict
        ``session_lengths`` : np.ndarray of per-user session lengths.
        ``survival_5``      : fraction of users surviving >= 5 steps.
        ``survival_10``     : fraction of users surviving >= 10 steps.
        ``survival_20``     : fraction of users surviving >= 20 steps.
        ``mean_session_length`` : float.
    """
    rng = np.random.default_rng(seed)
    session_lengths = np.zeros(num_users, dtype=np.int64)

    logger.info(
        "Starting replay: %d users, max_steps=%d, top_k=%d, seed=%d",
        num_users, max_steps, top_k, seed,
    )

    for user_idx in range(num_users):
        seen: set[int] = set()
        length = 0

        for step in range(max_steps):
            # Ask recommender for candidates
            recs = recommender.recommend(user_idx, top_k, exclude=seen)
            if not recs:
                # No more items to recommend — session ends
                break

            item_idx = recs[0]

            # Oracle click probability
            p = simulator.click_prob(user_idx, item_idx)

            # Stochastic Bernoulli sample
            clicked = bool(rng.random() < p)

            if clicked:
                seen.add(item_idx)
                length += 1
            else:
                # Session ends on first non-click
                break

        session_lengths[user_idx] = length

        if (user_idx + 1) % 500 == 0:
            logger.info(
                "  replayed %d / %d users (mean length so far: %.2f)",
                user_idx + 1,
                num_users,
                float(session_lengths[: user_idx + 1].mean()),
            )

    mean_len = float(session_lengths.mean())
    survival_5 = float((session_lengths >= 5).mean())
    survival_10 = float((session_lengths >= 10).mean())
    survival_20 = float((session_lengths >= 20).mean())

    logger.info(
        "Replay complete: mean_session=%.2f  surv@5=%.3f  surv@10=%.3f  surv@20=%.3f",
        mean_len, survival_5, survival_10, survival_20,
    )

    return {
        "session_lengths": session_lengths,
        "survival_5": survival_5,
        "survival_10": survival_10,
        "survival_20": survival_20,
        "mean_session_length": mean_len,
    }


# ---------------------------------------------------------------------------
# CLI entry point (smoke test / help)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python benchmarks/movielens_1m/simulator.py",
        description=(
            "Click-probability simulator for the MovieLens-1M benchmark.  "
            "This module is primarily used as a library import.  Running it "
            "directly prints usage information and optionally runs a quick "
            "smoke test with synthetic data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example library usage:\n"
            "\n"
            "    from benchmarks.movielens_1m.simulator import ClickSimulator\n"
            "\n"
            "    sim = ClickSimulator(num_users=6040, num_items=3952,\n"
            "                         item_features=item_feat_array)\n"
            "    metrics = sim.fit(train_df, u2idx, i2idx, epochs=20)\n"
            "    p = sim.click_prob(user_idx=0, item_idx=42)\n"
        ),
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with tiny synthetic data to verify "
             "that the simulator trains and predicts without errors.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    return p


def _smoke_test(seed: int = 42) -> None:
    """Quick end-to-end check with tiny synthetic data."""
    logger.info("Running smoke test with synthetic data (seed=%d)...", seed)

    num_users, num_items, feat_dim = 50, 100, 8
    rng = np.random.default_rng(seed)

    item_features = rng.standard_normal((num_items, feat_dim)).astype(np.float32)

    # Synthetic train data
    n_samples = 2000
    user_ids = rng.integers(0, num_users, size=n_samples)
    item_ids = rng.integers(0, num_items, size=n_samples)
    ratings = np.clip(rng.normal(3.5, 1.0, size=n_samples), 1.0, 5.0)
    labels = (ratings >= 4.0).astype(float)
    timestamps = np.arange(n_samples, dtype=float)

    train_df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "label": labels,
        "timestamp": timestamps,
    })

    u2idx = {uid: uid for uid in range(num_users)}
    i2idx = {iid: iid for iid in range(num_items)}

    sim = ClickSimulator(
        num_users=num_users,
        num_items=num_items,
        item_features=item_features,
        embed_dim=16,
        hidden_dim=32,
        device="cpu",
    )
    metrics = sim.fit(train_df, u2idx, i2idx, epochs=5, lr=0.001, seed=seed)
    logger.info("Training metrics: %s", metrics)

    # Single prediction
    p = sim.click_prob(0, 0)
    logger.info("click_prob(0, 0) = %.4f", p)
    assert 0.0 <= p <= 1.0, f"click_prob out of range: {p}"

    # Batch prediction
    u_batch = np.array([0, 1, 2, 3, 4])
    i_batch = np.array([10, 20, 30, 40, 50])
    probs = sim.click_probs_batch(u_batch, i_batch)
    logger.info("Batch probs: %s", probs)
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Replay with a trivial random recommender
    class _RandomRecommender:
        def __init__(self, num_items: int, seed: int) -> None:
            self._num_items = num_items
            self._rng = np.random.default_rng(seed)

        def recommend(
            self, user_idx: int, k: int, exclude: set[int]
        ) -> list[int]:
            available = [i for i in range(self._num_items) if i not in exclude]
            if not available:
                return []
            chosen = self._rng.choice(available, size=min(k, len(available)), replace=False)
            return list(chosen)

        def score(self, user_idx: int, item_idx: int) -> float:
            return 0.0

    rec = _RandomRecommender(num_items, seed=seed + 1)
    results = replay_sessions(
        simulator=sim,
        recommender=rec,
        num_users=num_users,
        max_steps=15,
        seed=seed,
    )
    logger.info(
        "Replay results: mean_session=%.2f  surv@5=%.3f  surv@10=%.3f",
        results["mean_session_length"],
        results["survival_5"],
        results["survival_10"],
    )
    logger.info("Smoke test PASSED.")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test(seed=args.seed)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
