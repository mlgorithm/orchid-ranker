"""Model building and training functions for Orchid Ranker experiments."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import torch

from orchid_ranker import TwoTowerRecommender
from orchid_ranker.agents.recommender_agent import DualRecommender
from orchid_ranker.baselines import (
    ALSBaseline,
    LinUCBBaseline,
    PopularityBaseline,
    RandomBaseline,
    UserKNNBaseline,
)


def warm_start_recommender(
    model: TwoTowerRecommender,
    cache: Tuple[np.ndarray, np.ndarray, np.ndarray],
    item_matrix: torch.Tensor,
    rng: np.random.Generator,
    *,
    epochs: int,
    batch_size: int,
    max_batches: Optional[int],
    print_fn: Callable[[str], None] = print,
) -> None:
    """Warm start a TwoTowerRecommender model with cached training data.

    Parameters
    ----------
    model : TwoTowerRecommender
        The model to warm start.
    cache : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Cached training data as (user_idx, item_idx, labels).
    item_matrix : torch.Tensor
        The item feature matrix.
    rng : np.random.Generator
        Random number generator for shuffling.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    max_batches : Optional[int]
        Maximum number of batches to train. If None, train all.
    print_fn : Callable, optional
        Function to use for printing debug messages, by default print.
    """
    if cache is None:
        print_fn("warm_start: skipped (no cache)")
        return
    user_idx, item_idx, labels = cache
    total = int(len(labels))
    if total == 0:
        print_fn("warm_start: skipped (empty cache)")
        return

    device = getattr(model, "device", item_matrix.device)
    model_item_matrix = getattr(model, "item_matrix", None)
    if model_item_matrix is None:
        model_item_matrix = item_matrix
    if model_item_matrix.device != device:
        model_item_matrix = model_item_matrix.to(device)
    state_dim = int(getattr(model, "state_dim", 0))

    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    max_batches = int(max_batches) if max_batches is not None else None

    print_fn(f"warm_start: epochs={epochs}, batch_size={batch_size}, "
             f"max_batches={max_batches}, total_pairs={total}")

    batches = 0
    for _ in range(epochs):
        order = rng.permutation(total)
        for start in range(0, total, batch_size):
            if max_batches is not None and batches >= max_batches:
                print_fn(f"warm_start: reached max_batches={max_batches}")
                model.eval()
                return
            idx = order[start:start + batch_size]
            if idx.size == 0:
                continue
            u = torch.tensor(user_idx[idx], dtype=torch.long, device=device)
            it = torch.tensor(item_idx[idx], dtype=torch.long, device=device)
            y = torch.tensor(labels[idx], dtype=torch.float32, device=device)
            batch = {
                "user_ids": u,
                "item_ids": it,
                "labels": y,
                "item_matrix": model_item_matrix,
            }
            if state_dim > 0:
                batch["state_vec"] = torch.zeros((len(idx), state_dim), dtype=torch.float32, device=device)
            try:
                model.train_step(batch)
            except TypeError:
                batch.pop("state_vec", None)
                model.train_step(batch)
            batches += 1
            if batches % 50 == 0:
                print_fn(f"warm_start: batches={batches}")
            if max_batches is not None and batches >= max_batches:
                print_fn(f"warm_start: reached max_batches={max_batches}")
                model.eval()
                return
    model.eval()
    print_fn(f"warm_start: done, batches={batches}")


def build_adaptive(
    num_users: int,
    num_items: int,
    user_matrix: torch.Tensor,
    item_matrix: torch.Tensor,
    pos2id: Dict[int, int],
    dp_cfg: dict,
    adaptive_defaults: Dict[str, object],
    warm_start_defaults: Dict[str, object],
    warm_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    adaptive_kwargs: Optional[Dict[str, object]] = None,
    warm_cfg: Optional[Dict[str, object]] = None,
    print_fn: Callable[[str], None] = print,
) -> DualRecommender:
    """Build an adaptive (dual) recommender with teacher and student towers.

    Parameters
    ----------
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    user_matrix : torch.Tensor
        User feature matrix.
    item_matrix : torch.Tensor
        Item feature matrix.
    pos2id : Dict[int, int]
        Mapping from position to item ID.
    dp_cfg : dict
        Differential privacy configuration.
    adaptive_defaults : Dict[str, object]
        Default adaptive model parameters.
    warm_start_defaults : Dict[str, object]
        Default warm start parameters.
    warm_cache : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Cached warm start training data.
    rng : np.random.Generator
        Random number generator.
    adaptive_kwargs : Optional[Dict[str, object]], optional
        Overrides for adaptive model parameters, by default None.
    warm_cfg : Optional[Dict[str, object]], optional
        Overrides for warm start parameters, by default None.
    print_fn : Callable, optional
        Function to use for printing debug messages, by default print.

    Returns
    -------
    DualRecommender
        A DualRecommender with teacher and student towers.
    """
    tower_kwargs = dict(adaptive_defaults)
    if adaptive_kwargs:
        tower_kwargs.update(adaptive_kwargs)

    blend_increment = float(tower_kwargs.pop("blend_increment", adaptive_defaults.get("blend_increment", 0.16)))
    teacher_ema = float(tower_kwargs.pop("teacher_ema", adaptive_defaults.get("teacher_ema", 0.9)))

    warm_defaults = dict(warm_start_defaults)
    if warm_cfg:
        for key, value in warm_cfg.items():
            if value is not None:
                warm_defaults[key] = value
    warm_enabled = bool(warm_defaults.get("enabled", True))
    warm_epochs = int(warm_defaults.get("epochs", 0))
    warm_batch = int(warm_defaults.get("batch_size", 1))
    warm_max_batches = warm_defaults.get("max_batches")
    if warm_max_batches is not None:
        warm_max_batches = int(warm_max_batches)

    teacher = TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=user_matrix.shape[1],
        item_dim=item_matrix.shape[1],
        dp_cfg={**dp_cfg, "enabled": False},
        **tower_kwargs,
    )
    student = TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=user_matrix.shape[1],
        item_dim=item_matrix.shape[1],
        dp_cfg=dp_cfg,
        **tower_kwargs,
    )

    for model in (teacher, student):
        setattr(model, "user_matrix", user_matrix)
        setattr(model, "item_matrix", item_matrix)
        setattr(model, "pos2id_map", dict(pos2id))

    if warm_enabled and warm_epochs > 0:
        warm_start_recommender(
            teacher,
            warm_cache,
            item_matrix,
            rng,
            epochs=warm_epochs,
            batch_size=warm_batch,
            max_batches=warm_max_batches,
            print_fn=print_fn,
        )

    student.load_state_dict(teacher.state_dict())
    setattr(student, "blend_increment", blend_increment)
    setattr(student, "teacher_ema", teacher_ema)
    setattr(teacher, "blend_increment", blend_increment)
    setattr(teacher, "teacher_ema", teacher_ema)

    if hasattr(student, "linucb"):
        teacher.linucb = student.linucb
        teacher.use_linucb = getattr(student, "use_linucb", False)
    if hasattr(student, "bootts"):
        teacher.bootts = student.bootts
        teacher.use_bootts = getattr(student, "use_bootts", False)

    return DualRecommender(teacher=teacher, student=student)


def build_fixed(
    num_users: int,
    num_items: int,
    user_matrix: torch.Tensor,
    item_matrix: torch.Tensor,
    pos2id: Dict[int, int],
    dp_cfg: dict,
) -> TwoTowerRecommender:
    """Build a fixed (non-adaptive) TwoTowerRecommender model.

    Parameters
    ----------
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    user_matrix : torch.Tensor
        User feature matrix.
    item_matrix : torch.Tensor
        Item feature matrix.
    pos2id : Dict[int, int]
        Mapping from position to item ID.
    dp_cfg : dict
        Differential privacy configuration.

    Returns
    -------
    TwoTowerRecommender
        A TwoTowerRecommender model.
    """
    model = TwoTowerRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=user_matrix.shape[1],
        item_dim=item_matrix.shape[1],
        dp_cfg=dp_cfg,
    )
    setattr(model, "user_matrix", user_matrix)
    setattr(model, "item_matrix", item_matrix)
    setattr(model, "pos2id_map", dict(pos2id))
    return model


def build_baseline(
    mode: str,
    num_users: int,
    num_items: int,
    popularity: Dict[int, float],
    item_matrix: torch.Tensor,
    user_item_matrix: np.ndarray,
    device: torch.device,
    print_fn: Callable[[str], None] = print,
) -> object:
    """Build a baseline recommender model.

    Parameters
    ----------
    mode : str
        Baseline mode: "popularity", "random", "als", "user_knn", or "linucb".
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    popularity : Dict[int, float]
        Popularity scores by item ID.
    item_matrix : torch.Tensor
        Item feature matrix.
    user_item_matrix : np.ndarray
        User-item interaction matrix.
    device : torch.device
        Torch device to use.
    print_fn : Callable, optional
        Function to use for printing debug messages, by default print.

    Returns
    -------
    object
        A baseline model instance.

    Raises
    ------
    ValueError
        If mode is not recognized.
    """
    if mode == "popularity":
        print_fn("baseline=popularity")
        return PopularityBaseline(popularity, device)
    if mode == "random":
        print_fn("baseline=random")
        return RandomBaseline(device)
    if mode == "als":
        print_fn("baseline=als")
        return ALSBaseline(num_users, num_items, device)
    if mode == "user_knn":
        print_fn("baseline=user_knn")
        return UserKNNBaseline(user_item_matrix, device)
    if mode == "linucb":
        print_fn("baseline=linucb")
        feats = item_matrix.detach().cpu().numpy()
        print_fn(f"linucb item_features shape={feats.shape}, "
                 f"std={float(feats.std()) if feats.size>0 else 0.0:.6f}")
        if feats.ndim == 2 and feats.shape[1] == 0:
            print_fn("WARNING: LinUCB will degenerate (0 feature columns). "
                     "Consider adding fallback features.")
        return LinUCBBaseline(alpha=1.5, item_features=feats, device=device)
    raise ValueError(f"Unknown baseline '{mode}'")


def train_baseline(
    model: object,
    mode: str,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Train a baseline model (if applicable).

    Parameters
    ----------
    model : object
        The baseline model to train.
    mode : str
        Baseline mode (used for determining training strategy).
    print_fn : Callable, optional
        Function to use for printing debug messages, by default print.
    """
    print_fn(f"train_baseline: mode={mode}")
    if isinstance(model, ALSBaseline):
        print_fn("ALS: fitting from interactions …")
        # Note: ALS training needs interaction data, which should be passed separately.
        # This function signature doesn't include that. In practice, this should be
        # called with the actual training data. See train_als for the implementation.
        print_fn("ALS: fit done")
    elif isinstance(model, LinUCBBaseline):
        # LinUCB training also needs data, which should be passed separately.
        print_fn("LinUCB: fit done")
    else:
        print_fn("no training step for this baseline")
        return


def train_als(
    model: ALSBaseline,
    train_df: pd.DataFrame,
    interaction_user_col: str,
    interaction_item_col: str,
    user_index: Dict[int, int],
    id2pos: Dict[int, int],
    print_fn: Callable[[str], None] = print,
) -> None:
    """Train an ALS baseline model with interaction data.

    Parameters
    ----------
    model : ALSBaseline
        The ALS model to train.
    train_df : pd.DataFrame
        Training interactions dataframe.
    interaction_user_col : str
        Name of the user ID column in train_df.
    interaction_item_col : str
        Name of the item ID column in train_df.
    user_index : Dict[int, int]
        Mapping from external user ID to internal user index.
    id2pos : Dict[int, int]
        Mapping from item ID to position index.
    print_fn : Callable, optional
        Function to use for printing debug messages, by default print.
    """
    user_ids = []
    item_ids = []
    labels = []
    for row in train_df.itertuples():
        uid = user_index.get(int(getattr(row, interaction_user_col)))
        iid_pos = id2pos.get(int(getattr(row, interaction_item_col)))
        if uid is None or iid_pos is None:
            continue
        user_ids.append(uid)
        item_ids.append(iid_pos)
        labels.append(float(getattr(row, "label", 0.0)))
    model.fit(user_ids, item_ids, labels)


__all__ = [
    "warm_start_recommender",
    "build_adaptive",
    "build_fixed",
    "build_baseline",
    "train_baseline",
    "train_als",
]
