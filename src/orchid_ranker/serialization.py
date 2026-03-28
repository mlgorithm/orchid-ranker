"""Serialization and deserialization utilities for Orchid Ranker models.

This module provides functions to save and load OrchidRecommender and TwoTowerRecommender
models to/from disk, preserving their internal state and enabling reproducible inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .recommender import OrchidRecommender
    from .agents.recommender_agent import TwoTowerRecommender


__all__ = ["save_model", "load_model"]

_CHECKPOINT_VERSION = "1.0"
_logger = logging.getLogger(__name__)


def save_model(model: Any, path: str | Path) -> None:
    """Save an OrchidRecommender or TwoTowerRecommender to disk.

    Saves the model's internal state including strategy configuration, user/item mappings,
    fitted model state, and item features. Uses torch.save() for serialization with pickle.

    Parameters
    ----------
    model : OrchidRecommender or TwoTowerRecommender
        The fitted recommender model to save.
    path : str or Path
        Destination file path for the checkpoint.

    Raises
    ------
    ValueError
        If model type is not supported or if model has not been fitted.
    RuntimeError
        If checkpoint writing fails.

    Examples
    --------
    >>> rec = OrchidRecommender(strategy="als")
    >>> rec.fit(interactions_df)
    >>> save_model(rec, "model.pt")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine model type and extract state
    model_type = type(model).__name__

    if model_type == "OrchidRecommender":
        state = _extract_orchid_state(model)
    elif model_type == "TwoTowerRecommender":
        state = _extract_two_tower_state(model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Expected OrchidRecommender or TwoTowerRecommender.")

    checkpoint = {
        "version": _CHECKPOINT_VERSION,
        "model_type": model_type,
        "state": state,
    }

    try:
        torch.save(checkpoint, path)
        _logger.info(f"Model saved to {path} (type={model_type}, version={_CHECKPOINT_VERSION})")
    except Exception as exc:
        raise RuntimeError(f"Failed to save model to {path}: {exc}") from exc


def load_model(path: str | Path) -> Any:
    """Load a previously saved OrchidRecommender or TwoTowerRecommender from disk.

    Restores the model's internal state including user/item mappings, fitted parameters,
    and model configuration.

    Parameters
    ----------
    path : str or Path
        Path to the saved checkpoint file.

    Returns
    -------
    OrchidRecommender or TwoTowerRecommender
        The restored model in fitted state.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    ValueError
        If checkpoint is corrupted, truncated, or has unknown model type.
    RuntimeError
        If checkpoint loading or restoration fails.

    Examples
    --------
    >>> rec = load_model("model.pt")
    >>> predictions = rec.predict(user_id=1, item_id=5)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    version = checkpoint.get("version")
    model_type = checkpoint.get("model_type")
    state = checkpoint.get("state")

    # Check all required keys exist
    required_keys = {"version", "model_type", "state"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Checkpoint is corrupted/truncated: missing keys {missing_keys}"
        )

    if not all([version, model_type, state]):
        raise ValueError("Invalid checkpoint format: version, model_type, or state is empty.")

    # Validate model_type is known
    supported_types = {"OrchidRecommender", "TwoTowerRecommender"}
    if model_type not in supported_types:
        raise ValueError(
            f"Unknown model type in checkpoint: {model_type}. "
            f"Supported types: {supported_types}"
        )

    if version != _CHECKPOINT_VERSION:
        _logger.warning(
            f"Checkpoint version mismatch: file is v{version} but loader expects v{_CHECKPOINT_VERSION}. "
            "Loading may succeed but behavior is not guaranteed."
        )

    if model_type == "OrchidRecommender":
        model = _restore_orchid_model(state)
    elif model_type == "TwoTowerRecommender":
        model = _restore_two_tower_model(state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    _logger.info(f"Model loaded from {path} (type={model_type})")
    return model


def _extract_orchid_state(model: OrchidRecommender) -> Dict[str, Any]:
    """Extract internal state from a fitted OrchidRecommender.

    Parameters
    ----------
    model : OrchidRecommender
        The fitted recommender model.

    Returns
    -------
    dict
        State dictionary containing strategy, mappings, and fitted model data.

    Raises
    ------
    RuntimeError
        If model has not been fitted yet.
    """
    if model._baseline is None:
        raise RuntimeError("Cannot save an unfitted OrchidRecommender. Call fit() first.")

    state: Dict[str, Any] = {
        "strategy": model.strategy,
        "strategy_kwargs": model.strategy_kwargs,
        "device": str(model.device),
        "user_map": model._user2idx,
        "item_map": model._item2idx,
        "seen_items": model._seen_items,
    }

    # Save baseline model state
    baseline = model._baseline
    baseline_name = type(baseline).__name__

    # For neural models (NeuralMatrixFactorizationBaseline, TwoTowerRecommender, etc.)
    if hasattr(baseline, "state_dict"):
        state["baseline_type"] = baseline_name
        state["baseline_state_dict"] = baseline.state_dict()
        _logger.debug(f"Saved neural baseline state: {baseline_name}")
    else:
        # For non-neural models (popularity, random, etc.), pickle the entire baseline
        state["baseline_type"] = baseline_name
        state["baseline_object"] = baseline
        _logger.debug(f"Saved non-neural baseline object: {baseline_name}")

    # Save item features if present (for linucb)
    if model._item_features is not None:
        state["item_features"] = model._item_features

    return state


def _restore_orchid_model(state: Dict[str, Any]) -> OrchidRecommender:
    """Restore an OrchidRecommender from saved state.

    Parameters
    ----------
    state : dict
        State dictionary from _extract_orchid_state.

    Returns
    -------
    OrchidRecommender
        Restored model in fitted state.
    """
    from .recommender import OrchidRecommender

    strategy = state["strategy"]
    strategy_kwargs = state.get("strategy_kwargs", {})
    device = state.get("device", "cpu")

    model = OrchidRecommender(strategy=strategy, device=device, **strategy_kwargs)

    # Restore mappings and seen items
    model._user2idx = state["user_map"]
    model._idx2user = {idx: uid for uid, idx in model._user2idx.items()}
    model._item2idx = state["item_map"]
    model._idx2item = {idx: iid for iid, idx in model._item2idx.items()}
    model._seen_items = state.get("seen_items", {})

    # Restore item features if present
    if "item_features" in state:
        model._item_features = state["item_features"]

    # Restore baseline model
    baseline_type = state.get("baseline_type")
    if "baseline_state_dict" in state:
        # Neural model with state_dict
        model._baseline = _restore_neural_baseline(
            baseline_type,
            state["baseline_state_dict"],
            strategy,
            len(model._user2idx),
            len(model._item2idx),
            device,
            strategy_kwargs,
        )
    elif "baseline_object" in state:
        # Non-neural model stored as pickled object
        model._baseline = state["baseline_object"]
        # Move to correct device if it has device awareness
        if hasattr(model._baseline, "device"):
            model._baseline.device = torch.device(device)
    else:
        raise RuntimeError(f"Invalid baseline state in checkpoint")

    return model


def _restore_neural_baseline(
    baseline_type: str,
    state_dict: Dict[str, torch.Tensor],
    strategy: str,
    num_users: int,
    num_items: int,
    device: str,
    strategy_kwargs: Dict[str, Any],
) -> Any:
    """Restore a neural baseline model from its state_dict.

    Parameters
    ----------
    baseline_type : str
        Name of the baseline class.
    state_dict : dict
        PyTorch state_dict.
    strategy : str
        Strategy name.
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    device : str
        Torch device string.
    strategy_kwargs : dict
        Additional strategy kwargs.

    Returns
    -------
    Baseline instance
        Restored baseline model.
    """
    from .baselines import NeuralMatrixFactorizationBaseline

    if baseline_type == "NeuralMatrixFactorizationBaseline":
        baseline = NeuralMatrixFactorizationBaseline(
            num_users=num_users,
            num_items=num_items,
            device=device,
            **strategy_kwargs,
        )
        baseline.load_state_dict(state_dict)
        _logger.debug(f"Restored {baseline_type} with {len(state_dict)} parameters")
        return baseline
    else:
        raise RuntimeError(f"Unknown neural baseline type: {baseline_type}")


def _extract_two_tower_state(model: TwoTowerRecommender) -> Dict[str, Any]:
    """Extract internal state from a fitted TwoTowerRecommender.

    Parameters
    ----------
    model : TwoTowerRecommender
        The fitted two-tower model.

    Returns
    -------
    dict
        State dictionary containing model architecture and weights.
    """
    state: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_users": model.num_users if hasattr(model, "num_users") else None,
            "num_items": model.num_items if hasattr(model, "num_items") else None,
            "user_dim": model.user_dim if hasattr(model, "user_dim") else None,
            "item_dim": model.item_dim if hasattr(model, "item_dim") else None,
            "hidden": model.hidden if hasattr(model, "hidden") else None,
            "emb_dim": model.emb_dim if hasattr(model, "emb_dim") else None,
            "state_dim": model.state_dim if hasattr(model, "state_dim") else None,
        },
        "device": str(model.device) if hasattr(model, "device") else "cpu",
    }

    _logger.debug("Extracted TwoTowerRecommender state")
    return state


def _restore_two_tower_model(state: Dict[str, Any]) -> TwoTowerRecommender:
    """Restore a TwoTowerRecommender from saved state.

    Parameters
    ----------
    state : dict
        State dictionary from _extract_two_tower_state.

    Returns
    -------
    TwoTowerRecommender
        Restored model.
    """
    from .agents.recommender_agent import TwoTowerRecommender

    config = state.get("config", {})
    device = state.get("device", "cpu")

    # Create model with saved config
    model = TwoTowerRecommender(
        num_users=config.get("num_users", 1),
        num_items=config.get("num_items", 1),
        user_dim=config.get("user_dim", 32),
        item_dim=config.get("item_dim", 32),
        hidden=config.get("hidden", 64),
        emb_dim=config.get("emb_dim", 32),
        state_dim=config.get("state_dim", 4),
        device=device,
    )

    # Restore weights
    model.load_state_dict(state["model_state_dict"])
    _logger.debug("Restored TwoTowerRecommender with saved weights")
    return model
