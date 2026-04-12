"""Serialization and deserialization utilities for Orchid Ranker models.

This module provides functions to save and load OrchidRecommender and TwoTowerRecommender
models to/from disk, preserving their internal state and enabling reproducible inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import numpy as np
import torch

# Register numpy types as safe for torch.load(weights_only=True).
# Our checkpoints contain numpy arrays/dtypes; this allows safe loading
# without falling back to full pickle.
_numpy_reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
_NUMPY_SAFE_GLOBALS = [
    np.ndarray,
    np.dtype,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
    np.dtypes.Int32DType,
    np.dtypes.Int64DType,
    np.dtypes.UInt8DType,
    np.dtypes.BoolDType,
]
if _numpy_reconstruct is not None:
    _NUMPY_SAFE_GLOBALS.insert(0, _numpy_reconstruct)
try:
    torch.serialization.add_safe_globals(cast(list[Any], _NUMPY_SAFE_GLOBALS))
except AttributeError:
    pass  # older torch versions without add_safe_globals

if TYPE_CHECKING:
    from .agents.recommender_agent import TwoTowerRecommender
    from .recommender import OrchidRecommender


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

    # Atomic write: save to temp file then os.replace to avoid partial/corrupt checkpoints
    import tempfile
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".orchid_save_"
        )
        os.close(fd)
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, str(path))
        _logger.info(f"Model saved to {path} (type={model_type}, version={_CHECKPOINT_VERSION})")
    except Exception as exc:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
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
        # Try safe loading first (weights_only=True rejects pickle payloads).
        try:
            checkpoint = torch.load(path, weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Safe loading failed for {path}. Unsafe pickle-based "
                "checkpoints are not supported. Re-save the model from a "
                "trusted environment using save_model()."
            ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    version = checkpoint.get("version")
    model_type_obj = checkpoint.get("model_type")
    state_obj = checkpoint.get("state")

    # Check all required keys exist
    required_keys = {"version", "model_type", "state"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Checkpoint is corrupted/truncated: missing keys {missing_keys}"
        )

    if not version or not model_type_obj or not state_obj:
        raise ValueError("Invalid checkpoint format: version, model_type, or state is empty.")
    if not isinstance(model_type_obj, str):
        raise ValueError(f"Invalid model_type in checkpoint: expected str, got {type(model_type_obj).__name__}")
    if not isinstance(state_obj, dict):
        raise ValueError(f"Invalid state in checkpoint: expected dict, got {type(state_obj).__name__}")
    model_type = model_type_obj
    state = cast(Dict[str, Any], state_obj)

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


def _extract_neural_mf_state(baseline) -> Dict[str, Any]:
    """Extract safe state from a NeuralMatrixFactorizationBaseline."""
    return {
        "mlp_state_dict": baseline.mlp.state_dict(),
        "user_emb_state_dict": baseline.user_emb.state_dict(),
        "item_emb_state_dict": baseline.item_emb.state_dict(),
        "num_users": baseline.num_users,
        "num_items": baseline.num_items,
        "emb_dim": baseline.emb_dim,
        "hidden": baseline.hidden,
        "loss_type": baseline.loss_type,
        "neg_k": baseline.neg_k,
        "batch_size": baseline.batch_size,
    }


def _extract_orchid_state(model: OrchidRecommender) -> Dict[str, Any]:
    """Extract internal state from a fitted OrchidRecommender.

    Private function that captures all necessary state for serialization,
    including user/item mappings, strategy configuration, and fitted baseline model.

    Parameters
    ----------
    model : OrchidRecommender
        The fitted recommender model.

    Returns
    -------
    dict
        State dictionary containing:
        - strategy: strategy name
        - strategy_kwargs: kwargs passed to baseline
        - device: torch device string
        - user_map, item_map: ID mappings
        - seen_items: items seen by each user
        - baseline_type: type of baseline model
        - baseline_state_dict or baseline_object: model weights/params
        - item_features: optional feature matrix for linucb

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
    state["baseline_type"] = baseline_name

    # Extract state as safe, pickle-free dicts/tensors depending on baseline type
    if baseline_name == "PopularityBaseline":
        state["baseline_data"] = {"popularity": baseline.popularity}
    elif baseline_name == "RandomBaseline":
        state["baseline_data"] = {}
    elif baseline_name == "ALSBaseline":
        state["baseline_data"] = {"model_state_dict": baseline.model.state_dict()}
    elif baseline_name == "ExplicitMFBaseline":
        state["baseline_data"] = {
            "model_state_dict": baseline.model.state_dict(),
            "global_mean": baseline._global_mean,
            "min_rating": baseline._min_rating,
            "max_rating": baseline._max_rating,
        }
    elif baseline_name == "UserKNNBaseline":
        state["baseline_data"] = {
            "matrix": baseline.matrix,
            "k": baseline.k,
        }
    elif baseline_name == "LinUCBBaseline":
        state["baseline_data"] = {
            "alpha": baseline.alpha,
            "A": baseline.A.cpu(),
            "b": baseline.b.cpu(),
        }
    elif baseline_name == "NeuralMatrixFactorizationBaseline":
        state["baseline_data"] = _extract_neural_mf_state(baseline)
    elif baseline_name in ("ImplicitALSBaseline", "ImplicitBPRBaseline"):
        state["baseline_data"] = {
            "user_factors": baseline.user_factors,
            "item_factors": baseline.item_factors,
        }
    else:
        raise ValueError(f"Cannot serialize unknown baseline type: {baseline_name}")

    _logger.debug(f"Saved baseline state: {baseline_name}")

    # Save item features if present (for linucb)
    if model._item_features is not None:
        state["item_features"] = model._item_features

    return state


def _restore_orchid_model(state: Dict[str, Any]) -> OrchidRecommender:
    """Restore an OrchidRecommender from saved state.

    Private function that reconstructs an OrchidRecommender from a saved state dictionary,
    restoring all internal mappings and fitted parameters.

    Parameters
    ----------
    state : dict
        State dictionary from _extract_orchid_state.

    Returns
    -------
    OrchidRecommender
        Restored model in fitted state, ready for inference.

    Raises
    ------
    RuntimeError
        If baseline state is invalid.
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
    baseline_type_obj = state.get("baseline_type")
    baseline_data_obj = state.get("baseline_data")
    baseline_type = baseline_type_obj if isinstance(baseline_type_obj, str) else None
    baseline_data = baseline_data_obj if isinstance(baseline_data_obj, dict) else None

    if baseline_data is not None:
        if baseline_type is None:
            raise RuntimeError("Invalid baseline type in checkpoint")
        model._baseline = _restore_baseline_from_data(
            baseline_type,
            baseline_data,
            len(model._user2idx),
            len(model._item2idx),
            device,
            strategy_kwargs,
            item_features=model._item_features,
        )
    elif "baseline_state_dict" in state:
        # Legacy: neural model with state_dict (pre-0.3.0 checkpoints)
        if baseline_type is None:
            raise RuntimeError("Invalid baseline type in legacy checkpoint")
        legacy_state_dict = state["baseline_state_dict"]
        if not isinstance(legacy_state_dict, dict):
            raise RuntimeError("Invalid legacy neural baseline state in checkpoint")
        model._baseline = _restore_neural_baseline(
            baseline_type,
            cast(Dict[str, Any], legacy_state_dict),
            strategy,
            len(model._user2idx),
            len(model._item2idx),
            device,
            strategy_kwargs,
        )
    elif "baseline_object" in state:
        # Legacy: pickled object (pre-0.3.0 checkpoints)
        _logger.warning(
            "Loading a pre-0.3.0 checkpoint that uses pickle deserialization. "
            "Re-save the model to upgrade to the safe format."
        )
        model._baseline = state["baseline_object"]
        if hasattr(model._baseline, "device"):
            model._baseline.device = torch.device(device)
    else:
        raise RuntimeError("Invalid baseline state in checkpoint")

    return model


def _restore_neural_baseline(
    baseline_type: str,
    state_dict: Dict[str, Any],
    strategy: str,
    num_users: int,
    num_items: int,
    device: str,
    strategy_kwargs: Dict[str, Any],
) -> Any:
    """Restore a neural baseline model from its state_dict.

    Private function that instantiates a neural baseline model and loads
    saved PyTorch weights.

    Parameters
    ----------
    baseline_type : str
        Name of the baseline class (e.g., "NeuralMatrixFactorizationBaseline").
    state_dict : dict
        PyTorch state_dict containing model weights.
    strategy : str
        Strategy name.
    num_users : int
        Number of users for model initialization.
    num_items : int
        Number of items for model initialization.
    device : str
        Torch device string (e.g., "cpu", "cuda").
    strategy_kwargs : dict
        Additional strategy kwargs for baseline instantiation.

    Returns
    -------
    Baseline instance
        Restored and initialized baseline model.

    Raises
    ------
    RuntimeError
        If baseline_type is not recognized.
    """
    from .baselines import NeuralMatrixFactorizationBaseline

    if baseline_type == "NeuralMatrixFactorizationBaseline":
        baseline = NeuralMatrixFactorizationBaseline(
            num_users=num_users,
            num_items=num_items,
            device=torch.device(device),
            **strategy_kwargs,
        )
        mlp_state = state_dict.get("mlp_state_dict")
        user_emb_state = state_dict.get("user_emb_state_dict")
        item_emb_state = state_dict.get("item_emb_state_dict")
        if not all(isinstance(blob, dict) for blob in (mlp_state, user_emb_state, item_emb_state)):
            raise RuntimeError("Legacy neural baseline checkpoint is missing component state dicts")
        baseline.mlp.load_state_dict(cast(Dict[str, Any], mlp_state))
        baseline.user_emb.load_state_dict(cast(Dict[str, Any], user_emb_state))
        baseline.item_emb.load_state_dict(cast(Dict[str, Any], item_emb_state))
        _logger.debug(f"Restored {baseline_type} with {len(state_dict)} parameters")
        return baseline
    else:
        raise RuntimeError(f"Unknown neural baseline type: {baseline_type}")


def _restore_baseline_from_data(
    baseline_type: str,
    data: Dict[str, Any],
    num_users: int,
    num_items: int,
    device: str,
    strategy_kwargs: Dict[str, Any],
    item_features: Optional[np.ndarray] = None,
) -> Any:
    """Restore a baseline from its safe state dict (no pickle).

    Parameters
    ----------
    baseline_type : str
        Name of the baseline class.
    data : dict
        Safe state data extracted during save.
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    device : str
        Torch device string.
    strategy_kwargs : dict
        Strategy kwargs for baseline instantiation.
    item_features : np.ndarray, optional
        Item features for LinUCB.

    Returns
    -------
    Baseline instance
        Restored baseline, ready for inference.
    """
    from .baselines import (
        ALSBaseline,
        ExplicitMFBaseline,
        ImplicitALSBaseline,
        ImplicitBPRBaseline,
        LinUCBBaseline,
        NeuralMatrixFactorizationBaseline,
        PopularityBaseline,
        RandomBaseline,
        UserKNNBaseline,
    )

    dev = torch.device(device)

    if baseline_type == "PopularityBaseline":
        return PopularityBaseline(data["popularity"], device=dev)

    if baseline_type == "RandomBaseline":
        return RandomBaseline(dev)

    if baseline_type == "ALSBaseline":
        als_baseline = ALSBaseline(num_users, num_items, device=dev, **strategy_kwargs)
        als_baseline.model.load_state_dict(cast(Dict[str, Any], data["model_state_dict"]))
        return als_baseline

    if baseline_type == "ExplicitMFBaseline":
        explicit_mf_baseline = ExplicitMFBaseline(num_users, num_items, device=dev, **strategy_kwargs)
        explicit_mf_baseline.model.load_state_dict(cast(Dict[str, Any], data["model_state_dict"]))
        explicit_mf_baseline._global_mean = data["global_mean"]
        explicit_mf_baseline._min_rating = data["min_rating"]
        explicit_mf_baseline._max_rating = data["max_rating"]
        return explicit_mf_baseline

    if baseline_type == "UserKNNBaseline":
        return UserKNNBaseline(
            data["matrix"],
            device=dev,
            k=data.get("k", 20),
        )

    if baseline_type == "LinUCBBaseline":
        if item_features is None:
            raise RuntimeError("item_features required to restore LinUCBBaseline")
        linucb_baseline = LinUCBBaseline(
            alpha=data["alpha"],
            item_features=item_features,
            device=dev,
        )
        linucb_baseline.A = data["A"].to(dev)
        linucb_baseline.b = data["b"].to(dev)
        return linucb_baseline

    if baseline_type == "NeuralMatrixFactorizationBaseline":
        hidden_dims = tuple(int(v) for v in data.get("hidden", (64, 32)))
        neural_mf_baseline = NeuralMatrixFactorizationBaseline(
            num_users=data.get("num_users", num_users),
            num_items=data.get("num_items", num_items),
            device=dev,
            emb_dim=data.get("emb_dim", 32),
            hidden=hidden_dims,
            loss=data.get("loss_type", "bce"),
            neg_k=data.get("neg_k", 10),
            batch_size=data.get("batch_size", 256),
        )
        neural_mf_baseline.mlp.load_state_dict(cast(Dict[str, Any], data["mlp_state_dict"]))
        neural_mf_baseline.user_emb.load_state_dict(cast(Dict[str, Any], data["user_emb_state_dict"]))
        neural_mf_baseline.item_emb.load_state_dict(cast(Dict[str, Any], data["item_emb_state_dict"]))
        return neural_mf_baseline

    if baseline_type in ("ImplicitALSBaseline", "ImplicitBPRBaseline"):
        cls = ImplicitALSBaseline if baseline_type == "ImplicitALSBaseline" else ImplicitBPRBaseline
        implicit_baseline = cls(**strategy_kwargs)
        implicit_baseline.user_factors = data["user_factors"]
        implicit_baseline.item_factors = data["item_factors"]
        return implicit_baseline

    raise RuntimeError(f"Unknown baseline type: {baseline_type}")


def _extract_two_tower_state(model: TwoTowerRecommender) -> Dict[str, Any]:
    """Extract internal state from a fitted TwoTowerRecommender.

    Private function that captures PyTorch model state and configuration
    for a two-tower neural recommendation model.

    Parameters
    ----------
    model : TwoTowerRecommender
        The fitted two-tower model.

    Returns
    -------
    dict
        State dictionary containing:
        - model_state_dict: PyTorch state_dict with all parameters
        - config: architecture configuration (num_users, num_items, dims, etc.)
        - device: torch device string
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

    Private function that reconstructs a two-tower model from saved state,
    instantiating the architecture and loading saved weights.

    Parameters
    ----------
    state : dict
        State dictionary from _extract_two_tower_state.

    Returns
    -------
    TwoTowerRecommender
        Restored model in fitted state, ready for inference.
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
