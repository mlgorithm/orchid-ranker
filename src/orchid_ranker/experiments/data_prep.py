"""Data preparation and feature matrix building utilities for Orchid Ranker experiments."""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _resolve_id_column(df: pd.DataFrame, preferred: str, fallbacks: tuple[str, ...] = ()) -> str:
    """Resolve the ID column name in a dataframe with fallback options.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to search.
    preferred : str
        The preferred column name.
    fallbacks : tuple[str, ...]
        Fallback column names to try in order.

    Returns
    -------
    str
        The resolved column name, or the first column if none match.
    """
    if preferred in df.columns:
        return preferred
    for col in fallbacks + ("u", "i", "user_id", "item_id", "id"):
        if col in df.columns:
            return col
    return df.columns[0]


def _build_feature_matrix(
    df: pd.DataFrame,
    id_col: str,
    device: torch.device,
    *,
    kind: str = "items",
    verbose: bool = True,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Build a feature matrix from a dataframe.

    Robust builder that:
    - Resolves the ID column.
    - Keeps ONLY numeric feature columns (avoids object->NaN coercion).
    - Cleans inf/-inf -> NaN -> 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to build from.
    id_col : str
        The name of the ID column.
    device : torch.device
        The torch device to place the tensor on.
    kind : str, optional
        Description label for logging ("items", "users", etc.), by default "items".
    verbose : bool, optional
        Whether to print debug info, by default True.

    Returns
    -------
    Tuple[np.ndarray, torch.Tensor]
        A tuple of (ids, feature_matrix) where ids is a numpy array of IDs
        and feature_matrix is a torch tensor of shape (n_samples, n_features).
    """
    if df.empty:
        return np.arange(0, dtype=int), torch.zeros((0, 0), dtype=torch.float32, device=device)

    resolved = _resolve_id_column(df, id_col)
    ids = df[resolved].to_numpy()

    rem = df.drop(columns=[resolved], errors="ignore")
    num = rem.select_dtypes(include=[np.number]).copy()
    dropped = [c for c in rem.columns if c not in num.columns]

    if not num.empty:
        num.replace([np.inf, -np.inf], np.nan, inplace=True)
        num.fillna(0.0, inplace=True)

    feats_np = num.to_numpy(dtype=np.float32) if not num.empty else np.zeros((len(df), 0), dtype=np.float32)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)

    if verbose:
        kept = list(num.columns)
        msg = f"[{kind}] id_col='{resolved}', kept_num={len(kept)}, dropped_non_num={len(dropped)}"
        if dropped:
            msg += f", dropped={dropped[:8]}{'…' if len(dropped) > 8 else ''}"
        logger.debug("%s", msg)

    return ids, feats


def _prepare_item_meta(df: pd.DataFrame) -> Dict[int, dict]:
    """Prepare item metadata dictionary from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to extract metadata from.

    Returns
    -------
    Dict[int, dict]
        A dictionary mapping item IDs to their metadata dictionaries.
    """
    meta = {}
    if df.empty:
        return meta
    id_col = _resolve_id_column(df, "i", ("item_id",))
    for row in df.to_dict(orient="records"):
        iid = int(row[id_col])
        meta[iid] = {k: v for k, v in row.items() if k != id_col}
    return meta


def _flatten_round_records(records: List[dict]) -> pd.DataFrame:
    """Flatten round summary records into a dataframe.

    Parameters
    ----------
    records : List[dict]
        A list of record dictionaries from experiment logging.

    Returns
    -------
    pd.DataFrame
        A dataframe with flattened round-level metrics.
    """
    rows = []
    for rec in records:
        if rec.get("type") != "round_summary":
            continue
        metric = rec.get("metrics", {})
        rows.append({"round": rec.get("round"), "mode": rec.get("mode"), **metric})
    return pd.DataFrame(rows)


def _flatten_user_records(records: List[dict]) -> pd.DataFrame:
    """Flatten user-round records into a dataframe.

    Parameters
    ----------
    records : List[dict]
        A list of record dictionaries from experiment logging.

    Returns
    -------
    pd.DataFrame
        A dataframe with flattened user-round records including telemetry, knobs, and state estimator data.
    """
    rows = []
    for rec in records:
        if rec.get("type") != "user_round":
            continue
        base = {
            "round": rec.get("round"),
            "mode": rec.get("mode"),
            "user_id": rec.get("user_id"),
            "student_method": rec.get("student_method"),
            "profile": rec.get("profile"),
        }
        tel = rec.get("telemetry", {}) or {}
        pre = ((rec.get("state_estimator") or {}).get("pre")) or {}
        post = ((rec.get("state_estimator") or {}).get("post")) or {}
        knobs = rec.get("knobs", {}) or {}

        # telemetry
        for k, v in tel.items():
            base[f"tel_{k}"] = v
        # knobs
        for k, v in knobs.items():
            base[f"knob_{k}"] = v
        # pre / post
        for k, v in pre.items():
            base[f"pre_{k}"] = v
        for k, v in post.items():
            base[f"post_{k}"] = v

        rows.append(base)
    return pd.DataFrame(rows)


__all__ = [
    "_resolve_id_column",
    "_build_feature_matrix",
    "_prepare_item_meta",
    "_flatten_round_records",
    "_flatten_user_records",
]
