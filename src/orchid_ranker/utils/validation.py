"""Input validation helpers targeted at enterprise deployments."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when user-provided data fails structural validation."""


REQUIRED_INTERACTION_COLUMNS = {"user_id", "item_id"}


def validate_interactions_frame(
    interactions: pd.DataFrame,
    *,
    required_columns: Optional[Iterable[str]] = None,
) -> None:
    """Ensure the interactions dataframe meets minimum structural expectations."""

    if interactions.empty:
        raise ValidationError("Interactions dataframe is empty.")

    cols = set(interactions.columns)
    missing = (required_columns or REQUIRED_INTERACTION_COLUMNS) - cols
    if missing:
        raise ValidationError(f"Missing required columns: {sorted(missing)}")

    for column in REQUIRED_INTERACTION_COLUMNS:
        if interactions[column].isnull().any():
            raise ValidationError(f"Column '{column}' contains null values; please clean inputs.")

    user_type = interactions["user_id"].dtype
    item_type = interactions["item_id"].dtype
    if not np.issubdtype(user_type, np.integer):
        raise ValidationError("'user_id' column must be integer typed.")
    if not np.issubdtype(item_type, np.integer):
        raise ValidationError("'item_id' column must be integer typed.")


def validate_item_features(item_features: np.ndarray, expected_items: int) -> None:
    """Validate dimensionality of item feature matrix."""

    if item_features.ndim != 2:
        raise ValidationError("item_features must be a 2D array.")
    if item_features.shape[0] != expected_items:
        raise ValidationError(
            f"item_features rows ({item_features.shape[0]}) do not match number of items ({expected_items})."
        )


__all__ = ["ValidationError", "validate_interactions_frame", "validate_item_features"]
