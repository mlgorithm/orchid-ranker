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
    check_cols = required_columns or REQUIRED_INTERACTION_COLUMNS
    missing = set(check_cols) - cols
    if missing:
        raise ValidationError(f"Missing required columns: {sorted(missing)}")

    for column in check_cols:
        if interactions[column].isnull().any():
            raise ValidationError(f"Column '{column}' contains null values; please clean inputs.")

    # Validate types for user and item columns if they exist
    cols_to_check = set(check_cols) & REQUIRED_INTERACTION_COLUMNS
    for column in cols_to_check:
        col_type = interactions[column].dtype
        if not np.issubdtype(col_type, np.integer):
            raise ValidationError(f"'{column}' column must be integer typed.")


def validate_item_features(item_features: np.ndarray, expected_items: int) -> None:
    """Validate dimensionality of item feature matrix."""

    if item_features.ndim != 2:
        raise ValidationError("item_features must be a 2D array.")
    if item_features.shape[0] != expected_items:
        raise ValidationError(
            f"item_features rows ({item_features.shape[0]}) do not match number of items ({expected_items})."
        )


__all__ = ["ValidationError", "validate_interactions_frame", "validate_item_features"]
