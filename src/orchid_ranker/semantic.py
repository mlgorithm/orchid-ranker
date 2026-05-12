"""Semantic exercise/item encoding for adaptive-learning cold start.

The encoder is deliberately lightweight: it uses scikit-learn's hashing vectorizer
to create deterministic normalized text embeddings from item text and metadata.
That gives Orchid a first semantic retrieval path without requiring an external
embedding service, model download, or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

__all__ = [
    "SemanticItemEncoder",
    "SemanticRecommendation",
    "SemanticExerciseRanker",
]


@dataclass(frozen=True)
class SemanticRecommendation:
    """Semantic retrieval result for one item."""

    item_id: Any
    score: float


class SemanticItemEncoder:
    """Hashing-vectorizer item encoder for semantic candidate generation.

    Parameters
    ----------
    n_features:
        Hashing dimension. Larger values reduce collisions; smaller values are
        useful in sketch mode.
    ngram_range:
        Word n-grams included in item representations.
    alternate_sign:
        Passed to ``HashingVectorizer``. ``False`` gives non-negative feature
        counts, which is easier to reason about for retrieval diagnostics.
    """

    def __init__(
        self,
        *,
        n_features: int = 2**14,
        ngram_range: tuple[int, int] = (1, 2),
        alternate_sign: bool = False,
        lowercase: bool = True,
    ) -> None:
        if n_features < 2:
            raise ValueError("n_features must be >= 2")
        self.n_features = int(n_features)
        self.ngram_range = ngram_range
        self.alternate_sign = bool(alternate_sign)
        self.lowercase = bool(lowercase)
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=self.alternate_sign,
            norm=None,
            lowercase=self.lowercase,
            ngram_range=self.ngram_range,
        )
        self.item_ids_: list[Any] = []
        self.item_text_: dict[Any, str] = {}
        self.embeddings_: Any = None
        self._item_index: dict[Any, int] = {}

    @property
    def is_fitted(self) -> bool:
        return self.embeddings_ is not None

    def fit(
        self,
        catalog: pd.DataFrame,
        *,
        item_col: str = "item_id",
        text_col: str = "item_text",
        metadata_cols: Optional[Sequence[str]] = None,
    ) -> "SemanticItemEncoder":
        """Fit item embeddings from item text and optional metadata columns."""
        if item_col not in catalog.columns:
            raise ValueError(f"catalog missing item_col={item_col!r}")
        metadata = list(metadata_cols or [])
        text_available = text_col in catalog.columns
        missing_meta = [col for col in metadata if col not in catalog.columns]
        if missing_meta:
            raise ValueError(f"catalog missing metadata columns: {missing_meta}")
        if not text_available and not metadata:
            raise ValueError("catalog must include text_col or at least one metadata column")

        work = catalog.drop_duplicates(subset=[item_col], keep="last").reset_index(drop=True)
        self.item_ids_ = work[item_col].tolist()
        texts = [
            self._row_text(row, text_col=text_col if text_available else None, metadata_cols=metadata)
            for _, row in work.iterrows()
        ]
        if not any(text.strip() for text in texts):
            raise ValueError("semantic item text is empty after preprocessing")
        self.item_text_ = {item_id: text for item_id, text in zip(self.item_ids_, texts)}
        self._item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids_)}
        self.embeddings_ = normalize(self.vectorizer.transform(texts), norm="l2", copy=False)
        return self

    def encode_texts(self, texts: Sequence[str]) -> Any:
        """Encode arbitrary query texts into normalized sparse vectors."""
        if not texts:
            raise ValueError("texts must be non-empty")
        vectors = self.vectorizer.transform([str(text) for text in texts])
        return normalize(vectors, norm="l2", copy=False)

    def scores(
        self,
        query_text: str,
        *,
        candidate_item_ids: Optional[Sequence[Any]] = None,
    ) -> dict[Any, float]:
        """Return cosine scores between query text and candidate items."""
        self._require_fitted()
        assert self.embeddings_ is not None
        query = self.encode_texts([query_text])
        if candidate_item_ids is None:
            raw_scores = (self.embeddings_ @ query.T).toarray().ravel()
            return {item_id: float(raw_scores[idx]) for idx, item_id in enumerate(self.item_ids_)}

        result: dict[Any, float] = {}
        for item_id in candidate_item_ids:
            idx = self._item_index.get(item_id)
            if idx is None:
                continue
            result[item_id] = float((self.embeddings_[idx] @ query.T).toarray().ravel()[0])
        return result

    def similar_items(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_item_ids: Optional[Sequence[Any]] = None,
    ) -> list[Any]:
        """Return item IDs ranked by semantic similarity to query text."""
        if top_k <= 0:
            return []
        scores = self.scores(query_text, candidate_item_ids=candidate_item_ids)
        ranked = sorted(scores, key=lambda item_id: (scores[item_id], str(item_id)), reverse=True)
        return ranked[: min(int(top_k), len(ranked))]

    def similar_to_items(
        self,
        item_ids: Sequence[Any],
        *,
        top_k: int = 10,
        candidate_item_ids: Optional[Sequence[Any]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> list[Any]:
        """Return items close to a weighted profile of known item IDs."""
        self._require_fitted()
        assert self.embeddings_ is not None
        if not item_ids:
            return []
        indices = [self._item_index[item_id] for item_id in item_ids if item_id in self._item_index]
        if not indices:
            return []
        if weights is None:
            raw_weights = np.ones((len(indices),), dtype=float)
        else:
            raw_weights = np.asarray(list(weights), dtype=float)[: len(indices)]
            if raw_weights.size != len(indices):
                raise ValueError("weights must match the number of known item IDs")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weights must be finite")
        profile = np.asarray(raw_weights @ self.embeddings_[indices].toarray(), dtype=float).reshape(1, -1)
        profile = normalize(profile, norm="l2", copy=False)
        candidates = self.item_ids_ if candidate_item_ids is None else list(candidate_item_ids)
        scores: dict[Any, float] = {}
        for item_id in candidates:
            idx = self._item_index.get(item_id)
            if idx is None or item_id in item_ids:
                continue
            scores[item_id] = float(np.asarray(self.embeddings_[idx] @ profile.T).ravel()[0])
        ranked = sorted(scores, key=lambda item_id: (scores[item_id], str(item_id)), reverse=True)
        return ranked[: min(int(top_k), len(ranked))]

    def diagnostics(self) -> dict[str, Any]:
        """Return encoder metadata for model cards and serving logs."""
        self._require_fitted()
        return {
            "n_items": len(self.item_ids_),
            "n_features": self.n_features,
            "ngram_range": self.ngram_range,
            "alternate_sign": self.alternate_sign,
        }

    def _row_text(self, row: pd.Series, *, text_col: Optional[str], metadata_cols: Sequence[str]) -> str:
        parts: list[str] = []
        if text_col is not None:
            value = row.get(text_col)
            if value is not None and not pd.isna(value):
                parts.append(str(value))
        for col in metadata_cols:
            value = row.get(col)
            if value is None or pd.isna(value):
                continue
            if isinstance(value, Mapping):
                parts.extend(f"{key}:{item}" for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
            else:
                parts.append(f"{col}:{value}")
        return " ".join(parts)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("SemanticItemEncoder must be fitted before use")


class SemanticExerciseRanker:
    """Small wrapper that returns scored semantic item recommendations."""

    def __init__(self, encoder: SemanticItemEncoder) -> None:
        if not encoder.is_fitted:
            raise RuntimeError("encoder must be fitted before constructing SemanticExerciseRanker")
        self.encoder = encoder

    def rank(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_item_ids: Optional[Sequence[Any]] = None,
    ) -> list[SemanticRecommendation]:
        scores = self.encoder.scores(query_text, candidate_item_ids=candidate_item_ids)
        ranked = sorted(scores, key=lambda item_id: (scores[item_id], str(item_id)), reverse=True)
        return [
            SemanticRecommendation(item_id=item_id, score=float(scores[item_id]))
            for item_id in ranked[: min(max(0, int(top_k)), len(ranked))]
        ]
