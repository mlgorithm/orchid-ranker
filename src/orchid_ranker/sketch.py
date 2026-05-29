"""Sketch-mode candidate generation utilities for adaptive serving."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "BloomFilter",
    "CountMinSketch",
    "ExactEmbeddingIndex",
    "ReservoirSampler",
    "SketchCandidateGenerator",
]


class CountMinSketch:
    """Small deterministic Count-Min sketch for streaming support priors."""

    def __init__(self, *, width: int = 2048, depth: int = 5, seed: int = 17) -> None:
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        self.table = np.zeros((self.depth, self.width), dtype=np.float64)

    def add(self, key: Any, count: float = 1.0) -> None:
        if count < 0.0:
            raise ValueError("count must be non-negative")
        for row, column in enumerate(self._hashes(key)):
            self.table[row, column] += float(count)

    def estimate(self, key: Any) -> float:
        return float(min(self.table[row, column] for row, column in enumerate(self._hashes(key))))

    def _hashes(self, key: Any) -> list[int]:
        payload = repr((self.seed, key)).encode("utf-8")
        return [
            int.from_bytes(hashlib.blake2b(payload + f":{row}".encode(), digest_size=8, person=b"cms").digest(), "big")
            % self.width
            for row in range(self.depth)
        ]


class BloomFilter:
    """Classical Bloom filter for seen-item and gate checks."""

    def __init__(self, *, num_bits: int = 1_000_003, num_hashes: int = 7, seed: int = 23) -> None:
        if num_bits < 1 or num_hashes < 1:
            raise ValueError("num_bits and num_hashes must be positive")
        self.num_bits = int(num_bits)
        self.num_hashes = int(num_hashes)
        self.seed = int(seed)
        self.bits = np.zeros(self.num_bits, dtype=bool)

    def add(self, key: Any) -> None:
        for index in self._hashes(key):
            self.bits[index] = True

    def contains(self, key: Any) -> bool:
        return bool(all(self.bits[index] for index in self._hashes(key)))

    def _hashes(self, key: Any) -> list[int]:
        payload = repr((self.seed, key)).encode("utf-8")
        return [
            int.from_bytes(
                hashlib.blake2b(payload + f":{idx}".encode(), digest_size=8, person=b"bf").digest(),
                "big",
            )
            % self.num_bits
            for idx in range(self.num_hashes)
        ]


class ReservoirSampler:
    """Bounded per-key reservoir for old learner events."""

    def __init__(self, *, max_size: int = 128, seed: int = 29) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._rng = random.Random(seed)
        self._items: dict[Any, list[Any]] = defaultdict(list)
        self._counts: dict[Any, int] = defaultdict(int)

    def update(self, key: Any, value: Any) -> None:
        self._counts[key] += 1
        seen = self._counts[key]
        values = self._items[key]
        if len(values) < self.max_size:
            values.append(value)
            return
        slot = self._rng.randrange(seen)
        if slot < self.max_size:
            values[slot] = value

    def items(self, key: Any) -> list[Any]:
        return list(self._items.get(key, []))


class ExactEmbeddingIndex:
    """Small exact vector index with the same API shape as ANN backends."""

    def __init__(self) -> None:
        self._vectors: dict[Any, np.ndarray] = {}

    def add(self, item_id: Any, vector: Sequence[float]) -> None:
        array = np.asarray(vector, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("vector must be a finite one-dimensional array")
        norm = float(np.linalg.norm(array))
        self._vectors[item_id] = array / max(norm, 1e-12)

    def search(self, query: Sequence[float], *, k: int = 10) -> list[Any]:
        if k <= 0 or not self._vectors:
            return []
        q = np.asarray(query, dtype=float)
        if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
            raise ValueError("query must be a finite one-dimensional array")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = [(item_id, float(np.dot(q, vector))) for item_id, vector in self._vectors.items()]
        scores.sort(key=lambda pair: (pair[1], str(pair[0])), reverse=True)
        return [item_id for item_id, _score in scores[: min(int(k), len(scores))]]


class SketchCandidateGenerator:
    """Candidate generator for sketch mode.

    It combines Count-Min support priors, tracked heavy hitters, optional vector
    retrieval, a seen-item Bloom filter, and bounded recent-history reservoirs.
    The final policy scorer should still rerank these candidates.
    """

    def __init__(
        self,
        *,
        cms: Optional[CountMinSketch] = None,
        ann_index: Optional[Any] = None,
        seen_bloom: Optional[BloomFilter] = None,
        recent_reservoir: Optional[ReservoirSampler] = None,
        top_m: int = 200,
        max_heavy_hitters_per_concept: Optional[int] = None,
    ) -> None:
        if top_m < 1:
            raise ValueError("top_m must be positive")
        self.cms = cms or CountMinSketch()
        self.ann = ann_index
        self.seen = seen_bloom or BloomFilter()
        self.recent = recent_reservoir or ReservoirSampler()
        self.top_m = int(top_m)
        self.max_heavy_hitters_per_concept = int(max_heavy_hitters_per_concept or max(self.top_m * 4, 1024))
        if self.max_heavy_hitters_per_concept < 1:
            raise ValueError("max_heavy_hitters_per_concept must be positive")
        self._concept_item_counts: dict[Any, dict[Any, float]] = defaultdict(dict)

    def update(self, learner_id: Any, concept_id: Any, item_id: Any, correct: Optional[Any] = None) -> None:
        self.cms.add((concept_id, item_id), 1.0)
        counts = self._concept_item_counts[concept_id]
        counts[item_id] = float(counts.get(item_id, 0.0) + 1.0)
        while len(counts) > self.max_heavy_hitters_per_concept:
            evict_id = min(counts, key=lambda key: (counts[key], str(key)))
            counts.pop(evict_id, None)
        self.recent.update(learner_id, (item_id, correct))
        self.mark_seen(learner_id, item_id)

    def mark_seen(self, learner_id: Any, item_id: Any) -> None:
        self.seen.add((learner_id, item_id))

    def candidates(
        self,
        learner_id: Any,
        concept_id: Any,
        *,
        item_query_vec: Optional[Sequence[float]] = None,
        top_m: Optional[int] = None,
    ) -> list[Any]:
        k = self.top_m if top_m is None else int(top_m)
        if k <= 0:
            return []
        heavy = self._heavy_hitters(concept_id, k=max(1, k // 2))
        ann_items: list[Any] = []
        if self.ann is not None and item_query_vec is not None:
            ann_items = list(self.ann.search(item_query_vec, k=k))
        merged = []
        seen_items = set()
        for item_id in [*heavy, *ann_items]:
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            if self.seen.contains((learner_id, item_id)):
                continue
            merged.append(item_id)
            if len(merged) >= k:
                break
        return merged

    def _heavy_hitters(self, concept_id: Any, *, k: int) -> list[Any]:
        rows = [
            (item_id, self.cms.estimate((concept_id, item_id)), count)
            for item_id, count in self._concept_item_counts.get(concept_id, {}).items()
        ]
        rows.sort(key=lambda pair: (pair[1], pair[2], str(pair[0])), reverse=True)
        return [item_id for item_id, _estimate, _count in rows[: min(int(k), len(rows))]]
