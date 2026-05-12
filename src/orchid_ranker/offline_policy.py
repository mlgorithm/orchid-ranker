"""Conservative offline policy learning for discrete next-item actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_schema import parse_candidate_list, validate_logged_decisions

__all__ = [
    "CQLDiscretePolicy",
    "CQLTrainingReport",
]


@dataclass(frozen=True)
class CQLTrainingReport:
    """Training summary for :class:`CQLDiscretePolicy`."""

    n_events: int
    n_contexts: int
    n_actions: int
    epochs: int
    learning_rate: float
    conservative_weight: float
    reward_mean: float
    final_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CQLDiscretePolicy:
    """A tabular CQL-style learner for logged discrete next-item choices.

    This is the small-data, dependency-light policy learner used by
    ``AdaptiveRanker`` before introducing deep offline RL dependencies. It uses
    the CQL penalty ``logsumexp(Q(s, A)) - Q(s, a_logged)`` over the logged
    candidate set, which lowers unsupported actions and favors choices backed by
    observed reward.
    """

    def __init__(
        self,
        *,
        conservative_weight: float = 1.0,
        learning_rate: float = 0.05,
        epochs: int = 50,
        random_state: Optional[int] = 42,
        unseen_penalty: float = 0.05,
    ) -> None:
        if conservative_weight < 0.0:
            raise ValueError("conservative_weight must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if unseen_penalty < 0.0:
            raise ValueError("unseen_penalty must be non-negative")
        self.conservative_weight = float(conservative_weight)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = random_state
        self.unseen_penalty = float(unseen_penalty)
        self.q_values_: dict[tuple[Any, Any], float] = {}
        self.global_action_values_: dict[Any, float] = {}
        self.actions_by_context_: dict[Any, set[Any]] = {}
        self.default_value_: float = 0.0
        self.report_: Optional[CQLTrainingReport] = None

    @property
    def is_fitted(self) -> bool:
        return self.report_ is not None

    def fit(
        self,
        decisions: pd.DataFrame,
        *,
        context_col: str = "context_hash",
        candidate_col: str = "candidate_item_ids",
        action_col: str = "chosen_item_id",
        reward_col: str = "reward",
        propensity_col: str = "propensity",
    ) -> "CQLDiscretePolicy":
        """Fit from logged decisions with rewards and propensities."""
        work = validate_logged_decisions(
            decisions,
            candidate_col=candidate_col,
            chosen_col=action_col,
            propensity_col=propensity_col,
            context_hash_col=context_col,
            reward_col=reward_col,
        ).reset_index(drop=True)
        rows = []
        action_rewards: dict[Any, list[float]] = {}
        for context, candidates_raw, action, reward in work[
            [context_col, candidate_col, action_col, reward_col]
        ].itertuples(index=False, name=None):
            candidates = parse_candidate_list(candidates_raw)
            rows.append((context, candidates, action, float(reward)))
            self.actions_by_context_.setdefault(context, set()).update(candidates)
            action_rewards.setdefault(action, []).append(float(reward))
            for candidate in candidates:
                self.q_values_.setdefault((context, candidate), 0.0)

        self.default_value_ = float(work[reward_col].astype(float).mean())
        self.global_action_values_ = {
            action: float(np.mean(values)) for action, values in action_rewards.items()
        }
        rng = np.random.RandomState(self.random_state)
        final_loss = 0.0
        for _epoch in range(self.epochs):
            order = rng.permutation(len(rows))
            losses = []
            for row_id in order:
                context, candidates, action, reward = rows[int(row_id)]
                losses.append(self._update_one(context, candidates, action, reward))
            final_loss = float(np.mean(losses)) if losses else 0.0

        all_actions = {action for actions in self.actions_by_context_.values() for action in actions}
        self.report_ = CQLTrainingReport(
            n_events=int(len(work)),
            n_contexts=int(len(self.actions_by_context_)),
            n_actions=int(len(all_actions)),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            conservative_weight=self.conservative_weight,
            reward_mean=self.default_value_,
            final_loss=final_loss,
        )
        return self

    def score(self, context: Any, candidate_actions: Sequence[Any]) -> dict[Any, float]:
        """Return conservative Q scores for candidate actions."""
        self._require_fitted()
        return {action: self._score_one(context, action) for action in candidate_actions}

    def recommend(self, context: Any, candidate_actions: Sequence[Any], *, top_k: int = 1) -> list[Any]:
        """Choose top actions by conservative value."""
        if top_k <= 0:
            return []
        scores = self.score(context, candidate_actions)
        ranked = sorted(scores, key=lambda action: (scores[action], str(action)), reverse=True)
        return ranked[: min(int(top_k), len(ranked))]

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.report_ is not None
        return self.report_.to_dict()

    def _update_one(self, context: Any, candidates: Sequence[Any], action: Any, reward: float) -> float:
        q_values = np.asarray([self.q_values_.get((context, candidate), 0.0) for candidate in candidates], dtype=float)
        action_index = list(candidates).index(action)
        q_sa = float(q_values[action_index])
        bellman_loss = (q_sa - reward) ** 2
        probs = _softmax(q_values)
        cql_loss = float(np.log(np.sum(np.exp(q_values - np.max(q_values)))) + np.max(q_values) - q_sa)
        for idx, candidate in enumerate(candidates):
            grad = self.conservative_weight * float(probs[idx])
            if idx == action_index:
                grad += 2.0 * (q_sa - reward) - self.conservative_weight
            key = (context, candidate)
            self.q_values_[key] = float(self.q_values_.get(key, 0.0) - self.learning_rate * grad)
        return float(bellman_loss + self.conservative_weight * cql_loss)

    def _score_one(self, context: Any, action: Any) -> float:
        if (context, action) in self.q_values_:
            return float(self.q_values_[(context, action)])
        if action in self.global_action_values_:
            return float(self.global_action_values_[action] - self.unseen_penalty)
        return float(self.default_value_ - self.unseen_penalty)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("CQLDiscretePolicy must be fitted before use")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=float)
    return exp / denom
