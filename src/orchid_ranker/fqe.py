"""Tabular fitted Q evaluation for logged adaptive-learning policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "FQEReport",
    "TabularFQE",
]


@dataclass(frozen=True)
class FQEReport:
    """Summary of fitted Q evaluation."""

    n_transitions: int
    n_contexts: int
    n_actions: int
    gamma: float
    epochs: int
    learning_rate: float
    estimated_value: float
    final_loss: float
    target_start_action_col: Optional[str] = None
    target_start_action_supported_fraction: float = 0.0
    target_start_action_is_explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TabularFQE:
    """Dependency-light FQE for discrete next-item policy evaluation."""

    def __init__(
        self,
        *,
        gamma: float = 0.95,
        learning_rate: float = 0.1,
        epochs: int = 100,
        random_state: Optional[int] = 42,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = random_state
        self.q_values_: dict[tuple[Any, Any], float] = {}
        self.report_: Optional[FQEReport] = None

    @property
    def is_fitted(self) -> bool:
        return self.report_ is not None

    def fit(
        self,
        transitions: pd.DataFrame,
        *,
        context_col: str = "context_hash",
        action_col: str = "chosen_item_id",
        reward_col: str = "reward",
        next_context_col: str = "next_context_hash",
        target_action_col: str = "target_action_id",
        target_start_action_col: Optional[str] = None,
        done_col: str = "done",
    ) -> "TabularFQE":
        """Fit Q values for a fixed target policy action per next context.

        The reported ``estimated_value`` is the policy value at the start states,
        ``V^π(s0) = Q^π(s0, π(s0))``. Supply ``target_start_action_col`` with the
        target policy's action at each start context for an exact estimate. When
        it is omitted, the start action defaults to the target policy's action
        recovered from the observed ``next_context -> target_action`` pairs (NOT
        the logged action), falling back to the logged action only for start
        contexts whose target action is unknown. Inspect
        ``target_start_action_supported_fraction`` to see how much of the start
        value is backed by logged ``(context, action)`` support.
        """
        required = {context_col, action_col, reward_col, next_context_col, target_action_col, done_col}
        if target_start_action_col is not None:
            required.add(target_start_action_col)
        missing = required - set(transitions.columns)
        if missing:
            raise ValueError(f"transitions missing required columns: {sorted(missing)}")
        work = transitions.reset_index(drop=True)
        self.q_values_ = {}
        rows = []
        contexts: set[Any] = set()
        actions: set[Any] = set()
        explicit_start_actions: list[Any] = []
        logged_pairs: set[tuple[Any, Any]] = set()
        # Deterministic target policy recovered from observed (state, π(state)) pairs: each
        # row reports the target action at its *next* context, i.e. π evaluated at that state.
        # We reuse it to recover the target action at *start* contexts as well.
        policy_action_by_context: dict[Any, Any] = {}
        columns = [context_col, action_col, reward_col, next_context_col, target_action_col, done_col]
        if target_start_action_col is not None:
            columns.append(target_start_action_col)
        for values in work[columns].itertuples(index=False, name=None):
            context, action, reward, next_context, target_action, done = values[:6]
            reward_value = float(reward)
            if not np.isfinite(reward_value):
                raise ValueError("reward values must be finite")
            done_value = bool(done)
            rows.append((context, action, reward_value, next_context, target_action, done_value))
            if target_start_action_col is not None:
                explicit_start_actions.append(values[6])
            logged_pairs.add((context, action))
            contexts.update([context, next_context])
            actions.update([action, target_action])
            policy_action_by_context.setdefault(next_context, target_action)
            self.q_values_.setdefault((context, action), 0.0)
            self.q_values_.setdefault((next_context, target_action), 0.0)

        # Resolve the start action used to report V^π(s0) = Q^π(s0, π(s0)). Explicit column
        # wins; otherwise use the target policy's action at the start context (NOT the logged
        # action), falling back to the logged action only for start contexts whose target
        # action is unknown. Defaulting to the logged action would report Q^π(s0, a_logged),
        # which is not the target policy value when behavior and target diverge at the root.
        start_actions: list[Any] = []
        for idx, (context, action, *_rest) in enumerate(rows):
            if target_start_action_col is not None:
                start_action = explicit_start_actions[idx]
            else:
                start_action = policy_action_by_context.get(context, action)
            start_actions.append(start_action)
            actions.add(start_action)
            self.q_values_.setdefault((context, start_action), 0.0)

        rng = np.random.RandomState(self.random_state)
        final_loss = 0.0
        for _ in range(self.epochs):
            losses = []
            for idx in rng.permutation(len(rows)):
                context, action, reward, next_context, target_action, done = rows[int(idx)]
                current = self.q_values_.get((context, action), 0.0)
                bootstrap = 0.0 if done else self.q_values_.get((next_context, target_action), 0.0)
                target = reward + self.gamma * bootstrap
                error = current - target
                self.q_values_[(context, action)] = float(current - self.learning_rate * error)
                losses.append(error * error)
            final_loss = float(np.mean(losses)) if losses else 0.0

        start_values = [
            self.q_values_.get((context, start_action), 0.0)
            for (context, *_rest), start_action in zip(rows, start_actions)
        ]
        supported = [
            1.0 if (context, start_action) in logged_pairs else 0.0
            for (context, *_rest), start_action in zip(rows, start_actions)
        ]
        self.report_ = FQEReport(
            n_transitions=int(len(rows)),
            n_contexts=int(len(contexts)),
            n_actions=int(len(actions)),
            gamma=self.gamma,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            estimated_value=float(np.mean(start_values)) if start_values else 0.0,
            final_loss=final_loss,
            target_start_action_col=target_start_action_col,
            target_start_action_supported_fraction=float(np.mean(supported)) if supported else 0.0,
            target_start_action_is_explicit=target_start_action_col is not None,
        )
        return self

    def score(self, context: Any, action: Any) -> float:
        self._require_fitted()
        return float(self.q_values_.get((context, action), 0.0))

    def value(self, contexts: list[Any], actions: list[Any]) -> float:
        self._require_fitted()
        if len(contexts) != len(actions):
            raise ValueError("contexts and actions must have the same length")
        if not contexts:
            return 0.0
        return float(np.mean([self.q_values_.get((context, action), 0.0) for context, action in zip(contexts, actions)]))

    def _require_fitted(self) -> None:
        if self.report_ is None:
            raise RuntimeError("TabularFQE must be fitted before use")
