"""Progression-oriented rewards for adaptive-learning recommendation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "ProgressionRewardBreakdown",
    "ProgressionRewardConfig",
    "expected_progression_reward",
    "observed_progression_reward",
]


@dataclass(frozen=True)
class ProgressionRewardConfig:
    """Weights and thresholds for progression-oriented item value."""

    target_correct: float = 0.70
    stretch_margin: float = 0.15
    stretch_width: float = 0.30
    default_competence: float = 0.50
    correctness_weight: float = 0.25
    mastery_gain_weight: float = 1.00
    stretch_weight: float = 0.75
    difficulty_weight: float = 0.20
    easy_penalty_weight: float = 0.35
    hard_penalty_weight: float = 0.15
    repetition_penalty_weight: float = 0.25
    easy_correct_threshold: float = 0.88
    hard_correct_threshold: float = 0.25
    incorrect_attempt_credit: float = 0.10
    repetition_window: int = 3
    clip_reward: bool = True


@dataclass(frozen=True)
class ProgressionRewardBreakdown:
    """Decomposed progression value for one candidate item."""

    p_correct: float
    difficulty: float
    competence: float
    expected_outcome_value: float
    mastery_gain: float
    stretch_fit: float
    difficulty_bonus: float
    easy_penalty: float
    hard_penalty: float
    repetition_penalty: float
    expected_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_progression_reward(
    *,
    p_correct: float,
    difficulty: Optional[float],
    competence: float,
    recent_repetition: int = 0,
    config: Optional[ProgressionRewardConfig] = None,
) -> ProgressionRewardBreakdown:
    """Estimate progression value before serving a candidate item."""
    cfg = config or ProgressionRewardConfig()
    p = _clamp01(p_correct)
    diff = _clamp01(0.5 if difficulty is None else difficulty)
    comp = _clamp01(competence)

    expected_outcome = p * (0.55 + 0.45 * diff) + (1.0 - p) * cfg.incorrect_attempt_credit * diff
    mastery_gain = p * (1.0 - p) * max(0.0, 1.0 - comp)
    stretch_target = _clamp01(comp + cfg.stretch_margin)
    difficulty_fit = max(0.0, 1.0 - abs(diff - stretch_target) / max(cfg.stretch_width, 1e-6))
    correctness_fit = max(
        0.0,
        1.0 - abs(p - cfg.target_correct) / max(cfg.target_correct, 1.0 - cfg.target_correct, 1e-6),
    )
    stretch_fit = 0.5 * difficulty_fit + 0.5 * correctness_fit
    difficulty_bonus = diff
    easy_penalty = max(0.0, p - cfg.easy_correct_threshold) / max(1.0 - cfg.easy_correct_threshold, 1e-6)
    hard_penalty = max(0.0, cfg.hard_correct_threshold - p) / max(cfg.hard_correct_threshold, 1e-6)
    repetition_penalty = min(1.0, max(0, recent_repetition) / max(1, cfg.repetition_window))

    reward = (
        cfg.correctness_weight * expected_outcome
        + cfg.mastery_gain_weight * mastery_gain
        + cfg.stretch_weight * stretch_fit
        + cfg.difficulty_weight * difficulty_bonus
        - cfg.easy_penalty_weight * easy_penalty
        - cfg.hard_penalty_weight * hard_penalty
        - cfg.repetition_penalty_weight * repetition_penalty
    )
    if cfg.clip_reward:
        reward = _clamp01(reward)

    return ProgressionRewardBreakdown(
        p_correct=p,
        difficulty=diff,
        competence=comp,
        expected_outcome_value=float(expected_outcome),
        mastery_gain=float(mastery_gain),
        stretch_fit=float(stretch_fit),
        difficulty_bonus=float(difficulty_bonus),
        easy_penalty=float(easy_penalty),
        hard_penalty=float(hard_penalty),
        repetition_penalty=float(repetition_penalty),
        expected_reward=float(reward),
    )


def observed_progression_reward(
    *,
    correct: Any,
    p_correct: float,
    difficulty: Optional[float],
    competence: float,
    recent_repetition: int = 0,
    config: Optional[ProgressionRewardConfig] = None,
) -> float:
    """Compute a progression reward for a logged outcome.

    The reward gives more credit for correct answers on harder items, some
    credit for attempting hard items, and subtracts the same easy/repetition
    penalties used by the expected reward.
    """
    cfg = config or ProgressionRewardConfig()
    expected = expected_progression_reward(
        p_correct=p_correct,
        difficulty=difficulty,
        competence=competence,
        recent_repetition=recent_repetition,
        config=cfg,
    )
    label = _label01(correct)
    outcome = label * (0.55 + 0.45 * expected.difficulty)
    if label == 0.0:
        outcome = cfg.incorrect_attempt_credit * expected.difficulty
    reward = (
        cfg.correctness_weight * outcome
        + cfg.mastery_gain_weight * expected.mastery_gain
        + cfg.stretch_weight * expected.stretch_fit
        + cfg.difficulty_weight * expected.difficulty_bonus
        - cfg.easy_penalty_weight * expected.easy_penalty
        - cfg.hard_penalty_weight * expected.hard_penalty
        - cfg.repetition_penalty_weight * expected.repetition_penalty
    )
    return _clamp01(reward) if cfg.clip_reward else float(reward)


def _clamp01(value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("progression reward inputs must be finite")
    return max(0.0, min(1.0, numeric))


def _label01(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("correct labels must be finite")
    return 1.0 if numeric >= 0.5 else 0.0
