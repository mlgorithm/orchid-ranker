"""Offline policy evaluation for adaptive recommenders.

The estimators in this module evaluate a candidate recommendation policy from
logged bandit data before that policy is served live. They are intentionally
small, dataframe-friendly, and independent of Torch so they can run in CI,
notebooks, and compliance review jobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import pandas as pd

__all__ = [
    "BootstrapLoggedPolicyReport",
    "BootstrapPolicyComparisonReport",
    "LoggedPolicyReport",
    "OPERolloutGateReport",
    "PolicyComparisonReport",
    "bootstrap_compare_logged_policies",
    "bootstrap_logged_policy",
    "compare_logged_policies",
    "deterministic_policy_probabilities",
    "evaluate_logged_policy",
    "evaluate_rollout_gate",
]


@dataclass(frozen=True)
class LoggedPolicyReport:
    """Offline value estimate for one candidate policy.

    Attributes
    ----------
    n_events:
        Number of logged events evaluated.
    logging_reward:
        Mean observed reward under the historical logging policy.
    ips:
        Inverse-propensity-score estimate.
    snips:
        Self-normalized IPS estimate.
    direct_method:
        Mean model-predicted target-policy value, if supplied.
    doubly_robust:
        Doubly robust value estimate, if value and logged-action estimates are supplied.
    estimator:
        Estimator used for ``value``, ``standard_error``, and confidence bounds.
    value:
        Preferred estimate, choosing doubly robust when available, then SNIPS, then IPS.
    standard_error:
        Normal-approximation standard error for the preferred estimate.
    ci_low / ci_high:
        Confidence interval for the preferred estimate.
    effective_sample_size:
        Importance-weight effective sample size.
    coverage:
        Fraction of logged events that the target policy could have produced.
    weight_mean / weight_max:
        Importance-weight diagnostics.
    clipped_fraction:
        Fraction of raw weights clipped by ``max_weight``.
    """

    n_events: int
    logging_reward: float
    ips: float
    snips: float
    direct_method: Optional[float]
    doubly_robust: Optional[float]
    estimator: str
    value: float
    standard_error: float
    ci_low: float
    ci_high: float
    effective_sample_size: float
    coverage: float
    weight_mean: float
    weight_max: float
    clipped_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))


@dataclass(frozen=True)
class PolicyComparisonReport:
    """Paired offline comparison between a target policy and a baseline."""

    target: LoggedPolicyReport
    baseline: LoggedPolicyReport
    estimator: str
    uplift: float
    standard_error: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.to_dict()
        data["baseline"] = self.baseline.to_dict()
        return cast(dict[str, Any], _jsonable(data))


@dataclass(frozen=True)
class BootstrapLoggedPolicyReport:
    """Logged-policy report with bootstrap confidence interval."""

    base: LoggedPolicyReport
    estimator: str
    n_bootstrap: int
    ci: float
    bootstrap_standard_error: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    cluster_col: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _jsonable(
                {
                    "base": self.base.to_dict(),
                    "estimator": self.estimator,
                    "n_bootstrap": self.n_bootstrap,
                    "ci": self.ci,
                    "bootstrap_standard_error": self.bootstrap_standard_error,
                    "bootstrap_ci_low": self.bootstrap_ci_low,
                    "bootstrap_ci_high": self.bootstrap_ci_high,
                    "cluster_col": self.cluster_col,
                }
            ),
        )


@dataclass(frozen=True)
class BootstrapPolicyComparisonReport:
    """Paired policy comparison with bootstrap uplift interval."""

    base: PolicyComparisonReport
    estimator: str
    n_bootstrap: int
    ci: float
    bootstrap_standard_error: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    cluster_col: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _jsonable(
                {
                    "base": self.base.to_dict(),
                    "estimator": self.estimator,
                    "n_bootstrap": self.n_bootstrap,
                    "ci": self.ci,
                    "bootstrap_standard_error": self.bootstrap_standard_error,
                    "bootstrap_ci_low": self.bootstrap_ci_low,
                    "bootstrap_ci_high": self.bootstrap_ci_high,
                    "cluster_col": self.cluster_col,
                }
            ),
        )


@dataclass(frozen=True)
class OPERolloutGateReport:
    """Decision-grade rollout gate derived from an OPE report."""

    allowed: bool
    estimator: str
    effect: float
    ci_low: float
    ci_high: float
    min_effect: float
    n_events: int
    effective_sample_size: float
    effective_sample_size_fraction: float
    coverage: float
    clipped_fraction: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))


@dataclass(frozen=True)
class _Estimate:
    report: LoggedPolicyReport
    terms: np.ndarray


def deterministic_policy_probabilities(
    logged_actions: Sequence[Any],
    policy_actions: Sequence[Any],
) -> np.ndarray:
    """Return logged-action probabilities for a deterministic target policy.

    The result is ``1`` when the candidate policy would have selected the logged
    action and ``0`` otherwise. This is useful for replay-style evaluation on
    uniformly randomized logs.
    """
    if len(logged_actions) != len(policy_actions):
        raise ValueError("logged_actions and policy_actions must have the same length")
    return np.asarray([float(logged == policy) for logged, policy in zip(logged_actions, policy_actions)], dtype=float)


def evaluate_logged_policy(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    target_value_col: Optional[str] = None,
    logged_action_value_col: Optional[str] = None,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = 1.0,
    min_propensity: float = 1e-6,
    max_weight: Optional[float] = None,
    ci: float = 0.95,
) -> LoggedPolicyReport:
    """Estimate a target policy's value from logged bandit data.

    ``target_probability_col`` must contain the probability that the target
    policy would assign to the logged action for that row. For deterministic
    replay policies this is usually 1 for matching rows and 0 otherwise.

    If ``target_value_col`` and ``logged_action_value_col`` are supplied, the
    report includes a doubly robust estimate:
    ``V_hat_pi(x) + w * (reward - Q_hat(x, logged_action))``.
    """
    return _estimate_logged_policy(
        events,
        reward_col=reward_col,
        propensity_col=propensity_col,
        policy_probability_col=target_probability_col,
        policy_value_col=target_value_col,
        logged_action_value_col=logged_action_value_col,
        reward_min=reward_min,
        reward_max=reward_max,
        min_propensity=min_propensity,
        max_weight=max_weight,
        ci=ci,
    ).report


def compare_logged_policies(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    baseline_probability_col: str,
    target_value_col: Optional[str] = None,
    baseline_value_col: Optional[str] = None,
    logged_action_value_col: Optional[str] = None,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = 1.0,
    min_propensity: float = 1e-6,
    max_weight: Optional[float] = None,
    ci: float = 0.95,
) -> PolicyComparisonReport:
    """Compare a candidate policy against a baseline on the same logged events."""
    target = _estimate_logged_policy(
        events,
        reward_col=reward_col,
        propensity_col=propensity_col,
        policy_probability_col=target_probability_col,
        policy_value_col=target_value_col,
        logged_action_value_col=logged_action_value_col,
        reward_min=reward_min,
        reward_max=reward_max,
        min_propensity=min_propensity,
        max_weight=max_weight,
        ci=ci,
    )
    baseline = _estimate_logged_policy(
        events,
        reward_col=reward_col,
        propensity_col=propensity_col,
        policy_probability_col=baseline_probability_col,
        policy_value_col=baseline_value_col,
        logged_action_value_col=logged_action_value_col,
        reward_min=reward_min,
        reward_max=reward_max,
        min_propensity=min_propensity,
        max_weight=max_weight,
        ci=ci,
    )

    estimator = target.report.estimator if target.report.estimator == baseline.report.estimator else "paired"
    diff_terms = target.terms - baseline.terms
    uplift = float(np.mean(diff_terms))
    se, low, high = _normal_ci(diff_terms, ci)
    return PolicyComparisonReport(
        target=target.report,
        baseline=baseline.report,
        estimator=estimator,
        uplift=uplift,
        standard_error=se,
        ci_low=low,
        ci_high=high,
    )


def bootstrap_logged_policy(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    target_value_col: Optional[str] = None,
    logged_action_value_col: Optional[str] = None,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = 1.0,
    min_propensity: float = 1e-6,
    max_weight: Optional[float] = None,
    ci: float = 0.95,
    n_bootstrap: int = 500,
    random_state: Optional[int] = 42,
    cluster_col: Optional[str] = None,
) -> BootstrapLoggedPolicyReport:
    """Evaluate a logged policy with percentile bootstrap confidence bounds.

    When ``cluster_col`` is supplied, resampling happens at the cluster level
    instead of the row level. This is useful for adaptive-learning logs where
    events from the same learner, classroom, or course are correlated.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    _validate_cluster_col(events, cluster_col)
    base = evaluate_logged_policy(
        events,
        reward_col=reward_col,
        propensity_col=propensity_col,
        target_probability_col=target_probability_col,
        target_value_col=target_value_col,
        logged_action_value_col=logged_action_value_col,
        reward_min=reward_min,
        reward_max=reward_max,
        min_propensity=min_propensity,
        max_weight=max_weight,
        ci=ci,
    )
    rng = np.random.RandomState(random_state)
    values = np.zeros((int(n_bootstrap),), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample = _bootstrap_sample(events, rng=rng, cluster_col=cluster_col)
        values[idx] = evaluate_logged_policy(
            sample,
            reward_col=reward_col,
            propensity_col=propensity_col,
            target_probability_col=target_probability_col,
            target_value_col=target_value_col,
            logged_action_value_col=logged_action_value_col,
            reward_min=reward_min,
            reward_max=reward_max,
            min_propensity=min_propensity,
            max_weight=max_weight,
            ci=ci,
        ).value
    low, high = _percentile_ci(values, ci)
    return BootstrapLoggedPolicyReport(
        base=base,
        estimator=base.estimator,
        n_bootstrap=int(n_bootstrap),
        ci=float(ci),
        bootstrap_standard_error=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
        cluster_col=cluster_col,
    )


def bootstrap_compare_logged_policies(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    baseline_probability_col: str,
    target_value_col: Optional[str] = None,
    baseline_value_col: Optional[str] = None,
    logged_action_value_col: Optional[str] = None,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = 1.0,
    min_propensity: float = 1e-6,
    max_weight: Optional[float] = None,
    ci: float = 0.95,
    n_bootstrap: int = 500,
    random_state: Optional[int] = 42,
    cluster_col: Optional[str] = None,
) -> BootstrapPolicyComparisonReport:
    """Compare two logged policies with a paired bootstrap uplift interval.

    When ``cluster_col`` is supplied, resampling happens at the cluster level.
    Use learner/classroom/course IDs here when logged outcomes are correlated
    within those groups.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    _validate_cluster_col(events, cluster_col)
    base = compare_logged_policies(
        events,
        reward_col=reward_col,
        propensity_col=propensity_col,
        target_probability_col=target_probability_col,
        baseline_probability_col=baseline_probability_col,
        target_value_col=target_value_col,
        baseline_value_col=baseline_value_col,
        logged_action_value_col=logged_action_value_col,
        reward_min=reward_min,
        reward_max=reward_max,
        min_propensity=min_propensity,
        max_weight=max_weight,
        ci=ci,
    )
    rng = np.random.RandomState(random_state)
    uplifts = np.zeros((int(n_bootstrap),), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample = _bootstrap_sample(events, rng=rng, cluster_col=cluster_col)
        uplifts[idx] = compare_logged_policies(
            sample,
            reward_col=reward_col,
            propensity_col=propensity_col,
            target_probability_col=target_probability_col,
            baseline_probability_col=baseline_probability_col,
            target_value_col=target_value_col,
            baseline_value_col=baseline_value_col,
            logged_action_value_col=logged_action_value_col,
            reward_min=reward_min,
            reward_max=reward_max,
            min_propensity=min_propensity,
            max_weight=max_weight,
            ci=ci,
        ).uplift
    low, high = _percentile_ci(uplifts, ci)
    return BootstrapPolicyComparisonReport(
        base=base,
        estimator=base.estimator,
        n_bootstrap=int(n_bootstrap),
        ci=float(ci),
        bootstrap_standard_error=float(np.std(uplifts, ddof=1)) if uplifts.size > 1 else 0.0,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
        cluster_col=cluster_col,
    )


def evaluate_rollout_gate(
    report: (
        LoggedPolicyReport
        | BootstrapLoggedPolicyReport
        | PolicyComparisonReport
        | BootstrapPolicyComparisonReport
    ),
    *,
    min_effect: float = 0.0,
    min_ess_fraction: float = 0.05,
    min_coverage: float = 0.05,
    max_clipped_fraction: float = 0.20,
) -> OPERolloutGateReport:
    """Gate a candidate adaptive policy before live rollout.

    For single-policy reports, ``effect`` is the estimated policy value. For
    comparison reports, ``effect`` is target-minus-baseline uplift. Bootstrap
    reports use bootstrap percentile intervals; plain reports use normal
    approximation intervals.
    """
    if not 0.0 <= min_ess_fraction <= 1.0:
        raise ValueError("min_ess_fraction must be in [0, 1]")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    if not 0.0 <= max_clipped_fraction <= 1.0:
        raise ValueError("max_clipped_fraction must be in [0, 1]")

    effect, ci_low, ci_high, estimator, diagnostics = _rollout_gate_parts(report)
    n_events = diagnostics.n_events
    ess = diagnostics.effective_sample_size
    ess_fraction = float(ess / n_events) if n_events > 0 else 0.0
    coverage = diagnostics.coverage
    clipped = diagnostics.clipped_fraction

    reasons: list[str] = []
    if not np.isfinite(effect) or not np.isfinite(ci_low) or not np.isfinite(ci_high):
        reasons.append("non-finite OPE estimate")
    if ci_low <= float(min_effect):
        reasons.append(
            f"lower confidence bound {ci_low:.6g} is not above required effect {float(min_effect):.6g}"
        )
    if ess_fraction < float(min_ess_fraction):
        reasons.append(
            f"effective sample size fraction {ess_fraction:.3f} is below {float(min_ess_fraction):.3f}"
        )
    if coverage < float(min_coverage):
        reasons.append(f"coverage {coverage:.3f} is below {float(min_coverage):.3f}")
    if clipped > float(max_clipped_fraction):
        reasons.append(f"clipped fraction {clipped:.3f} exceeds {float(max_clipped_fraction):.3f}")

    return OPERolloutGateReport(
        allowed=not reasons,
        estimator=estimator,
        effect=float(effect),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        min_effect=float(min_effect),
        n_events=int(n_events),
        effective_sample_size=float(ess),
        effective_sample_size_fraction=ess_fraction,
        coverage=float(coverage),
        clipped_fraction=float(clipped),
        reasons=tuple(reasons),
    )


def _estimate_logged_policy(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    policy_probability_col: str,
    policy_value_col: Optional[str],
    logged_action_value_col: Optional[str],
    reward_min: Optional[float],
    reward_max: Optional[float],
    min_propensity: float,
    max_weight: Optional[float],
    ci: float,
) -> _Estimate:
    _validate_config(min_propensity=min_propensity, max_weight=max_weight, ci=ci)
    _require_columns(
        events,
        reward_col,
        propensity_col,
        policy_probability_col,
        *(col for col in [policy_value_col, logged_action_value_col] if col is not None),
    )
    if events.empty:
        raise ValueError("events DataFrame is empty")

    rewards = _numeric(events[reward_col], reward_col)
    if reward_min is not None or reward_max is not None:
        low = -np.inf if reward_min is None else float(reward_min)
        high = np.inf if reward_max is None else float(reward_max)
        if low > high:
            raise ValueError("reward_min must be <= reward_max")
        rewards = np.clip(rewards, low, high)

    propensities = _numeric(events[propensity_col], propensity_col)
    policy_probs = _numeric(events[policy_probability_col], policy_probability_col)
    if np.any(propensities < min_propensity) or np.any(propensities > 1.0):
        raise ValueError(f"{propensity_col} values must be in [{min_propensity}, 1]")
    if np.any(policy_probs < 0.0) or np.any(policy_probs > 1.0):
        raise ValueError(f"{policy_probability_col} values must be in [0, 1]")

    raw_weights = policy_probs / propensities
    clipped_fraction = 0.0
    if max_weight is not None:
        clipped_fraction = float(np.mean(raw_weights > float(max_weight)))
        weights = np.minimum(raw_weights, float(max_weight))
    else:
        weights = raw_weights

    weighted_rewards = weights * rewards
    ips = float(np.mean(weighted_rewards))
    weight_sum = float(np.sum(weights))
    snips = float(np.sum(weighted_rewards) / weight_sum) if weight_sum > 0.0 else float("nan")
    weight_mean = float(np.mean(weights))
    weight_max = float(np.max(weights))
    coverage = float(np.mean(policy_probs > 0.0))
    ess = _effective_sample_size(weights)
    direct_method = None
    doubly_robust = None

    if policy_value_col is not None:
        policy_values = _numeric(events[policy_value_col], policy_value_col)
        direct_method = float(np.mean(policy_values))
    else:
        policy_values = None

    if policy_values is not None and logged_action_value_col is not None:
        logged_action_values = _numeric(events[logged_action_value_col], logged_action_value_col)
        terms = policy_values + weights * (rewards - logged_action_values)
        doubly_robust = float(np.mean(terms))
        estimator = "doubly_robust"
        value = doubly_robust
    elif weight_mean > 0.0 and np.isfinite(snips):
        terms = weighted_rewards / weight_mean
        estimator = "snips"
        value = snips
    else:
        terms = weighted_rewards
        estimator = "ips"
        value = ips

    se, low, high = _normal_ci(terms, ci)
    report = LoggedPolicyReport(
        n_events=int(len(events)),
        logging_reward=float(np.mean(rewards)),
        ips=ips,
        snips=snips,
        direct_method=direct_method,
        doubly_robust=doubly_robust,
        estimator=estimator,
        value=float(value),
        standard_error=se,
        ci_low=low,
        ci_high=high,
        effective_sample_size=ess,
        coverage=coverage,
        weight_mean=weight_mean,
        weight_max=weight_max,
        clipped_fraction=clipped_fraction,
    )
    return _Estimate(report=report, terms=terms)


def _rollout_gate_parts(
    report: (
        LoggedPolicyReport
        | BootstrapLoggedPolicyReport
        | PolicyComparisonReport
        | BootstrapPolicyComparisonReport
    ),
) -> tuple[float, float, float, str, LoggedPolicyReport]:
    if isinstance(report, BootstrapPolicyComparisonReport):
        return (
            report.base.uplift,
            report.bootstrap_ci_low,
            report.bootstrap_ci_high,
            report.estimator,
            report.base.target,
        )
    if isinstance(report, PolicyComparisonReport):
        return report.uplift, report.ci_low, report.ci_high, report.estimator, report.target
    if isinstance(report, BootstrapLoggedPolicyReport):
        return (
            report.base.value,
            report.bootstrap_ci_low,
            report.bootstrap_ci_high,
            report.estimator,
            report.base,
        )
    if isinstance(report, LoggedPolicyReport):
        return report.value, report.ci_low, report.ci_high, report.estimator, report
    raise TypeError(f"Unsupported OPE report type: {type(report).__name__}")


def _require_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"events missing required columns: {missing}")


def _numeric(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_config(*, min_propensity: float, max_weight: Optional[float], ci: float) -> None:
    if not 0.0 < min_propensity <= 1.0:
        raise ValueError("min_propensity must be in (0, 1]")
    if max_weight is not None and max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")


def _validate_cluster_col(events: pd.DataFrame, cluster_col: Optional[str]) -> None:
    if cluster_col is None:
        return
    if cluster_col not in events.columns:
        raise ValueError(f"cluster_col={cluster_col!r} is not present in events")
    if events[cluster_col].isna().any():
        raise ValueError(f"cluster_col={cluster_col!r} contains missing values")


def _bootstrap_sample(
    events: pd.DataFrame,
    *,
    rng: np.random.RandomState,
    cluster_col: Optional[str],
) -> pd.DataFrame:
    if cluster_col is None:
        return events.iloc[rng.randint(0, len(events), size=len(events))].reset_index(drop=True)

    clusters = np.asarray(events[cluster_col].drop_duplicates().tolist(), dtype=object)
    sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
    by_cluster = {cluster: group for cluster, group in events.groupby(cluster_col, sort=False)}
    parts = [by_cluster[cluster] for cluster in sampled_clusters]
    return pd.concat(parts, ignore_index=True)


def _effective_sample_size(weights: np.ndarray) -> float:
    denom = float(np.sum(weights**2))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(weights) ** 2) / denom)


def _normal_ci(terms: np.ndarray, ci: float) -> tuple[float, float, float]:
    terms = np.asarray(terms, dtype=float)
    mean = float(np.mean(terms))
    if terms.size <= 1:
        return 0.0, mean, mean
    se = float(np.std(terms, ddof=1) / np.sqrt(terms.size))
    z = NormalDist().inv_cdf(0.5 + ci / 2.0)
    return se, float(mean - z * se), float(mean + z * se)


def _percentile_ci(values: np.ndarray, ci: float) -> tuple[float, float]:
    lower = 100.0 * (1.0 - ci) / 2.0
    upper = 100.0 * (1.0 + ci) / 2.0
    bounds = np.percentile(np.asarray(values, dtype=float), [lower, upper])
    return float(bounds[0]), float(bounds[1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
