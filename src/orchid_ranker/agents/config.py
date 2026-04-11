"""Configuration dataclasses for multi-user agentic orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class MultiConfig:
    """Configuration for multi-user agentic orchestration experiments.

    Parameters
    ----------
    rounds : int, optional
        Number of recommendation rounds (default: 10).
    """
    rounds: int = 10
    top_k_base: int = 5
    zpd_margin: float = 0.12
    min_candidates: int = 100
    # Deterministic candidate pool sampling per round (for controlled comparisons)
    deterministic_pool: bool = False
    pool_seed: int = 12345
    # Persist candidate pools to disk so all modes/runs reuse identical per-round pools
    # When enabled, pools are saved under runs/<exp>/<name>/candidate_pools/ by default
    # (derived from cfg.log_path parent). You can override the folder via pool_cache_dir.
    persistent_pool: bool = False
    pool_cache_dir: Optional[str] = None
    epsilon_total_global: float = 0.0
    novelty_bonus: float = 0.10
    mmr_lambda: float = 0.25
    log_path: Optional[str] = None
    console: bool = True
    timing_log_path: Optional[str] = None
    timing_rounds: int = 0
    shuffle_users_each_round: bool = True

    # privacy flags (left here for compatibility with prior code)
    privacy_mode: str = "standard"
    share_signals: bool = False
    per_round_eps_target: float = 0.0

    # adaptive policy bounds (shown in banner; tuning occurs in students)
    alpha_bounds: Tuple[float, float] = (0.10, 0.80)
    lam_bounds: Tuple[float, float] = (0.05, 0.60)
    k_bounds: Tuple[int, int] = (2, 6)
    zpd_bounds: Tuple[float, float] = (0.08, 0.18)
    console: bool = True
    console_user: bool = True
    policy_gain: float = 1.0
    # Training augmentation knobs (adaptive only)
    train_on_all_shown: bool = False  # if True, treat shown-but-not-accepted as negatives
    train_steps_per_round: int = 1    # extra gradient steps per user-round
    # Warmup (adaptive only): collect batches for the first N rounds,
    # then pretrain the student for `warmup_steps` passes before online updates
    warmup_rounds: int = 0
    warmup_steps: int = 1
    warmup_top_k_boost: int = 0
    warmup_diversity_scale: float = 1.0  # multiply mmr_lambda and novelty_bonus during warmup
    warmup_preloop: bool = False  # if True, run a pseudo-labeled warmup before online rounds
    # FunkSVD-style distillation (optional auxiliary guidance)
    funk_distill: bool = False
    funk_lambda: float = 0.3
    # Use FunkSVD candidate generation (top-N) before agentic re-rank
    use_funk_candidates: bool = False
    funk_pool_size: int = 0  # 0 => use min_candidates


@dataclass
class UserCtx:
    """User context and state for agentic recommendations.

    Parameters
    ----------
    user_id : int
        External user identifier.
    user_idx : int
        Internal index into user feature matrix.
    student : Any
        StudentAgent instance for adaptive personalization.
    user_vec : torch.Tensor
        Shape (1, Du), user feature vector.
    profile : str, optional
        Profile tag for fairness tracking (default: None).
    name : str, optional
        User name or identifier (default: None).
    """
    user_id: int           # external user ID
    user_idx: int          # internal index into user_matrix
    student: Any           # StudentAgent instance
    user_vec: torch.Tensor # [1, Du] — dense side-features row for the user
    profile: Optional[str] = None  # profile tag used for fairness stats
    name: Optional[str] = None


@dataclass
class PolicyState:
    """Adaptive policy parameters and moving-average statistics.

    Parameters
    ----------
    alpha : float
        Exploration strength parameter.
    lam : float
        Diversity/MMR weight.
    top_k : int
        Top-k for ranking and candidate filtering.
    zpd_delta : float
        Zone of proximal development margin.
    novelty : float
        Novelty bonus weight.
    accept_ma : float, optional
        Moving average of acceptance rate (default: 0.5).
    acc_ma : float, optional
        Moving average of accuracy (default: 0.6).
    novelty_ma : float, optional
        Moving average of novelty rate (default: 0.5).
    reward_ma : float, optional
        Moving average of reward/engagement (default: 0.55).
    knowledge_ma : float, optional
        Moving average of user knowledge (default: 0.5).
    knowledge_delta_ma : float, optional
        Moving average of knowledge change (default: 0.0).
    rounds : int, optional
        Number of rounds executed (default: 0).
    """
    alpha: float
    lam: float
    top_k: int
    zpd_delta: float
    novelty: float
    accept_ma: float = 0.5
    acc_ma: float = 0.6
    novelty_ma: float = 0.5
    reward_ma: float = 0.55
    knowledge_ma: float = 0.5
    knowledge_delta_ma: float = 0.0
    rounds: int = 0


class OnlineState:
    """Online user state tracking for knowledge, fatigue, trust, engagement, and uncertainty.

    Parameters
    ----------
    None

    Attributes
    ----------
    _state : dict
        Mapping of user_id to state dict with keys: knowledge, fatigue, trust, engagement, uncertainty.
    """
    def __init__(self):
        self._state: Dict[int, Dict[str, float]] = {}  # user_id -> {k,f,t,e,uncertainty}

    def set_initial(self, uid: int, *, knowledge: float, fatigue: float, engagement: float, trust: float, uncertainty: float):
        self._state[int(uid)] = dict(knowledge=float(knowledge),
                                     fatigue=float(fatigue),
                                     engagement=float(engagement),
                                     trust=float(trust),
                                     uncertainty=float(uncertainty))

    def get(self, uid: int) -> Dict[str, float]:
        return dict(self._state.get(int(uid), dict(knowledge=0.5, fatigue=0.2, trust=0.5, engagement=0.6, uncertainty=0.5)))


__all__ = [
    "MultiConfig",
    "UserCtx",
    "PolicyState",
    "OnlineState",
]
