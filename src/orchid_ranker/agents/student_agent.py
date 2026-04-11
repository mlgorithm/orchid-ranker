from __future__ import annotations

import logging
import math
import os
import random
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# ------------------------ opt-in debug logging -------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_DEBUG_STUDENT = os.getenv("ORCHID_DEBUG_STUDENT", "0").lower() in {"1", "true", "yes", "on"}

def enable_debug_student_logs(flag: bool = True) -> None:
    """Enable/disable StudentAgent debug logs (env var also supported)."""
    global _DEBUG_STUDENT
    _DEBUG_STUDENT = bool(flag)
    if flag:
        logger.setLevel(logging.DEBUG)

def _p(*args) -> None:
    if _DEBUG_STUDENT or logger.isEnabledFor(logging.DEBUG):
        logger.debug(" ".join(str(a) for a in args))

def _safe(v):
    try:
        return float(v)
    except Exception:
        return v

# ----------------------------------------------------------------------





@dataclass
class ItemMeta:
    """Metadata about an educational item.

    Attributes
    ----------
    difficulty : float
        Difficulty level in [0, 1] (default: 0.5).
    skills : list of int, optional
        Skill indices (or boolean mask) associated with this item.
    """

    difficulty: float = 0.5
    skills: Optional[List[int]] = None  # indices or boolean mask


class AdaptiveAgent:
    """Simulator of a user with adaptive behavior in progression systems.

    Maintains latent variables (knowledge, fatigue, engagement, trust) and emits
    platform-like telemetry (accepted items, feedback, dwell time, etc.).
    Implements realistic progression dynamics with position bias, zone of proximal
    development (ZPD), and three-parameter logistic (3PL) response theory.

    Works for any adaptive domain: education (learners), corporate training
    (employees), rehabilitation (patients), fitness (athletes), gaming (players).

    Parameters
    ----------
    user_id : int
        Unique identifier for the user.
    knowledge_dim : int, optional
        Number of knowledge dimensions (default: 10). Used when knowledge_mode="vector".
    knowledge_mode : str, optional
        "scalar" for single ability, "vector" for multi-dimensional (default: "scalar").
    lr : float, optional
        Learning rate for knowledge updates (default: 0.2).
    decay : float, optional
        Knowledge decay/forgetting rate per round (default: 0.1).
    trust_influence : bool, optional
        Whether trust affects fatigue and engagement dynamics (default: True).
    fatigue_growth : float, optional
        Base fatigue growth rate per interaction (default: 0.05).
    fatigue_recovery : float, optional
        Fatigue recovery rate per idle round (default: 0.02).
    base_topk : int, optional
        Base number of items the student wants to review (default: 2).
    min_topk : int, optional
        Minimum items to engage with (default: 1).
    act_mode : str, optional
        Ability model: "IRT", "MIRT", "ZPD", or "ContextualZPD" (default: "ZPD").
    seed : int, optional
        Random seed (default: 42).
    zpd_delta : float, optional
        Target difficulty offset from current ability (default: 0.10).
    zpd_width : float, optional
        Width of zone of proximal development (default: 0.25).
    pos_eta : float, optional
        Position bias exponent, lower = stronger bias (default: 0.85).
    budget_mu : float, optional
        Lognormal mean for attention budget (default: 1.3).
    budget_sigma : float, optional
        Lognormal std for attention budget (default: 0.5).
    budget_scale_eng : float, optional
        Engagement multiplier on attention budget (default: 2.0).
    budget_fatigue_penalty : float, optional
        Fatigue penalty on attention budget (default: 2.0).
    a_mean : float, optional
        Mean discrimination parameter for 3PL (default: 1.2).
    a_std : float, optional
        Std of discrimination parameter (default: 0.2).
    c_alpha : float, optional
        Beta prior alpha for guessing probability (default: 2.0).
    c_beta : float, optional
        Beta prior beta for guessing probability (default: 20.0).
    s_alpha : float, optional
        Beta prior alpha for slip probability (default: 2.0).
    s_beta : float, optional
        Beta prior beta for slip probability (default: 20.0).
    w_rel_mu : float, optional
        Mean relevance weight in utility (default: 1.0).
    w_zpd_mu : float, optional
        Mean ZPD fit weight in utility (default: 0.6).
    w_nov_mu : float, optional
        Mean novelty weight in utility (default: 0.4).
    w_pos_mu : float, optional
        Mean position bias weight in utility (default: 0.6).
    w_std : float, optional
        Std of utility weights across students (default: 0.20).
    forgetting_rate : float, optional
        Per-round knowledge forgetting rate (default: 0.01).
    verbose : bool, optional
        Enable debug logging (default: False).
    """

    def __init__(
        self,
        user_id: int,
        knowledge_dim: int = 10,
        knowledge_mode: str = "scalar",
        learning_rate: float = 0.2,
        decay: float = 0.1,
        trust_influence: bool = True,
        fatigue_growth: float = 0.05,
        fatigue_recovery: float = 0.02,
        base_topk: int = 2,
        min_topk: int = 1,
        ability_model: str = "ZPD",
        seed: int = 42,
        # --- realism knobs ---
        zpd_delta: float = 0.10,
        zpd_width: float = 0.25,
        position_bias: float = 0.85,
        budget_mu: float = 1.3,
        budget_sigma: float = 0.5,
        budget_scale_eng: float = 2.0,
        budget_fatigue_penalty: float = 2.0,
        # 3PL IRT parameters (drawn per agent from priors)
        discrimination_mean: float = 1.2,
        discrimination_std: float = 0.2,
        guess_prior_alpha: float = 2.0,
        guess_prior_beta: float = 20.0,
        slip_prior_alpha: float = 2.0,
        slip_prior_beta: float = 20.0,
        # Acceptance utility weight priors (means; noise added per agent)
        relevance_weight: float = 1.0,
        zpd_fit_weight: float = 0.6,
        novelty_weight: float = 0.4,
        position_weight: float = 0.6,
        weight_noise_std: float = 0.20,
        forgetting_rate: float = 0.01,
        verbose: bool = False,
        *,
        # Backward-compatible abbreviated aliases
        lr: Optional[float] = None,
        act_mode: Optional[str] = None,
        pos_eta: Optional[float] = None,
        a_mean: Optional[float] = None,
        a_std: Optional[float] = None,
        c_alpha: Optional[float] = None,
        c_beta: Optional[float] = None,
        s_alpha: Optional[float] = None,
        s_beta: Optional[float] = None,
        w_rel_mu: Optional[float] = None,
        w_zpd_mu: Optional[float] = None,
        w_nov_mu: Optional[float] = None,
        w_pos_mu: Optional[float] = None,
        w_std: Optional[float] = None,
    ):
        # Resolve backward-compatible aliases
        learning_rate = lr if lr is not None else learning_rate
        ability_model = act_mode if act_mode is not None else ability_model
        position_bias = pos_eta if pos_eta is not None else position_bias
        discrimination_mean = a_mean if a_mean is not None else discrimination_mean
        discrimination_std = a_std if a_std is not None else discrimination_std
        guess_prior_alpha = c_alpha if c_alpha is not None else guess_prior_alpha
        guess_prior_beta = c_beta if c_beta is not None else guess_prior_beta
        slip_prior_alpha = s_alpha if s_alpha is not None else slip_prior_alpha
        slip_prior_beta = s_beta if s_beta is not None else slip_prior_beta
        relevance_weight = w_rel_mu if w_rel_mu is not None else relevance_weight
        zpd_fit_weight = w_zpd_mu if w_zpd_mu is not None else zpd_fit_weight
        novelty_weight = w_nov_mu if w_nov_mu is not None else novelty_weight
        position_weight = w_pos_mu if w_pos_mu is not None else position_weight
        weight_noise_std = w_std if w_std is not None else weight_noise_std

        # --- Input validation ---
        for name, val in [
            ("zpd_delta", zpd_delta),
            ("zpd_width", zpd_width),
            ("position_bias", position_bias),
            ("decay", decay),
            ("fatigue_growth", fatigue_growth),
            ("fatigue_recovery", fatigue_recovery),
            ("forgetting_rate", forgetting_rate),
        ]:
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        for name, val in [
            ("discrimination_std", discrimination_std),
            ("weight_noise_std", weight_noise_std),
            ("budget_sigma", budget_sigma),
        ]:
            if float(val) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        # opt-in per-instance verbosity (in addition to global)
        self.verbose = bool(verbose) or _DEBUG_STUDENT

        self.user_id = int(user_id)
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        # Knowledge representation
        self.knowledge_mode = str(knowledge_mode)
        if self.knowledge_mode == "scalar":
            self.knowledge = float(np.clip(self.rng.rand(), 0.3, 0.7))
        else:
            self.knowledge = np.clip(self.rng.rand(knowledge_dim), 0.3, 0.7).tolist()

        # Latent states (not visible to the recommender)
        self.fatigue = 0.0
        self.trust = 0.5
        self.engagement = 1.0

        # Behavior knobs
        self.lr = float(learning_rate)
        self.decay = float(decay)
        self.trust_influence = bool(trust_influence)
        self.fatigue_growth = float(fatigue_growth)
        self.fatigue_recovery = float(fatigue_recovery)
        self.base_topk = int(base_topk)
        self.min_topk = int(min_topk)
        self.act_mode = str(ability_model)

        # Item difficulty fallback
        self.item_difficulty = defaultdict(lambda: 0.5)

        # History of per-round correctness (for internal updates)
        self._history: List[Dict[int, int]] = []

        # --- realism state ---
        self.zpd_delta = float(zpd_delta)
        self.zpd_width = float(zpd_width)
        self.pos_eta = float(np.clip(position_bias, 0.5, 0.99))
        self.budget_mu = float(budget_mu)
        self.budget_sigma = float(budget_sigma)
        self.budget_scale_eng = float(budget_scale_eng)
        self.budget_fatigue_penalty = float(budget_fatigue_penalty)

        # Per-agent 3PL draws (Item Response Theory parameters)
        self.a = float(max(0.2, self.rng.normal(discrimination_mean, discrimination_std)))
        self.c = float(np.clip(self.rng.beta(guess_prior_alpha, guess_prior_beta), 0.01, 0.35))
        self.s = float(np.clip(self.rng.beta(slip_prior_alpha, slip_prior_beta), 0.01, 0.35))

        # Random coefficients (heterogeneous preferences)
        def _w(mu: float) -> float:
            return float(max(0.0, self.rng.normal(mu, weight_noise_std)))

        self.w_rel = _w(relevance_weight)
        self.w_zpd = _w(zpd_fit_weight)
        self.w_nov = _w(novelty_weight)
        self.w_pos = _w(position_weight)

        # novelty memory
        self.recent = deque(maxlen=200)

        # forgetting
        self.forgetting_rate = float(np.clip(forgetting_rate, 0.0, 0.10))

        # init snapshot
        if self.verbose:
             _p(
                f"init user={self.user_id} mode={self.act_mode} kmode={self.knowledge_mode} "
                f"k={'%.3f'%self._k_mean()} f={'%.3f'%self.fatigue} t={'%.3f'%self.trust} e={'%.3f'%self.engagement} "
                f"a={self.a:.3f} c={self.c:.3f} s={self.s:.3f} "
                f"w(rel/zpd/nov/pos)=({self.w_rel:.3f}/{self.w_zpd:.3f}/{self.w_nov:.3f}/{self.w_pos:.3f}) "
                f"zpdΔ={self.zpd_delta:.3f}"   # <<< NEW
            )

    # ------------------- profile & init -------------------

    def set_initial_latents(
        self,
        *,
        engagement: Optional[float] = None,
        trust: Optional[float] = None,
        knowledge: Optional[float | List[float]] = None,
        fatigue: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Authoritatively set starting latents (used by run_all / RankingExperiment).
        Values are clamped; vector knowledge is respected when knowledge_mode='vector'.
        """
        if engagement is not None:
            self.engagement = float(np.clip(engagement, 0.0, 1.2))
        if trust is not None:
            self.trust = float(np.clip(trust, 0.0, 1.0))
        if fatigue is not None:
            self.fatigue = float(np.clip(fatigue, 0.0, 1.0))
        if knowledge is not None:
            if self.knowledge_mode == "scalar":
                self.knowledge = float(np.clip(_safe(knowledge), 0.0, 1.0))
            else:
                kvec = np.array(knowledge, dtype=float).ravel()
                if kvec.size == 0:
                    kvec = np.zeros_like(np.array(self.knowledge, dtype=float))
                self.knowledge = np.clip(kvec, 0.0, 1.0).tolist()
        if self.verbose:
            _p(
                f"set_initial_latents user={self.user_id} -> "
                f"k={'%.3f'%self._k_mean()} f={'%.3f'%self.fatigue} t={'%.3f'%self.trust} e={'%.3f'%self.engagement}"
            )

    def profile(self) -> Dict[str, Any]:
        """Get a snapshot of the student's current latent state.

        Returns
        -------
        dict
            Profile with keys: user_id, act_mode, knowledge_mode, knowledge,
            fatigue, trust, engagement.
        """
        return {
            "user_id": self.user_id,
            "act_mode": self.act_mode,
            "knowledge_mode": self.knowledge_mode,
            "knowledge": (
                float(self.knowledge)
                if self.knowledge_mode == "scalar"
                else np.array(self.knowledge).tolist()
            ),
            "fatigue": float(self.fatigue),
            "trust": float(self.trust),
            "engagement": float(self.engagement),
        }

    # ------------------- helpers -------------------

    def _k_mean(self) -> float:
        if self.knowledge_mode == "scalar":
            return float(self.knowledge)
        return float(np.mean(np.array(self.knowledge, dtype=float)))

    def _topk_given_state(self, base_k: int) -> int:
        k = max(self.min_topk, int(round(base_k * (1.0 - 0.5 * self.fatigue))))
        if self.trust_influence:
            k = int(round(k * (0.5 + self.trust)))
        return max(self.min_topk, k)

    def _coerce_meta(self, items_meta, item_id: int):
        if not items_meta or item_id not in items_meta:
            _p(f"item={item_id} meta=NONE -> fallback diff={self.item_difficulty[item_id]:.3f}")
            return None
        m = items_meta[item_id]
        if isinstance(m, ItemMeta):
            _p(f"item={item_id} meta=ItemMeta diff={m.difficulty:.3f}")
            return m
        if isinstance(m, dict):
            diff = float(m.get("difficulty", 0.5))
            _p(f"item={item_id} meta=dict diff={diff:.3f}")
            return ItemMeta(difficulty=diff, skills=m.get("skills"))
        return None



    def _ability_scalar(self, item_id: int, items_meta: Optional[Dict[int, ItemMeta]]) -> float:
        if self.knowledge_mode == "scalar":
            return float(self.knowledge)
        m = self._coerce_meta(items_meta, item_id)
        if m and m.skills:
            mask = np.array(m.skills, dtype=bool)
            sel = np.array(self.knowledge)[mask]
            if sel.size > 0:
                return float(sel.mean())
        return float(np.mean(self.knowledge))

    # old logistic (kept for compatibility)
    def _prob_correct(self, theta: float, d: float) -> float:
        alpha, beta = 1.0, 0.5
        logit = alpha * (theta - d) - beta * self.fatigue
        return 1.0 / (1.0 + np.exp(-logit))

    # ---- new realism pieces ----
    def _position_bias(self, rank_zero_based: int) -> float:
        # η^(rank), with rank starting at 0 (0=top)
        return float(self.pos_eta ** float(rank_zero_based))

    def _zpd_match_score(self, theta: float, diff: float, delta: float = None, width: float = None) -> float:
        # bell-shaped score around (theta + delta), normalized ~[0,1]
        # if delta is None:
        #     delta = self.zpd_delta
        # tgt = float(np.clip(theta + delta, 0.0, 1.0))
        # num = (diff - tgt) ** 2
        # den = (0.10**2)
        # return float(max(0.0, 1.0 - min(1.0, num / (den + 1e-9))))

        delta = self.zpd_delta if delta is None else float(delta)
        width = self.zpd_width if width is None else float(max(1e-6, width))
        tgt = float(np.clip(theta + delta, 0.0, 1.0))
        err = diff - tgt
        # Parabolic bell in [0,1]: positive when |err| < width
        # return float(max(0.0, 1.0 - (err * err) / (width * width + 1e-9)))
        return float(np.exp(-0.5 * (err / (width + 1e-9))**2))

    def _base_relevance(self, theta: float, diff: float) -> float:
        # simple logistic relevance pre-3PL (without guess/slip)
        return 1.0 / (1.0 + math.exp(-self.a * (theta - diff)))

    def _novelty(self, item_id: int) -> float:
        return 1.0 if (item_id not in self.recent) else 0.2

    def _sample_budget(self) -> int:
        # lognormal budget scaled by engagement and reduced by fatigue
        base = float(np.exp(self.rng.normal(self.budget_mu, self.budget_sigma)))
        base *= (1.0 + self.budget_scale_eng * float(self.engagement))
        base *= (1.0 - self.budget_fatigue_penalty * float(self.fatigue))
        return max(1, int(round(max(1.0, base))))

    def _gumbel(self, size: int) -> np.ndarray:
        u = self.rng.rand(size)
        return -np.log(-np.log(np.clip(u, 1e-8, 1 - 1e-8)))

    def _prob_correct_3pl(self, theta: float, diff: float) -> float:
        # p = c + (1-c-s)*sigmoid(a(θ-b) - β*fatigue)
        beta_fatigue = 0.5
        logit = self.a * (theta - diff) - beta_fatigue * float(self.fatigue)
        sig = 1.0 / (1.0 + math.exp(-logit))
        return float(np.clip(self.c + (1.0 - self.c - self.s) * sig, 0.0, 1.0))

        self._last_items_meta: Optional[Dict[int, Any]] = None
        self._last_action_ids: List[int] = []

    def _apply_forgetting(self) -> None:
        # decay knowledge a bit each round
        if self.knowledge_mode == "scalar":
            self.knowledge = float(max(0.0, self.knowledge * (1.0 - self.forgetting_rate)))
        else:
            k = np.array(self.knowledge, dtype=float)
            self.knowledge = np.clip(k * (1.0 - self.forgetting_rate), 0.0, 1.0).tolist()

    # ------------------- learning dynamics -------------------

    def _apply_learning_update(
        self,
        item_id: int,
        correct: int,
        p_correct: float,
        items_meta: Optional[Dict[int, ItemMeta]],
    ) -> None:
        if self.knowledge_mode == "scalar":
            delta = self.lr * (correct - p_correct)
            t = max(1, len(self._history))
            delta += self.rng.normal(0, 0.005 * (1.0 / np.sqrt(t)))
            self.knowledge = float(
                np.clip((1 - self.decay) * self.knowledge + delta, 0.0, 1.0)
            )
        else:
            m = self._coerce_meta(items_meta, item_id)
            skills = m.skills if (m and m.skills is not None) else None
            if not skills:
                grad = self.lr * (correct - p_correct)
                self.knowledge = np.clip(
                    (1 - self.decay) * np.array(self.knowledge) + grad, 0.0, 1.0
                ).tolist()
            else:
                kvec = np.array(self.knowledge, dtype=float)
                smask = np.array(skills, dtype=bool)
                grad = np.zeros_like(kvec, dtype=float)
                grad[smask] = self.lr * (correct - p_correct)
                t = max(1, len(self._history))
                grad += self.rng.normal(0, 0.005 * (1.0 / np.sqrt(t)), size=grad.shape)
                self.knowledge = np.clip((1 - self.decay) * kvec + grad, 0.0, 1.0).tolist()

    def _update_latents_after_round(
        self,
        feedback: Dict[int, int],
        items_meta: Optional[Dict[int, ItemMeta]],
    ) -> None:
        if not feedback:
            self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery)
            return

        corrects = sum(feedback.values())
        total = len(feedback)
        acc = corrects / max(1, total)

        # Fatigue recovery, then growth with workload × avg difficulty
        self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery)

        diffs = []
        for i in feedback.keys():
            m = self._coerce_meta(items_meta, i)
            if m is not None:
                diffs.append(float(m.difficulty))
            else:
                diffs.append(float(self.item_difficulty[i]))
        avg_d = float(np.mean(diffs)) if diffs else 0.5

        factor = (1.0 - 0.3 * self.trust) if self.trust_influence else 1.0
        self.fatigue = min(
            1.0, self.fatigue + self.fatigue_growth * total * avg_d * factor
        )

        # Trust & engagement updates
        self.trust = float(np.clip(self.trust + 0.05 * (acc - 0.5), 0.0, 1.0))
        if self.trust_influence:
            self.engagement = float(
                np.clip(
                    self.engagement
                    + 0.1 * (acc - 0.5)
                    - 0.05 * self.fatigue
                    + 0.05 * (self.trust - 0.5),
                    0.2,
                    1.2,
                )
            )
        else:
            self.engagement = float(
                np.clip(
                    self.engagement + 0.1 * (acc - 0.5) - 0.05 * self.fatigue,
                    0.2,
                    1.2,
                )
            )

        # small extra coupling
        try:
            if feedback:
                self.engagement = float(
                    np.clip(self.engagement + 0.04 * (acc - 0.5), 0.2, 1.2)
                )
        except Exception:
            pass

    # ------------------- interaction loop -------------------

    def interact(
        self,
        recommended_ids: List[int],
        items_meta: Optional[Dict[int, ItemMeta]] = None,
        rng_explain_prob: float = 0.15,
    ) -> Dict[str, Any]:
        """Simulate student interaction with a recommended slate of items.

        The student's attention budget, utility preferences, and position bias
        determine which items to engage with. Outcomes are simulated via 3PL
        and learning updates are applied.

        Parameters
        ----------
        recommended_ids : list of int
            Recommended item IDs (in order of preference/ranking).
        items_meta : dict, optional
            Mapping from item ID to ItemMeta. If None, uses default difficulty.
        rng_explain_prob : float, optional
            Probability of viewing explanation when fatigued (default: 0.15).

        Returns
        -------
        dict
            Interaction telemetry with keys:
            - accepted_ids: items the student decided to work on
            - skipped_ids: items not engaged
            - feedback: {item_id: 1 or 0} for correctness on accepted items
            - dwell_s: time spent in seconds
            - latency_s: response time in seconds
            - explanation_viewed: whether explanation was viewed
        """
        # ----- decide how many items the student will actually engage with -----
        k_sys = self._topk_given_state(self.base_topk)  # system’s intended k
        budget = self._sample_budget()  # user's attention budget
        k_take = max(self.min_topk, min(k_sys, budget, len(recommended_ids)))

        R = len(recommended_ids)
        if R == 0:
            if self.verbose:
                _p(f"user={self.user_id} empty slate -> k_sys={k_sys} budget={budget} k_take=0")
            return {
                "accepted_ids": [],
                "skipped_ids": [],
                "feedback": {},
                "dwell_s": float(0.0),
                "latency_s": float(0.0),
                "explanation_viewed": 0,
            }

        # ----- compute utility components for Plackett–Luce -----
        utils = []
        rels, zpds, novs, pposs = [], [], [], []
        thetas, diffs = [], []
        for rnk, item in enumerate(recommended_ids):
            m = self._coerce_meta(items_meta, item)
            d = float(m.difficulty) if m is not None else float(self.item_difficulty[item])
            theta = self._ability_scalar(item, items_meta)
            if self.verbose and rnk < 2:  # only first two to keep logs tidy
                _p(f"user={self.user_id} rnk={rnk} item={item} θ={theta:.3f} d={d:.3f} "
                    f"rel={self._base_relevance(theta,d):.3f} zpd={self._zpd_match_score(theta,d):.3f}")

            rel = self._base_relevance(theta, d)  # [0,1]
            zpd = self._zpd_match_score(theta, d, self.zpd_delta)  # [0,1]
            nov = self._novelty(item)  # {1.0, 0.2}
            ppos = self._position_bias(rnk)  # η^rank

            u = (
                self.w_rel * rel
                + self.w_zpd * zpd
                + self.w_nov * nov
                + self.w_pos * ppos
            )
            utils.append(float(u))
            rels.append(rel); zpds.append(zpd); novs.append(nov); pposs.append(ppos)
            thetas.append(theta); diffs.append(d)

        # Gumbel-Top-k sampling (approx Plackett–Luce)
        g = self._gumbel(R)
        noisy = np.array(utils, dtype=float) + g
        order = noisy.argsort()[::-1].tolist()

        # accept top-k_take
        accepted_idx = order[:k_take]
        accepted = [int(recommended_ids[i]) for i in accepted_idx]
        skipped = [int(recommended_ids[i]) for i in range(R) if i not in accepted_idx]

        # ----- Telemetry -----
        latency_s = max(0.2, self.rng.gamma(shape=2.0 + 3.0 * self.fatigue, scale=0.8))
        dwell_base = 6.0 + 10.0 * self.engagement
        dwell_s = float(max(1.0, self.rng.normal(dwell_base, 2.0)))
        explanation_viewed = int(
            self.rng.rand()
            < rng_explain_prob * (0.8 + 0.4 * (1.0 - self.fatigue))
        )

        # ----- Outcomes with 3PL -----
        feedback: Dict[int, int] = {}
        for item in accepted:
            m = self._coerce_meta(items_meta, item)
            d = float(m.difficulty) if m is not None else float(self.item_difficulty[item])
            theta = self._ability_scalar(item, items_meta)
            p_correct = self._prob_correct_3pl(theta, d)
            correct = 1 if self.rng.rand() < p_correct else 0
            feedback[item] = int(correct)

            # learning update
            self._apply_learning_update(item, correct, p_correct, items_meta)

        # novelty memory
        self.recent.extend(accepted)

        # record + update latents and forgetting
        self._history.append(feedback)
        self._apply_forgetting()
        self._update_latents_after_round(feedback, items_meta)

        # ----- round debug (compact) -----
        if self.verbose:
            try:
                u_arr = np.array(utils, dtype=float)
                msg = (
                    f"user={self.user_id} slate={R} "
                    f"k_sys={k_sys} budget={budget} k_take={k_take} "
                    f"util[μ/σ/min/max]={u_arr.mean():.3f}/{u_arr.std():.3f}/{u_arr.min():.3f}/{u_arr.max():.3f} "
                    f"relμ={np.mean(rels):.3f} zpdμ={np.mean(zpds):.3f} novμ={np.mean(novs):.3f} posμ={np.mean(pposs):.3f} "
                    f"θμ={np.mean(thetas):.3f} dμ={np.mean(diffs):.3f} "
                    f"accepted={len(accepted)} correct={sum(feedback.values())} "
                    f"K={self._k_mean():.3f} F={self.fatigue:.3f} T={self.trust:.3f} E={self.engagement:.3f}"
                )
                _p(msg)
            except Exception as e:
                _p(f"debug print error: {e.__class__.__name__}: {e}")


        return {
            "accepted_ids": accepted,
            "skipped_ids": skipped,
            "feedback": feedback,
            "dwell_s": float(dwell_s),
            "latency_s": float(latency_s),
            "explanation_viewed": int(explanation_viewed),
        }

    # ------------------- reward summary -------------------

    def reward(self, feedback: Dict[int, int]) -> float:
        """Compute a reward signal based on interaction feedback.

        Combines accuracy, fatigue, and engagement into a single reward score.

        Parameters
        ----------
        feedback : dict
            Mapping from item_id to {0, 1} indicating correctness.

        Returns
        -------
        float
            Reward in [0, 1.2], where 0.6*accuracy + 0.25*(1-fatigue) + 0.15*engagement.
        """
        if not feedback:
            acc = 0.0
        else:
            acc = float(np.mean(list(feedback.values())))
        r = float(
            np.clip(
                0.60 * acc
                + 0.25 * (1.0 - float(self.fatigue))
                + 0.15 * float(self.engagement),
                0.0,
                1.2,
            )
        )
        if self.verbose:
            _p(f"user={self.user_id} reward: acc={acc:.3f} -> r={r:.3f} "
               f"(F={self.fatigue:.3f}, E={self.engagement:.3f})")
        return r

    # ------------------------------------------------------------------
    # Legacy compatibility API (act/update)
    # ------------------------------------------------------------------

    def act(
        self,
        decision: Dict[str, Any],
        items_meta: Optional[Dict[int, Any]] = None,
    ) -> Dict[int, int]:
        """
        Backwards-compatible hook for legacy experiments expecting ``act``.

        Parameters
        ----------
        decision:
            Dictionary containing an ``"accepted"`` list (legacy format). Any
            missing list yields an empty interaction.
        items_meta:
            Optional per-item metadata mapping. Dict values may already be
            ``ItemMeta`` instances or plain dictionaries with ``difficulty`` /
            ``skills`` keys.

        Returns
        -------
        Dict[int, int]
            Binary feedback per accepted item (1=correct, 0=incorrect).
        """

        accepted = decision.get("accepted") or decision.get("accepted_ids") or []
        accepted_ids = [int(i) for i in accepted]
        self._last_items_meta = items_meta
        self._last_action_ids = accepted_ids

        feedback: Dict[int, int] = {}
        for item in accepted_ids:
            meta = self._coerce_meta(items_meta, item)
            diff = float(meta.difficulty) if meta is not None else float(self.item_difficulty[item])
            theta = self._ability_scalar(item, items_meta)
            p_correct = self._prob_correct_3pl(theta, diff)
            feedback[item] = int(self.rng.rand() < p_correct)

        return feedback

    def update(
        self,
        feedback: Dict[int, int],
        items_meta: Optional[Dict[int, Any]] = None,
    ) -> None:
        """
        Legacy post-act update. Applies learning updates and latent adjustments
        using the provided feedback (previously produced by :meth:`act`).
        """

        if items_meta is None:
            items_meta = self._last_items_meta

        items_meta = items_meta or {}
        accepted_ids = list(feedback.keys()) if feedback else list(self._last_action_ids)

        for item in accepted_ids:
            meta = self._coerce_meta(items_meta, item)
            diff = float(meta.difficulty) if meta is not None else float(self.item_difficulty[item])
            theta = self._ability_scalar(item, items_meta)
            correct = int(feedback.get(item, 0))
            p_correct = self._prob_correct_3pl(theta, diff)
            self._apply_learning_update(item, correct, p_correct, items_meta)

        if accepted_ids:
            self.recent.extend(accepted_ids)

        self._history.append(dict(feedback))
        self._apply_forgetting()
        self._update_latents_after_round(dict(feedback), items_meta)


class AdaptiveAgentFactory:
    """Factory for creating pre-configured AdaptiveAgent instances.

    Provides convenient factory methods to instantiate agents with different
    ability models (IRT, MIRT, ZPD, etc.).
    """

    _registry = {
        "irt": lambda **kw: AdaptiveAgent(act_mode="IRT", **kw),
        "mirt": lambda **kw: AdaptiveAgent(act_mode="MIRT", knowledge_mode="vector", **kw),
        "zpd": lambda **kw: AdaptiveAgent(act_mode="ZPD", **kw),
        "contextual_zpd": lambda **kw: AdaptiveAgent(act_mode="ContextualZPD", **kw),
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> AdaptiveAgent:
        """Create an AdaptiveAgent of a specific type.

        Parameters
        ----------
        name : str
            Agent type: "irt", "mirt", "zpd", or "contextual_zpd".
        **kwargs
            Additional keyword arguments passed to AdaptiveAgent.__init__.
            Supports initial latent aliases: E, T, K, F, theta.

        Returns
        -------
        AdaptiveAgent
            Configured adaptive agent.

        Raises
        ------
        ValueError
            If name is not a registered type.
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown AdaptiveAgent type '{name}'. Available: {list(cls._registry.keys())}"
            )
        agent = cls._registry[key](**kwargs)
        # If caller passed initial latents in kwargs (common from run_all), apply them now.
        # We look for common keys and aliases.
        init_keys = {"engagement", "trust", "knowledge", "fatigue", "E", "T", "K", "F", "theta"}
        if any(k in kwargs for k in init_keys):
            # normalize a few aliases
            lat = {}
            if "E" in kwargs: lat["engagement"] = kwargs["E"]
            if "T" in kwargs: lat["trust"] = kwargs["T"]
            if "K" in kwargs: lat["knowledge"] = kwargs["K"]
            if "theta" in kwargs: lat["knowledge"] = kwargs["theta"]
            if "F" in kwargs: lat["fatigue"] = kwargs["F"]
            for k in ("engagement", "trust", "knowledge", "fatigue"):
                if k in kwargs:
                    lat[k] = kwargs[k]
            try:
                agent.set_initial_latents(**lat)
            except Exception:
                traceback.print_exc()
        return agent

    @classmethod
    def available(cls) -> List[str]:
        """Return list of available agent types.

        Returns
        -------
        list of str
            Registered factory keys.
        """
        return list(cls._registry.keys())


__all__ = [
    "ItemMeta",
    "AdaptiveAgent",
    "AdaptiveAgentFactory",
    # Backward-compatible aliases (deprecated)
    "StudentAgent",
    "StudentAgentFactory",
    "enable_debug_student_logs",
]


# --- Deprecation handling for renamed symbols (PEP 562) ---
_DEPRECATED_NAMES = {
    "StudentAgent": "AdaptiveAgent",
    "StudentAgentFactory": "AdaptiveAgentFactory",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import warnings
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED_NAMES[name]} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[_DEPRECATED_NAMES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
