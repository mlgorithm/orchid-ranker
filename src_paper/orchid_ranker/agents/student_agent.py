from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# ------------------------ opt-in debug printing ------------------------

import os, sys, traceback

_DEBUG_STUDENT = os.getenv("ORCHID_DEBUG_STUDENT", "1").lower() in {"1", "true", "yes", "on"}

def enable_debug_student_logs(flag: bool = True) -> None:
    """Enable/disable StudentAgent debug logs globally (env var also supported)."""
    global _DEBUG_STUDENT
    _DEBUG_STUDENT = bool(flag)

def _p(*args) -> None:
    if _DEBUG_STUDENT:
        print("[StudentAgent]", *args)

def _safe(v):
    try:
        return float(v)
    except Exception:
        return v

# ----------------------------------------------------------------------





@dataclass
class ItemMeta:
    difficulty: float = 0.5
    skills: Optional[List[int]] = None  # indices or boolean mask


class StudentAgent:
    """
    Simulator of a learner. Maintains latent variables (knowledge, fatigue,
    engagement, trust) and emits only platform-like telemetry.

    Public methods:
      - profile()
      - set_initial_latents(**kwargs)   <-- added (engagement, trust, knowledge, fatigue)
      - set_item_difficulty(...)
      - interact(recommended_ids, items_meta)
      - reward(feedback)
    """

    def __init__(
        self,
        user_id: int,
        knowledge_dim: int = 10,
        knowledge_mode: str = "scalar",  # "scalar" | "vector"
        lr: float = 0.2,
        decay: float = 0.1,
        trust_influence: bool = True,
        fatigue_growth: float = 0.05,
        fatigue_recovery: float = 0.02,
        base_topk: int = 2,
        min_topk: int = 1,
        act_mode: str = "ZPD",  # "IRT" | "MIRT" | "ZPD" | "ContextualZPD"
        seed: int = 42,
        # --- realism knobs ---
        zpd_delta: float = 0.10,  # shift target difficulty toward (θ + delta)
        zpd_width: float = 0.25,  # width of the ZPD
        pos_eta: float = 0.85,  # position bias parameter η (0<η<=1); lower -> stronger bias
        budget_mu: float = 1.3,  # lognormal mean for attention budget
        budget_sigma: float = 0.5,  # lognormal std for attention budget
        budget_scale_eng: float = 2.0,  # engagement multiplier to budget
        budget_fatigue_penalty: float = 2.0,  # fatigue reduces budget
        # 3PL parameters (drawn per-student)
        a_mean: float = 1.2,
        a_std: float = 0.2,  # discrimination
        c_alpha: float = 2.0,
        c_beta: float = 20.0,  # guess prior Beta(α,β)
        s_alpha: float = 2.0,
        s_beta: float = 20.0,  # slip prior Beta(α,β)
        # acceptance utility weight priors (means; noise added per student)
        w_rel_mu: float = 1.0,
        w_zpd_mu: float = 0.6,
        w_nov_mu: float = 0.4,
        w_pos_mu: float = 0.6,
        w_std: float = 0.20,
        forgetting_rate: float = 0.01,  # per-round forgetting of knowledge
        # debug
        verbose: bool = True,         
    
    ):
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
        self.lr = float(lr)
        self.decay = float(decay)
        self.trust_influence = bool(trust_influence)
        self.fatigue_growth = float(fatigue_growth)
        self.fatigue_recovery = float(fatigue_recovery)
        self.base_topk = int(base_topk)
        self.min_topk = int(min_topk)
        self.act_mode = str(act_mode)

        # Item difficulty fallback
        self.item_difficulty = defaultdict(lambda: 0.5)

        # History of per-round correctness (for internal updates)
        self._history: List[Dict[int, int]] = []

        # --- realism state ---
        self.zpd_delta = float(zpd_delta)
        self.zpd_width = float(zpd_width)
        self.pos_eta = float(np.clip(pos_eta, 0.5, 0.99))
        self.budget_mu = float(budget_mu)
        self.budget_sigma = float(budget_sigma)
        self.budget_scale_eng = float(budget_scale_eng)
        self.budget_fatigue_penalty = float(budget_fatigue_penalty)

        # Per-student 3PL draws
        self.a = float(max(0.2, self.rng.normal(a_mean, a_std)))
        self.c = float(np.clip(self.rng.beta(c_alpha, c_beta), 0.01, 0.35))  # guessing floor
        self.s = float(np.clip(self.rng.beta(s_alpha, s_beta), 0.01, 0.35))  # slip

        # Random coefficients (heterogeneous preferences)
        def _w(mu: float) -> float:
            return float(max(0.0, self.rng.normal(mu, w_std)))

        self.w_rel = _w(w_rel_mu)
        self.w_zpd = _w(w_zpd_mu)
        self.w_nov = _w(w_nov_mu)
        self.w_pos = _w(w_pos_mu)

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


class StudentAgentFactory:
    _registry = {
        "irt": lambda **kw: StudentAgent(act_mode="IRT", **kw),
        "mirt": lambda **kw: StudentAgent(act_mode="MIRT", knowledge_mode="vector", **kw),
        "zpd": lambda **kw: StudentAgent(act_mode="ZPD", **kw),
        "contextual_zpd": lambda **kw: StudentAgent(act_mode="ContextualZPD", **kw),
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> StudentAgent:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown StudentAgent type '{name}'. Available: {list(cls._registry.keys())}"
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
        return list(cls._registry.keys())


__all__ = ["ItemMeta", "StudentAgent", "StudentAgentFactory", "enable_debug_student_logs"]
