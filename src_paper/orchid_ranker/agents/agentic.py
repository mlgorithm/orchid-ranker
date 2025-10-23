# src/agents/agentic.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from collections import defaultdict
from orchid_ranker.agents.recommender_agent import JSONLLogger, TwoTowerRecommender, DualRecommender, ItemMeta
from orchid_ranker.agents.student_agent import StudentAgent

# ------------------------------------------------------------------------------------
# Small config + user context
# ------------------------------------------------------------------------------------
@dataclass
class MultiConfig:
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

@dataclass
class UserCtx:
    user_id: int           # external user ID
    user_idx: int          # internal index into user_matrix
    student: Any           # StudentAgent instance
    user_vec: torch.Tensor # [1, Du] — dense side-features row for the user
    profile: Optional[str] = None  # profile tag used for fairness stats
    name: Optional[str] = None

@dataclass
class PolicyState:
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


# ------------------------------------------------------------------------------------
# A tiny “online state” placeholder (kept for compatibility)
# ------------------------------------------------------------------------------------
class OnlineState:
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


# ------------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------------
class MultiUserOrchestrator:
    class _PrivacyDummy:
        def __init__(self):
            self.mia_threshold: float = 0.55

    def __init__(
        self,
        rec: TwoTowerRecommender | DualRecommender,
        users: List[UserCtx],
        item_matrix_normal: torch.Tensor,
        item_matrix_sanitized: Optional[torch.Tensor],
        item_ids_pos: torch.Tensor,
        pos2id: List[int],
        id2pos: Dict[int, int],
        item_meta_by_id: Dict[int, dict],
        cfg: MultiConfig,
        device: torch.device,
        *,
        mode_label: str = "fixed",
        sanitize_user_cols_idx: List[int] = None,
        sanitize_item_cols_idx: List[int] = None,
    ):
        self.rec = rec
        self.users = list(users)
        self.item_matrix = item_matrix_normal
        self.item_matrix_sanitized = item_matrix_sanitized
        self.item_ids_pos = item_ids_pos  # positions [0..N-1]
        self.pos2id = pos2id              # pos -> ext item id
        self.id2pos = id2pos              # ext item id -> pos
        self.item_meta_by_id = item_meta_by_id

        self.cfg = cfg
        self.device = device
        self.state = OnlineState()
        self.priv = self._PrivacyDummy()
        self._sanitize_user_cols_idx = sanitize_user_cols_idx or []
        self._sanitize_item_cols_idx = sanitize_item_cols_idx or []
        self._mode_label = str(mode_label)

        # difficulty map by ext item id
        self._difficulty_map: Dict[int, float] = {}
        for iid, meta in item_meta_by_id.items():
            try:
                self._difficulty_map[int(iid)] = float(meta.get("difficulty", 0.5))
            except Exception:
                self._difficulty_map[int(iid)] = 0.5

        # JSONL logger
        self.logger = JSONLLogger(Path(self.cfg.log_path) if self.cfg.log_path else Path("runs/run.jsonl"))

        # Popularity (online) — exposures & accepts (by ext item id)
        self._pop_expose: Dict[int, int] = {}
        self._pop_accept: Dict[int, int] = {}

        # Per-user memory (ext ids)
        self._seen_by_user: Dict[int, set] = {int(u.user_id): set() for u in self.users}
        self._accepted_by_user: Dict[int, List[int]] = {int(u.user_id): [] for u in self.users}

        # Per-user history centroid in item-embedding space (computed on-demand)
        self._hist_centroid: Dict[int, torch.Tensor] = {}

        # Handy handles
        self._core = rec.teacher if hasattr(rec, "teacher") else rec
        self._device = self.device
        self._user_hist = defaultdict(lambda: {"seen": set(), "mean_diff": 0.5, "n": 0})

        # Engagement-aware bandit statistics (Yang et al. 2024)
        self._ulcb_stats: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"visits": 0.0, "avg_reward": 0.0, "last_engagement": 0.6}
        )
        self._total_visits: float = 0.0
        self._user_profiles: Dict[int, Optional[str]] = {int(u.user_id): u.profile for u in self.users}
        self._group_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"engagement_sum": 0.0, "knowledge_sum": 0.0, "visits": 0.0}
        )
        self._global_engagement_sum: float = 0.0
        self._global_engagement_visits: float = 0.0
        self._global_knowledge_sum: float = 0.0

        # Per-round accumulators
        self._round = {}
        self._reset_round_counters()

        # adaptive policy book-keeping (only used when DualRecommender is provided)
        self._is_adaptive = isinstance(rec, DualRecommender)
        self._policy_state: Dict[int, PolicyState] = {}
        if self._is_adaptive:
            for ux in self.users:
                self._policy_state[int(ux.user_id)] = self._init_policy_state()

        # ----- Paper-aligned State (EWMA + uncertainty scalar) -----
        # Per-user EWMA estimates: k_hat and e_hat
        self._khat: Dict[int, float] = {int(u.user_id): 0.5 for u in self.users}
        self._ehat: Dict[int, float] = {int(u.user_id): 0.6 for u in self.users}
        # per-user uncertainty scalar in [0,1]
        self._uunc: Dict[int, float] = {int(u.user_id): 0.5 for u in self.users}
        # gains (defaults per paper; can be made config knobs later)
        self._eta_k: float = 0.25
        self._eta_e: float = 0.25

    # ------------------ helpers ------------------
    def _reset_round_counters(self):
        """Clear per-round counters before starting a new round."""
        self._round = dict(
            shown=0,
            accepted=0,
            correct=0,
            dwell_sum=0.0,
            dwell_n=0,
            novelty_hits=0,     # number of accepted items that were novel
            novelty_den=0,      # total accepted items (for novelty rate)
            serend_sum=0.0,     # sum of per-user serendipity scores this round
            serend_n=0,
        )

    # ------------------ adaptive policy helpers ------------------
    def _init_policy_state(self) -> PolicyState:
        alpha_lo, alpha_hi = self.cfg.alpha_bounds
        lam_lo, lam_hi = self.cfg.lam_bounds
        k_lo, k_hi = self.cfg.k_bounds
        z_lo, z_hi = self.cfg.zpd_bounds

        alpha_mid = float(np.clip(np.mean([alpha_lo, alpha_hi]), alpha_lo, alpha_hi))
        lam_mid = float(np.clip(self.cfg.mmr_lambda, lam_lo, lam_hi))
        top_mid = int(np.clip(self.cfg.top_k_base, k_lo, k_hi))
        zpd_mid = float(np.clip(self.cfg.zpd_margin, z_lo, z_hi))
        novelty_mid = float(np.clip(self.cfg.novelty_bonus, 0.0, 1.0))
        return PolicyState(
            alpha=alpha_mid,
            lam=lam_mid,
            top_k=top_mid,
            zpd_delta=zpd_mid,
            novelty=novelty_mid,
        )

    def _apply_policy(self, params: PolicyState) -> None:
        targets: List[Any] = []
        if isinstance(self.rec, DualRecommender):
            targets.extend([self.rec.teacher, self.rec.student])
        else:
            targets.append(self.rec)

        for model in targets:
            if model is None:
                continue
            if hasattr(model, "mmr_lambda"):
                model.mmr_lambda = float(params.lam)
            if hasattr(model, "novelty_bonus"):
                model.novelty_bonus = float(params.novelty)
            if hasattr(model, "linucb_alpha"):
                model.linucb_alpha = float(params.alpha)
            if hasattr(model, "ts_alpha"):
                model.ts_alpha = float(max(0.05, params.alpha * 0.6))

    def _policy_next(self, uid: int, state_dict: Dict[str, float]) -> PolicyState:
        """Affine-and-clip dial mapping per paper §3.2 using EWMA state.

        α_t = clip(α0 + c1 u_unc + c2 (ê-0.5), [α_lo, α_hi])
        λ_t = clip(λ0 + d1 (1-ê), [λ_lo, λ_hi])
        K_t = clip(K_base * (0.8 + 0.4 ê), [K_lo, K_hi])
        Δ_t = clip(Δ_base * (0.8 + 0.6 ê), [Δ_lo, Δ_hi])  # mapped to zpd_delta in implementation
        """
        ps = self._policy_state.setdefault(uid, self._init_policy_state())
        alpha_lo, alpha_hi = self.cfg.alpha_bounds
        lam_lo, lam_hi = self.cfg.lam_bounds
        k_lo, k_hi = self.cfg.k_bounds
        z_lo, z_hi = self.cfg.zpd_bounds

        # pull EWMA state
        ehat = float(np.clip(self._ehat.get(uid, 0.6), 0.0, 1.0))
        uunc = float(np.clip(self._uunc.get(uid, 0.5), 0.0, 1.0))

        # base params from config
        alpha0 = float(alpha_lo + 0.5 * (alpha_hi - alpha_lo))
        lam0   = float(self.cfg.mmr_lambda)
        Kbase  = int(self.cfg.top_k_base)
        Dbase  = float(self.cfg.zpd_margin)
        gain   = float(getattr(self.cfg, "policy_gain", 1.0) or 1.0)

        # affine maps (with small, stable coefficients)
        c1, c2 = 0.6 * gain, 0.4 * gain
        d1 = 0.5 * gain

        ps.alpha = float(np.clip(alpha0 + c1 * uunc + c2 * (ehat - 0.5), alpha_lo, alpha_hi))
        ps.lam   = float(np.clip(lam0 + d1 * (1.0 - ehat), lam_lo, lam_hi))
        ps.top_k = int(np.clip(int(round(Kbase * (0.8 + 0.4 * ehat))), k_lo, k_hi))
        ps.zpd_delta = float(np.clip(Dbase * (0.8 + 0.6 * ehat), z_lo, z_hi))
        # novelty follows inverse engagement slightly
        ps.novelty = float(np.clip(self.cfg.novelty_bonus * (1.0 + 0.4 * (0.5 - ehat)), 0.0, 0.9))
        return ps

    def _policy_update_metrics(
        self,
        uid: int,
        accepted_cnt: int,
        correct_cnt: int,
        top_k: int,
        accepted_ids: List[int],
        prev_state: Dict[str, float] | None,
        post_state: Dict[str, float] | None,
    ) -> None:
        ps = self._policy_state.get(uid)
        if ps is None or top_k <= 0:
            return
        beta = 0.3 if ps.rounds < 5 else 0.15
        accept_rate = float(accepted_cnt) / max(1, top_k)
        accuracy = float(correct_cnt) / max(1, accepted_cnt) if accepted_cnt else 0.0
        novelty_obs = self._novelty_user(uid, accepted_ids) if accepted_ids else 0.0
        prev_eng = float(prev_state.get("engagement", 0.6)) if prev_state else 0.6
        post_eng = float(post_state.get("engagement", prev_eng)) if post_state else prev_eng
        engagement_delta = post_eng - prev_eng
        prev_k = float(prev_state.get("knowledge", 0.5)) if prev_state else 0.5
        post_k = float(post_state.get("knowledge", prev_k)) if post_state else prev_k
        knowledge_delta = post_k - prev_k
        knowledge_component = knowledge_delta if knowledge_delta >= 0.0 else 1.5 * knowledge_delta
        reward = (
            0.33 * accuracy
            + 0.24 * accept_rate
            + 0.28 * knowledge_component
            + 0.15 * max(0.0, engagement_delta)
        )

        ps.accept_ma = (1.0 - beta) * ps.accept_ma + beta * accept_rate
        ps.acc_ma = (1.0 - beta) * ps.acc_ma + beta * accuracy
        ps.novelty_ma = (1.0 - beta) * ps.novelty_ma + beta * novelty_obs
        ps.reward_ma = (1.0 - beta) * ps.reward_ma + beta * reward
        ps.knowledge_ma = (1.0 - beta) * ps.knowledge_ma + beta * post_k
        ps.knowledge_delta_ma = (1.0 - beta) * ps.knowledge_delta_ma + beta * knowledge_delta
        ps.rounds += 1

    def _update_user_state_from_sim(self, uid: int, student: StudentAgent) -> None:
        prev = self.state.get(uid)
        prev_k = float(prev.get("knowledge", 0.5)) if prev else 0.5
        prev_eng = float(prev.get("engagement", 0.6)) if prev else 0.6
        try:
            if isinstance(student.knowledge, (list, tuple, np.ndarray)):
                observed_k = float(np.clip(np.mean(student.knowledge), 0.0, 1.0))
            else:
                observed_k = float(np.clip(student.knowledge, 0.0, 1.0))
        except Exception:
            observed_k = prev_k
        knowledge = max(prev_k, observed_k)
        knowledge = float(np.clip(knowledge, 0.0, 1.0))

        fatigue = float(np.clip(getattr(student, "fatigue", 0.2), 0.0, 1.0))
        try:
            if isinstance(student.engagement, (list, tuple, np.ndarray)):
                observed_e = float(np.clip(np.mean(student.engagement), 0.0, 1.0))
            else:
                observed_e = float(np.clip(student.engagement, 0.0, 1.0))
        except Exception:
            observed_e = prev_eng
        engagement = max(prev_eng, observed_e)
        engagement = float(np.clip(engagement, 0.0, 1.0))
        trust = float(np.clip(getattr(student, "trust", 0.5), 0.0, 1.0))
        uncertainty = float(np.clip(0.65 - 0.25 * (trust - 0.5) - 0.20 * (engagement - 0.6), 0.05, 0.95))

        self.state.set_initial(
            uid,
            knowledge=knowledge,
            fatigue=fatigue,
            engagement=engagement,
            trust=trust,
            uncertainty=uncertainty,
        )

    def _ingest_feedback(
        self,
        *,
        user_id: int,
        accepted_ids: list[int],
        feedback: dict[int, int],
        dwell_s: float,
        latency_s: float,
        top_k: int,
        item_meta_by_id: dict[int, dict] | None = None,
    ):
        """
        Update per-round aggregates + per-user novelty/serendipity state.
        - novelty: fraction of accepted items not previously seen by the user
        - serendipity (simple): mean |difficulty - user's running mean difficulty| (pre-update), clipped to [0,1]
        """
        # --------- basic counters ---------
        acc_cnt = int(len(accepted_ids))
        cor_cnt = int(sum(int(v) for v in feedback.values())) if feedback else 0

        self._round["shown"]     += int(top_k)
        self._round["accepted"]  += acc_cnt
        self._round["correct"]   += cor_cnt
        if np.isfinite(dwell_s):
            self._round["dwell_sum"] += float(dwell_s)
            self._round["dwell_n"]   += 1

        # --------- novelty / serendipity ---------
        hist = self._user_hist[user_id]
        seen_before = hist["seen"]
        prev_mean_d = float(hist["mean_diff"])
        prev_n      = int(hist["n"])

        # difficulties for accepted items
        diffs = []
        novel_hits = 0
        if accepted_ids:
            for iid in accepted_ids:
                if iid not in seen_before:
                    novel_hits += 1
                if item_meta_by_id and iid in item_meta_by_id:
                    d = float(item_meta_by_id[iid].get("difficulty", 0.5))
                else:
                    d = 0.5
                d = float(np.clip(d, 0.0, 1.0))
                diffs.append(d)

        # serendipity: mean absolute deviation from user's *previous* mean difficulty
        # (normalize by 0.5 so it roughly lies in [0,1])
        if diffs:
            mad = float(np.mean([abs(d - prev_mean_d) for d in diffs]))
            serend = float(np.clip(mad / 0.5, 0.0, 1.0))
            self._round["serend_sum"] += serend
            self._round["serend_n"]   += 1

        # novelty bookkeeping
        self._round["novelty_hits"] += novel_hits
        self._round["novelty_den"]  += acc_cnt

        # update user history (seen set + running mean difficulty)
        if diffs:
            # running mean update
            m = prev_mean_d
            n = prev_n
            for d in diffs:
                m = (m * n + d) / (n + 1)
                n += 1
            hist["mean_diff"] = float(m)
            hist["n"] = int(n)
        seen_before.update(accepted_ids)

    def _candidate_pool(self, min_candidates: int) -> torch.Tensor:
        N = self.item_matrix.shape[0]
        cand = torch.randperm(N, device=self._device)[:max(min_candidates, 1)]
        return cand.long()

    def _user_state_vec(self, student) -> torch.Tensor:
        # [knowledge, fatigue, trust, engagement] — if knowledge is vector, average it
        if isinstance(student.knowledge, (list, np.ndarray)):
            k = float(np.mean(student.knowledge))
        else:
            k = float(student.knowledge)
        s = torch.tensor([[k, float(student.fatigue), float(student.trust), float(student.engagement)]],
                         dtype=torch.float32, device=self._device)
        return s

    @torch.no_grad()
    def _item_reps(self, ids_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute L2-normalized item embeddings for the given catalog positions.
        Uses the underlying tower (teacher) to ensure consistency with decide().
        """
        model = self._core
        mat = self.item_matrix.to(self._device)
        ids_pos = ids_pos.to(self._device).long()
        I_base = model.item_net(mat.index_select(0, ids_pos))
        I = model.item_proj(I_base) + model.item_emb(ids_pos)
        I = torch.nn.functional.normalize(I, dim=-1)
        return I  # [K, D]

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def _pop_percentile(self, counts: List[int], val: int) -> float:
        if not counts:
            return 0.0
        arr = np.asarray(counts, dtype=float)
        return float((arr <= float(val)).mean())


    # ------------------ metrics ------------------
    def _novelty_user(self, uid: int, items_ext: List[int]) -> float:
        seen = self._seen_by_user.get(int(uid), set())
        if not items_ext:
            return 0.0
        return float(np.mean([1.0 if (i not in seen) else 0.0 for i in items_ext]))

    def _novelty_pop(self, items_ext: List[int]) -> float:
        if not items_ext:
            return 0.0
        counts = list(self._pop_expose.values())
        vals = []
        for iid in items_ext:
            c = int(self._pop_expose.get(int(iid), 0))
            pct = self._pop_percentile(counts, c)  # in [0,1]
            vals.append(1.0 - pct)
        return float(np.mean(vals)) if vals else 0.0

    @torch.no_grad()
    def _serendipity(self, uid: int, item_pos: List[int], logits_1xK: torch.Tensor) -> float:
        """
        Serendipity ≈ mean over accepted items of:
            (1 - cosine_sim(item, user_history_centroid)) * relevance(item)
        Where relevance(item) = sigmoid(model_logit).
        If user has no history, we use serendipity = novelty_pop as a fallback.
        """
        if not item_pos:
            return 0.0
        # relevance per position
        logits = logits_1xK.view(-1)
        # build centroid from previously accepted items (ext ids -> pos)
        hist_ext = self._accepted_by_user.get(int(uid), [])
        if not hist_ext:
            # fallback to pop novelty only
            items_ext = [self.pos2id[p] for p in item_pos]
            return self._novelty_pop(items_ext)

        # cache centroid if not already
        if int(uid) not in self._hist_centroid:
            hist_pos = [self.id2pos.get(int(e), None) for e in hist_ext]
            hist_pos = [p for p in hist_pos if p is not None]
            if not hist_pos:
                items_ext = [self.pos2id[p] for p in item_pos]
                return self._novelty_pop(items_ext)
            H = torch.tensor(hist_pos, dtype=torch.long, device=self._device)
            Hrep = self._item_reps(H)  # [H, D]
            cen = torch.nn.functional.normalize(Hrep.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
            self._hist_centroid[int(uid)] = cen.detach()
        cen = self._hist_centroid[int(uid)]

        # item reps for current items
        P = torch.tensor(item_pos, dtype=torch.long, device=self._device)
        I = self._item_reps(P)  # [K, D]
        sim = (I @ cen.T).squeeze(1)  # [K], cosine (since normalized)
        sim = torch.clamp(sim, -1.0, 1.0)

        rel = self._sigmoid(logits[P]) if logits.numel() >= I.shape[0] else torch.ones(I.shape[0], device=self._device)
        ser = ((1.0 - sim) * rel).mean().item()
        return float(ser)

    # ------------------ main loop ------------------
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def _top_k_for_user(self, engagement: float, trust: float) -> int:
        base = int(self.cfg.top_k_base)
        lo, hi = self.cfg.k_bounds
        adj = base + int(round((engagement - 0.6) * 2.0 + (trust - 0.5)))
        return max(lo, min(hi, adj))

    def _pop_percentile(self, counts: List[int], val: int) -> float:
        if not counts:
            return 0.0
    # 
    def run(self):
        """
        Online simulation loop with rich console printing and JSONL logging.

        CHANGES vs. your current version
        --------------------------------
        • Logs per-student rows with:
            - student_method (ux.student.mode)
            - profile (self._user_profiles[uid])
            - PRE and POST latent state snapshots taken directly from the student
            (knowledge, engagement, trust, fatigue)
        • Uses PRE state for scoring/decisions; POST state is used for telemetry
        and cohort means.
        • Round-summary JSONL is still emitted (type="round_summary"), but the
        per-student table (type="user_round") is now complete for downstream
        flattening.
        """
        import math
        import random
        import numpy as np
        import torch

        # ---- small helper to pull state from the student object ----
        def _extract_student_state(student):
            def _avg(x, default):
                try:
                    if isinstance(x, (list, tuple, np.ndarray)):
                        return float(np.clip(np.mean(x), 0.0, 1.0))
                    return float(np.clip(x, 0.0, 1.0))
                except Exception:
                    return float(default)
            k = _avg(getattr(student, "knowledge", 0.5), 0.5)
            e = _avg(getattr(student, "engagement", 0.6), 0.6)
            t = float(np.clip(getattr(student, "trust", 0.5), 0.0, 1.0))
            f = float(np.clip(getattr(student, "fatigue", 0.2), 0.0, 1.0))
            return {"knowledge": k, "engagement": e, "trust": t, "fatigue": f}

        # ---------- setup ----------
        rounds = int(self.cfg.rounds)
        is_adaptive = hasattr(self.rec, "teacher") and hasattr(self.rec, "student")
        mode_label = "adaptive (teacher+student)" if is_adaptive else "fixed (single model)"

        # DP owner: student if adaptive, otherwise the single model
        dp_owner = self.rec.student if is_adaptive else self.rec
        dp_cfg = getattr(dp_owner, "dp_settings", None)
        dp_enabled  = bool(getattr(dp_cfg, "enabled", False)) if dp_cfg else False
        dp_sigma    = float(getattr(dp_cfg, "noise_multiplier", float("nan"))) if dp_cfg else float("nan")
        dp_q        = float(getattr(dp_cfg, "sample_rate", float("nan"))) if dp_cfg else float("nan")
        dp_delta    = float(getattr(dp_cfg, "delta", float("nan"))) if dp_cfg else float("nan")
        dp_max_grad = float(getattr(dp_cfg, "max_grad_norm", float("nan"))) if dp_cfg else float("nan")

        # Ensure we have a logger
        if not hasattr(self, "logger") or self.logger is None:
            try:
                self.logger = JSONLLogger(Path(self.cfg.log_path))
            except Exception:
                self.logger = None

        # Per-user seen sets and embedding histories (for novelty/serendipity)
        seen_by_user = {int(ux.user_id): set() for ux in self.users}
        emb_hist_by_user = {int(ux.user_id): [] for ux in self.users}   # list of torch tensors (item reps)
        from collections import defaultdict as _dd
        pop_counts = _dd(int)  # optional popularity proxy

        # ---------- header ----------
        if self.cfg.console:
            print("\n--- Orchestrator run info --------------------------------")
            print(f"  policy mode              : {mode_label}")
            print(f"  rounds / base top_k      : {self.cfg.rounds} / {self.cfg.top_k_base}")
            print(f"  zpd_margin               : {self.cfg.zpd_margin}")
            print(f"  mmr_lambda / novelty     : {self.cfg.mmr_lambda} / {self.cfg.novelty_bonus}")
            print(f"  min_candidates           : {self.cfg.min_candidates}")
            print(f"  DP online training       : {dp_enabled}")
            if dp_cfg:
                print(f"  DP sigma / q / delta     : {dp_sigma} / {dp_q} / {dp_delta}")
                print(f"  DP max_grad_norm         : {dp_max_grad}")
            else:
                print("  DP config                : <none>")
            print(f"  log_path                 : {self.cfg.log_path}")
            print("-----------------------------------------------------------\n")

        eps_cum = float(getattr(dp_owner, "eps_cum", 0.0) or 0.0)

        # ---------- main loop ----------
        for r in range(1, rounds + 1):
            # round accumulators
            shown_total = 0
            accept_total = 0
            correct_total = 0
            novel_hits = 0
            novel_den  = 0
            serend_sum = 0.0
            serend_den = 0

            # cohort means of k/f/e/t and Accept@4 (we’ll use POST for means)
            sum_k = sum_f = sum_e = sum_t = 0.0
            n_users = 0
            sum_accept_at4 = 0.0
            n_users_accept_at4 = 0

            # shuffle users per config
            order = list(range(len(self.users)))
            if self.cfg.shuffle_users_each_round:
                random.shuffle(order)

            # Optional DP per-round budget
            if dp_enabled and hasattr(self.rec, "begin_dp_step") and callable(self.rec.begin_dp_step):
                eps_target = float(getattr(self.cfg, "per_round_eps_target", 0.0) or 0.0)
                try:
                    self.rec.begin_dp_step(eps_target)
                except Exception:
                    pass

            for idx in order:
                ux = self.users[idx]
                uid_ext = int(ux.user_id)
                uid = int(ux.user_idx)

                # ---- PRE state from the student (authoritative) ----
                pre = _extract_student_state(ux.student)
                k_val, f_val, t_val, e_val = pre["knowledge"], pre["fatigue"], pre["trust"], pre["engagement"]

                # state vector for the recommender
                state_vec = torch.tensor([[k_val, f_val, t_val, e_val]], dtype=torch.float32, device=self.device)

                # ---- candidate sampling (optionally deterministic per round) ----
                cand_pos_all = self.item_ids_pos
                Kmin = int(self.cfg.min_candidates)
                if cand_pos_all.numel() > Kmin:
                        # Optional persistent cache (shared across modes within a run directory)
                        used_cache = False
                        cache_enabled = bool(getattr(self.cfg, "persistent_pool", False))
                        cache_dir_opt = getattr(self.cfg, "pool_cache_dir", None)
                        cache_dir = None
                        cache_file = None
                        if cache_enabled:
                            try:
                                base_dir = Path(self.cfg.log_path).parent if self.cfg.log_path else Path("runs")
                                cache_dir = Path(cache_dir_opt) if cache_dir_opt else (base_dir / "candidate_pools")
                                cache_dir.mkdir(parents=True, exist_ok=True)
                                cache_file = cache_dir / f"round_{int(r):04d}_k{Kmin}.npy"
                                if cache_file.exists():
                                    arr = np.load(str(cache_file))
                                    if isinstance(arr, np.ndarray) and arr.size == Kmin:
                                        arr = arr.astype(np.int64)
                                        idxs = torch.tensor(arr, dtype=torch.long, device=cand_pos_all.device)
                                        cand_pos = cand_pos_all.index_select(0, idxs)
                                        used_cache = True
                            except Exception:
                                used_cache = False

                        if not used_cache:
                            if bool(getattr(self.cfg, "deterministic_pool", False)):
                                try:
                                    g = torch.Generator(device=cand_pos_all.device)
                                    # per-round seed ensures same pool across modes/runs with same config
                                    g.manual_seed(int(getattr(self.cfg, "pool_seed", 12345)) + int(r))
                                    perm = torch.randperm(cand_pos_all.numel(), generator=g, device=cand_pos_all.device)
                                except Exception:
                                    # safe fallback
                                    perm = torch.randperm(cand_pos_all.numel(), device=cand_pos_all.device)
                            else:
                                perm = torch.randperm(cand_pos_all.numel(), device=cand_pos_all.device)
                            idxs = perm[:Kmin]
                            cand_pos = cand_pos_all.index_select(0, idxs)

                            # Save to cache if enabled
                            if cache_enabled and cache_file is not None:
                                try:
                                    np.save(str(cache_file), idxs.detach().cpu().numpy())
                                except Exception:
                                    pass
                else:
                    cand_pos = cand_pos_all
                cand_item_ids = cand_pos.long().to(self.device)

                # ---- user tensors ----
                user_ids = torch.tensor([uid], dtype=torch.long, device=self.device)
                if hasattr(self.rec, "user_matrix") and isinstance(self.rec.user_matrix, torch.Tensor):
                    user_vec = self.rec.user_matrix[uid].unsqueeze(0).to(self.device)
                else:
                    user_vec = None  # let model fetch internally

                # ---- score BEFORE decide ----
                logits = self.rec.infer(
                    user_vec=user_vec,
                    item_matrix=self.item_matrix,
                    user_ids=user_ids,
                    item_ids=cand_item_ids,
                    state_vec=state_vec,
                )

                # ---- choose slate ----
                params = None
                if self._is_adaptive:
                    # update EWMA uncertainty scalar u_unc from model widths on current slate
                    try:
                        scorer = self.rec.teacher if is_adaptive else self.rec
                        if hasattr(scorer, "_last_item_ids") and hasattr(scorer, "_last_phi_np"):
                            ids_list = list(getattr(scorer, "_last_item_ids", []) or [])
                            phi_np = getattr(scorer, "_last_phi_np", None)
                            if ids_list and phi_np is not None and hasattr(scorer, "uncertainty_widths"):
                                widths = scorer.uncertainty_widths(phi_np, ids_list)
                                # normalize widths to [0,1]
                                w = np.array(widths, dtype=float)
                                w = np.where(np.isfinite(w), w, 0.0)
                                w = (w - w.min()) / (w.ptp() + 1e-6)
                                self._uunc[uid_ext] = float(np.clip(np.percentile(w, 75), 0.0, 1.0))
                    except Exception:
                        pass

                    params = self._policy_next(uid_ext, dict(pre))
                    self._apply_policy(params)
                    top_k = int(params.top_k)
                    zpd_delta = float(params.zpd_delta)
                else:
                    top_k = int(self.cfg.top_k_base)
                    zpd_delta = float(self.cfg.zpd_margin)

                sel_pos, _ = self.rec.decide(
                    logits=logits,
                    top_k=top_k,
                    item_ids=cand_item_ids,
                    user_id=uid_ext,
                    engagement=e_val,
                    trust=t_val,
                    difficulty_map=self._difficulty_map,
                    knowledge=k_val,
                    zpd_delta=zpd_delta,
                )

                chosen_item_positions = sel_pos
                chosen_item_ids = [int(self.pos2id[int(p)]) for p in chosen_item_positions]
                shown_total += len(chosen_item_ids)

                # Try to fetch item reps for serendipity
                item_reps = None
                try:
                    scorer = self.rec.teacher if is_adaptive else self.rec
                    if hasattr(scorer, "_last_item_ids") and hasattr(scorer, "_last_item_reps"):
                        last_ids = list(getattr(scorer, "_last_item_ids", []) or [])
                        last_reps = getattr(scorer, "_last_item_reps", None)
                        if last_reps is not None and last_ids:
                            idxs = [last_ids.index(int(p)) for p in chosen_item_positions if int(p) in last_ids]
                            if idxs:
                                item_reps = last_reps[idxs].detach()  # [k, D]
                except Exception:
                    item_reps = None

                # ---- simulate student interaction ----
                out = ux.student.interact(
                    recommended_ids=chosen_item_ids,
                    items_meta=self.item_meta_by_id,
                    rng_explain_prob=0.15,
                )

                accepted_ids = list(out.get("accepted_ids", []) or [])
                fb = dict(out.get("feedback", {}) or {})
                dwell_s = float(out.get("dwell_s", 0.0))
                latency_s = float(out.get("latency_s", 0.0))

                accepted_cnt = len(accepted_ids)
                correct_cnt = int(sum(fb.values())) if fb else 0
                accept_total += accepted_cnt
                correct_total += correct_cnt

                # ---- Update EWMAs (paper Eq. 1 & 2 with β≡1) ----
                # acceptance event a_t: 1 if accepted_cnt>0 (shown per-user slate), else 0
                a_t = 1.0 if accepted_cnt > 0 else 0.0
                y_t = (float(correct_cnt) / max(1, accepted_cnt)) if accepted_cnt else 0.0
                self._khat[uid_ext] = (1.0 - self._eta_k) * self._khat.get(uid_ext, 0.5) + self._eta_k * y_t
                self._ehat[uid_ext] = (1.0 - self._eta_e) * self._ehat.get(uid_ext, 0.6) + self._eta_e * a_t

                # ---- POST state from the student after interact() ----
                self._update_user_state_from_sim(uid_ext, ux.student)  # keep if OnlineState is used elsewhere
                post = _extract_student_state(ux.student)

                # cohort means use POST
                sum_k += post["knowledge"]; sum_f += post["fatigue"]; sum_e += post["engagement"]; sum_t += post["trust"]
                n_users += 1

                # bandit stats / rewards (unchanged in spirit)
                prev_state = dict(pre)
                post_state = dict(post)
                prev_eng = float(prev_state.get("engagement", e_val))
                post_eng = float(post_state.get("engagement", prev_eng))
                prev_k = float(prev_state.get("knowledge", k_val))
                post_k = float(post_state.get("knowledge", prev_k))
                knowledge_delta = post_k - prev_k
                engagement_delta = post_eng - prev_eng
                top_k_eff = max(1, len(chosen_item_ids))
                accept_rate_chosen = accepted_cnt / top_k_eff
                accuracy_user = float(correct_cnt) / max(1, accepted_cnt) if accepted_cnt else 0.0
                knowledge_component = knowledge_delta if knowledge_delta >= 0.0 else 1.5 * knowledge_delta
                reward_signal = (
                    0.33 * accuracy_user
                    + 0.24 * accept_rate_chosen
                    + 0.28 * knowledge_component
                    + 0.15 * max(0.0, engagement_delta)
                )
                stats = self._ulcb_stats[uid_ext]
                visits_prev = stats["visits"]
                visits_new = visits_prev + 1.0
                stats["avg_reward"] = (stats["avg_reward"] * visits_prev + reward_signal) / max(1.0, visits_new)
                stats["visits"] = visits_new
                stats["last_engagement"] = post_eng
                self._total_visits += 1.0
                profile_key = self._user_profiles.get(uid_ext)
                if profile_key:
                    grp = self._group_metrics[profile_key]
                    grp["engagement_sum"] += post_eng
                    grp["knowledge_sum"] += post_k
                    grp["visits"] += 1.0
                self._global_engagement_sum += post_eng
                self._global_engagement_visits += 1.0
                self._global_knowledge_sum += post_k
                if self._is_adaptive:
                    self._policy_update_metrics(
                        uid_ext, accepted_cnt, correct_cnt, top_k, accepted_ids, prev_state, post_state
                    )

                # engagement metrics (per student)
                accept_at4 = (accepted_cnt / max(1, min(len(chosen_item_ids), 4))) if chosen_item_ids else float("nan")
                if not math.isnan(accept_at4):
                    sum_accept_at4 += accept_at4
                    n_users_accept_at4 += 1

                # novelty (per student)
                prev_seen = seen_by_user[uid_ext]
                novelty_rate_user = float("nan")
                if accepted_cnt:
                    novel_flags = [(1.0 if (iid not in prev_seen) else 0.0) for iid in accepted_ids]
                    novelty_rate_user = float(np.mean(novel_flags))

                # serendipity (per student)
                serend_user = float("nan")
                try:
                    if accepted_cnt and (item_reps is not None) and (item_reps.shape[0] == len(chosen_item_ids)):
                        acc_pos = [chosen_item_ids.index(a) for a in accepted_ids if a in chosen_item_ids]
                        if acc_pos:
                            acc_reps = item_reps[acc_pos]
                            hist = emb_hist_by_user[uid_ext]
                            if hist:
                                H = torch.stack(hist, dim=0)
                                H = H / (H.norm(dim=1, keepdim=True) + 1e-8)
                                A = acc_reps / (acc_reps.norm(dim=1, keepdim=True) + 1e-8)
                                sims = torch.einsum("ad,md->am", A, H)
                                max_sim, _ = sims.max(dim=1)
                                serend_user = float((1.0 - max_sim.clamp(0.0, 1.0)).mean().item())
                except Exception:
                    pass

                # ---------- PER-STUDENT JSONL LOG ----------
                    # ---- PER-STUDENT JSONL LOG (flat + nested) ----
                if self.logger is not None:
                    self._log_user_round(
                        r=r,
                        uid_ext=uid_ext,
                        ux=ux,
                        mode_label=mode_label,
                        dp_enabled=dp_enabled,
                        eps_cum=float(getattr(dp_owner, "eps_cum", eps_cum)),
                        top_k=int(top_k),
                        zpd_delta=float(zpd_delta),
                        mmr_lambda=float(params.lam if params else self.cfg.mmr_lambda),
                        novelty_bonus=float(params.novelty if params else self.cfg.novelty_bonus),
                        chosen_item_ids=chosen_item_ids,
                        accepted_ids=accepted_ids,
                        correct_cnt=correct_cnt,
                        dwell_s=dwell_s,
                        latency_s=latency_s,
                        novelty_rate_user=(None if not accepted_ids else float(novelty_rate_user)),
                        serend_user=(None if math.isnan(serend_user) else float(serend_user)) if isinstance(serend_user, float) else None,
                        pre={**pre, "khat": float(self._khat.get(uid_ext, 0.5)), "ehat": float(self._ehat.get(uid_ext, 0.6)), "u_unc": float(self._uunc.get(uid_ext, 0.5))},
                        post={**post, "khat": float(self._khat.get(uid_ext, 0.5)), "ehat": float(self._ehat.get(uid_ext, 0.6)), "u_unc": float(self._uunc.get(uid_ext, 0.5))},
                    )


                # ----------- PER-STUDENT CONSOLE PRINT -----------
                if self.cfg.console and getattr(self.cfg, "console_user", True):
                    print(
                        f"R{r:03d} | U{uid_ext:<7d} | "
                        f"top_k={top_k:<2d} "
                        f"pre(k/f/e/t)={k_val:0.2f}/{f_val:0.2f}/{e_val:0.2f}/{t_val:0.2f} "
                        f"post(k/e)={post['knowledge']:0.2f}/{post['engagement']:0.2f} | "
                        f"shown={len(chosen_item_ids):<2d} acc={accepted_cnt:<2d} cor={correct_cnt:<2d} "
                        f"acc_rate={(accepted_cnt / max(1, len(chosen_item_ids))):0.3f} at4={(accept_at4 if not math.isnan(accept_at4) else float('nan')):0.3f} "
                        f"novel={('nan' if not accepted_cnt else f'{novelty_rate_user:0.3f}')} "
                        f"serend={('nan' if math.isnan(serend_user) else f'{serend_user:0.3f}')} "
                        f"dwell={dwell_s:0.1f}s lat={latency_s:0.1f}s"
                    )

                # ---- cohort-level novelty/serendipity accounting (after logging) ----
                if accepted_ids:
                    for iid in accepted_ids:
                        if iid not in prev_seen:
                            novel_hits += 1
                        prev_seen.add(iid)
                        pop_counts[iid] += 1
                    novel_den += len(accepted_ids)

                    if (item_reps is not None) and (item_reps.shape[0] == len(chosen_item_ids)):
                        acc_pos = [chosen_item_ids.index(a) for a in accepted_ids if a in chosen_item_ids]
                        if acc_pos:
                            acc_reps = item_reps[acc_pos]
                            hist = emb_hist_by_user[uid_ext]
                            if hist:
                                try:
                                    H = torch.stack(hist, dim=0)
                                    H = H / (H.norm(dim=1, keepdim=True) + 1e-8)
                                    A = acc_reps / (acc_reps.norm(dim=1, keepdim=True) + 1e-8)
                                    sims = torch.einsum("ad,md->am", A, H)
                                    max_sim, _ = sims.max(dim=1)
                                    ser = (1.0 - max_sim.clamp(0.0, 1.0)).mean().item()
                                    serend_sum += float(ser) * len(acc_pos)
                                    serend_den += len(acc_pos)
                                except Exception:
                                    pass
                            # extend history with current accepted reps (cap memory)
                            try:
                                for t in acc_reps:
                                    emb_hist_by_user[uid_ext].append(t.detach())
                                if len(emb_hist_by_user[uid_ext]) > 256:
                                    emb_hist_by_user[uid_ext] = emb_hist_by_user[uid_ext][-256:]
                            except Exception:
                                pass

                # ---- estimator update / round-level bookkeeping ----
                self._ingest_feedback(
                    user_id=uid_ext,
                    accepted_ids=accepted_ids,
                    feedback=fb,
                    dwell_s=dwell_s,
                    latency_s=latency_s,
                    top_k=int(top_k),
                    item_meta_by_id=self.item_meta_by_id,
                )

                # ---- online training step ----
                if fb:
                    pos = [self.id2pos[i] for i in fb.keys() if i in self.id2pos]
                    if pos:
                        batch = {
                            "user_ids": user_ids.expand(len(pos)),
                            "item_ids": torch.tensor(pos, dtype=torch.long, device=self.device),
                            "labels": torch.tensor([fb[self.pos2id[p]] for p in pos], dtype=torch.float32, device=self.device),
                            "item_matrix": self.item_matrix,
                            "state_vec": state_vec.expand(len(pos), -1),
                        }
                        if hasattr(self.rec, "train_step") and callable(self.rec.train_step):
                            self.rec.train_step(batch)

            # round-level metrics
            acc_rate = (correct_total / max(1, accept_total)) if accept_total else 0.0
            accept_rate = (accept_total / max(1, shown_total)) if shown_total else 0.0
            novelty_rate = (novel_hits / max(1, novel_den)) if novel_den else 0.0
            serendipity = (serend_sum / max(1, serend_den)) if serend_den else float("nan")
            mean_k = (sum_k / max(1, n_users))
            mean_f = (sum_f / max(1, n_users))
            mean_e = (sum_e / max(1, n_users))
            mean_t = (sum_t / max(1, n_users))
            mean_accept_at4 = (sum_accept_at4 / max(1, n_users_accept_at4)) if n_users_accept_at4 else float("nan")

            # update epsilon cumulative (if available)
            eps_cum = float(getattr(dp_owner, "eps_cum", eps_cum))

            # ---------- console round summary ----------
            if self.cfg.console:
                print(
                    f"Round {r:03d}/{rounds} | "
                    f"mode={self._mode_label.upper()} "
                    f"DP={'ON' if dp_enabled else 'OFF'} "
                    f"eps_cum={eps_cum:.3f} | "
                    f"shown={shown_total} acc={accept_total} cor={correct_total} "
                    f"acc_rate={acc_rate:.3f} accept_rate={accept_rate:.3f} at4={(mean_accept_at4 if not math.isnan(mean_accept_at4) else float('nan')):0.3f} "
                    f"novel={novelty_rate:.3f} serend={('nan' if math.isnan(serendipity) else f'{serendipity:.3f}')} | "
                    f"mean k/f/e/t = {mean_k:.2f}/{mean_f:.2f}/{mean_e:.2f}/{mean_t:.2f}"
                )

            # ---------- JSONL round summary ----------
            if self.logger is not None:
                try:
                    self.logger.log({
                        "type": "round_summary",
                        "round": int(r),
                        "mode": self._mode_label,
                        "dp": {
                            "enabled": bool(dp_enabled),
                            "sigma": float(dp_sigma),
                            "sample_rate": float(dp_q),
                            "delta": float(dp_delta),
                            "max_grad_norm": float(dp_max_grad),
                            "epsilon_cum": float(eps_cum),
                        },
                        "metrics": {
                            "shown": int(shown_total),
                            "accepted": int(accept_total),
                            "correct": int(correct_total),
                            "accept_rate": float(accept_rate),
                            "accuracy": float(acc_rate),
                            "accept_at4": (None if math.isnan(mean_accept_at4) else float(mean_accept_at4)),
                            "novelty_rate": float(novelty_rate),
                            "serendipity": (None if math.isnan(serendipity) else float(serendipity)),
                            "mean_knowledge": float(mean_k),
                            "mean_fatigue": float(mean_f),
                            "mean_engagement": float(mean_e),
                            "mean_trust": float(mean_t),
                        },
                    })
                except Exception:
                    pass

        return {"epsilon_cum": float(eps_cum)}

    # add this tiny helper inside MultiUserOrchestrator (top of class)
    def _extract_student_state(self, student):
        import numpy as np
        def _avg(x, default):
            try:
                if isinstance(x, (list, tuple, np.ndarray)):
                    return float(np.clip(np.mean(x), 0.0, 1.0))
                return float(np.clip(x, 0.0, 1.0))
            except Exception:
                return float(default)
        k = _avg(getattr(student, "knowledge", 0.5), 0.5)
        e = _avg(getattr(student, "engagement", 0.6), 0.6)
        t = float(np.clip(getattr(student, "trust", 0.5), 0.0, 1.0))
        f = float(np.clip(getattr(student, "fatigue", 0.2), 0.0, 1.0))
        return {"knowledge": k, "engagement": e, "trust": t, "fatigue": f}
    

    def _log_user_round(
        self,
        *,
        r: int,
        uid_ext: int,
        ux: "UserCtx",
        mode_label: str,
        dp_enabled: bool,
        eps_cum: float,
        top_k: int,
        zpd_delta: float,
        mmr_lambda: float,
        novelty_bonus: float,
        chosen_item_ids: list[int],
        accepted_ids: list[int],
        correct_cnt: int,
        dwell_s: float,
        latency_s: float,
        novelty_rate_user: float | None,
        serend_user: float | None,
        pre: dict,
        post: dict,
    ) -> None:
        # derive telemetry
        shown = int(len(chosen_item_ids))
        accepted = int(len(accepted_ids))
        accept_rate = float(accepted) / max(1, shown)
        accept_at4 = float(accepted) / max(1, min(shown, 4)) if shown else None

        # names
        user_name = getattr(ux, "name", None) or f"user_{uid_ext}"
        student_method = str(getattr(ux.student, "act_mode", "unknown")).lower()
        profile = self._user_profiles.get(uid_ext)

        # flattened record (plus nested blocks for backwards-compat)
        record = {
            "type": "user_round",
            "round": int(r),
            "mode": self._mode_label,              # e.g., "adaptive"
            "user_id": int(uid_ext),
            "user_name": user_name,
            "student_method": student_method,
            "profile": profile,

            # telemetry (flat)
            "tel_shown": shown,
            "tel_accepted": accepted,
            "tel_correct": int(correct_cnt),
            "tel_accept_rate": float(accept_rate),
            "accept_at4": (None if accept_at4 is None else float(accept_at4)),
            "novelty_rate": (None if novelty_rate_user is None else float(novelty_rate_user)),
            "serendipity": (None if serend_user is None else float(serend_user)),
            "dwell_s": float(dwell_s),
            "latency_s": float(latency_s),

            # knobs (flat)
            "knob_top_k": int(top_k),
            "knob_zpd_margin": float(zpd_delta),
            "knob_mmr_lambda": float(mmr_lambda),
            "knob_novelty_bonus": float(novelty_bonus),

            # pre/post state (flat)
            "pre_knowledge": float(pre.get("knowledge", 0.5)),
            "pre_engagement": float(pre.get("engagement", 0.6)),
            "pre_trust": float(pre.get("trust", 0.5)),
            "pre_fatigue": float(pre.get("fatigue", 0.2)),
            "post_knowledge": float(post.get("knowledge", 0.5)),
            "post_engagement": float(post.get("engagement", 0.6)),
            "post_trust": float(post.get("trust", 0.5)),
            "post_fatigue": float(post.get("fatigue", 0.2)),

            # keep nested blocks too (RankingExperiment flatteners may already expect them)
            "dp": {
                "enabled": bool(dp_enabled),
                "epsilon_cum": float(eps_cum),
            },
            "knobs": {
                "top_k": int(top_k),
                "zpd_margin": float(zpd_delta),
                "mmr_lambda": float(mmr_lambda),
                "novelty_bonus": float(novelty_bonus),
            },
            "state_estimator": {
                "pre":  dict(pre),
                "post": dict(post),
            },
            "telemetry": {
                "shown": shown,
                "accepted": accepted,
                "correct": int(correct_cnt),
                "accept_rate": float(accept_rate),
                "accept_at4": (None if accept_at4 is None else float(accept_at4)),
                "dwell_s": float(dwell_s),
                "latency_s": float(latency_s),
                "novelty_rate": (None if novelty_rate_user is None else float(novelty_rate_user)),
                "serendipity": (None if serend_user is None else float(serend_user)),
            },
            "slate": {
                "chosen_item_ids": [int(x) for x in chosen_item_ids],
                "accepted_item_ids": [int(x) for x in accepted_ids],
            },
        }
        try:
            self.logger.log(record)
        except Exception:
            pass

