"""Two-tower neural recommender with state-dependent gating and diversity mechanisms."""

from __future__ import annotations

import copy
import logging
import math
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from orchid_ranker.agents.policies import BootTS, LinUCBPolicy
from orchid_ranker.agents.simple_dp import SimpleDPConfig
from orchid_ranker.dp_accountant import build_accountant
from orchid_ranker.native.fast_score import fast_score
from orchid_ranker.security import AuditLogger

logger = logging.getLogger(__name__)


# Global debug logging
_DEBUG_REC = os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}


def enable_debug_rec_logs(flag: bool = True) -> None:
    """Enable/disable extra recommender debug logs globally."""
    global _DEBUG_REC
    _DEBUG_REC = bool(flag)


def _d(*args) -> None:
    """Debug logging helper."""
    if _DEBUG_REC:
        logger.debug("%s", " ".join(str(a) for a in args))


class TwoTowerRecommender(nn.Module):
    """Two-tower neural recommender with state-dependent gating and diversity mechanisms.

    Dual encoder architecture with FiLM gating for online learning, MMR diversity, and
    zone-of-proximal-development constraints.

    Parameters
    ----------
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    user_dim : int
        User feature dimension.
    item_dim : int
        Item feature dimension.
    hidden : int, optional
        Hidden layer dimension (default: 64).
    emb_dim : int, optional
        Embedding dimension (default: 32).
    state_dim : int, optional
        Cognitive state dimension [knowledge, fatigue, trust, engagement] (default: 4).
    lr : float, optional
        Learning rate (default: 1e-2).
    seed : int, optional
        Random seed (default: 42).
    dp_cfg : dict, optional
        Differential privacy config (default: None).
    mmr_lambda : float, optional
        Relevance–diversity tradeoff for Maximal Marginal Relevance (MMR).
        0.0 = pure relevance (no diversity), 1.0 = pure diversity (ignore
        relevance scores). Values around 0.2–0.4 work well in practice.
        Default: 0.3.
    novelty_bonus : float, optional
        Novelty bonus factor (default: 0.10).
    zpd_width : float, optional
        Zone of proximal development width (default: 0.10).
    zpd_weight : float, optional
        ZPD constraint weight (default: 0.05).
    sigma_min : float, optional
        Minimum uncertainty/noise (default: 0.05).
    item_bias : bool, optional
        Whether to use per-item bias (default: True).
    device : str, optional
        Torch device (default: "cpu").
    use_native_scoring : bool, optional
        Use native C++ scoring kernel (default: False).
    compile : bool, optional
        Enable torch.compile graph optimization on the scoring path.
        Requires PyTorch >= 2.0 and a CUDA device (not supported on MPS).
        Can provide 20-40% inference latency reduction (default: False).
    **kwargs
        Additional parameters (num_cohorts, adapter_slots, etc.).
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_dim: int,
        item_dim: int,
        hidden: int = 64,
        emb_dim: int = 32,
        state_dim: int = 4,   # [knowledge, fatigue, trust, engagement]
        lr: float = 1e-2,
        seed: int = 42,
        dp_cfg: Optional[dict] = None,
        mmr_lambda: float = 0.3,
        novelty_bonus: float = 0.10,
        zpd_width: float = 0.10,
        zpd_weight: float = 0.05,
        sigma_min: float = 0.05,
        item_bias: bool = True,
        device: str = "cpu",
        use_native_scoring: bool = False,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__()
        torch.manual_seed(seed)

        # --- validate dimensions ---
        for name, val in [("num_users", num_users), ("num_items", num_items),
                          ("user_dim", user_dim), ("item_dim", item_dim),
                          ("hidden", hidden), ("emb_dim", emb_dim)]:
            if int(val) <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        # --- model dims/config ---
        self.user_dim = int(user_dim)
        self.item_dim = int(item_dim)
        self.hidden = int(hidden)
        self.emb_dim = int(emb_dim)
        self.state_dim = int(state_dim)
        self.mmr_lambda = float(mmr_lambda)
        self.novelty_bonus = float(novelty_bonus)
        # paper-aligned knobs (configurable)
        self.zpd_width = float(max(1e-6, zpd_width))
        self.zpd_weight = float(max(0.0, zpd_weight))
        self.sigma_min = float(max(1e-6, sigma_min))
        self.item_bias_enabled = bool(item_bias)
        self._extra_kwargs = dict(kwargs) if kwargs else {}
        self.use_native_scoring = bool(use_native_scoring)

        _d(f"TwoTower init users={num_users} items={num_items} Du={user_dim} Di={item_dim} "
           f"H={hidden} D={emb_dim} state_dim={state_dim} mmr={mmr_lambda} nov={novelty_bonus} lr={lr}")

        # --- towers ---
        self.user_net = nn.Sequential(
            nn.Linear(self.user_dim + self.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.item_net = nn.Sequential(
            nn.Linear(self.item_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # --- FiLM gating from state_vec ---
        self.state_to_gate = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden * 2),
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden * 2),
        )

        # optional state -> (lambda, novelty) head
        self.state_head = nn.Sequential(
            nn.Linear(self.state_dim, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )

        # --- embeddings & projections ---
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        num_cohorts = self._extra_kwargs.get("num_cohorts", None)
        self.cohort_emb = nn.Embedding(int(num_cohorts), emb_dim) if num_cohorts else None

        self.user_proj = nn.Linear(hidden, emb_dim)
        self.item_proj = nn.Linear(hidden, emb_dim)

        # per-user calibration
        self.user_temp = nn.Embedding(num_users, 1)
        self.user_bias = nn.Embedding(num_users, 1)
        nn.init.constant_(self.user_temp.weight, 1.0)
        nn.init.zeros_(self.user_bias.weight)

        # per-user adapters (frozen by default; can be enabled for per-user fine-tune)
        self.num_adapter_slots = int(self._extra_kwargs.get("adapter_slots", 256))
        self.adapters = nn.ModuleDict({
            str(i): nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            for i in range(self.num_adapter_slots)
        })
        for m in self.adapters.values():
            for p in m.parameters():
                p.requires_grad_(False)

        # optional item bias
        self.item_bias = nn.Embedding(num_items, 1) if self.item_bias_enabled else None
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)

        # user preference EMA (no grads)
        self.user_pref = nn.Embedding(num_users, emb_dim)
        nn.init.zeros_(self.user_pref.weight)
        for p in self.user_pref.parameters():
            p.requires_grad_(False)
        self.pref_map = nn.Linear(hidden, emb_dim, bias=False)

        # base optimizer
        core_params = (
            list(self.user_net.parameters())
            + list(self.item_net.parameters())
            + (list(self.cohort_emb.parameters()) if self.cohort_emb is not None else [])
            + list(self.user_proj.parameters())
            + list(self.item_proj.parameters())
            + list(self.state_to_gate.parameters())
            + list(self.state_head.parameters())
            + list(self.user_temp.parameters())
            + list(self.user_bias.parameters())
            + list(self.pref_map.parameters())
        )
        if self.item_bias is not None:
            core_params += list(self.item_bias.parameters())
        self.optimizer = torch.optim.Adam([p for p in core_params if p.requires_grad], lr=lr)

        # --- Teacher + KL anchor (teacher overridden by DualRecommender) ---
        self.teacher = copy.deepcopy(self).to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.kl_beta = float(self._extra_kwargs.get("kl_beta", 0.05))

        # --- Exploration policies ---
        self.use_linucb: bool = bool(self._extra_kwargs.get("use_linucb", False))
        self.linucb_alpha: float = float(self._extra_kwargs.get("linucb_alpha", 1.0))
        self.linucb = LinUCBPolicy(d=3 * emb_dim, alpha=self.linucb_alpha, l2=1.0) if self.use_linucb else None

        self.use_bootts = bool(self._extra_kwargs.get("use_bootts", True))
        self.bootts = BootTS(d=3 * emb_dim,
                             heads=int(self._extra_kwargs.get("ts_heads", 10)),
                             l2=1.0, rng=int(self._extra_kwargs.get("ts_rng", 42))) if self.use_bootts else None
        self.ts_alpha = float(self._extra_kwargs.get("ts_alpha", 0.8))
        self.entropy_lambda = float(self._extra_kwargs.get("entropy_lambda", 0.0))
        self.info_gain_lambda = float(self._extra_kwargs.get("info_gain_lambda", 0.0))
        self.pos2id_map = dict(self._extra_kwargs.get("pos2id_map", {}))

        # optional dwell head
        self.use_dwell = bool(self._extra_kwargs.get("use_dwell", False))
        if self.use_dwell:
            self.dwell_head = nn.Sequential(
                nn.Linear(3 * emb_dim + 1, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        # --- DP config ---
        self.dp_cfg = (dp_cfg or {})
        self.dp_engine = str(self.dp_cfg.get("engine", "per_sample")).lower()
        self.dp_settings = SimpleDPConfig(
            enabled=bool(self.dp_cfg.get("enabled", True)),
            noise_multiplier=float(self.dp_cfg.get("noise_multiplier", self.dp_cfg.get("sigma", 1.0))),
            max_grad_norm=float(self.dp_cfg.get("max_grad", 1.0)),
            sample_rate=float(self.dp_cfg.get("sample_rate", 0.02)),
            delta=float(self.dp_cfg.get("delta", 1e-5)),
        )
        self._dp_accountant = build_accountant(self.dp_engine, self.dp_settings)
        self.eps_cum = 0.0
        self.eps_last = 0.0
        self._dp_steps_total = 0

        # Loss
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # caches & runtime
        self.recent_by_user = defaultdict(lambda: deque(maxlen=200))
        self._rep_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._rep_cache_cap = 16
        self._cached_item_matrix: Optional[torch.Tensor] = None
        self._last_item_reps: Optional[torch.Tensor] = None
        self._last_item_ids: Optional[List[int]] = None
        if hasattr(self, "teacher"):
            self.teacher._rep_cache = {}
            self.teacher._rep_cache_cap = self._rep_cache_cap
            self.teacher._cached_item_matrix = None
            self.teacher._last_item_reps = None
            self.teacher._last_item_ids = None
        self._adapted_lam: Optional[float] = None
        self._adapted_nov: Optional[float] = None

        self.device = device
        self.per_user_only = False
        self._compiled = False
        self.to(self.device)

        # Optional torch.compile for scoring path (PyTorch >= 2.0, CUDA only)
        if compile:
            _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            _is_cuda = "cuda" in str(device)
            if _torch_version >= (2, 0) and _is_cuda:
                try:
                    self._scores_logits = torch.compile(  # type: ignore[assignment]
                        self._scores_logits, mode="reduce-overhead"
                    )
                    self._compiled = True
                except Exception:
                    pass  # fallback to eager if compile fails

        self.audit_logger: Optional[AuditLogger] = None
        self._audit_actor = "TwoTowerRecommender"

    # ---------------- helpers ----------------
    def _slot_for(self, user_id: int) -> str:
        return str(hash(user_id) % self.num_adapter_slots)

    def attach_audit_logger(self, logger: AuditLogger, *, actor: str | None = None) -> None:
        self.audit_logger = logger
        if actor:
            self._audit_actor = str(actor)

    def _apply_film(self, u_base: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.state_to_gate(state_vec)   # [B, 2H]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return u_base * (1 + torch.tanh(gamma)) + beta

    def _apply_adapters(self, u: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "adapters") or len(self.adapters) == 0:
            return u
        if getattr(self, "per_user_only", False):
            if user_ids is None or user_ids.numel() == 0:
                return u
            flat = user_ids.view(-1)
            slot = self._slot_for(int(flat[0].item()))
            if slot not in self.adapters:
                return u
            return self.adapters[slot](u)
        return u

    def _user_context_vec(self, x_u, user_ids, state_vec, cohort_ids=None) -> torch.Tensor:
        xu_cat = torch.cat([x_u, state_vec], dim=1)
        u_base = self.user_net(xu_cat)
        u_base = self._apply_film(u_base, state_vec)
        u = self.user_proj(u_base) + self.user_emb(user_ids)
        if (cohort_ids is not None) and (self.cohort_emb is not None):
            u = u + self.cohort_emb(cohort_ids)
        u = self._apply_adapters(u, user_ids)
        u = u + self.user_pref(user_ids)
        # cache for dwell features
        self._user_context_vec_cache = u.detach()
        return u

    def _scores_logits(
        self,
        user_vec: Optional[torch.Tensor],      # <- now Optional
        item_matrix: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        state_vec: Optional[torch.Tensor] = None,
        cohort_ids: Optional[torch.Tensor] = None,
    ):
        # --- NEW: build user_vec when None so train_step can pass None ---
        if user_vec is None:
            user_vec = self._user_vec_from_ids(user_ids)   # [B, Du]
        elif user_vec.ndim == 1:
            user_vec = user_vec.unsqueeze(0)

        # ensure device/dtype
        dev = item_matrix.device
        user_vec = user_vec.to(dev)

        B = user_vec.size(0)
        if state_vec is None:
            state_vec = torch.zeros((B, self.state_dim), device=dev, dtype=user_vec.dtype)
        else:
            state_vec = state_vec.to(dev)

        # user path
        u = self._user_context_vec(user_vec, user_ids.to(dev), state_vec, cohort_ids)  # [B,D]

        # item path
        item_ids = item_ids.long().to(dev)
        I_base = self.item_net(item_matrix.index_select(0, item_ids))                  # [K,H]
        I = self.item_proj(I_base) + self.item_emb(item_ids)                           # [K,D]

        # normalize
        u = torch.nn.functional.normalize(u, dim=-1)
        I = torch.nn.functional.normalize(I, dim=-1)

        # logits + per-user temp/bias
        logits = fast_score(u, I, use_native=self.use_native_scoring)  # [B,K]
        tau = torch.clamp(self.user_temp(user_ids.to(dev)), 0.25, 4.0)
        b_u = self.user_bias(user_ids.to(dev))
        logits = logits / tau + b_u

        if self.item_bias is not None:
            logits = logits + self.item_bias(item_ids).squeeze(1)

        if _DEBUG_REC:
            arr = logits.detach().float().cpu().numpy().ravel()
            _d(f"scores: B={B} K={len(arr)} mu={arr.mean():.4f} sd={arr.std():.4f} "
               f"min={arr.min():.4f} max={arr.max():.4f}")

        return logits, u, I

    # Forward for scoring/ranking (adds optional BootTS/linUCB bonuses)
    def think(
        self,
        user_vec: Optional[torch.Tensor],
        item_matrix: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        state_vec: Optional[torch.Tensor] = None,
        cohort_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """Score candidate items with optional exploration bonuses.

        Runs the two-tower forward pass to produce relevance logits, then
        optionally adds LinUCB confidence or BootTS uncertainty bonuses when
        the model is in training mode. Call :meth:`decide` on the returned
        logits to apply MMR diversity re-ranking and select the final slate.

        Parameters
        ----------
        user_vec : torch.Tensor or None
            User feature vector of shape ``(1, user_dim)``. If ``None``,
            user features are looked up from ``user_ids``.
        item_matrix : torch.Tensor
            Item feature matrix of shape ``(num_items, item_dim)``.
        user_ids : torch.Tensor
            User index tensor of shape ``(1,)``.
        item_ids : torch.Tensor
            Candidate item indices of shape ``(K,)``.
        state_vec : torch.Tensor, optional
            Cognitive state ``[knowledge, fatigue, trust, engagement]`` of
            shape ``(1, state_dim)``. Defaults to zeros.
        cohort_ids : torch.Tensor, optional
            Cohort index for cohort-aware embedding (if configured).

        Returns
        -------
        torch.Tensor
            Logit scores of shape ``(1, K)``. Higher means more relevant.
            Pass these to :meth:`decide` for final slate selection.
        """
        logits, u, I = self._scores_logits(user_vec, item_matrix, user_ids, item_ids, state_vec, cohort_ids)
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        # cache item reps for decide()
        self._last_item_reps = I.detach()
        self._last_item_ids = item_ids.cpu().tolist()
        self._cached_item_matrix = item_matrix.detach()
        key = tuple(self._last_item_ids)
        if key in self._rep_cache:
            val = self._rep_cache.pop(key)
            self._rep_cache[key] = val
        else:
            self._rep_cache[key] = self._last_item_reps
            if len(self._rep_cache) > self._rep_cache_cap:
                self._rep_cache.pop(next(iter(self._rep_cache)), None)

        # optional diversity conditioning by state
        if (state_vec is None) or (not self._extra_kwargs.get("state_condition_diversity", False)):
            lam_eff = self.mmr_lambda
            nov_eff = self.novelty_bonus
        else:
            h = self.state_head(state_vec)
            lam_eff = torch.sigmoid(h[:, 0:1]).clamp(0.05, 0.9).item()
            nov_eff = (0.5 * torch.sigmoid(h[:, 1:2])).item()
        self._adapted_lam = lam_eff
        self._adapted_nov = nov_eff

        if _DEBUG_REC:
            _d(f"think: K={logits.numel()} lam_eff={lam_eff:.3f} nov_eff={nov_eff:.3f} "
               f"linucb={self.use_linucb} bootTS={self.use_bootts}")

        # --------- Optional exploration bonuses (training mode only) ----------
        u_vec = u[0]
        phi = torch.cat([u_vec.expand_as(I), I, u_vec.unsqueeze(0) * I], dim=1)  # [K,3D]
        phi_np = phi.detach().cpu().numpy()

        if self.training and torch.is_grad_enabled():
            if self.use_linucb and self.linucb is not None:
                bonuses = np.array([self.linucb.score(phi_np[j], base=0.0, i=int(iid))
                                    for j, iid in enumerate(self._last_item_ids)])
                b = torch.from_numpy(bonuses).to(device=logits.device, dtype=logits.dtype)
                b = (b - b.mean()) / (b.std(unbiased=False) + 1e-6)
                logits = logits + b.unsqueeze(0) * self.linucb_alpha
                if _DEBUG_REC:
                    _d(f"think: linUCB bonus applied alpha={self.linucb_alpha:.3f}")

            if self.use_bootts and self.bootts is not None:
                bonuses = np.array([self.bootts.score_vec(phi_np[j])
                                    for j in range(len(self._last_item_ids))])
                b = torch.from_numpy(bonuses).to(device=logits.device, dtype=logits.dtype)
                b = (b - b.mean()) / (b.std(unbiased=False) + 1e-6)
                logits = logits + b.unsqueeze(0) * self.ts_alpha
                if _DEBUG_REC:
                    _d(f"think: BootTS bonus applied alpha={self.ts_alpha:.3f}")

        # keep features for update()
        self._last_phi_np = phi_np
        return logits  # [1,K]

    # ---------------- uncertainty widths for policy mapping (paper §3.1) ----------------
    def uncertainty_widths(self, phi_np: np.ndarray, item_ids: List[int]) -> List[float]:
        """Return per-item uncertainty widths σ(i) in [0, +∞), with a small floor.

        Sources:
        - If LinUCB is enabled: σ(i) = sqrt(x^T (A_i + l2 I)^-1 x)
        - Else if BootTS is enabled: σ(i) = std_h(x^T θ_h) across bootstrap heads
        - Else: constant small width

        The caller is responsible for optional [0,1] normalization across the slate.
        """
        try:
            x = np.asarray(phi_np, dtype=np.float64)
            ids = [int(i) for i in item_ids]
            K = x.shape[0]
            widths = np.full((K,), float(getattr(self, "sigma_min", 0.05) or 0.05), dtype=np.float64)

            # LinUCB confidence radius
            if getattr(self, "use_linucb", False) and (self.linucb is not None):
                d = int(getattr(self.linucb, "d", x.shape[1]))
                l2 = float(getattr(self.linucb, "l2", 1.0))
                I = np.eye(d, dtype=np.float64)
                for j, iid in enumerate(ids):
                    A = self.linucb.A.get(int(iid))
                    if A is None:
                        A = l2 * I
                    try:
                        A_reg = A + l2 * I
                        Ax = np.linalg.solve(A_reg, x[j])
                        conf = float(np.sqrt(max(0.0, float(x[j] @ Ax))))
                    except np.linalg.LinAlgError:
                        A_pinv = np.linalg.pinv(A + l2 * I)
                        conf = float(np.sqrt(max(0.0, float(x[j] @ (A_pinv @ x[j])))))
                    widths[j] = max(widths[j], conf)

            # Bootstrapped TS predictive spread
            elif getattr(self, "use_bootts", False) and (self.bootts is not None):
                H = int(getattr(self.bootts, "H", 0))
                if H and self.bootts.As and self.bootts.bs:
                    d = x.shape[1]
                    I = np.eye(d, dtype=np.float64)
                    preds = np.zeros((K, H), dtype=np.float64)
                    for h in range(H):
                        A = self.bootts.As[h]
                        b = self.bootts.bs[h]
                        try:
                            theta = np.linalg.solve(A + 1e-8 * I, b)
                        except np.linalg.LinAlgError:
                            theta = np.linalg.pinv(A + 1e-8 * I) @ b
                        preds[:, h] = x @ theta
                    widths = np.maximum(widths, preds.std(axis=1))

            # small positive floor
            sigma_min = float(getattr(self, "sigma_min", 0.05) or 0.05)
            widths = np.maximum(widths, sigma_min)
            return [float(w) for w in widths.tolist()]
        except Exception:
            # fallback constant widths
            sigma_min = float(getattr(self, "sigma_min", 0.05) or 0.05)
            return [sigma_min for _ in item_ids]

    # ---------------- decide (MMR + novelty + ZPD soft shaping + optional dwell reorder) ----------------
    @torch.no_grad()
    def decide(self, *, logits, top_k: int, item_ids, user_id: int,
           engagement: float, trust: float, difficulty_map: dict,
           knowledge: float, zpd_delta: float, policy: str = None):
        """Select a diverse, difficulty-appropriate slate from scored candidates.

        Applies Maximal Marginal Relevance (MMR) re-ranking to balance
        relevance against intra-list diversity, plus novelty bonuses and
        zone-of-proximal-development (ZPD) soft shaping.

        Must be called after :meth:`think` (or :meth:`infer`), which
        caches the item representations needed for similarity computation.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor of shape ``(1, K)`` from :meth:`think` or :meth:`infer`.
        top_k : int
            Number of items to select for the final slate.
        item_ids : torch.Tensor or list[int]
            Candidate item IDs corresponding to each column of ``logits``.
        user_id : int
            Active user ID (for novelty memory tracking).
        engagement : float
            User engagement level in ``[0, 1]`` (currently used for logging).
        trust : float
            User trust level in ``[0, 1]``. Higher trust reduces novelty
            bonus (trusted users get more exploitation, less exploration).
        difficulty_map : dict
            Mapping ``{item_id: difficulty}`` where difficulty is in ``[0, 1]``.
        knowledge : float
            User's current knowledge/ability estimate in ``[0, 1]``.
        zpd_delta : float
            Target gap above current knowledge for ZPD centering.
            Items at ``knowledge + zpd_delta`` get the highest ZPD bonus.
        policy : str, optional
            Unused; reserved for future policy routing.

        Returns
        -------
        tuple[list[int], dict[int, float]]
            ``(selected_item_ids, scores_dict)`` where ``scores_dict`` maps
            each selected item ID to its MMR-adjusted score.

        Notes
        -----
        The MMR score for each candidate combines four signals:

        ``MMR = (1 - mmr_lambda) * relevance - mmr_lambda * max_similarity
        + novelty_bonus * novelty + zpd_weight * zpd_fit``

        - **mmr_lambda** (set at init): Controls relevance–diversity tradeoff.
          0.0 = pure relevance, 1.0 = pure diversity. Default 0.3.
        - **novelty_bonus**: Extra score for unseen items (scaled by ``1 - trust``).
        - **zpd_weight**: Soft bonus for items near the user's ZPD.
        """
        assert logits.ndim == 2 and logits.size(0) == 1, "decide() expects logits of shape [1, K]"
        scores = logits[0].detach().cpu().numpy()

        if torch.is_tensor(item_ids):
            item_ids_list = item_ids.detach().cpu().tolist()
        else:
            item_ids_list = list(item_ids)

        pos_map = getattr(self, "pos2id_map", {})
        entropy_lambda = float(getattr(self, "entropy_lambda", 0.0))
        info_lambda = float(getattr(self, "info_gain_lambda", 0.0))

        # get (or compute) item reps for this candidate set
        key = tuple(item_ids_list)
        if getattr(self, "_last_item_reps", None) is not None and self._last_item_ids == item_ids_list:
            I = self._last_item_reps
            cache_hit = True
        elif key in self._rep_cache:
            I = self._rep_cache[key]
            cache_hit = True
        else:
            ids_t = torch.tensor(item_ids_list, dtype=torch.long, device=logits.device)
            mat = self._cached_item_matrix.to(logits.device) if self._cached_item_matrix is not None else None
            if mat is None:
                raise RuntimeError("Item matrix cache missing; call think() before decide().")
            I_base = self.item_net(mat.index_select(0, ids_t))
            I = self.item_proj(I_base) + self.item_emb(ids_t)
            self._rep_cache[key] = I.detach()
            if len(self._rep_cache) > self._rep_cache_cap:
                oldest = next(iter(self._rep_cache))
                self._rep_cache.pop(oldest, None)
            cache_hit = False

        I = I / (I.norm(dim=1, keepdim=True) + 1e-8)   # [K,D]

        if info_lambda and difficulty_map is not None and knowledge is not None:
            knowledge_val = float(knowledge)
            scale = max(0.05, float(abs(zpd_delta)) + 0.05)
            denom = 2.0 * (scale ** 2)
            target_gap = 0.08
            # Vectorized difficulty lookup and info gain computation
            diffs = np.array([float(difficulty_map.get(int(pos_map.get(int(pos), int(pos))),
                             difficulty_map.get(int(pos), 0.5))) for pos in item_ids_list])
            gaps = diffs - knowledge_val
            info_bonus = np.where(gaps >= 0.0,
                                  np.exp(-((gaps - target_gap) ** 2) / denom),
                                  -0.05 * np.exp(-(gaps ** 2) / denom))
            scores = scores + info_lambda * info_bonus

        # effective λ, novelty (from state if available)
        lam = float(self._adapted_lam) if self._adapted_lam is not None else self.mmr_lambda
        nov = float(self._adapted_nov) if self._adapted_nov is not None else self.novelty_bonus
        if trust is not None:
            nov = nov * (1.0 - float(trust))

        # novelty signal from per-user memory
        seen = set(self.recent_by_user.get(user_id, [])) if user_id is not None else set()

        def novelty_of(iid: int) -> float:
            return 1.0 if (iid not in seen) else 0.2

        def zpd_bonus(iid: int) -> float:
            if (difficulty_map is None) or (knowledge is None):
                return 0.0
            ext_id = int(pos_map.get(int(iid), int(iid)))
            d = float(difficulty_map.get(ext_id, difficulty_map.get(int(iid), 0.5)))
            tgt = float(np.clip(knowledge + zpd_delta, 0.0, 1.0))
            width = float(getattr(self, "zpd_width", 0.10) or 0.10)
            return 1.0 - min(1.0, ((d - tgt) ** 2) / (max(1e-6, width) ** 2))

        K = len(item_ids_list)
        remaining = list(range(K))
        selected, selected_scores = [], {}

        # Pre-compute similarity matrix for MMR: O(K²D) once vs O(K²D) per iteration
        sim_matrix = (I @ I.T).cpu().numpy()  # [K, K]

        # Pre-compute novelty and ZPD bonuses for all candidates
        novelty_arr = np.array([novelty_of(item_ids_list[j]) for j in range(K)])
        zpd_w = float(getattr(self, "zpd_weight", 0.05))
        zpd_arr = np.array([zpd_bonus(item_ids_list[j]) for j in range(K)])

        # DEBUG preamble
        if _DEBUG_REC:
            _d(f"decide: top_k={top_k} K={K} lam={lam:.3f} nov={nov:.3f} zpd={zpd_delta:.3f} "
               f"trust={trust:.2f} eng={engagement:.2f} cache_hit={cache_hit}")

        while remaining and len(selected) < top_k:
            best_j, best_val = None, -1e18
            for j in remaining:
                if selected:
                    sim = float(sim_matrix[j, selected].max())
                else:
                    sim = 0.0
                div_bonus = entropy_lambda * (1.0 - sim)
                mmr = (1.0 - lam) * float(scores[j]) - lam * sim \
                      + nov * novelty_arr[j] + zpd_w * zpd_arr[j] + div_bonus
                mmr += 1e-9 * (j + 1)  # tiny jitter
                if mmr > best_val:
                    best_val, best_j = mmr, j
            if best_j is None:
                # Fallback: fill remaining by base score
                if remaining:
                    ordered = sorted(remaining, key=lambda idx: float(scores[idx]), reverse=True)
                    take = ordered[: max(0, top_k - len(selected))]
                    selected.extend(take)
                    for ridx in take:
                        selected_scores[item_ids_list[ridx]] = float(scores[ridx])
                    remaining = []
                break
            selected.append(best_j)
            selected_scores[item_ids_list[best_j]] = best_val
            if best_j in remaining:
                remaining.remove(best_j)

        sel_item_ids = [item_ids_list[i] for i in selected]

        # Optional dwell-based reordering within the chosen slate
        if self._extra_kwargs.get("use_dwell", False) and hasattr(self, "dwell_head"):
            try:
                idxs = [self._last_item_ids.index(i) for i in sel_item_ids]
                I_sel = self._last_item_reps[idxs]
                U = getattr(self, "_user_context_vec_cache", None)
                if U is not None:
                    U = U.expand_as(I_sel)
                    feats = torch.cat([U, I_sel, U * I_sel,
                                       torch.zeros((len(idxs), 1), device=I_sel.device)], dim=1)
                    dwell_pred = self.dwell_head(feats).squeeze(1)
                    order = torch.argsort(dwell_pred, descending=True).tolist()
                    sel_item_ids = [sel_item_ids[i] for i in order]
                    _d("decide: dwell reorder applied")
            except Exception:
                pass

        # maintain novelty memory
        if user_id is not None:
            self.recent_by_user[user_id].extend(sel_item_ids)

        if _DEBUG_REC:
            _d(f"decide: chose={len(sel_item_ids)} ids_head={sel_item_ids[:min(5,len(sel_item_ids))]}")

        return sel_item_ids, selected_scores

    # ---------------- training/update (adds KL anchor; optional TS updates) ----------------
    def begin_dp_step(self, eps_target: float):
        tgt = max(0.0, float(eps_target))
        if not getattr(self, "dp_settings", None) or not self.dp_settings.enabled or tgt <= 0.0:
            self._dp_steps_budget = 0
            _d(f"DP begin: disabled or tgt<=0 (tgt={tgt})")
            return
        sigma = float(self.dp_settings.noise_multiplier or 0.0)
        q     = float(self.dp_settings.sample_rate or 0.0)
        if sigma <= 0.0 or q <= 0.0:
            self._dp_steps_budget = 0
            _d("DP begin: invalid sigma/q -> no budget")
            return
        preview = self._dp_accountant.fork()
        steps = 0
        MAX_STEPS = 32
        while steps < MAX_STEPS:
            inc, _ = preview.step(1)
            if inc <= 0.0:
                if inc == 0.0:
                    steps += 1
                break
            if inc > tgt:
                break
            steps += 1

        self._dp_steps_budget = max(0, steps)
        _d(f"DP begin: tgt_eps={tgt:.4f} -> steps_budget={self._dp_steps_budget}")

    @torch.no_grad()
    def infer(
        self,
        *,
        user_vec: Optional[torch.Tensor] = None,
        item_matrix: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        state_vec: Optional[torch.Tensor] = None,
        cohort_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pure scoring without exploration bonuses.

        In fixed-policy runs the orchestrator calls ``rec.infer(...)``
        for deterministic scoring.  Use :meth:`think` when exploration
        bonuses (BootTS / LinUCB) are desired.
        """
        logits, _u, _I = self._scores_logits(
            user_vec, item_matrix, user_ids, item_ids,
            state_vec=state_vec, cohort_ids=cohort_ids,
        )
        out = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        # Cache item reps so decide() can use them without requiring think()
        self._last_item_reps = _I.detach()
        self._last_item_ids = item_ids.cpu().tolist()
        self._cached_item_matrix = item_matrix.detach()

        if _DEBUG_REC:
            _d(f"infer: logits shape={tuple(out.shape)}")
        return out

    def infer_batch(
        self,
        *,
        user_vecs: torch.Tensor,
        item_matrix: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        state_vecs: torch.Tensor,
        cohort_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched inference for multiple users in a single forward pass.

        Instead of calling infer() N times (once per user), this method
        processes all users simultaneously, reducing device sync overhead
        and enabling GPU parallelism.

        Parameters
        ----------
        user_vecs : torch.Tensor
            User feature vectors, shape (N, user_dim).
        item_matrix : torch.Tensor
            Full item feature matrix, shape (num_items, item_dim).
        user_ids : torch.Tensor
            User indices, shape (N,).
        item_ids : torch.Tensor
            Candidate item indices, shape (K,). Same candidates for all users.
        state_vecs : torch.Tensor
            Cognitive state vectors, shape (N, state_dim).
        cohort_ids : torch.Tensor, optional
            Cohort indices, shape (N,).

        Returns
        -------
        torch.Tensor
            Logits of shape (N, K) — scores for each user × candidate pair.
        """
        # Delegate to _scores_logits which handles all user/item tower logic,
        # FiLM gating, adapters, user_pref, calibration, and item bias.
        # This ensures infer_batch is consistent with the per-user infer() path.
        logits, _u, _I = self._scores_logits(
            user_vecs, item_matrix, user_ids, item_ids,
            state_vec=state_vecs, cohort_ids=cohort_ids,
        )
        return torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    def update(self, *,
           feedback: Dict[int, int],
           user_vec: torch.Tensor,
           state_vec: torch.Tensor,
           user_ids: torch.Tensor,
           item_matrix: torch.Tensor,
           item_ids: torch.Tensor,
           epochs: int = 1,
           scope: str = "global",
           **kwargs) -> Dict[str, float]:
        """
        DP-aware update + KL anchor to a frozen teacher.
        """
        device = next(self.parameters()).device
        self.train()

        # freeze/unfreeze by scope
        all_params = list(self.parameters())
        if scope == "per_user":
            for p in all_params:
                p.requires_grad_(False)
            slot_key = None
            if user_ids is not None and user_ids.numel() > 0:
                try:
                    slot_key = self._slot_for(int(user_ids.view(-1)[0].item()))
                except Exception:
                    slot_key = None
            if slot_key in getattr(self, "adapters", {}):
                for p in self.adapters[slot_key].parameters():
                    p.requires_grad_(True)
            trainable = [p for p in self.parameters() if p.requires_grad]
        else:
            for p in all_params:
                p.requires_grad_(True)
            trainable = [p for p in all_params if p.requires_grad]

        lr = getattr(self, "lr", 1e-2)
        if not hasattr(self, "_opt") or getattr(self, "_opt_scope", None) != scope:
            self._opt = torch.optim.Adam(trainable, lr=lr)
            self._opt_scope = scope
        else:
            self._opt.param_groups[0]["params"] = trainable

        if _DEBUG_REC:
            _d(f"update(scope={scope}): feedback_n={len(feedback)} epochs={epochs} "
               f"trainable_params={sum(p.numel() for p in trainable)}")

        if not feedback:
            return {
                "loss": 0.0,
                "epsilon_delta": 0.0,
                "epsilon_cum": float(getattr(self, "eps_cum", 0.0)),
                "noise_multiplier": None,
                "delta": None,
                "sample_rate": None,
            }

        # map catalog positions -> local indices within item_ids tensor
        with torch.no_grad():
            pos2idx = {int(p): i for i, p in enumerate(item_ids.detach().cpu().tolist())}
            idx = [pos2idx[p] for p in feedback.keys() if p in pos2idx]
            if not idx:
                _d("update: no matching item_ids for feedback keys; skip")
                return {
                    "loss": 0.0,
                    "epsilon_delta": 0.0,
                    "epsilon_cum": float(getattr(self, "eps_cum", 0.0)),
                    "noise_multiplier": None,
                    "delta": None,
                    "sample_rate": None,
                }
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            y = torch.tensor([feedback[int(p)] for p in feedback.keys() if p in pos2idx],
                             dtype=torch.float32, device=device)

        # DP settings
        sigma = float(getattr(self.dp_settings, "noise_multiplier", 0.0) or 0.0)
        q = float(getattr(self.dp_settings, "sample_rate", 0.0) or 0.0)
        delta = float(getattr(self.dp_settings, "delta", 1e-5))
        C = float(getattr(self.dp_settings, "max_grad_norm", 1.0))
        dp_enabled = bool(getattr(self.dp_settings, "enabled", False) and sigma > 0.0 and q > 0.0)

        float(getattr(self, "eps_cum", 0.0) or 0.0)
        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

        steps_budget = int(getattr(self, "_dp_steps_budget", 0)) if dp_enabled else max(1, int(epochs))
        steps_done = max(1, min(int(epochs), steps_budget)) if dp_enabled else max(1, int(epochs))
        if dp_enabled:
            self._dp_steps_budget = max(0, steps_budget - steps_done)

        if dp_enabled and self.dp_engine == "per_sample":
            loss_val = self._dp_update_per_sample(
                idx_t=idx_t,
                y=y,
                user_vec=user_vec,
                state_vec=state_vec,
                user_ids=user_ids,
                item_matrix=item_matrix,
                item_ids=item_ids,
                trainable=trainable,
                sigma=sigma,
                C=C,
            )
        else:
            loss_val = 0.0
            for step in range(steps_done):
                self._opt.zero_grad(set_to_none=True)

                logits_all = self.think(
                    user_vec=user_vec.to(device),
                    item_matrix=item_matrix.to(device),
                    user_ids=user_ids.to(device),
                    item_ids=item_ids.to(device),
                    state_vec=state_vec.to(device),
                    cohort_ids=None,
                )
                logits = logits_all[0, idx_t]

                with torch.no_grad():
                    t_logits_all = self.teacher.think(
                        user_vec=user_vec.to(device),
                        item_matrix=item_matrix.to(device),
                        user_ids=user_ids.to(device),
                        item_ids=item_ids.to(device),
                        state_vec=state_vec.to(device),
                        cohort_ids=None,
                    )
                log_p = torch.nn.functional.logsigmoid(logits_all[0, idx_t])
                log_pt = torch.nn.functional.logsigmoid(t_logits_all[0, idx_t])
                kl_stu = torch.exp(log_pt) * (log_pt - log_p) + torch.exp(log_p) * (log_p - log_pt)
                kl_pen = self.kl_beta * kl_stu.mean()

                loss = bce(logits, y) + kl_pen
                loss.backward()

                if dp_enabled:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=C)
                    with torch.no_grad():
                        for p in trainable:
                            if p.grad is None:
                                continue
                            noise = torch.normal(
                                mean=0.0,
                                std=sigma * C,
                                size=p.grad.shape,
                                device=p.grad.device,
                                dtype=p.grad.dtype,
                            )
                            p.grad.add_(noise)

                self._opt.step()
                loss_val = float(loss.detach().item())
                if _DEBUG_REC:
                    _d(f"update: step {step+1}/{steps_done} loss={loss_val:.4f}")

        # ----- DP accountant (RDP approx) -----
        eps_delta = 0.0
        if dp_enabled:
            eps_delta, self.eps_cum = self._dp_accountant.step(int(steps_done))
            self._dp_steps_total += int(steps_done)
            self.eps_last = eps_delta
        else:
            self.eps_cum = 0.0
            self.eps_last = 0.0
            self._dp_steps_total = 0

        rewards_np = y.detach().cpu().numpy()

        # ----- Optional: update LinUCB head with observed reward -----
        if self.use_linucb and self.linucb is not None and hasattr(self, "_last_phi_np"):
            try:
                for j_local, rwd in zip(idx, rewards_np):
                    iid = int(self._last_item_ids[j_local])
                    phi_row = self._last_phi_np[j_local]
                    self.linucb.update(iid, phi_row, float(rwd))
            except Exception:
                pass

        # ----- Optional: update BootTS with simple reward shaping -----
        if self.use_bootts and self.bootts is not None and hasattr(self, "_last_phi_np"):
            try:
                rewards = 0.5 * rewards_np + 0.5
                for j_local, rwd in zip(idx, rewards):
                    phi_row = self._last_phi_np[j_local]
                    self.bootts.update(phi_row, float(rwd), k=2)
            except Exception:
                pass

        if _DEBUG_REC:
            _d(f"update: done loss={loss_val:.4f} dp={'ON' if dp_enabled else 'OFF'} "
               f"eps+={eps_delta:.4f} eps_cum={getattr(self, 'eps_cum', 0.0):.4f}")

        stats_out = {
            "loss": float(loss_val),
            "epsilon_delta": float(eps_delta),
            "epsilon_cum": float(getattr(self, "eps_cum", 0.0)),
            "noise_multiplier": (sigma if dp_enabled else None),
            "delta": (delta if dp_enabled else None),
            "sample_rate": (q if dp_enabled else None),
        }

        if self.audit_logger is not None:
            payload = {
                "loss": stats_out["loss"],
                "epsilon_delta": stats_out["epsilon_delta"],
                "epsilon_cum": stats_out["epsilon_cum"],
                "noise_multiplier": stats_out["noise_multiplier"],
                "delta": stats_out["delta"],
                "sample_rate": stats_out["sample_rate"],
                "steps_total": getattr(self, "_dp_steps_total", 0),
                "dp_enabled": bool(dp_enabled),
            }
            try:
                self.audit_logger.log_event("dp_update", self._audit_actor, payload)
            except Exception:
                if _DEBUG_REC:
                    _d("audit logger failed", payload)

        return stats_out

    def _dp_update_per_sample(
        self,
        *,
        idx_t: torch.Tensor,
        y: torch.Tensor,
        user_vec: torch.Tensor,
        state_vec: torch.Tensor,
        user_ids: torch.Tensor,
        item_matrix: torch.Tensor,
        item_ids: torch.Tensor,
        trainable: List[torch.nn.Parameter],
        sigma: float,
        C: float,
    ) -> float:
        device = next(self.parameters()).device

        logits_all = self.think(
            user_vec=user_vec.to(device),
            item_matrix=item_matrix.to(device),
            user_ids=user_ids.to(device),
            item_ids=item_ids.to(device),
            state_vec=state_vec.to(device),
            cohort_ids=None,
        )
        teacher = self.teacher
        if teacher is not None:
            # ensure frozen teacher has runtime attributes expected by think()
            defaults = {
                "use_linucb": getattr(self, "use_linucb", False),
                "linucb": copy.deepcopy(getattr(self, "linucb", None)) if getattr(self, "use_linucb", False) else None,
                "use_bootts": getattr(self, "use_bootts", False),
                "bootts": copy.deepcopy(getattr(self, "bootts", None)) if getattr(self, "use_bootts", False) else None,
                "ts_alpha": getattr(self, "ts_alpha", 0.0),
                "entropy_lambda": getattr(self, "entropy_lambda", 0.0),
                "info_gain_lambda": getattr(self, "info_gain_lambda", 0.0),
                "pos2id_map": getattr(self, "pos2id_map", {}),
                "use_dwell": getattr(self, "use_dwell", False),
                "dwell_head": copy.deepcopy(getattr(self, "dwell_head", None)),
                "_rep_cache": {},
                "_rep_cache_cap": getattr(self, "_rep_cache_cap", 16),
                "_cached_item_matrix": None,
                "_last_item_reps": None,
                "_last_item_ids": None,
            }
            for attr, val in defaults.items():
                if not hasattr(teacher, attr):
                    setattr(teacher, attr, val)

        with torch.no_grad():
            t_logits_all = teacher.think(
                user_vec=user_vec.to(device),
                item_matrix=item_matrix.to(device),
                user_ids=user_ids.to(device),
                item_ids=item_ids.to(device),
                state_vec=state_vec.to(device),
                cohort_ids=None,
            )

        logits = logits_all[0, idx_t]
        log_p = torch.nn.functional.logsigmoid(logits)
        log_pt = torch.nn.functional.logsigmoid(t_logits_all[0, idx_t])
        kl_stu = torch.exp(log_pt) * (log_pt - log_p) + torch.exp(log_p) * (log_p - log_pt)
        bce_per = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        per_sample_losses = bce_per + self.kl_beta * kl_stu

        sum_grads = [torch.zeros_like(p, device=device) for p in trainable]
        total_loss = 0.0
        B = len(per_sample_losses)
        for idx, loss_sample in enumerate(per_sample_losses):
            for p in trainable:
                if p.grad is not None:
                    p.grad.zero_()
            loss_sample.backward(retain_graph=idx < B - 1)
            grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p, device=device) for p in trainable]
            norm_sq = sum(float(torch.sum(g.pow(2))) for g in grads)
            l2 = math.sqrt(max(norm_sq, 1e-12))
            clip = min(1.0, C / l2)
            for j, g in enumerate(grads):
                sum_grads[j].add_(g * clip)
            total_loss += float(loss_sample.detach().item())

        std = float(sigma) * float(C)
        for j, p in enumerate(trainable):
            noise = torch.normal(mean=0.0, std=std, size=p.shape, device=p.device, dtype=p.dtype)
            p.grad = (sum_grads[j] + noise) / max(1, B)

        self._opt.step()
        return float(total_loss / max(1, B))

    # src/agents/agents.py  (inside the TwoTowerRecommender class)
    def _user_vec_from_ids(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Build the user feature vectors for training so they MATCH the model's expected input.
        Priority (reversed): 1) user_matrix (Du), 2) user_encoder, 3) embeddings, 4) zeros(Du)
        """
        dev = self.device if hasattr(self, "device") else user_ids.device

        # 1) use the same matrix you used at inference (Du columns)
        Umat = getattr(self, "user_matrix", None)
        if isinstance(Umat, torch.Tensor) and Umat.ndimension() == 2:
            return Umat[user_ids.to(dev)]

        # 2) encoder if you have one that outputs Du
        enc = getattr(self, "user_encoder", None)
        if callable(enc):
            try:
                return enc(user_ids.to(dev))
            except Exception:
                pass

        # 3) embeddings (only if embedding dim matches user_dim expected by user_net)
        expected_dim = int(getattr(self, "user_dim", 0))
        for attr in ("user_embedding", "user_emb", "emb_user", "u_emb"):
            emb = getattr(self, attr, None)
            if emb is not None and hasattr(emb, "weight"):
                if expected_dim and emb.weight.shape[1] == expected_dim:
                    return emb(user_ids.to(dev))

        # 4) zeros with the model's expected user_dim (NOT emb_dim)
        d = int(getattr(self, "user_dim", getattr(self, "emb_dim", 64)))
        return torch.zeros((user_ids.shape[0], d), dtype=torch.float32, device=dev)

    def _pairwise_logits(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        state_vec: torch.Tensor,
        item_matrix: torch.Tensor,
    ) -> torch.Tensor:
        dev = self.device if hasattr(self, "device") else user_ids.device
        user_ids = user_ids.to(dev)
        item_ids = item_ids.to(dev)
        state_vec = state_vec.to(dev)

        user_vec = self._user_vec_from_ids(user_ids)
        u = self._user_context_vec(user_vec, user_ids, state_vec, cohort_ids=None)

        item_feats = item_matrix.to(dev).index_select(0, item_ids)
        I_base = self.item_net(item_feats)
        I = self.item_proj(I_base) + self.item_emb(item_ids)

        u = torch.nn.functional.normalize(u, dim=-1)
        I = torch.nn.functional.normalize(I, dim=-1)

        logits = torch.sum(u * I, dim=-1)

        tau = torch.clamp(self.user_temp(user_ids), 0.25, 4.0).squeeze(1)
        b_u = self.user_bias(user_ids).squeeze(1)
        logits = logits / tau + b_u

        if self.item_bias is not None:
            logits = logits + self.item_bias(item_ids).squeeze(1)

        return logits

    def train_step(self, batch: dict) -> dict:
        self.train()

        user_ids = batch["user_ids"].to(self.device)
        item_ids = batch["item_ids"].to(self.device)
        labels = batch["labels"].to(self.device).float()
        Xi = batch["item_matrix"].to(self.device)

        state_vec = batch.get("state_vec")
        if state_vec is None:
            state_vec = torch.zeros((user_ids.shape[0], self.state_dim), dtype=torch.float32, device=self.device)
        else:
            state_vec = state_vec.to(self.device)

        logits = self._pairwise_logits(
            user_ids=user_ids,
            item_ids=item_ids,
            state_vec=state_vec,
            item_matrix=Xi,
        )

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        # Optional Funk-style distillation: encourage tower scores to match auxiliary MF scores
        distill = batch.get("distill_scores")
        lam = float(batch.get("distill_lambda", 0.0)) if "distill_lambda" in batch else 0.0
        if distill is not None and lam > 0.0:
            try:
                # Ensure shape match; interpret distill as probabilities in [0,1]
                distill_t = distill.to(self.device).view_as(logits)
                probs = torch.sigmoid(logits)
                loss = loss + lam * torch.nn.functional.mse_loss(probs, distill_t)
            except Exception:
                pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(self, "dp_settings", None) and self.dp_settings.enabled:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.dp_settings.max_grad_norm)
        self.optimizer.step()
        out = {"loss": float(loss.item())}
        if _DEBUG_REC:
            _d(f"train_step: B={int(user_ids.numel())} loss={out['loss']:.4f}")
        return out


__all__ = [
    "TwoTowerRecommender",
    "enable_debug_rec_logs",
]
