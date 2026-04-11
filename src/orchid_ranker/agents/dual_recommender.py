"""Dual recommender wrapper with teacher and student models."""

from __future__ import annotations

import copy
import inspect
import logging
import os
from collections import deque
from typing import Any, Dict

import numpy as np
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender

logger = logging.getLogger(__name__)


def _d(*args) -> None:
    """Debug logging (respects ORCHID_DEBUG_REC env var)."""
    if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
        logger.debug("%s", " ".join(str(a) for a in args))


class DualRecommender:
    """
    Wrapper around two recommenders:
      - teacher: inference-only (used for scoring/selection)
      - student: trainable (updated by train_step / update)
    """

    def __init__(self, teacher, student, device=None, *, warm_start: bool = False, replay_size: int = 0, replay_steps: int = 0):
        self.teacher = teacher
        self.student = student
        self.device  = device or getattr(student, "device", torch.device("cpu"))
        # Optional: start teacher from student params to reduce cold-start gap
        if warm_start:
            try:
                with torch.no_grad():
                    for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                        t_param.data.copy_(s_param.data)
            except Exception:
                pass

        # mirror common knobs onto both so orchestrator can set once
        self._novelty_bonus = getattr(student, "novelty_bonus", 0.0)
        self._mmr_lambda    = getattr(student, "mmr_lambda", 0.25)
        self._student_weight = 0.0
        self._blend_increment = float(getattr(student, "blend_increment", 0.12))
        self._teacher_ema = float(getattr(student, "teacher_ema", 0.94))
        self._ensure_recent_map()

        # push initial knobs into both models
        self._sync_knobs()

        _d(f"DualRec init: blend_inc={self._blend_increment} teacher_ema={self._teacher_ema}")

        # Lightweight replay buffer for additional per-call updates
        self._replay_buf = deque(maxlen=int(replay_size)) if int(replay_size) > 0 else None
        self._replay_steps = int(replay_steps)

    # ---------- small utils ----------
    def _ensure_recent_map(self):
        if not hasattr(self, "recent_by_user"):
            self.recent_by_user = {}
        for m in (self.teacher, self.student):
            if not hasattr(m, "recent_by_user"):
                m.recent_by_user = self.recent_by_user

    def _sync_knobs(self):
        for m in (self.teacher, self.student):
            if hasattr(m, "novelty_bonus"):
                m.novelty_bonus = self._novelty_bonus
            if hasattr(m, "mmr_lambda"):
                m.mmr_lambda = self._mmr_lambda

    @staticmethod
    def _call_with_supported_args(fn, **kwargs):
        """Filter kwargs to only what `fn` accepts."""
        sig = inspect.signature(fn)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**allowed)

    def _infer_component(self, component: str, **kwargs):
        model = self.teacher if component == "teacher" else self.student
        fn = getattr(model, "infer", None)
        if callable(fn):
            return self._call_with_supported_args(fn, **kwargs)
        return self._call_with_supported_args(model.think, **kwargs)

    def infer_policy(self, policy: str = "adaptive", **kwargs):
        pol = (policy or "adaptive").lower()
        if pol == "teacher":
            return self._infer_component("teacher", **kwargs)
        if pol in {"student", "adaptive_student"}:
            return self._infer_component("student", **kwargs)
        return self.infer(**kwargs)

    def _after_student_update(self) -> None:
        self._student_weight = float(min(1.0, self._student_weight + self._blend_increment))
        tau = float(self._teacher_ema)
        tau = min(max(tau, 0.0), 0.999)
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            _d(f"DualRec: student_weight={self._student_weight:.3f} (after update)")

    # ---------- public knobs (proxied) ----------
    @property
    def novelty_bonus(self):
        return self._novelty_bonus
    @novelty_bonus.setter
    def novelty_bonus(self, v):
        self._novelty_bonus = float(v)
        self._sync_knobs()
        _d(f"DualRec: set novelty_bonus={v}")

    @property
    def mmr_lambda(self):
        return self._mmr_lambda
    @mmr_lambda.setter
    def mmr_lambda(self, v):
        self._mmr_lambda = float(v)
        self._sync_knobs()
        _d(f"DualRec: set mmr_lambda={v}")

    # ---------- DP / epsilon passthrough to student ----------
    @property
    def dp_settings(self):
        # let orchestrator read/modify DP flags on the trainable model
        return getattr(self.student, "dp_settings", None)

    @property
    def eps_cum(self):
        return float(getattr(self.student, "eps_cum", 0.0))
    @eps_cum.setter
    def eps_cum(self, v):
        if hasattr(self.student, "eps_cum"):
            self.student.eps_cum = float(v)

    @property
    def eps_last(self):
        return float(getattr(self.student, "eps_last", 0.0))
    @eps_last.setter
    def eps_last(self, v):
        if hasattr(self.student, "eps_last"):
            self.student.eps_last = float(v)

    # ---------- inference ----------
    def infer(self, **kwargs):
        """
        Blend teacher + student scores (teacher keeps caches for decide()).
        Before the student has trained, this reduces to the teacher scores.
        """
        if hasattr(self.teacher, "infer") and callable(self.teacher.infer):
            logits_teacher = self._call_with_supported_args(self.teacher.infer, **kwargs)
        else:
            logits_teacher = self._call_with_supported_args(self.teacher.think, **kwargs)

        weight = float(np.clip(getattr(self, "_student_weight", 0.0), 0.0, 1.0))
        if weight <= 0.0 or not hasattr(self.student, "think"):
            if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
                _d("DualRec.infer: weight=0 -> teacher only")
            return logits_teacher

        logits_student = self._call_with_supported_args(self.student.think, **kwargs)
        try:
            out = torch.lerp(logits_teacher, logits_student, weight)
        except Exception:
            out = (1.0 - weight) * logits_teacher + weight * logits_student
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            _d(f"DualRec.infer: blend weight={weight:.3f} logits_shape={tuple(out.shape)}")
        return out

    def think(self, **kwargs):
        """
        Generic 'score' entry point.
        If you still call rec.think() from elsewhere, we route:
          - policy == "adaptive" -> student
          - otherwise            -> teacher
        Any extra kwargs (like policy) are filtered out if unsupported.
        """
        policy = kwargs.get("policy", "adaptive")
        target = self.student if str(policy).lower() == "adaptive" else self.teacher
        # prefer infer if available (read-only); think otherwise
        if hasattr(target, "think") and callable(target.think):
            out = self._call_with_supported_args(target.think, **kwargs)
        elif hasattr(target, "infer") and callable(target.infer):
            out = self._call_with_supported_args(target.infer, **kwargs)
        else:
            raise AttributeError("Underlying model lacks think()/infer().")
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            _d(f"DualRec.think: policy={policy} -> {target.__class__.__name__} shape={tuple(out.shape)}")
        return out

    # ---------- selection ----------
    def decide(self, **kwargs):
        """
        Use teacher's selection logic if present; otherwise provide a robust fallback.
        Expected kwargs (we'll filter): logits, top_k, item_ids, user_id, engagement, trust, ...
        """
        if hasattr(self.teacher, "decide") and callable(self.teacher.decide):
            out = self._call_with_supported_args(self.teacher.decide, **kwargs)
            if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
                top_k = kwargs.get("top_k", None)
                _d(f"DualRec.decide: teacher path top_k={top_k}")
            return out

        # Fallback: top-k by logits (expects logits: (1, C) or (C,))
        logits = kwargs.get("logits")
        top_k  = int(kwargs.get("top_k", 5))
        item_ids = kwargs.get("item_ids")  # list of candidate *positions*
        if logits is None:
            raise ValueError("decide() fallback needs logits.")
        scores = logits.view(-1)
        k = min(top_k, scores.numel())
        vals, idx = torch.topk(scores, k=k, largest=True)
        # If item_ids provided, map local indices to candidate positions
        if item_ids is not None:
            chosen_pos = [int(item_ids[int(i)]) for i in idx.tolist()]
        else:
            chosen_pos = idx.tolist()
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            _d(f"DualRec.decide: fallback k={k}")
        return chosen_pos, vals.tolist()

    # ---------- training / updates ----------
    def update(self, **kwargs):
        """
        For compatibility with your older path (privacy-aware update).
        We forward only to the student. Extra args are filtered.
        """
        if hasattr(self.student, "update") and callable(self.student.update):
            out = self._call_with_supported_args(self.student.update, **kwargs)
            self._after_student_update()
            return out
        # If there is no privacy-aware update, fall back to train_step when possible
        if hasattr(self.student, "train_step") and callable(self.student.train_step):
            out = self._call_with_supported_args(self.student.train_step, **kwargs)
            self._after_student_update()
            if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
                _d(f"DualRec.update: train_step fallback loss={out.get('loss') if isinstance(out, dict) else out}")
            return out if isinstance(out, dict) else {"loss": float(out)}
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            _d("DualRec.update: no update method available")
        return {"loss": 0.0, "note": "no-update-implemented"}

    def train_step(self, batch: dict):
        """
        Single-writer training step used by the orchestrator's Phase B.
        Always trains the student.
        """
        if hasattr(self.student, "train_step") and callable(self.student.train_step):
            out = self.student.train_step(batch)
            # push to replay and perform extra small updates
            if self._replay_buf is not None:
                try:
                    self._replay_buf.append(batch)
                except Exception:
                    pass
                for _ in range(max(0, self._replay_steps)):
                    try:
                        b = self._replay_buf[0]
                        self.student.train_step(b)
                    except Exception:
                        break
            self._after_student_update()
            if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
                _d(f"DualRec.train_step: loss={out.get('loss') if isinstance(out, dict) else out}")
            return out
        # best-effort fallback to update()
        out = self.update(**batch)
        return out


__all__ = [
    "DualRecommender",
]
