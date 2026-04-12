# rec_shim.py
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class RecShim:
    """Adapts your TwoTowerRecommender to the simple interface the agents expect."""
    def __init__(self, rec) -> None:
        self.rec = rec
        self.device = rec.device
        self.dp_cfg = getattr(rec, "dp_cfg", {"enabled": False})
        self.eps_cum = getattr(rec, "eps_cum", 0.0)

    def _to_t(self, x, dtype=torch.float32):
        if isinstance(x, torch.Tensor): return x.to(self.device)
        a = np.array(x)
        t = torch.from_numpy(a)
        if dtype is not None and t.dtype != dtype and not t.dtype.is_floating_point:
            return t.to(self.device)
        return t.to(self.device)

    @torch.no_grad()
    def think(self, *, user_vec, item_matrix, user_ids, item_ids, state_vec):
        u = self._to_t(user_vec, torch.float32)
        Xi = self._to_t(item_matrix, torch.float32)
        uids = self._to_t(user_ids, torch.long)
        iids = self._to_t(item_ids, torch.long)
        s = self._to_t(state_vec, torch.float32)
        return self.rec.think(user_vec=u, item_matrix=Xi, user_ids=uids, item_ids=iids, state_vec=s)

    @torch.no_grad()
    def decide(self, logits, top_k: int, item_ids, user_id: int, engagement: float, trust: float):
        return self.rec.decide(logits=logits, top_k=int(top_k), item_ids=item_ids,
                               user_id=int(user_id), engagement=float(engagement), trust=float(trust))

    def _normalize_feedback_keys(self, feedback: Dict[int, int], item_ids):
        """Map feedback keys to the key space used by the wrapped recommender."""
        if not feedback:
            return {}

        try:
            candidate_ids = [int(i) for i in self._to_t(item_ids, torch.long).detach().cpu().tolist()]
        except Exception:
            candidate_ids = [int(i) for i in item_ids]
        candidate_lookup = {iid: idx for idx, iid in enumerate(candidate_ids)}

        pos2id_map = getattr(self.rec, "pos2id_map", {}) or {}
        id2pos_map = {int(ext): int(pos) for pos, ext in pos2id_map.items()}

        normalized = {}
        for raw_key, raw_val in feedback.items():
            key = int(raw_key)
            resolved = None

            if key in candidate_lookup:
                resolved = key
            elif key in pos2id_map:
                ext_id = int(pos2id_map[key])
                if ext_id in candidate_lookup:
                    resolved = ext_id
            elif key in id2pos_map:
                pos = int(id2pos_map[key])
                if pos in candidate_lookup:
                    resolved = pos

            if resolved is not None:
                normalized[int(resolved)] = int(raw_val)

        return normalized

    def update(self, *, feedback: Dict[int,int], user_vec, state_vec, user_ids, item_matrix, item_ids, epochs: int=5):
        u = self._to_t(user_vec, torch.float32)
        s = self._to_t(state_vec, torch.float32)
        uids = self._to_t(user_ids, torch.long)
        Xi = self._to_t(item_matrix, torch.float32)
        iids = self._to_t(item_ids, torch.long)
        normalized_feedback = self._normalize_feedback_keys(feedback, iids)
        out = self.rec.update(feedback=normalized_feedback, user_vec=u, state_vec=s, user_ids=uids,
                              item_matrix=Xi, item_ids=iids, epochs=int(epochs))
        self.eps_cum = float(getattr(self.rec, "eps_cum", self.eps_cum))
        return out

    # pass-through setters the agents change
    @property
    def per_user_only(self): return getattr(self.rec, "per_user_only", False)
    @per_user_only.setter
    def per_user_only(self, v): setattr(self.rec, "per_user_only", bool(v))

    @property
    def mmr_lambda(self): return getattr(self.rec, "mmr_lambda", 0.30)
    @mmr_lambda.setter
    def mmr_lambda(self, v): setattr(self.rec, "mmr_lambda", float(v))


__all__ = [
    "RecShim",
]
