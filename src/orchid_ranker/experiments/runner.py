"""High-level experiment driver for Orchid Ranker."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt  # kept if you plot elsewhere
import pandas as pd
import torch

from orchid_ranker import (
    MultiConfig,
    MultiUserOrchestrator,
    StudentAgentFactory,
    TwoTowerRecommender,
)
from orchid_ranker.agents.agentic import UserCtx
from orchid_ranker.agents.recommender_agent import DualRecommender
from orchid_ranker.data import DatasetLoader
from orchid_ranker.dp import get_dp_config
from orchid_ranker.baselines import (
    ALSBaseline,
    LinUCBBaseline,
    PopularityBaseline,
    RandomBaseline,
    UserKNNBaseline,
)
from itertools import product


# ---------------- internal loggers ----------------

def _fmt_scalar_or_mean(x, ndigits=3):
    try:
        import numpy as np
        if isinstance(x, (list, tuple, np.ndarray)):
            return f"{float(np.mean(x)):.{ndigits}f}"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "nan"

class _CompositeLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.records: List[dict] = []

    def log(self, obj: dict) -> None:
        self.records.append(obj)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


class _MemoryLogger:
    def __init__(self) -> None:
        self.records: List[dict] = []

    def log(self, obj: dict) -> None:
        self.records.append(obj)


# ---------------- helpers ----------------

def _resolve_id_column(df: pd.DataFrame, preferred: str, fallbacks: tuple[str, ...] = ()) -> str:
    if preferred in df.columns:
        return preferred
    for col in fallbacks + ("u", "i", "user_id", "item_id", "id"):
        if col in df.columns:
            return col
    return df.columns[0]


def _build_feature_matrix(
    df: pd.DataFrame,
    id_col: str,
    device: torch.device,
    *,
    kind: str = "items",
    verbose: bool = True,
):
    """
    Robust builder:
    - Resolves the ID column.
    - Keeps ONLY numeric feature columns (avoids object->NaN coercion).
    - Cleans inf/-inf -> NaN -> 0.0.
    """
    if df.empty:
        return np.arange(0, dtype=int), torch.zeros((0, 0), dtype=torch.float32, device=device)

    resolved = _resolve_id_column(df, id_col)
    ids = df[resolved].to_numpy()

    rem = df.drop(columns=[resolved], errors="ignore")
    num = rem.select_dtypes(include=[np.number]).copy()
    dropped = [c for c in rem.columns if c not in num.columns]

    if not num.empty:
        num.replace([np.inf, -np.inf], np.nan, inplace=True)
        num.fillna(0.0, inplace=True)

    feats_np = num.to_numpy(dtype=np.float32) if not num.empty else np.zeros((len(df), 0), dtype=np.float32)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)

    if verbose:
        kept = list(num.columns)
        msg = f"[{kind}] id_col='{resolved}', kept_num={len(kept)}, dropped_non_num={len(dropped)}"
        if dropped:
            msg += f", dropped={dropped[:8]}{'…' if len(dropped) > 8 else ''}"
        print("[RankingExperiment]", msg)

    return ids, feats


def _prepare_item_meta(df: pd.DataFrame) -> Dict[int, dict]:
    meta = {}
    if df.empty:
        return meta
    id_col = _resolve_id_column(df, "i", ("item_id",))
    for row in df.to_dict(orient="records"):
        iid = int(row[id_col])
        meta[iid] = {k: v for k, v in row.items() if k != id_col}
    return meta


def _flatten_round_records(records: List[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("type") != "round_summary":
            continue
        metric = rec.get("metrics", {})
        rows.append({"round": rec.get("round"), "mode": rec.get("mode"), **metric})
    return pd.DataFrame(rows)


def _flatten_user_records(records: List[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("type") != "user_round":
            continue
        base = {
            "round": rec.get("round"),
            "mode": rec.get("mode"),
            "user_id": rec.get("user_id"),
            "student_method": rec.get("student_method"),
            "profile": rec.get("profile"),
        }
        tel = rec.get("telemetry", {}) or {}
        pre = ((rec.get("state_estimator") or {}).get("pre")) or {}
        post = ((rec.get("state_estimator") or {}).get("post")) or {}
        knobs = rec.get("knobs", {}) or {}

        # telemetry
        for k, v in tel.items():
            base[f"tel_{k}"] = v
        # knobs
        for k, v in knobs.items():
            base[f"knob_{k}"] = v
        # pre / post
        for k, v in pre.items():
            base[f"pre_{k}"] = v
        for k, v in post.items():
            base[f"post_{k}"] = v

        rows.append(base)
    return pd.DataFrame(rows)



# ---------------- data classes ----------------

@dataclass
class SummaryRow:
    mode: str
    accuracy: float
    accept_rate: float
    novelty_rate: float
    serendipity: float
    mean_knowledge: float
    epsilon_cum: float
    mean_engagement: float = float("nan")


# ---------------- main class ----------------

class RankingExperiment:
    """One-stop helper to run adaptive vs. baseline policies."""

    def __init__(
        self,
        config_path: str,
        dataset: Optional[str] = None,
        *,
        encoding: str = "onehot",
        cohort_size: int = 16,
        seed: int = 42,
        device: Optional[str] = None,
        student_methods: Optional[List[str]] = None,
        initial_profiles: Optional[List[dict]] = None,
        assignment_mode: str = "cycle",  # "cycle" or "cartesian"
        verbose: bool = True,
    ) -> None:

        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        loader = DatasetLoader(encoding=encoding)
        bundle, meta = loader.load_from_yaml(config_path, dataset=dataset, encode_side_info=True)

        # Raw tables
        self.dataset_cfg = meta["config"]
        self.train = bundle["train"].copy()
        self.val = bundle["val"].copy()
        self.test = bundle["test"].copy()
        self.side_users = bundle["side_information_users"].copy()
        self.side_items = bundle["side_information_items"].copy()

        # --- Repair/derive item difficulty for StudentAgent ---
        si = self.side_items.copy()

        def _derive_difficulty_from_side(si: pd.DataFrame) -> Optional[pd.Series]:
            # prefer success-like signals if available
            for cand in ("decayed_success", "recent_success_4w", "mean_success"):
                if cand in si.columns:
                    s = pd.to_numeric(si[cand], errors="coerce")
                    s = (s - s.min()) / (s.max() - s.min() + 1e-8)  # normalize to [0,1]
                    return 1.0 - s  # harder if success is low
            return None

        def _derive_difficulty_from_interactions(self) -> Optional[pd.Series]:
            # fallback: use per-item correctness/label from train
            label_col = None
            for c in ("correct", "label"):
                if c in self.train.columns:
                    label_col = c; break
            if label_col is None or self.train.empty:
                return None
            suc = (
                self.train.groupby(self.interaction_item_col)[label_col]
                .mean()
                .rename("success")
            )
            suc = (suc - suc.min()) / (suc.max() - suc.min() + 1e-8)
            diff = 1.0 - suc
            # map back onto side_items by item id
            sid = _resolve_id_column(si, "i", ("item_id",))
            return pd.Series(diff, name="difficulty").reindex(si[sid].astype(int)).reset_index(drop=True)

        # 1) decide whether the existing column is usable
        use_existing = ("difficulty" in si.columns) and (pd.to_numeric(si["difficulty"], errors="coerce").std(skipna=True) > 1e-8)

        if not use_existing:
            diff = _derive_difficulty_from_side(si)
            if diff is None:
                diff = _derive_difficulty_from_interactions(self)
            if diff is None:
                diff = pd.Series(0.5, index=si.index)  # last-resort constant
            si["difficulty"] = diff

        # 2) clamp & de-zero
        si["difficulty"] = pd.to_numeric(si["difficulty"], errors="coerce").fillna(0.5)
        d = si["difficulty"]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d = d.clip(0.0, 1.0)
        d.loc[d <= 0.0] = 0.05  # tiny floor to avoid all-zeros downstream
        si["difficulty"] = d

        print("[Difficulty] describe:\n", si["difficulty"].describe())
        print("[Difficulty] zeros%:", (si["difficulty"] <= 0).mean() * 100)

        # 3) build meta from the repaired frame
        self.side_items = si
        self.item_meta = _prepare_item_meta(self.side_items)


        # 1) Resolve interaction columns FIRST (truth source for IDs)
        self.interaction_user_col = _resolve_id_column(self.train, "u") if not self.train.empty else "u"
        self.interaction_item_col = _resolve_id_column(self.train, "i") if not self.train.empty else "i"

        # ---------- FIX/REPAIR side_users to ensure correct (unique) user IDs ----------
        if not self.side_users.empty:
            # Prefer to use the same ID name as interactions when present
            if self.interaction_user_col in self.side_users.columns:
                self.user_col = self.interaction_user_col
            else:
                self.user_col = _resolve_id_column(self.side_users, "u")

            su = self.side_users.copy()

            # Coerce ID column to int where possible
            try:
                su[self.user_col] = su[self.user_col].astype("int64")
            except Exception:
                # if coercion fails (e.g., mixed types), use pandas to_numeric (coerce), drop NaNs
                su[self.user_col] = pd.to_numeric(su[self.user_col], errors="coerce").astype("Int64")

            # Drop rows with missing user IDs
            before = len(su)
            su = su.dropna(subset=[self.user_col]).copy()
            after = len(su)
            if self.verbose and after != before:
                self._p(f"[repair] dropped {before - after} side_users rows with null IDs")

            # Deduplicate on the ID column
            before = len(su)
            su = su.drop_duplicates(subset=[self.user_col], keep="first").copy()
            if self.verbose and len(su) != before:
                self._p(f"[repair] removed {before - len(su)} duplicate side_users rows on '{self.user_col}'")

            # If IDs are still suspicious (e.g., constant or tiny), rebuild from interactions
            su_ids = su[self.user_col].dropna().astype("int64").to_numpy()
            unique_count = np.unique(su_ids).size
            needs_rebuild = (unique_count != su_ids.size) or (unique_count <= 1)

            # Also rebuild if the interaction table has many more users than side_users
            if not self.train.empty:
                inter_unique = (
                    self.train[self.interaction_user_col]
                    .dropna()
                    .astype("int64")
                    .unique()
                )
                if unique_count < max(2, int(0.1 * len(inter_unique))):  # clearly underspecified
                    needs_rebuild = True
            else:
                inter_unique = np.array([], dtype=np.int64)

            if needs_rebuild:
                if self.verbose:
                    self._p("[repair] rebuilding side_users from interactions’ unique users")

                base = pd.DataFrame({self.interaction_user_col: inter_unique.astype("int64")})
                # Try to bring over any compatible columns by merging on whichever col matches
                merge_key = self.user_col if self.user_col in su.columns else None
                if merge_key and merge_key != self.interaction_user_col:
                    su_renamed = su.rename(columns={merge_key: self.interaction_user_col})
                else:
                    su_renamed = su

                # Only keep one row per user in su_renamed
                su_renamed = su_renamed.drop_duplicates(subset=[self.interaction_user_col], keep="first")
                side_users_fixed = base.merge(su_renamed, on=self.interaction_user_col, how="left")

                # Fill numeric columns’ NaNs with 0.0
                num_cols = side_users_fixed.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    side_users_fixed[num_cols] = side_users_fixed[num_cols].fillna(0.0)

                self.side_users = side_users_fixed
                self.user_col = self.interaction_user_col
            else:
                # Use the cleaned version
                self.side_users = su
        else:
            self.user_col = "u"

        # ---------- Force side_items to share the interaction item ID name when present ----------
        if not self.side_items.empty:
            if self.interaction_item_col in self.side_items.columns:
                self.item_side_col = self.interaction_item_col
            else:
                self.item_side_col = _resolve_id_column(self.side_items, "i")
        else:
            self.item_side_col = "i"

        # 3) Build numeric-only feature matrices (robust to objects)
        if not self.side_users.empty:
            su = self.side_users.rename(columns={self.user_col: "__u__"})
            self.user_ids, self.user_matrix = _build_feature_matrix(
                su, "__u__", self.device, kind="users", verbose=self.verbose
            )
        else:
            self.user_ids, self.user_matrix = (
                np.arange(0, dtype=int),
                torch.zeros((0, 0), dtype=torch.float32, device=self.device),
            )

        if not self.side_items.empty:
            si = self.side_items.rename(columns={self.item_side_col: "__i__"})
            self.item_ids, self.item_matrix = _build_feature_matrix(
                si, "__i__", self.device, kind="items", verbose=self.verbose
            )
        else:
            self.item_ids, self.item_matrix = (
                np.arange(0, dtype=int),
                torch.zeros((0, 0), dtype=torch.float32, device=self.device),
            )

        # 4) sanity: IDs must be unique
        if self.verbose:
            self._p(f"user_id sample: {self.user_ids[:5]}")
            self._p(f"item_id sample: {self.item_ids[:5]}")
        assert len(np.unique(self.user_ids)) == len(self.user_ids), "Side-users ID column seems wrong (duplicates/constant)."
        assert len(np.unique(self.item_ids)) == len(self.item_ids), "Side-items ID column seems wrong (duplicates/constant)."

        # 5) mappings + meta
        self.user_index = {int(uid): idx for idx, uid in enumerate(self.user_ids)}
        self.id2pos = {int(i): idx for idx, i in enumerate(self.item_ids)}
        self.pos2id = {idx: int(i) for idx, i in enumerate(self.item_ids)}
        self.item_meta = _prepare_item_meta(self.side_items)

        # cohort + methods + (authoritative) initial profiles
        self.student_methods = student_methods or ["irt", "mirt", "zpd", "contextual_zpd"]
        self.initial_profiles = initial_profiles or []
        self.assignment_mode = str(assignment_mode or "cycle").lower()

        cohort = self._sample_users(cohort_size)
        self._user_specs: List[tuple[int, int, str, Optional[dict]]] = []
        # Prepare assignment strategy
        use_cartesian = (
            self.assignment_mode == "cartesian"
            and len(self.student_methods) > 0
            and len(self.initial_profiles) > 0
        )

        combos: List[tuple[str, dict]] = []
        if use_cartesian:
            combos = list(product(self.student_methods, self.initial_profiles))

        for j, uid_ext in enumerate(cohort):
            uid_idx = self.user_index.get(uid_ext)
            if uid_idx is None:
                continue
            if use_cartesian and len(combos) > 0:
                # First take unique combos, then cycle if cohort exceeds Cartesian size
                m, p = combos[j] if j < len(combos) else combos[j % len(combos)]
                method = str(m)
                profile = dict(p)
            else:
                # Default: cycle each list independently by index
                method = self.student_methods[j % len(self.student_methods)]
                profile = self.initial_profiles[j % len(self.initial_profiles)] if self.initial_profiles else None
            self._user_specs.append((uid_ext, uid_idx, method, profile))

        # config
        self.rounds = int(self.dataset_cfg.get("rounds", 50))
        self.top_k_base = int(self.dataset_cfg.get("top_k_base", 5))
        self.zpd_margin = float(self.dataset_cfg.get("zpd_margin", 0.12))
        self.student_zpd_delta = float(self.dataset_cfg.get("student_zpd_delta", self.zpd_margin if "zpd_margin" in self.dataset_cfg else 0.10))

        self.min_candidates = int(self.dataset_cfg.get("min_candidates", 100))
        self.default_policy_gain = float(self.dataset_cfg.get("policy_gain", 1.25)) if "policy_gain" in self.dataset_cfg else 1.25

        # cached signals
        self.popularity = self._compute_popularity()
        self.user_item_matrix = self._build_user_item_matrix()

        # model defaults
        self.adaptive_defaults: Dict[str, object] = {
            "hidden": 96,
            "emb_dim": 48,
            "use_linucb": True,
            "linucb_alpha": 1.6,
            "use_bootts": True,
            "ts_heads": 12,
            "ts_alpha": 0.9,
            "adapter_slots": 512,
            "kl_beta": 0.02,
            "blend_increment": 0.16,
            "teacher_ema": 0.9,
            "entropy_lambda": 0.08,
            "info_gain_lambda": 0.12,
        }
        self.warm_start_defaults: Dict[str, object] = {
            "enabled": True,
            "epochs": 3,
            "batch_size": 256,
            "max_batches": 360,
        }
        self._warm_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        # --- DEBUG SNAPSHOT ---
        self._p(f"device={self.device}, seed={seed}, rounds={self.rounds}, "
                f"top_k_base={self.top_k_base}, min_candidates={self.min_candidates}")
        self._p(f"users: {len(self.user_ids)}  items: {len(self.item_ids)}  cohort={len(self._user_specs)}")
        self._vec_stats(self.user_matrix, "user_matrix")
        self._vec_stats(self.item_matrix, "item_matrix")

        try:
            if len(self.popularity) > 0:
                pop_vals = np.array(list(self.popularity.values()), dtype=np.float32)
                self._p(f"popularity: items={len(pop_vals)}, mean={pop_vals.mean():.4f}, "
                        f"std={pop_vals.std():.4f}, min={pop_vals.min():.4f}, max={pop_vals.max():.4f}")
            else:
                self._p("popularity: empty")
        except Exception as e:
            self._p(f"popularity: stats error -> {e}")

        wc = self._build_warm_cache()
        if wc is None:
            self._p("warm_cache: None")
        else:
            u, i, y = wc
            self._p(f"warm_cache: n={len(y)} (user_idx {u.min() if len(u)>0 else 'NA'}..{u.max() if len(u)>0 else 'NA'})")
        # config
        self.rounds = int(self.dataset_cfg.get("rounds", 50))
        self.top_k_base = int(self.dataset_cfg.get("top_k_base", 5))

        # ZPD settings (experiment-level)
        self.zpd_margin = float(self.dataset_cfg.get("zpd_margin", 0.12))
        self.zpd_bounds = tuple(self.dataset_cfg.get("zpd_bounds", (0.06, 0.22)))
        self.student_zpd_delta = float(self.dataset_cfg.get("student_zpd_delta", self.zpd_margin))
        # choose a generous width unless explicitly provided
        self.student_zpd_width = float(self.dataset_cfg.get(
            "student_zpd_width",
            max(0.30, (self.zpd_bounds[1] - self.zpd_bounds[0]) / 2.0)
        ))
    # ---------- small debug helpers ----------
    def _p(self, *args):
        if self.verbose:
            print("[RankingExperiment]", *args)

    def _vec_stats(self, t: torch.Tensor, name: str):
        try:
            if t.numel() == 0:
                self._p(f"{name}: empty tensor")
                return
            nan_cnt = int(torch.isnan(t).sum().item())
            std_all = float(t.std().item()) if t.numel() > 1 else 0.0
            zero_var_cols = 0
            if t.ndim == 2 and t.shape[1] > 0:
                col_std = t.std(dim=0)
                zero_var_cols = int((col_std == 0).sum().item())
            self._p(f"{name}: shape={tuple(t.shape)}, std(all)={std_all:.6f}, "
                    f"NaNs={nan_cnt}, zero-var-cols={zero_var_cols}")
        except Exception as e:
            self._p(f"{name}: stats error -> {e}")
    # -----------------------------------------

    # ---------- profile plumbing (authoritative) ----------
    def _normalize_profile(self, profile: dict) -> dict:
        if not isinstance(profile, dict):
            return {}
        alias_map = {
            # engagement
            "E": "engagement", "eng": "engagement", "Engagement": "engagement", "engagement": "engagement",
            # trust
            "T": "trust", "trs": "trust", "Trust": "trust", "trust": "trust",
            # knowledge / ability / theta
            "K": "knowledge", "kn": "knowledge", "know": "knowledge", "Knowledge": "knowledge",
            "ability": "knowledge", "theta": "knowledge", "Theta": "knowledge", "skill": "knowledge",
            # fatigue
            "F": "fatigue", "fat": "fatigue", "Fatigue": "fatigue", "fatigue": "fatigue",
        }
        out: dict = {}
        for k, v in profile.items():
            if k == "name":
                out["name"] = v
                continue
            key = alias_map.get(k, k)
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if key in {"engagement", "trust", "knowledge", "fatigue"}:
                val = max(0.0, min(1.0, val))
            out[key] = val
        return out

    def _apply_initial_latents(self, student, profile_norm: dict):
        if not profile_norm:
            return
        # preferred: explicit setter if provided
        if hasattr(student, "set_initial_latents") and callable(student.set_initial_latents):
            kwargs = {k: v for k, v in profile_norm.items() if k != "name"}
            try:
                student.set_initial_latents(**kwargs)
                return
            except TypeError:
                pass
        # fallback: setattr
        for k, v in profile_norm.items():
            if k == "name":
                continue
            if hasattr(student, k):
                setattr(student, k, v)
    # -----------------------------------------------------

    # ------------------------------------------------------------------
    def _sample_users(self, cohort_size: int) -> List[int]:
        if self.side_users.empty:
            return []
        col = self.user_col
        unique_users = [int(uid) for uid in self.side_users[col].unique()]
        if cohort_size < len(unique_users):
            sampled = self.rng.choice(unique_users, size=cohort_size, replace=False)
        else:
            sampled = unique_users
        return [int(u) for u in sampled]

    def _make_user_contexts(self) -> List[UserCtx]:
        self._p(f"building user contexts for {len(self._user_specs)} users (methods cycle={self.student_methods})")
        users = []
        for uid_ext, uid_idx, method, profile in self._user_specs:
            # use experiment-level ZPD knobs stored on self
            zpd_delta = self.student_zpd_delta
            zpd_width = self.student_zpd_width

            student = StudentAgentFactory.create(
                method,
                user_id=uid_ext,
                zpd_delta=zpd_delta,
                zpd_width=zpd_width,   # <-- use the computed width
                verbose=True,
            )

            # apply your profile (authoritative if provided)
            profile_norm = self._normalize_profile(profile) if profile else {}
            if profile_norm:
                self._apply_initial_latents(student, profile_norm)
                prof_name = profile_norm.get("name", None)
            else:
                prof_name = getattr(student, "profile_name", None)

            if prof_name is not None:
                setattr(student, "profile_name", prof_name)

            user_vec = self.user_matrix[uid_idx].unsqueeze(0)

            if self.verbose:
                self._p(
                    f"user {uid_ext} -> idx {uid_idx}  method={method}  "
                    f"profile_name={prof_name}  "
                    f"init(E,T,K,F)=("
                    # f"{getattr(student, 'engagement', float('nan')):.3f},"
                    # f"{getattr(student, 'trust', float('nan')):.3f},"
                    # f"{getattr(student, 'knowledge', float('nan')):.3f},"
                    # f"{getattr(student, 'fatigue', float('nan')):.3f})"
                    f"{_fmt_scalar_or_mean(getattr(student, 'engagement', 0.6))},"
                    f"{_fmt_scalar_or_mean(getattr(student, 'trust', 0.5))},"
                    f"{_fmt_scalar_or_mean(getattr(student, 'knowledge', 0.5))},"
                    f"{_fmt_scalar_or_mean(getattr(student, 'fatigue', 0.2))}"
                )

            users.append(UserCtx(
                user_id=uid_ext,
                user_idx=uid_idx,
                student=student,
                user_vec=user_vec,
                profile=prof_name
            ))
        return users
    

    def _compute_popularity(self) -> Dict[int, float]:
        grouped = self.train.groupby(self.interaction_item_col)["label"].mean().to_dict()
        return {int(k): float(v) for k, v in grouped.items()}

    def _build_user_item_matrix(self) -> np.ndarray:
        mat = np.zeros((len(self.user_ids), len(self.item_ids)), dtype=np.float32)
        for row in self.train.itertuples():
            uid = self.user_index.get(int(getattr(row, self.interaction_user_col)))
            iid_raw = int(getattr(row, self.interaction_item_col))
            iid_pos = self.id2pos.get(iid_raw)
            if uid is None or iid_pos is None:
                continue
            mat[uid, iid_pos] += float(getattr(row, "label", 0.0))
        counts = (mat > 0).astype(np.float32).sum(axis=1, keepdims=True) + 1e-6
        return mat / counts

    def _build_warm_cache(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._warm_cache is not None:
            return self._warm_cache
        if self.train.empty:
            self._warm_cache = None
            return None
        label_col = None
        for candidate in ("label", "correct", "accept"):
            if candidate in self.train.columns:
                label_col = candidate
                break
        if label_col is None:
            self._warm_cache = None
            return None

        users: List[int] = []
        items: List[int] = []
        labels: List[float] = []
        for row in self.train.itertuples():
            uid_ext = int(getattr(row, self.interaction_user_col))
            iid_ext = int(getattr(row, self.interaction_item_col))
            uid = self.user_index.get(uid_ext)
            iid = self.id2pos.get(iid_ext)
            if uid is None or iid is None:
                continue
            users.append(uid)
            items.append(iid)
            labels.append(float(getattr(row, label_col, 0.0)))

        if not users:
            self._warm_cache = None
            return None

        self._warm_cache = (
            np.asarray(users, dtype=np.int64),
            np.asarray(items, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
        )
        return self._warm_cache

    def _warm_start_recommender(
        self,
        model: TwoTowerRecommender,
        *,
        epochs: int,
        batch_size: int,
        max_batches: Optional[int],
    ) -> None:
        cache = self._build_warm_cache()
        if cache is None:
            self._p("warm_start: skipped (no cache)")
            return
        user_idx, item_idx, labels = cache
        total = int(len(labels))
        if total == 0:
            self._p("warm_start: skipped (empty cache)")
            return

        device = getattr(model, "device", self.device)
        item_matrix = getattr(model, "item_matrix", None)
        if item_matrix is None:
            item_matrix = self.item_matrix
        if item_matrix.device != device:
            item_matrix = item_matrix.to(device)
        state_dim = int(getattr(model, "state_dim", 0))

        epochs = max(1, int(epochs))
        batch_size = max(1, int(batch_size))
        max_batches = int(max_batches) if max_batches is not None else None

        self._p(f"warm_start: epochs={epochs}, batch_size={batch_size}, "
                f"max_batches={max_batches}, total_pairs={total}")

        batches = 0
        for _ in range(epochs):
            order = self.rng.permutation(total)
            for start in range(0, total, batch_size):
                if max_batches is not None and batches >= max_batches:
                    self._p(f"warm_start: reached max_batches={max_batches}")
                    model.eval()
                    return
                idx = order[start:start + batch_size]
                if idx.size == 0:
                    continue
                u = torch.tensor(user_idx[idx], dtype=torch.long, device=device)
                it = torch.tensor(item_idx[idx], dtype=torch.long, device=device)
                y = torch.tensor(labels[idx], dtype=torch.float32, device=device)
                batch = {
                    "user_ids": u,
                    "item_ids": it,
                    "labels": y,
                    "item_matrix": item_matrix,
                }
                if state_dim > 0:
                    batch["state_vec"] = torch.zeros((len(idx), state_dim), dtype=torch.float32, device=device)
                try:
                    model.train_step(batch)
                except TypeError:
                    batch.pop("state_vec", None)
                    model.train_step(batch)
                batches += 1
                if batches % 50 == 0:
                    self._p(f"warm_start: batches={batches}")
                if max_batches is not None and batches >= max_batches:
                    self._p(f"warm_start: reached max_batches={max_batches}")
                    model.eval()
                    return
        model.eval()
        self._p(f"warm_start: done, batches={batches}")

    def _make_config(self, *, console: bool, policy_gain: float) -> MultiConfig:
        return MultiConfig(
            rounds=self.rounds,
            top_k_base=self.top_k_base,
            zpd_margin=self.zpd_margin,
            min_candidates=self.min_candidates,
            novelty_bonus=0.10,
            mmr_lambda=0.25,
            log_path="runs/auto_log.jsonl",
            console=console,
            shuffle_users_each_round=True,
            privacy_mode="standard",
            share_signals=False,
            per_round_eps_target=0.0,
            policy_gain=policy_gain,
        )

    def _build_adaptive(
        self,
        dp_cfg: dict,
        adaptive_kwargs: Optional[Dict[str, object]] = None,
        warm_cfg: Optional[Dict[str, object]] = None,
    ):
        tower_kwargs = dict(self.adaptive_defaults)
        if adaptive_kwargs:
            tower_kwargs.update(adaptive_kwargs)

        blend_increment = float(tower_kwargs.pop("blend_increment", self.adaptive_defaults.get("blend_increment", 0.16)))
        teacher_ema = float(tower_kwargs.pop("teacher_ema", self.adaptive_defaults.get("teacher_ema", 0.9)))

        warm_defaults = dict(self.warm_start_defaults)
        if warm_cfg:
            for key, value in warm_cfg.items():
                if value is not None:
                    warm_defaults[key] = value
        warm_enabled = bool(warm_defaults.get("enabled", True))
        warm_epochs = int(warm_defaults.get("epochs", 0))
        warm_batch = int(warm_defaults.get("batch_size", 1))
        warm_max_batches = warm_defaults.get("max_batches")
        if warm_max_batches is not None:
            warm_max_batches = int(warm_max_batches)

        teacher = TwoTowerRecommender(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            user_dim=self.user_matrix.shape[1],
            item_dim=self.item_matrix.shape[1],
            dp_cfg={**dp_cfg, "enabled": False},
            **tower_kwargs,
        )
        student = TwoTowerRecommender(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            user_dim=self.user_matrix.shape[1],
            item_dim=self.item_matrix.shape[1],
            dp_cfg=dp_cfg,
            **tower_kwargs,
        )

        for model in (teacher, student):
            setattr(model, "user_matrix", self.user_matrix)
            setattr(model, "item_matrix", self.item_matrix)
            setattr(model, "pos2id_map", dict(self.pos2id))

        if warm_enabled and warm_epochs > 0:
            self._warm_start_recommender(
                teacher,
                epochs=warm_epochs,
                batch_size=warm_batch,
                max_batches=warm_max_batches,
            )

        student.load_state_dict(teacher.state_dict())
        setattr(student, "blend_increment", blend_increment)
        setattr(student, "teacher_ema", teacher_ema)
        setattr(teacher, "blend_increment", blend_increment)
        setattr(teacher, "teacher_ema", teacher_ema)

        if hasattr(student, "linucb"):
            teacher.linucb = student.linucb
            teacher.use_linucb = getattr(student, "use_linucb", False)
        if hasattr(student, "bootts"):
            teacher.bootts = student.bootts
            teacher.use_bootts = getattr(student, "use_bootts", False)

        return DualRecommender(teacher=teacher, student=student)

    def _build_fixed(self, dp_cfg: dict):
        model = TwoTowerRecommender(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            user_dim=self.user_matrix.shape[1],
            item_dim=self.item_matrix.shape[1],
            dp_cfg=dp_cfg,
        )
        setattr(model, "user_matrix", self.user_matrix)
        setattr(model, "item_matrix", self.item_matrix)
        setattr(model, "pos2id_map", dict(self.pos2id))
        return model

    def _build_baseline(self, mode: str) -> object:
        if mode == "popularity":
            self._p("baseline=popularity")
            return PopularityBaseline(self.popularity, self.device)
        if mode == "random":
            self._p("baseline=random")
            return RandomBaseline(self.device)
        if mode == "als":
            self._p("baseline=als")
            return ALSBaseline(len(self.user_ids), len(self.item_ids), self.device)
        if mode == "user_knn":
            self._p("baseline=user_knn")
            return UserKNNBaseline(self.user_item_matrix, self.device)
        if mode == "linucb":
            self._p("baseline=linucb")
            feats = self.item_matrix.detach().cpu().numpy()
            self._p(f"linucb item_features shape={feats.shape}, "
                    f"std={float(feats.std()) if feats.size>0 else 0.0:.6f}")
            if feats.ndim == 2 and feats.shape[1] == 0:
                self._p("WARNING: LinUCB will degenerate (0 feature columns). "
                        "Consider adding fallback features.")
            return LinUCBBaseline(alpha=1.5, item_features=feats, device=self.device)
        raise ValueError(f"Unknown baseline '{mode}'")

    def _train_baseline(self, model, mode: str) -> None:
        self._p(f"train_baseline: mode={mode}")
        if isinstance(model, ALSBaseline):
            self._p("ALS: fitting from interactions …")
            self._train_als(model)
            self._p("ALS: fit done")
        elif isinstance(model, LinUCBBaseline):
            rewards = self.train.groupby(self.interaction_item_col)["label"].mean().to_dict()
            self._p(f"LinUCB: reward dict size={len(rewards)}")
            model.fit({self.id2pos[int(k)]: float(v) for k, v in rewards.items() if int(k) in self.id2pos})
            self._p("LinUCB: fit done")
        else:
            self._p("no training step for this baseline")
            return

    def _train_als(self, model: ALSBaseline) -> None:
        user_ids = []
        item_ids = []
        labels = []
        for row in self.train.itertuples():
            uid = self.user_index.get(int(getattr(row, self.interaction_user_col)))
            iid_pos = self.id2pos.get(int(getattr(row, self.interaction_item_col)))
            if uid is None or iid_pos is None:
                continue
            user_ids.append(uid)
            item_ids.append(iid_pos)
            labels.append(float(getattr(row, "label", 0.0)))
        model.fit(user_ids, item_ids, labels)

    def run(
        self,
        mode: str = "adaptive",
        *,
        console: bool = False,
        dp_enabled: bool = False,
        dp_params: Optional[dict] = None,
        config_overrides: Optional[dict] = None,
        log_path: Optional[str] = None,
        adaptive_kwargs: Optional[dict] = None,
        warm_start: Optional[dict] = None,
    ) -> Dict[str, object]:
        mode = mode.lower()
        dp_cfg = get_dp_config(dp_params if dp_params is not None else ("off" if not dp_enabled else "eps_1"))
        dp_cfg.setdefault("enabled", dp_enabled)
        policy_gain = self.default_policy_gain if mode == "adaptive" else 1.0
        cfg = self._make_config(console=console, policy_gain=policy_gain)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                elif key in dp_cfg:
                    dp_cfg[key] = value

        self._p(f"RUN mode={mode}")
        self._p(f"cfg: rounds={cfg.rounds}, top_k_base={cfg.top_k_base}, zpd_margin={cfg.zpd_margin}, "
                f"min_cand={cfg.min_candidates}, privacy_mode={getattr(cfg, 'privacy_mode', 'standard')}, "
                f"policy_gain={getattr(cfg, 'policy_gain', 1.0)}")
        self._p(f"dp_cfg: enabled={dp_cfg.get('enabled', False)}, "
                f"eps={dp_cfg.get('epsilon', dp_cfg.get('target_epsilon'))}, "
                f"delta={dp_cfg.get('delta')}, noise={dp_cfg.get('noise_scale')}")
        log_file = Path(log_path) if log_path else Path("runs") / f"auto_{mode}.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.log_path = str(log_file)
        self._p(f"log_file={log_file}")

        if mode == "adaptive":
            rec = self._build_adaptive(dp_cfg, adaptive_kwargs=adaptive_kwargs, warm_cfg=warm_start)
            print(f"[PolicyCfg] zpd_margin={adaptive_kwargs.get('zpd_margin')} "
                  f"use_linucb={adaptive_kwargs.get('use_linucb')} use_bootts={adaptive_kwargs.get('use_bootts')}")
        elif mode == "fixed":
            rec = self._build_fixed(dp_cfg)
        else:
            rec = self._build_baseline(mode)
            self._train_baseline(rec, mode)
            setattr(rec, "user_matrix", self.user_matrix)
            setattr(rec, "item_matrix", self.item_matrix)

        users = self._make_user_contexts()
        self._p(f"orchestrator: item_matrix_normal shape={tuple(self.item_matrix.shape)}, "
                f"item_matrix_sanitized=None")

        orchestrator = MultiUserOrchestrator(
            rec=rec,
            users=users,
            item_matrix_normal=self.item_matrix,
            item_matrix_sanitized=None,
            item_ids_pos=torch.arange(len(self.item_ids), device=self.device, dtype=torch.long),
            pos2id=[self.pos2id[i] for i in range(len(self.item_ids))],
            id2pos=self.id2pos,
            item_meta_by_id=self.item_meta,
            cfg=cfg,
            device=self.device,
            mode_label=mode,
        )

        memory_logger = _MemoryLogger()
        orchestrator.logger = memory_logger
        orchestrator.cfg.console = console

        result = orchestrator.run()
        df_round = _flatten_round_records(memory_logger.records)   # keep if you still want cohort per-round
        df_user  = _flatten_user_records(memory_logger.records)    # <-- NEW

        print(f"df_user: {df_user}")
        print(f"df_round: {df_round}")


        # print(f"result: {result}")
        # df_round = _flatten_round_records(memory_logger.records)
        # print(f"df_round: {df_round}")
        # summary = self._summarise(mode, df_round, result)

        # self._p(f"SUMMARY[{mode}]: "
        #         f"accuracy={summary.accuracy:.4f}  accept_rate={summary.accept_rate:.4f}  "
        #         f"novelty_rate={summary.novelty_rate:.4f}  serendipity={summary.serendipity:.4f}  "
        #         f"mean_knowledge={summary.mean_knowledge:.4f}  epsilon_cum={summary.epsilon_cum:.4f}  "
        #         f"mean_engagement={summary.mean_engagement:.4f}")

        # return {
        #     "mode": mode,
        #     "round_metrics": df_round,
        #     "summary": summary,
        # }
        return {
        "mode": mode,
        "round_metrics": df_round,   # per-round cohort (optional)
        "user_rounds": df_user,      # <-- per-student, per-round with model & pre/post states
    }

    def _summarise(self, mode: str, df_round: pd.DataFrame, result: dict) -> SummaryRow:
        def _coerce(value: object) -> float:
            if value is None:
                return float("nan")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
        if df_round.empty:
            return SummaryRow(
                mode=mode,
                accuracy=np.nan,
                accept_rate=np.nan,
                novelty_rate=np.nan,
                serendipity=np.nan,
                mean_knowledge=np.nan,
                epsilon_cum=float(result.get("epsilon_cum", np.nan)),
                mean_engagement=np.nan,
            )
        last = df_round.sort_values("round").iloc[-1]
        return SummaryRow(
            mode=mode,
            accuracy=_coerce(last.get("accuracy", np.nan)),
            accept_rate=_coerce(last.get("accept_rate", np.nan)),
            novelty_rate=_coerce(last.get("novelty_rate", np.nan)),
            serendipity=_coerce(last.get("serendipity", np.nan)),
            mean_knowledge=_coerce(last.get("mean_knowledge", np.nan)),
            epsilon_cum=float(result.get("epsilon_cum", np.nan)),
            mean_engagement=_coerce(last.get("mean_engagement", np.nan)),
        )

    def run_many(self, modes: Iterable[str], **kwargs) -> pd.DataFrame:
        rows = [asdict(self.run(m, **kwargs)["summary"]) for m in modes]
        return pd.DataFrame(rows)


__all__ = ["RankingExperiment"]
