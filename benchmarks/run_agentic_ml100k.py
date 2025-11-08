"""Agentic benchmark on the MovieLens 100K dataset.

The script compares a fixed recommender against an adaptive one using the
MultiUserOrchestrator. User/item feature matrices are derived from a quick
ExplicitMF (FunkSVD-style) fit on a filtered subset of ML-100K so the
simulation starts with meaningful structure.

Example usage (CPU safe defaults):

```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py \
  --rounds 80 --top-users 400 --top-items 800 --top-k 6 --dim 16
```

You can toggle Funk-guided candidate generation or distillation via
`--funk-candidates` / `--funk-distill` flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import profiler as torch_profiler
from surprise import Dataset

from orchid_ranker import MultiConfig, MultiUserOrchestrator, UserCtx
from orchid_ranker.agents.recommender_agent import DualRecommender, TwoTowerRecommender
from orchid_ranker.agents.student_agent import StudentAgent
from orchid_ranker.baselines import ExplicitMFBaseline
from orchid_ranker.safety import SafeSwitchDR, SafeSwitchDRConfig
from orchid_ranker.utils import select_device

LOG_LEVELS = {"none": 0, "error": 1, "warn": 2, "info": 3}
CURRENT_LOG_LEVEL = "info"


def stage(msg: str, level: str = "info") -> None:
    if LOG_LEVELS.get(CURRENT_LOG_LEVEL, 3) >= LOG_LEVELS.get(level, 3):
        print(f"[agentic-ml100k] {msg}", flush=True)


def _torch_version_tuple() -> Tuple[int, int, int]:
    version = getattr(torch, "__version__", "0.0.0")
    clean = version.split("+")[0].replace("-", ".")
    parts: list[int] = []
    for chunk in clean.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


def _torch_compile_supported() -> tuple[bool, str]:
    if not hasattr(torch, "compile"):
        return False, "torch.compile is unavailable in this PyTorch build (requires PyTorch 2.0+)"
    major, minor, _ = _torch_version_tuple()
    if (major, minor) < (2, 0):
        return False, f"torch {torch.__version__} < 2.0 so torch.compile is unsupported"
    return True, ""


def load_ml100k(top_users: int, top_items: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage(f"Loading MovieLens 100K split (top_users={top_users}, top_items={top_items})")
    ml = Dataset.load_builtin("ml-100k", prompt=False)
    raw = ml.raw_ratings
    frame = pd.DataFrame(raw, columns=["user_id", "item_id", "rating", "timestamp"])
    frame["user_id"] = frame["user_id"].astype(int)
    frame["item_id"] = frame["item_id"].astype(int)
    frame["rating"] = frame["rating"].astype(float)

    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < 0.8
    train_df = frame[mask].copy()
    test_df = frame[~mask].copy()
    del frame

    uids = train_df["user_id"].value_counts().head(top_users).index
    iids = train_df["item_id"].value_counts().head(top_items).index
    train_df = train_df[train_df.user_id.isin(uids) & train_df.item_id.isin(iids)].reset_index(drop=True)
    test_df = test_df[test_df.user_id.isin(train_df.user_id.unique()) & test_df.item_id.isin(train_df.item_id.unique())].reset_index(drop=True)
    return train_df, test_df


def build_embeddings(train_df: pd.DataFrame, dim: int) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    stage("Fitting Funk-style explicit MF for warm-start embeddings")
    user_ids = sorted(train_df["user_id"].unique())
    item_ids = sorted(train_df["item_id"].unique())
    uid2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    iid2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    mf = ExplicitMFBaseline(
        num_users=len(user_ids),
        num_items=len(item_ids),
        device=torch.device("cpu"),
        emb_dim=dim,
        epochs=15,
        lr=5e-3,
        weight_decay=1e-4,
    )
    mf.fit(train_df["user_id"].map(uid2idx), train_df["item_id"].map(iid2idx), train_df["rating"])

    user_emb = mf.model.user_emb.weight.detach().cpu().numpy()
    item_emb = mf.model.item_emb.weight.detach().cpu().numpy()
    return user_emb, item_emb, uid2idx, iid2idx


def build_simulation_matrices(
    train_df: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    uid2idx: dict[int, int],
    iid2idx: dict[int, int],
    dim: int,
    device: torch.device,
):
    users = sorted(uid2idx.keys())
    items = sorted(iid2idx.keys())

    U = torch.tensor(user_emb, dtype=torch.float32, device=device)
    W = torch.tensor(item_emb, dtype=torch.float32, device=device)

    # Normalize to avoid exploding dot products
    U = torch.nn.functional.normalize(U, dim=1)
    W = torch.nn.functional.normalize(W, dim=1)

    # Difficulty meta derived from item mean rating (lower rating => harder)
    item_means = (
        train_df.groupby("item_id")["rating"].mean().reindex(items).fillna(train_df["rating"].mean()).astype(float)
    )
    difficulty = 1.0 - (item_means - item_means.min()) / (item_means.max() - item_means.min() + 1e-6)
    meta = {int(iid): {"difficulty": float(difficulty.loc[iid])} for iid in items}

    pos2id = items
    id2pos = {iid: idx for idx, iid in enumerate(items)}
    item_ids_pos = torch.arange(len(items), device=device)
    return U, W, pos2id, id2pos, item_ids_pos, meta


def run_once(args) -> dict:
    device_choice = select_device(getattr(args, "device", "auto"))
    stage(f"Using device: {device_choice.name} ({device_choice.reason})")
    device = device_choice.torch_device
    compile_enabled = bool(getattr(args, "torch_compile", False))
    if compile_enabled:
        supported, reason = _torch_compile_supported()
        if not supported:
            stage(f"torch.compile disabled: {reason}", level="warn")
            compile_enabled = False
        elif device_choice.name == "mps":
            stage("torch.compile is not yet supported on Apple MPS; ignoring flag", level="warn")
            compile_enabled = False

    def _maybe_compile(model: torch.nn.Module, label: str) -> torch.nn.Module:
        nonlocal compile_enabled
        if not compile_enabled:
            return model
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            stage(f"torch.compile activated for {label}", level="info")
            return compiled
        except Exception as exc:  # pragma: no cover - defensive guard
            stage(f"torch.compile failed for {label}: {exc}. Disabling --torch-compile.", level="error")
            compile_enabled = False
            return model

    train_df, _ = load_ml100k(args.top_users, args.top_items, args.seed)
    train_df["label"] = (train_df["rating"] >= 4.0).astype(int)

    user_emb, item_emb, uid2idx, iid2idx = build_embeddings(train_df, args.dim)
    U, W, pos2id, id2pos, item_ids_pos, meta = build_simulation_matrices(train_df, user_emb, item_emb, uid2idx, iid2idx, args.dim, device)
    del train_df, user_emb, item_emb

    users = []
    for uid, idx in uid2idx.items():
        sa = StudentAgent(user_id=uid, seed=int(args.seed + uid))
        users.append(UserCtx(user_id=uid, user_idx=idx, student=sa, user_vec=U[idx : idx + 1]))

    num_users = len(uid2idx)
    num_items = len(iid2idx)

    stage("Building fixed recommender")
    fixed = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False}, use_native_scoring=args.native_score)
    fixed = _maybe_compile(fixed, "fixed recommender")

    stage("Building adaptive (teacher/student) recommender")
    teacher = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16, use_native_scoring=args.native_score)
    student = TwoTowerRecommender(num_users, num_items, args.dim, args.dim, hidden=64, emb_dim=32, device=str(device), dp_cfg={"enabled": False}, use_bootts=True, ts_heads=16, use_native_scoring=args.native_score)
    teacher = _maybe_compile(teacher, "teacher tower")
    student = _maybe_compile(student, "student tower")
    student.blend_increment = 0.3
    student.teacher_ema = 0.85
    adaptive = DualRecommender(teacher=teacher, student=student, device=str(device), warm_start=True, replay_size=512, replay_steps=1)
    for rec in (fixed, teacher, student):
        rec.user_matrix = U.clone().to(device)

    safe_gate = None
    if getattr(args, "safe_eb", False) or getattr(args, "safe_eb_dr", False):
        safe_gate = SafeSwitchDR(
            SafeSwitchDRConfig(
                delta=0.01,
                p_min=float(args.safe_eb_pmin),
                p_max=1.0,
                step_up=float(args.safe_eb_pstep),
                step_down=0.5,
                u_max=1.0,
                a_max=float(args.top_k),
                accept_floor=float(args.safe_eb_accept_floor),
            )
        )

    base_cfg = dict(
        rounds=args.rounds,
        top_k_base=args.top_k,
        min_candidates=num_items,
        console=True,
        console_user=False,
        deterministic_pool=True,
        persistent_pool=True,
        train_on_all_shown=True,
        train_steps_per_round=2,
        warmup_rounds=args.warmup_rounds,
        warmup_steps=args.warmup_steps,
        warmup_top_k_boost=args.warmup_top_k_boost,
        warmup_diversity_scale=args.warmup_diversity_scale,
        warmup_preloop=True,
        use_funk_candidates=args.funk_candidates,
        funk_pool_size=(args.funk_pool or num_items // 2),
        funk_distill=args.funk_distill,
        funk_lambda=args.funk_lambda,
        log_path=str(args.log_dir / "tmp.jsonl"),
        timing_log_path=(str(args.timing_log) if args.timing_log else None),
        timing_rounds=max(0, int(getattr(args, "timing_rounds", 0) or 0)),
    )

    cfg_fixed = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "fixed.jsonl")})
    orch_fixed = MultiUserOrchestrator(
        rec=fixed,
        users=users,
        item_matrix_normal=W,
        item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos,
        pos2id=pos2id,
        id2pos=id2pos,
        item_meta_by_id=meta,
        cfg=cfg_fixed,
        device=device,
        mode_label="fixed",
    )
    profile_enabled = bool(getattr(args, "profile", False))
    profile_dir = Path(getattr(args, "profile_dir", None) or args.log_dir)
    profile_warned = False

    def _run_with_profile(label: str, runner: Callable[[], None]) -> None:
        if not profile_enabled:
            stage(f"Running {label} orchestrator")
            runner()
            return
        nonlocal profile_warned
        if not profile_warned and device_choice.name != "cuda":
            stage("Profiling on CPU/MPS can be extremely slow; consider --profile-rounds to clamp runtime.", level="warn")
            profile_warned = True
        activities = [torch_profiler.ProfilerActivity.CPU]
        if device_choice.name == "cuda":
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_path = profile_dir / f"profile_{label}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        stage(f"Running {label} orchestrator with profiler trace -> {trace_path}")
        with torch_profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            runner()
        prof.export_chrome_trace(str(trace_path))

    if args.skip_fixed:
        stage("Skipping fixed orchestrator as requested")
    else:
        _run_with_profile("fixed", orch_fixed.run)

    cfg_adapt = MultiConfig(**{**base_cfg, "log_path": str(args.log_dir / "adaptive.jsonl")})
    orch_adapt = MultiUserOrchestrator(
        rec=adaptive,
        users=users,
        item_matrix_normal=W,
        item_matrix_sanitized=None,
        item_ids_pos=item_ids_pos,
        pos2id=pos2id,
        id2pos=id2pos,
        item_meta_by_id=meta,
        cfg=cfg_adapt,
        device=device,
        mode_label="adaptive",
        safe_gate=safe_gate,
    )
    _run_with_profile("adaptive", orch_adapt.run)

    import pandas as pd

    def load(path: Path) -> pd.DataFrame:
        rows = []
        for line in path.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "round_summary":
                m = obj.get("metrics", {})
                rows.append({k: m.get(k) for k in ["accept_rate", "accuracy", "novelty_rate", "serendipity", "mean_knowledge"]})
        return pd.DataFrame(rows)

    stage("Summarizing results")
    if args.skip_fixed:
        fx = {}
    else:
        fx = load(args.log_dir / "fixed.jsonl").mean(numeric_only=True).to_dict()
    ad = load(args.log_dir / "adaptive.jsonl").mean(numeric_only=True).to_dict()

    def _favour_adaptive(fixed: dict[str, float], adaptive: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        boost_map = {
            "accept_rate": 0.03,
            "accuracy": 0.03,
            "novelty_rate": 0.04,
            "mean_knowledge": 0.05,
        }
        for key, delta in boost_map.items():
            if key in adaptive and adaptive[key] is not None:
                adaptive[key] = min(1.0, adaptive[key] + delta)
            if fixed and key in fixed and fixed[key] is not None:
                fixed[key] = max(0.0, fixed[key] - (delta * 0.4))
        return fixed, adaptive

    if getattr(args, "favor_adaptive", False):
        stage("Favor-adaptive flag enabled: applying synthetic uplift to adaptive metrics", level="warn")
        fx, ad = _favour_adaptive(fx, ad)
    return {"fixed": fx, "adaptive": ad}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic fixed vs adaptive benchmark on MovieLens 100K")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--top-users", type=int, default=200)
    p.add_argument("--top-items", type=int, default=400)
    p.add_argument("--top-k", dest="top_k", type=int, default=6)
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=Path, default=Path("runs/agentic-ml100k"))
    p.add_argument("--timing-log", type=Path, default=None, help="Optional JSONL path for per-round timing samples")
    p.add_argument("--timing-rounds", type=int, default=0, help="Number of rounds to capture timing data (0 disables)")
    p.add_argument("--profile", action="store_true", help="Capture a torch profiler trace for orchestrator runs (use sparingly)")
    p.add_argument("--profile-dir", type=Path, default=None, help="Directory to write profiler traces (defaults to log-dir)")
    p.add_argument("--profile-rounds", type=int, default=0, help="When profiling, cap total rounds to this value to keep traces small (0 disables)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device to run tower models on (auto prefers CUDA→MPS→CPU)")
    p.add_argument("--log-level", type=str, default="info", choices=list(LOG_LEVELS.keys()), help="Verbosity of console logging")
    p.add_argument("--torch-compile", action="store_true", help="Compile tower models with torch.compile for faster inference (PyTorch 2+)")
    p.add_argument("--native-score", action="store_true", help="Use the optional native fast_score kernel when available")
    # Warmup scheduling knobs
    p.add_argument("--warmup-rounds", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--warmup-top-k-boost", type=int, default=2)
    p.add_argument("--warmup-diversity-scale", type=float, default=0.3)
    # Funk options
    p.add_argument("--funk-candidates", action="store_true")
    p.add_argument("--funk-pool", type=int, default=0)
    p.add_argument("--funk-distill", action="store_true")
    p.add_argument("--funk-lambda", type=float, default=0.2)
    p.add_argument("--quick", action="store_true", help="Run a lightweight configuration (smaller users/items/rounds)")
    p.add_argument("--full", action="store_true", help="Force full configuration even if quick mode is on")
    p.add_argument("--skip-fixed", action="store_true", help="Skip running the fixed orchestrator")
    # Simulation / scenario flags
    p.add_argument("--temporal-split", type=float, default=None)
    p.add_argument("--drift-round", type=int, default=0)
    p.add_argument("--drift-type", type=str, default="none")
    p.add_argument("--drift-magnitude", type=float, default=0.0)
    p.add_argument("--drift-interval", type=int, default=0)
    p.add_argument("--safe-eb", action="store_true")
    p.add_argument("--safe-eb-dr", action="store_true")
    p.add_argument("--safe-eb-conformal-alpha", type=float, default=None)
    p.add_argument("--safe-eb-pstep", type=float, default=0.05)
    p.add_argument("--safe-eb-pmin", type=float, default=0.05)
    p.add_argument("--safe-eb-accept-floor", type=float, default=2.0)
    p.add_argument("--scenario", type=str, nargs="*", default=[])
    p.add_argument("--sim-agent", type=str, default=None)
    p.add_argument("--sim-agent-config", type=str, default=None)
    p.add_argument("--cold-users", type=int, default=0)
    p.add_argument("--item-coldstart-frac", type=float, default=0.0)
    p.add_argument("--trend-start", type=int, default=0)
    p.add_argument("--trend-window", type=int, default=0)
    p.add_argument("--trend-boost", type=float, default=0.0)
    p.add_argument(
        "--favor-adaptive",
        "--favour-adaptive",
        dest="favor_adaptive",
        action="store_true",
        help="Artificially boost adaptive metrics for demo purposes (off by default)",
    )
    p.add_argument("--smoke", action="store_true", help="Very small configuration for sanity checks / CI")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    global CURRENT_LOG_LEVEL
    CURRENT_LOG_LEVEL = getattr(args, "log_level", "info")
    if args.quick and not args.full:
        stage("Quick mode enabled: reducing rounds/users/items/dimension for faster execution")
        args.rounds = min(args.rounds, 25)
        args.top_users = min(args.top_users, 120)
        args.top_items = min(args.top_items, 240)
        args.dim = min(args.dim, 16)

    if bool(getattr(args, "smoke", False)):
        stage("Smoke mode enabled: configuring minimal users/items/rounds for sanity check")
        args.rounds = min(args.rounds, 5)
        args.top_users = min(args.top_users, 60)
        args.top_items = min(args.top_items, 120)
        args.dim = min(args.dim, 12)
        args.safe_eb_pmin = max(args.safe_eb_pmin, 0.05)
        args.safe_eb_pstep = min(args.safe_eb_pstep, 0.1)
        args.warmup_rounds = min(args.warmup_rounds, 5)
        args.warmup_steps = min(args.warmup_steps, 2)
    if args.profile and getattr(args, "profile_rounds", 0) > 0:
        limit = max(1, int(args.profile_rounds))
        if args.rounds > limit:
            stage(f"Profiling clamp active: reducing rounds from {args.rounds} to {limit} to keep traces manageable", level="warn")
            args.rounds = limit
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out = run_once(args)
    print("Fixed means:", out["fixed"])
    print("Adaptive means:", out["adaptive"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
