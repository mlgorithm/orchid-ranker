"""Micro benchmark for tower inference throughput.

Keeps the workload intentionally tiny so it can run on CI boxes (CPU or Apple
MPS) to sanity-check the hottest scoring path without downloading datasets.
"""
from __future__ import annotations

import argparse
import time

import torch

from orchid_ranker.agents.recommender_agent import TwoTowerRecommender
from orchid_ranker.utils import select_device


def _torch_compile_supported() -> bool:
    return hasattr(torch, "compile") and tuple(int(x) for x in torch.__version__.split(".")[0:2]) >= (2, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the TwoTowerRecommender inference path.")
    parser.add_argument("--iters", type=int, default=25, help="Measurement iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations excluded from timing")
    parser.add_argument("--users", type=int, default=64, help="Number of synthetic users")
    parser.add_argument("--items", type=int, default=512, help="Number of synthetic items")
    parser.add_argument("--candidates", type=int, default=128, help="Number of candidate items scored per call")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device to benchmark")
    parser.add_argument("--torch-compile", action="store_true", help="Run the model through torch.compile when supported")
    parser.add_argument("--native-score", action="store_true", help="Use the optional native fast_score kernel when available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device_choice = select_device(args.device)
    device = device_choice.torch_device

    model = TwoTowerRecommender(
        num_users=args.users,
        num_items=args.items,
        user_dim=args.dim,
        item_dim=args.dim,
        hidden=32,
        emb_dim=16,
        device=str(device),
        dp_cfg={"enabled": False},
        use_native_scoring=args.native_score,
    )
    model.eval()
    model.user_matrix = torch.randn(args.users, args.dim, device=device)
    item_matrix = torch.randn(args.items, args.dim, device=device)
    state_vec = torch.zeros(1, 4, device=device)

    compile_requested = bool(args.torch_compile)
    if compile_requested:
        if _torch_compile_supported() and device_choice.name != "mps":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:  # pragma: no cover - depends on runtime
                print(f"[bench_infer] torch.compile failed: {exc}. Falling back to eager mode.")
        else:
            print("[bench_infer] torch.compile not available on this platform.")

    cand_count = min(args.candidates, args.items)
    candidates = torch.arange(cand_count, dtype=torch.long, device=device)

    def _sync():
        if device_choice.name == "cuda":
            torch.cuda.synchronize()
        elif device_choice.name == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    @torch.inference_mode()
    def _run_step(step_idx: int):
        uid = step_idx % args.users
        user_ids = torch.tensor([uid], dtype=torch.long, device=device)
        user_vec = model.user_matrix[uid : uid + 1]
        logits = model.infer(
            user_vec=user_vec,
            item_matrix=item_matrix,
            user_ids=user_ids,
            item_ids=candidates,
            state_vec=state_vec,
        )
        # prevent DCE by touching the tensor
        return float(logits.mean().item())

    total_steps = args.warmup + args.iters
    with torch.inference_mode():
        for i in range(args.warmup):
            _run_step(i)
        _sync()
        start = time.perf_counter()
        last = 0.0
        for i in range(args.warmup, total_steps):
            last = _run_step(i)
        _sync()
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / max(1, args.iters)) * 1000.0
    print(
        f"[bench_infer] device={device_choice.name} dim={args.dim} "
        f"candidates={cand_count} iters={args.iters} -> {avg_ms:.3f} ms/call "
        f"(last_mean={last:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
