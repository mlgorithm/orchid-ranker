from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import json
import pandas as pd

__all__ = ["create_report"]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _filter_round(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("type") != "round_summary":
            continue
        rows.append({"round": rec.get("round"), "mode": rec.get("mode"), **rec.get("metrics", {})})
    return pd.DataFrame(rows)


def _latest_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.sort_values("round").iloc[-1]
    return {k: float(last.get(k, float("nan"))) for k in df.columns if k not in {"round", "mode"}}


def create_report(run_dirs: Iterable[str | Path], output_dir: str | Path, modes: Optional[Iterable[str]] = None) -> Path:
    """Aggregate JSONL logs (round summaries) into CSV tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    by_round = []
    modes = set(m.lower() for m in modes) if modes else None

    for path in run_dirs:
        path = Path(path)
        if not path.exists():
            continue
        records = _load_jsonl(path)
        df_round = _filter_round(records)
        if df_round.empty:
            continue
        if modes:
            df_round = df_round[df_round["mode"].str.lower().isin(modes)]
        by_round.append(df_round.assign(log=str(path)))
        latest = _latest_metrics(df_round)
        latest.update({"mode": df_round.iloc[0]["mode"], "log": str(path)})
        rows.append(latest)

    round_df = pd.concat(by_round, ignore_index=True) if by_round else pd.DataFrame()
    summary_df = pd.DataFrame(rows)
    round_path = output_dir / "round_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    round_df.to_csv(round_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return summary_path
