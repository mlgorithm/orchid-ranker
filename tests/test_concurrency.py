"""Concurrency tests for thread safety of SafeSwitchDR, AuditLogger, and MultiConfig.

Uses ThreadPoolExecutor to exercise shared state from multiple threads simultaneously,
verifying that internal locks protect against data races and corruption.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytest

from orchid_ranker.connectors.bigquery import BigQueryConnector
from orchid_ranker.recommender import OrchidRecommender
from orchid_ranker.safety.safeswitch_dr import SafeSwitchDR, SafeSwitchDRConfig
from orchid_ranker.security.audit import AuditEvent, AuditLogger, verify_log_integrity
from orchid_ranker.agents.config import MultiConfig


# ---------------------------------------------------------------------------
# SafeSwitchDR thread safety
# ---------------------------------------------------------------------------

class TestSafeSwitchDRThreadSafety:
    """Verify that concurrent decide() + update() calls do not corrupt state."""

    NUM_THREADS = 4
    ITERS_PER_THREAD = 50

    def _worker(self, ss: SafeSwitchDR, errors: list) -> None:
        """Run decide/update in a tight loop, recording any exceptions."""
        try:
            for _ in range(self.ITERS_PER_THREAD):
                use_adaptive, p_used = ss.decide()
                ss.update(
                    served_adaptive=use_adaptive,
                    reward=0.5,
                    accepts_per_user=3.0,
                    Qa_pred=0.4,
                    Qf_pred=0.3,
                    p_used=max(p_used, 0.05),  # avoid zero p_used
                )
        except Exception as exc:
            errors.append(exc)

    def test_concurrent_decide_update(self) -> None:
        """4 threads x 50 iterations of decide+update should not raise."""
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.05,
            p_max=1.0,
            step_up=0.05,
            step_down=0.1,
        )
        ss = SafeSwitchDR(cfg)
        errors: list = []

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as pool:
            futures = [
                pool.submit(self._worker, ss, errors)
                for _ in range(self.NUM_THREADS)
            ]
            for f in as_completed(futures):
                f.result()  # re-raise if worker raised

        assert errors == [], f"Threads raised exceptions: {errors}"

        total_updates = self.NUM_THREADS * self.ITERS_PER_THREAD
        assert ss.t == total_updates, (
            f"Expected t={total_updates} after all updates, got t={ss.t}"
        )
        assert cfg.p_min <= ss.p <= cfg.p_max or ss.p == 0.0, (
            f"Deployment probability out of bounds: p={ss.p}"
        )


# ---------------------------------------------------------------------------
# AuditLogger thread safety
# ---------------------------------------------------------------------------

class TestAuditLoggerThreadSafety:
    """Verify that concurrent log() calls produce a valid HMAC chain."""

    NUM_THREADS = 4
    EVENTS_PER_THREAD = 25  # 4 * 25 = 100 total events

    def test_concurrent_logging_with_hmac(self, tmp_path) -> None:
        """100 events from 4 threads must produce an intact HMAC hash chain."""
        log_file = tmp_path / "audit.jsonl"
        hmac_key = b"supersecretkey1234567890abcdef00"
        logger = AuditLogger(log_file, hmac_key=hmac_key)

        def _write_events(thread_id: int) -> None:
            for i in range(self.EVENTS_PER_THREAD):
                event = AuditEvent(
                    event_type="test.concurrent",
                    actor=f"thread-{thread_id}",
                    payload={"iteration": i},
                )
                logger.log(event)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as pool:
            futures = [
                pool.submit(_write_events, tid)
                for tid in range(self.NUM_THREADS)
            ]
            for f in as_completed(futures):
                f.result()

        result = verify_log_integrity(log_file, hmac_key)
        assert result.valid, (
            f"Log integrity check failed: {result.error_message} "
            f"(first error at line {result.first_error_line})"
        )
        expected_lines = self.NUM_THREADS * self.EVENTS_PER_THREAD
        assert result.lines_checked == expected_lines, (
            f"Expected {expected_lines} lines, checked {result.lines_checked}"
        )


# ---------------------------------------------------------------------------
# MultiConfig concurrent creation
# ---------------------------------------------------------------------------

class TestMultiConfigThreadSafety:
    """Verify that creating MultiConfig from multiple threads does not crash."""

    NUM_THREADS = 4
    CONFIGS_PER_THREAD = 25

    def test_concurrent_config_creation(self) -> None:
        """Creating 100 MultiConfig instances from 4 threads should be safe."""
        configs: list = []
        errors: list = []

        def _create_configs(thread_id: int) -> None:
            try:
                for i in range(self.CONFIGS_PER_THREAD):
                    cfg = MultiConfig(
                        rounds=10 + thread_id,
                        top_k_base=5 + i % 3,
                    )
                    configs.append(cfg)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as pool:
            futures = [
                pool.submit(_create_configs, tid)
                for tid in range(self.NUM_THREADS)
            ]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"Threads raised exceptions: {errors}"
        expected = self.NUM_THREADS * self.CONFIGS_PER_THREAD
        assert len(configs) == expected, (
            f"Expected {expected} configs, got {len(configs)}"
        )


# ---------------------------------------------------------------------------
# OrchidRecommender fit/read serialization
# ---------------------------------------------------------------------------

class TestOrchidRecommenderThreadSafety:
    """Verify that retraining does not interleave with recommendation reads."""

    def test_recommend_blocks_while_fit_holds_state_lock(self, monkeypatch) -> None:
        """recommend() should wait for an in-flight fit() instead of observing mixed state."""
        rec = OrchidRecommender(strategy="popularity")
        initial = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [10, 11, 10],
                "rating": [1.0, 2.0, 3.0],
            }
        )
        updated = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [20, 21, 20],
                "rating": [4.0, 5.0, 2.0],
            }
        )
        rec.fit(initial, rating_col="rating")

        entered_fit = threading.Event()
        release_fit = threading.Event()
        recommend_finished = threading.Event()
        errors: list[Exception] = []

        original_fit_baseline = rec._fit_baseline

        def blocking_fit_baseline(*args, **kwargs):
            entered_fit.set()
            assert release_fit.wait(timeout=2.0)
            return original_fit_baseline(*args, **kwargs)

        monkeypatch.setattr(rec, "_fit_baseline", blocking_fit_baseline)

        def run_fit():
            try:
                rec.fit(updated, rating_col="rating")
            except Exception as exc:  # pragma: no cover - failure path assertion below
                errors.append(exc)

        fit_thread = threading.Thread(target=run_fit)
        fit_thread.start()
        assert entered_fit.wait(timeout=2.0)

        recommendations: list = []

        def run_recommend():
            try:
                recommendations.extend(rec.recommend(user_id=1, top_k=1, filter_seen=False))
            except Exception as exc:  # pragma: no cover - failure path assertion below
                errors.append(exc)
            finally:
                recommend_finished.set()

        recommend_thread = threading.Thread(target=run_recommend)
        recommend_thread.start()
        time.sleep(0.05)
        assert not recommend_finished.is_set()

        release_fit.set()
        fit_thread.join(timeout=2.0)
        recommend_thread.join(timeout=2.0)

        assert errors == []
        assert recommend_finished.is_set()
        assert len(recommendations) == 1
        assert recommendations[0].item_id in {20, 21}


# ---------------------------------------------------------------------------
# BigQueryConnector client serialization
# ---------------------------------------------------------------------------

class TestBigQueryConnectorThreadSafety:
    """Verify that query/load/close do not race the shared cached client."""

    def test_close_waits_for_inflight_query(self, monkeypatch) -> None:
        """close() should not tear down the cached client during an active query."""
        import orchid_ranker.connectors.bigquery as bigquery_mod

        query_entered = threading.Event()
        release_query = threading.Event()
        close_finished = threading.Event()

        class FakeJob:
            def __init__(self, client):
                self.client = client

            def result(self):
                if self.client.closed:
                    raise RuntimeError("client closed during query")
                return self

            def to_dataframe(self, create_bqstorage_client=False):
                return pd.DataFrame({"value": [1]})

        class FakeClient:
            def __init__(self, project=None):
                self.project = project
                self.closed = False

            def query(self, sql, timeout=None):
                query_entered.set()
                assert release_query.wait(timeout=2.0)
                if self.closed:
                    raise RuntimeError("client closed during query")
                return FakeJob(self)

            def close(self):
                self.closed = True
                close_finished.set()

        fake_module = type("FakeBigQueryModule", (), {"Client": FakeClient})
        monkeypatch.setattr(bigquery_mod, "bigquery", fake_module)

        conn = BigQueryConnector(project="demo-project")
        errors: list[Exception] = []
        results: list[pd.DataFrame] = []

        def run_query():
            try:
                results.append(conn.query_dataframe("SELECT 1"))
            except Exception as exc:  # pragma: no cover - failure path assertion below
                errors.append(exc)

        query_thread = threading.Thread(target=run_query)
        query_thread.start()
        assert query_entered.wait(timeout=2.0)

        closer_thread = threading.Thread(target=conn.close)
        closer_thread.start()
        time.sleep(0.05)
        assert not close_finished.is_set()

        release_query.set()
        query_thread.join(timeout=2.0)
        closer_thread.join(timeout=2.0)

        assert errors == []
        assert close_finished.is_set()
        assert len(results) == 1
        assert list(results[0]["value"]) == [1]
