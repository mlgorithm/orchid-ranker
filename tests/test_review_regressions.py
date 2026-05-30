from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from orchid_ranker.adaptive_ranker import AdaptiveRanker
from orchid_ranker.adaptive_schema import hash_identifier
from orchid_ranker.agents.policies import LinUCBPolicy
from orchid_ranker.agents.rec_shim import RecShim
from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.curriculum import DependencyGraph
from orchid_ranker.evaluation import engagement_score
from orchid_ranker.fqe import TabularFQE
from orchid_ranker.irt import IRTAdaptiveSelector
from orchid_ranker.offline_policy import CQLDiscretePolicy
from orchid_ranker.ope import evaluate_logged_policy
from orchid_ranker.security.access import AccessControl
from orchid_ranker.security.auth import InvalidTokenError, JWTAuthenticator
from orchid_ranker.streaming import StreamingAdaptiveRanker
from orchid_ranker.streaming_bus import KafkaEventBus


def test_dp_train_step_adds_noise_and_accounts_privacy() -> None:
    torch.manual_seed(0)
    rec = TwoTowerRecommender(
        num_users=2,
        num_items=3,
        user_dim=2,
        item_dim=2,
        hidden=4,
        emb_dim=3,
        device="cpu",
        dp_cfg={"enabled": True, "noise_multiplier": 1.0, "sample_rate": 0.5, "delta": 1e-5},
    )
    out = rec.train_step(
        {
            "user_ids": torch.tensor([0, 0]),
            "item_ids": torch.tensor([0, 1]),
            "labels": torch.tensor([1.0, 0.0]),
            "item_matrix": torch.randn(3, 2),
            "state_vec": torch.zeros(2, 4),
        }
    )
    assert out["epsilon_cum"] > 0.0
    assert rec.eps_cum == pytest.approx(out["epsilon_cum"])


def test_rec_shim_casts_list_inputs_and_passes_decide_defaults() -> None:
    rec = TwoTowerRecommender(1, 2, 2, 2, hidden=4, emb_dim=3, device="cpu", dp_cfg={"enabled": False})
    shim = RecShim(rec)
    logits = shim.think(
        user_vec=[[0.1, 0.2]],
        item_matrix=[[0.1, 0.0], [0.0, 0.2]],
        user_ids=[0],
        item_ids=[0, 1],
        state_vec=[[0.0, 0.0, 0.5, 0.5]],
    )
    chosen, _scores = shim.decide(logits, top_k=1, item_ids=torch.tensor([0, 1]), user_id=0, engagement=0.5, trust=0.5)
    assert len(chosen) == 1


def test_fqe_report_uses_target_start_action_value() -> None:
    transitions = pd.DataFrame(
        [
            {
                "context_hash": "s0",
                "chosen_item_id": "a",
                "reward": 1.0,
                "next_context_hash": "terminal",
                "target_action_id": "b",
                "target_start_action_id": "b",
                "done": True,
            }
        ]
    )
    fqe = TabularFQE(gamma=0.0, epochs=20, learning_rate=0.5).fit(
        transitions,
        target_start_action_col="target_start_action_id",
    )
    assert fqe.score("s0", "a") > 0.9
    assert fqe.report_ is not None
    assert fqe.report_.estimated_value == pytest.approx(0.0)


def test_fqe_default_start_action_uses_target_policy_not_logged() -> None:
    # Start context "s0" is logged with high-value action "a", but the target policy's
    # action at "s0" is "b" (recovered from the x -> s0 transition's target_action).
    # Without an explicit target_start_action_col, the reported policy value must be
    # Q(s0, "b") (~0), not the logged Q(s0, "a") (~1).
    transitions = pd.DataFrame(
        [
            {"context_hash": "s0", "chosen_item_id": "a", "reward": 1.0,
             "next_context_hash": "terminal", "target_action_id": "b", "done": True},
            {"context_hash": "s0", "chosen_item_id": "b", "reward": 0.0,
             "next_context_hash": "terminal", "target_action_id": "b", "done": True},
            {"context_hash": "x", "chosen_item_id": "c", "reward": 0.0,
             "next_context_hash": "s0", "target_action_id": "b", "done": False},
        ]
    )
    fqe = TabularFQE(gamma=0.9, epochs=80, learning_rate=0.3).fit(transitions)
    assert fqe.score("s0", "a") > 0.9
    assert fqe.score("s0", "b") == pytest.approx(0.0, abs=1e-6)
    # Logged-action default would give ~1/3; target-policy default gives ~0.
    assert fqe.report_.estimated_value < 0.2
    assert fqe.report_.target_start_action_is_explicit is False


def test_irt_3pl_update_matches_likelihood_gradient() -> None:
    selector = IRTAdaptiveSelector(initial_theta=0.2, learning_rate=1.0).fit_items(
        [{"item_id": "q", "difficulty": -0.3, "discrimination": 1.7, "guessing": 0.25}]
    )
    p = selector.probability("q")
    expected_grad = 1.7 * (1.0 - p) * (p - 0.25) / (p * (1.0 - 0.25))
    theta0 = selector.theta
    selector.observe("q", True)
    assert selector.theta - theta0 == pytest.approx(expected_grad)


def test_streaming_observe_is_transactional_on_bad_item() -> None:
    torch.manual_seed(0)
    user_features = torch.randn(2, 3)
    item_features = torch.randn(2, 3)
    tower = TwoTowerRecommender(2, 2, 3, 3, hidden=4, emb_dim=3, device="cpu", dp_cfg={"enabled": False}).eval()
    ranker = StreamingAdaptiveRanker(tower, user_features, item_features)

    with pytest.raises(IndexError):
        ranker.observe(user_id=0, item_id=999, correct=True)

    assert ranker.updates_for(0) == 0
    assert ranker.competence(0) == pytest.approx(0.1)


def test_kafka_malformed_message_committed_and_counted() -> None:
    class Message:
        def error(self):
            return None

        def value(self):
            return b"{not json"

    class Consumer:
        def __init__(self) -> None:
            self.calls = 0
            self.commits = []

        def poll(self, timeout):
            self.calls += 1
            return Message() if self.calls == 1 else None

        def commit(self, *, message, asynchronous):
            self.commits.append((message, asynchronous))

    bus = KafkaEventBus.__new__(KafkaEventBus)
    bus._consumer = Consumer()
    bus._closed = False
    bus._pending = {}
    bus.parse_errors = 0

    assert bus.poll(max_events=1, timeout_s=0.01) == []
    assert bus.parse_errors == 1
    assert len(bus._consumer.commits) == 1


def test_jwt_auth_requires_exp_claim() -> None:
    jwt = pytest.importorskip("jwt")

    auth = JWTAuthenticator.__new__(JWTAuthenticator)
    auth.issuer = "https://issuer.example"
    auth.audience = "orchid"
    auth.role_claim = "role"
    auth.algorithms = ["HS256"]
    auth._jwks_cache = SimpleNamespace(get_signing_key=lambda kid: SimpleNamespace(key="secret"))
    token = jwt.encode(
        {"sub": "user-1", "role": "viewer", "iss": auth.issuer, "aud": auth.audience},
        "secret",
        algorithm="HS256",
        headers={"kid": "test"},
    )

    with pytest.raises(InvalidTokenError, match="exp"):
        auth.authenticate(token)


def test_access_control_policy_is_immutable() -> None:
    acl = AccessControl()
    with pytest.raises(AttributeError):
        acl.policy["viewer"].add("*")  # type: ignore[attr-defined]
    assert not acl.can("viewer", "experiment")


def test_hash_identifier_requires_secret_salt(monkeypatch) -> None:
    monkeypatch.delenv("ORCHID_HASH_SALT", raising=False)
    with pytest.raises(ValueError, match="salt"):
        hash_identifier("student@example.com")
    monkeypatch.setenv("ORCHID_HASH_SALT", "deployment-secret")
    assert hash_identifier("student@example.com") == hash_identifier("student@example.com")


def test_linucb_uses_single_ridge_regularizer() -> None:
    bandit = LinUCBPolicy(d=1, alpha=0.0, l2=1.0)
    bandit.update(0, np.array([1.0]), 1.0)
    assert bandit.score(np.array([1.0]), i=0) == pytest.approx(0.5, rel=1e-6)


def test_snips_constant_rewards_have_zero_standard_error() -> None:
    report = evaluate_logged_policy(
        pd.DataFrame(
            {
                "reward": [0.5, 0.5, 0.5],
                "propensity": [0.2, 0.5, 1.0],
                "target_probability": [1.0, 0.5, 0.2],
            }
        ),
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_probability",
    )
    assert report.estimator == "snips"
    assert report.standard_error == pytest.approx(0.0)


def test_cql_fit_clears_previous_state() -> None:
    policy = CQLDiscretePolicy(epochs=1, random_state=0)
    first = pd.DataFrame(
        {
            "context_hash": ["a"],
            "candidate_item_ids": [["x"]],
            "chosen_item_id": ["x"],
            "reward": [1.0],
            "propensity": [1.0],
            "learner_id": ["u"],
            "ts": [1],
            "policy_name": ["logged"],
            "policy_version": ["1"],
            "scores": [[1.0]],
        }
    )
    second = first.assign(context_hash=["b"], candidate_item_ids=[["y"]], chosen_item_id=["y"])
    policy.fit(first)
    policy.fit(second)
    assert set(policy.actions_by_context_) == {"b"}
    assert ("a", "x") not in policy.q_values_


def test_adaptive_ranker_ope_report_honors_custom_propensity_col() -> None:
    report = AdaptiveRanker().ope_report(
        pd.DataFrame(
            {
                "learner_id": ["u"],
                "ts": [1],
                "candidate_item_ids": [["i"]],
                "chosen_item_id": ["i"],
                "logging_propensity": [1.0],
                "policy_name": ["logged"],
                "policy_version": ["1"],
                "scores": [[1.0]],
                "context_hash": ["ctx"],
                "reward": [1.0],
                "target_probability": [1.0],
            }
        ),
        propensity_col="logging_propensity",
    )
    assert report.value == pytest.approx(1.0)


def test_unknown_curriculum_candidates_are_not_ready() -> None:
    graph = DependencyGraph([("a", "b")])
    assert graph.prerequisites_met("ghost", set()) is False
    assert graph.available(set()) == ["a"]


def test_engagement_score_is_bounded() -> None:
    assert engagement_score(["a", "b", "c"], total_recommended=2) == 1.0


def test_audit_ship_rejects_insecure_url_and_malformed_records(tmp_path: Path) -> None:
    from scripts.ship_audit_logs import iter_records, ship

    log_path = tmp_path / "audit.jsonl"
    log_path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="https"):
        ship(log_path, "http://siem.example", api_key="token")
    with pytest.raises(json.JSONDecodeError):
        list(iter_records(log_path))


def test_connector_retry_backoff_handles_more_than_four_retries(monkeypatch) -> None:
    from orchid_ranker.connectors.exceptions import RetryExhaustedError
    from orchid_ranker.connectors.snowflake import SnowflakeConnector

    attempts = 0

    def always_transient():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary network failure")

    monkeypatch.setattr("orchid_ranker.connectors.snowflake.time.sleep", lambda _delay: None)
    connector = SnowflakeConnector("acct", "user", "pw", max_retries=5)
    with pytest.raises(RetryExhaustedError):
        connector._retry_with_backoff(always_transient)
    assert attempts == 5


def test_snowflake_fetch_dataframe_passes_query_params(monkeypatch) -> None:
    from orchid_ranker.connectors.snowflake import SnowflakeConnector

    calls = []

    class Cursor:
        description = [("answer",)]

        def execute(self, query, params=None):
            calls.append((query, params))

        def fetchall(self):
            return [(42,)]

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    connector = SnowflakeConnector("acct", "user", "pw")
    monkeypatch.setattr(connector, "_require_lib", lambda: None)
    monkeypatch.setattr(connector, "_connect", lambda: Connection())
    frame = connector.fetch_dataframe("select %(x)s as answer", params={"x": 42})
    assert calls == [("select %(x)s as answer", {"x": 42})]
    assert frame["answer"].tolist() == [42]


def test_strict_outcome_parsing_rejects_fractional_and_false_string() -> None:
    from orchid_ranker.live_metrics import RollingProgressionMonitor

    monitor = RollingProgressionMonitor()
    with pytest.raises(ValueError, match="correct"):
        monitor.record(user_id=1, item_id=1, correct=0.5)
    monitor.record(user_id=1, item_id=1, correct="false")
    assert monitor.snapshot().accept_rate == 0.0
