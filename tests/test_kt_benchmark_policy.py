"""Tests for KT benchmarking and KT-guided policy ranking."""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from orchid_ranker.delayed_gain import (
    build_delayed_gain_training_frame,
    diagnose_delayed_gain_predictions,
    fit_delayed_gain_reward_model,
    fit_delayed_gain_reward_model_from_frame,
)
from orchid_ranker.kt import SAKTTracer
from orchid_ranker.kt_benchmark import (
    KTHoldoutSplit,
    evaluate_item_mean_baseline,
    evaluate_sakt_replay,
    run_akt_benchmark,
    run_sakt_benchmark,
    time_ordered_user_split,
)
from orchid_ranker.learning_policy import DelayedGainValuePolicy, KTValuePolicy, SupportConstrainedDelayedGainPolicy
from orchid_ranker.policy_benchmark import (
    attach_delayed_gain_rewards,
    build_kt_policy_ope_events,
    estimate_delayed_gain_priors,
    run_kt_policy_ope_benchmark,
    run_kt_policy_ope_seed_sweep,
)


def _events() -> pd.DataFrame:
    rows = []
    items = [(10, 0.2), (20, 0.4), (30, 0.6), (10, 0.2), (20, 0.4)]
    for user_id, ability in [(1, 0.30), (2, 0.50), (3, 0.70), (4, 0.90)]:
        for step, (item_id, difficulty) in enumerate(items):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "correct": int(ability + 0.1 >= difficulty),
                    "difficulty": difficulty,
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


def _delayed_gain_events() -> pd.DataFrame:
    rows = []
    items = [(10, 0.2, "a"), (20, 0.4, "b"), (10, 0.2, "a"), (20, 0.4, "b"), (10, 0.2, "a"), (20, 0.4, "b")]
    for user_id, ability in [(1, 0.30), (2, 0.50), (3, 0.70), (4, 0.90)]:
        for step, (item_id, difficulty, skill_id) in enumerate(items):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "correct": int(ability + 0.1 >= difficulty),
                    "difficulty": difficulty,
                    "skill_id": skill_id,
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


def test_time_ordered_user_split_keeps_future_events_in_test():
    split = time_ordered_user_split(
        _events(),
        timestamp_col="timestamp",
        test_fraction=0.4,
        min_train_events=2,
    )

    assert len(split.train) > 0
    assert len(split.test) > 0
    for user_id, train_group in split.train.groupby("user_id"):
        test_group = split.test[split.test["user_id"] == user_id]
        assert train_group["timestamp"].max() < test_group["timestamp"].min()


def test_item_mean_baseline_returns_metrics():
    split = time_ordered_user_split(_events(), timestamp_col="timestamp", test_fraction=0.4)
    report = evaluate_item_mean_baseline(split)

    assert report.n_events == len(split.test)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.brier <= 1.0
    assert report.log_loss > 0.0


def test_sakt_replay_predicts_before_observing_heldout_events():
    split = time_ordered_user_split(_events(), timestamp_col="timestamp", test_fraction=0.4)
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=17,
        device="cpu",
    ).fit(split.train, timestamp_col="timestamp")
    before_history = tracer.history_for(1)

    report = evaluate_sakt_replay(tracer, split)

    assert report.n_events == len(split.test)
    assert tracer.history_for(1) != before_history
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.brier <= 1.0


def test_run_sakt_benchmark_returns_sakt_and_baseline_metrics():
    metrics = run_sakt_benchmark(
        _events(),
        timestamp_col="timestamp",
        test_fraction=0.4,
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        device="cpu",
    )

    assert set(metrics) == {"sakt", "item_mean", "split"}
    assert metrics["sakt"]["n_events"] == metrics["item_mean"]["n_events"]
    assert metrics["split"]["train_events"] > 0


def test_run_akt_benchmark_returns_akt_and_baseline_metrics():
    metrics = run_akt_benchmark(
        _events(),
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        test_fraction=0.4,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        device="cpu",
    )

    assert set(metrics) == {"akt", "item_mean", "split"}
    assert metrics["akt"]["n_events"] == metrics["item_mean"]["n_events"]


class _FakeTracer:
    def __init__(self) -> None:
        self.observed = []

    def predict_many(self, user_id, item_ids):
        del user_id
        values = {10: 0.30, 20: 0.69, 30: 0.92}
        return {item_id: values[item_id] for item_id in item_ids}

    def observe(self, user_id, item_id, correct):
        self.observed.append((user_id, item_id, correct))
        return len(self.observed)


class _FakeDelayedRewardModel:
    def predict_one(self, features):
        return 0.6 * features["delayed_gain_prior"] + 0.4 * features["support_score"]

    def to_dict(self):
        return {"fallback_only": False, "n_examples": 3}


class _ReplayFeatureTracer:
    def __init__(self) -> None:
        self._histories = {}
        self._history_times = {"existing": [123.0]}

    def predict_correct(self, user_id, item_id):
        del item_id
        return 0.10 + 0.10 * len(self._histories.get(user_id, []))

    def observe(self, user_id, item_id, correct, timestamp=None):
        self._histories.setdefault(user_id, []).append((item_id, int(correct)))
        if timestamp is not None:
            self._history_times.setdefault(user_id, []).append(float(timestamp))
        return len(self._histories[user_id])


def test_kt_value_policy_ranks_by_stretch_and_gain():
    tracer = _FakeTracer()
    policy = KTValuePolicy(tracer, target_correct=0.70)

    ranked = policy.rank("u", [10, 20, 30], top_k=3)
    updates = policy.observe("u", 20, True)

    assert ranked[0].item_id == 20
    assert ranked[0].stretch_fit > ranked[1].stretch_fit
    assert updates == 1
    assert tracer.observed == [("u", 20, True)]


def test_delayed_gain_policy_prefers_historical_gain_prior():
    tracer = _FakeTracer()
    policy = DelayedGainValuePolicy(
        tracer,
        difficulty_by_item={10: 0.2, 20: 0.4, 30: 0.6},
        concept_by_item={10: "a", 20: "b", 30: "c"},
        item_gain_prior={10: 0.10, 20: 0.20, 30: 0.95},
        global_gain_prior=0.50,
    )

    ranked = policy.rank("u", [10, 20, 30], top_k=3)

    assert ranked[0].item_id == 30
    assert ranked[0].delayed_gain_prior == 0.95
    assert ranked[0].score > ranked[-1].score


def test_support_constrained_delayed_gain_policy_uses_model_and_support():
    tracer = _FakeTracer()
    policy = SupportConstrainedDelayedGainPolicy(
        tracer,
        reward_model=_FakeDelayedRewardModel(),
        difficulty_by_item={10: 0.2, 20: 0.4, 30: 0.6},
        concept_by_item={10: "a", 20: "b", 30: "c"},
        item_gain_prior={10: 0.10, 20: 0.72, 30: 0.78},
        global_gain_prior=0.50,
        item_support={10: 10, 20: 10, 30: 0},
        concept_support={"a": 100, "b": 100, "c": 0},
    )

    ranked = policy.rank("u", [10, 20, 30], top_k=3)

    assert ranked[0].item_id == 20
    assert ranked[0].model_prediction is not None
    unsupported = next(rec for rec in ranked if rec.item_id == 30)
    assert ranked[0].support_penalty < unsupported.support_penalty


def test_kt_benchmark_cli_smoke(tmp_path):
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "metrics.json"
    _events().to_csv(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/kt_sakt_benchmark.py",
            "--data",
            str(data_path),
            "--model",
            "akt",
            "--timestamp-col",
            "timestamp",
            "--item-difficulty-col",
            "difficulty",
            "--test-fraction",
            "0.4",
            "--max-seq-len",
            "3",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_path.read_text())
    assert metrics["akt"]["n_events"] > 0


def test_build_kt_policy_ope_events_creates_logged_policy_frame():
    split = time_ordered_user_split(_events(), timestamp_col="timestamp", test_fraction=0.4)
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=31,
        device="cpu",
    ).fit(split.train, timestamp_col="timestamp")

    events = build_kt_policy_ope_events(
        tracer,
        split,
        candidate_size=3,
        max_events=5,
        random_state=11,
    )

    assert len(events) == 5
    assert set(
        [
            "reward",
            "logging_propensity",
            "target_probability",
            "random_probability",
            "target_value",
            "random_value",
            "logged_action_value",
        ]
    ).issubset(events.columns)
    assert events["logging_propensity"].between(0.0, 1.0).all()
    assert events["target_probability"].isin([0.0, 1.0]).all()


def test_build_kt_policy_ope_events_uses_real_propensity_column():
    source = _events()
    source["propensity"] = 0.25
    split = time_ordered_user_split(source, timestamp_col="timestamp", test_fraction=0.4)
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=33,
        device="cpu",
    ).fit(split.train, timestamp_col="timestamp")

    events = build_kt_policy_ope_events(
        tracer,
        split,
        candidate_size=3,
        max_events=4,
        random_state=11,
        logging_propensity_col="propensity",
    )

    assert events["logging_propensity"].tolist() == [0.25, 0.25, 0.25, 0.25]


def test_run_kt_policy_ope_benchmark_returns_policy_uplift_report():
    metrics = run_kt_policy_ope_benchmark(
        _events(),
        model="akt",
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        test_fraction=0.4,
        candidate_size=3,
        max_events=6,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=37,
        device="cpu",
    )

    assert metrics["n_events"] == 6
    assert metrics["assumptions"]["logging"] == "synthetic_uniform_over_candidate_set"
    assert metrics["comparison"]["target"]["estimator"] == "doubly_robust"
    assert "uplift" in metrics["comparison"]


def test_run_kt_policy_ope_benchmark_supports_progression_policy_and_reward():
    metrics = run_kt_policy_ope_benchmark(
        _events(),
        model="akt",
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        concept_col="item_id",
        policy="progression",
        reward_mode="progression",
        test_fraction=0.4,
        candidate_size=3,
        max_events=6,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=41,
        device="cpu",
    )

    assert metrics["n_events"] == 6
    assert metrics["assumptions"]["policy"] == "progression"
    assert metrics["assumptions"]["reward_mode"] == "progression"
    assert metrics["assumptions"]["reward"] == "observed_progression_reward"
    assert 0.0 <= metrics["logging_reward"] <= 1.0


def test_attach_delayed_gain_rewards_adds_future_same_concept_reward():
    source = _delayed_gain_events()
    split = time_ordered_user_split(source, timestamp_col="timestamp", test_fraction=0.5)
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=43,
        device="cpu",
    ).fit(split.train, timestamp_col="timestamp")
    events = build_kt_policy_ope_events(
        tracer,
        split,
        candidate_size=2,
        max_events=8,
        random_state=11,
        policy="progression",
        reward_mode="progression",
        concept_by_item={10: "a", 20: "b"},
        difficulty_by_item={10: 0.2, 20: 0.4},
    )

    with_delayed = attach_delayed_gain_rewards(events, split, concept_col="skill_id", future_window=1)

    assert "delayed_gain_reward" in with_delayed.columns
    assert with_delayed["delayed_gain_reward"].notna().any()
    assert with_delayed["future_same_concept_count"].max() >= 1


def test_estimate_delayed_gain_priors_uses_train_split_only():
    split = time_ordered_user_split(_delayed_gain_events(), timestamp_col="timestamp", test_fraction=0.5)

    priors = estimate_delayed_gain_priors(split, concept_col="skill_id", future_window=1, shrinkage=0.0)

    assert 0.0 <= priors["global_gain_prior"] <= 1.0
    assert priors["item_gain_prior"]
    assert priors["concept_gain_prior"]
    assert priors["shrinkage"] == 0.0


def test_fit_delayed_gain_reward_model_returns_bounded_predictions():
    split = time_ordered_user_split(_delayed_gain_events(), timestamp_col="timestamp", test_fraction=0.5)
    priors = estimate_delayed_gain_priors(split, concept_col="skill_id", future_window=1, shrinkage=0.0)

    model = fit_delayed_gain_reward_model(
        split,
        concept_col="skill_id",
        item_difficulty_col="difficulty",
        item_gain_prior=priors["item_gain_prior"],
        concept_gain_prior=priors["concept_gain_prior"],
        global_gain_prior=priors["global_gain_prior"],
        future_window=1,
        max_examples=20,
        random_state=5,
    )

    assert model.report.n_examples >= 0
    assert 0.0 <= model.default_prediction <= 1.0
    assert 0.0 <= model.predict_one({}) <= 1.0


def test_fit_delayed_gain_reward_model_supports_weighting_and_cross_fit_diagnostics():
    split = time_ordered_user_split(_delayed_gain_events(), timestamp_col="timestamp", test_fraction=0.5)
    priors = estimate_delayed_gain_priors(split, concept_col="skill_id", future_window=1, shrinkage=0.0)
    examples = build_delayed_gain_training_frame(
        split,
        concept_col="skill_id",
        item_difficulty_col="difficulty",
        item_gain_prior=priors["item_gain_prior"],
        concept_gain_prior=priors["concept_gain_prior"],
        global_gain_prior=priors["global_gain_prior"],
        future_window=1,
    )
    examples["target_probability"] = 1.0
    examples["logging_propensity"] = 0.5

    model = fit_delayed_gain_reward_model_from_frame(
        examples,
        max_examples=20,
        example_weighting="mrdr",
        cross_fit_folds=2,
        random_state=7,
    )
    diagnostics = diagnose_delayed_gain_predictions(
        examples["delayed_gain_reward"].tolist(),
        model.predict_many(examples[model.feature_names].to_dict("records")),
    )

    assert model.report.example_weighting == "mrdr"
    assert model.report.sample_weight_mean is not None
    assert diagnostics["n"] == len(examples)
    assert diagnostics["bins"]


def test_delayed_gain_training_frame_can_use_tracer_replay_predictions():
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": 10, "correct": 1, "skill_id": "a", "timestamp": 0},
            {"user_id": "u1", "item_id": 10, "correct": 0, "skill_id": "a", "timestamp": 1},
            {"user_id": "u1", "item_id": 10, "correct": 1, "skill_id": "a", "timestamp": 2},
        ]
    )
    split = KTHoldoutSplit(
        train=train,
        test=train.iloc[0:0].copy(),
        user_col="user_id",
        item_col="item_id",
        correct_col="correct",
        timestamp_col="timestamp",
    )
    tracer = _ReplayFeatureTracer()

    examples = build_delayed_gain_training_frame(split, concept_col="skill_id", future_window=1, tracer=tracer)

    assert examples["p_correct"].round(6).tolist() == [0.1, 0.2]
    assert tracer._histories == {}
    assert tracer._history_times == {"existing": [123.0]}


def test_run_kt_policy_ope_benchmark_supports_delayed_gain_reward():
    metrics = run_kt_policy_ope_benchmark(
        _delayed_gain_events(),
        model="akt",
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        concept_col="skill_id",
        policy="progression",
        reward_mode="delayed_gain",
        delayed_gain_window=1,
        test_fraction=0.5,
        candidate_size=2,
        max_events=8,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=47,
        device="cpu",
    )

    assert metrics["n_events"] > 0
    assert metrics["assumptions"]["reward_mode"] == "delayed_gain"
    assert metrics["assumptions"]["reward"] == "delayed_same_concept_gain_proxy"
    assert metrics["comparison"]["target"]["estimator"] == "snips"
    assert metrics["target_value_mean"] is None
    assert metrics["delayed_gain"]["future_same_concept_count_mean"] >= 1.0


def test_run_kt_policy_ope_benchmark_supports_delayed_gain_policy():
    metrics = run_kt_policy_ope_benchmark(
        _delayed_gain_events(),
        model="akt",
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        concept_col="skill_id",
        policy="delayed_gain",
        reward_mode="delayed_gain",
        delayed_gain_window=1,
        test_fraction=0.5,
        candidate_size=2,
        max_events=8,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=49,
        device="cpu",
    )

    assert metrics["n_events"] > 0
    assert metrics["assumptions"]["policy"] == "delayed_gain"
    assert metrics["comparison"]["target"]["estimator"] == "snips"
    assert metrics["delayed_gain_policy"]["item_priors"] > 0


def test_run_kt_policy_ope_benchmark_supports_support_constrained_delayed_gain_policy():
    metrics = run_kt_policy_ope_benchmark(
        _delayed_gain_events(),
        model="akt",
        timestamp_col="timestamp",
        item_difficulty_col="difficulty",
        concept_col="skill_id",
        policy="support_delayed_gain",
        reward_mode="delayed_gain",
        delayed_gain_window=1,
        test_fraction=0.5,
        candidate_size=2,
        max_events=8,
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=51,
        device="cpu",
        reward_model_max_examples=20,
        reward_model_example_weighting="support_inverse",
        reward_model_cross_fit_folds=2,
    )

    assert metrics["n_events"] > 0
    assert metrics["assumptions"]["policy"] == "support_delayed_gain"
    assert metrics["comparison"]["target"]["estimator"] == "doubly_robust"
    assert metrics["target_value_mean"] is not None
    assert metrics["delayed_gain_reward_model"]["n_examples"] >= 0
    assert metrics["delayed_gain_reward_model"]["example_weighting"] == "support_inverse"


def test_run_kt_policy_ope_seed_sweep_returns_aggregate_summary():
    metrics = run_kt_policy_ope_seed_sweep(
        _events(),
        seeds=[3, 5],
        model="sakt",
        timestamp_col="timestamp",
        test_fraction=0.4,
        candidate_size=3,
        max_events=5,
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        device="cpu",
    )

    assert metrics["summary"]["n_runs"] == 2.0
    assert metrics["summary"]["seeds"] == [3, 5]
    assert len(metrics["runs"]) == 2
    assert "uplift_mean" in metrics["summary"]


def test_kt_policy_ope_cli_smoke(tmp_path):
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "policy_ope.json"
    _events().to_csv(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/kt_policy_ope_benchmark.py",
            "--data",
            str(data_path),
            "--model",
            "akt",
            "--timestamp-col",
            "timestamp",
            "--item-difficulty-col",
            "difficulty",
            "--concept-col",
            "item_id",
            "--policy",
            "progression",
            "--reward-mode",
            "progression",
            "--test-fraction",
            "0.4",
            "--candidate-size",
            "3",
            "--max-events",
            "6",
            "--max-seq-len",
            "3",
            "--d-model",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--seed",
            "37",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_path.read_text())
    assert metrics["n_events"] == 6
    assert metrics["assumptions"]["policy"] == "progression"


def test_kt_policy_ope_cli_seed_sweep_smoke(tmp_path):
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "policy_ope_sweep.json"
    _events().to_csv(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/kt_policy_ope_benchmark.py",
            "--data",
            str(data_path),
            "--model",
            "sakt",
            "--timestamp-col",
            "timestamp",
            "--test-fraction",
            "0.4",
            "--candidate-size",
            "3",
            "--max-events",
            "5",
            "--max-seq-len",
            "3",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "3",
            "5",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_path.read_text())
    assert metrics["summary"]["seeds"] == [3, 5]


def test_adaptive_efficiency_benchmark_cli_smoke(tmp_path):
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "adaptive_efficiency.json"
    report_path = tmp_path / "adaptive_efficiency.md"
    _delayed_gain_events().to_csv(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/adaptive_efficiency_benchmark.py",
            "--data",
            str(data_path),
            "--models",
            "akt",
            "--seeds",
            "3",
            "--timestamp-col",
            "timestamp",
            "--item-difficulty-col",
            "difficulty",
            "--concept-col",
            "skill_id",
            "--policy-targets",
            "0.7",
            "--policy-rewards",
            "progression",
            "delayed_gain",
            "--test-fraction",
            "0.5",
            "--candidate-size",
            "2",
            "--max-events",
            "8",
            "--max-seq-len",
            "3",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output_path),
            "--report-md",
            str(report_path),
            "--benchmark-name",
            "Smoke credibility benchmark",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_path.read_text())
    assert "akt" in metrics["quality"]["summary"]
    assert metrics["policy"]["summary"]["table"]
    assert metrics["summary"]["best_policy"]["policy"] in {"progression", "delayed_gain", "support_delayed_gain"}
    report = report_path.read_text()
    assert "# Smoke credibility benchmark" in report
    assert "## KT Prediction Quality" in report
    assert "## Policy OPE" in report
    assert "research evidence" in report


def test_delayed_gain_model_benchmark_cli_smoke(tmp_path):
    data_path = tmp_path / "events.csv"
    output_path = tmp_path / "delayed_gain_model.json"
    _delayed_gain_events().to_csv(data_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/delayed_gain_model_benchmark.py",
            "--data",
            str(data_path),
            "--model",
            "akt",
            "--timestamp-col",
            "timestamp",
            "--item-difficulty-col",
            "difficulty",
            "--concept-col",
            "skill_id",
            "--test-fraction",
            "0.5",
            "--candidate-size",
            "2",
            "--max-events",
            "8",
            "--delayed-gain-window",
            "1",
            "--reward-model-weightings",
            "uniform",
            "--reward-model-max-examples",
            "20",
            "--reward-model-cross-fit-folds",
            "2",
            "--max-seq-len",
            "3",
            "--d-model",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_path.read_text())
    assert metrics["summary"]["table"]
    assert metrics["runs"][0]["target_policy"]["logged_action_diagnostics"]["n"] > 0
