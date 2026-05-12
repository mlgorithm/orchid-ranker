# API Reference

Complete reference for public classes and functions in Orchid Ranker v0.5.0.

---

## Start here: adaptive learning

The primary public workflow is `AdaptiveLearningEngine`
(`AdaptiveLearningRecommender` is the existing class name): fit from learner
outcomes, rank eligible learning items, observe the next outcome, and re-rank
from updated learner state.

```python
from orchid_ranker import AdaptiveLearningEngine

rec = AdaptiveLearningEngine(policy="auto").fit(
    events,
    correct_col="correct",
    concept_col="skill_id",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)
ranked = rec.rank("learner-7", [10, 20, 30], top_k=3)
rec.observe("learner-7", ranked[0].item_id, correct=True)
```

Use `OrchidRecommender` when you only have ordinary user-item interactions and
need a batch/generic recommender fallback.

---

## orchid_ranker.adaptive_learning

High-level adaptive-learning recommender that composes KT prediction,
progression reward, delayed-gain priors, support-aware reward modeling, and
optional prerequisite gating.

```python
class AdaptiveLearningRecommender:
    def __init__(self, config: AdaptiveLearningConfig | None = None, **overrides)
    def fit(
        self,
        interactions,
        *,
        user_col="user_id",
        item_col="item_id",
        correct_col="correct",
        timestamp_col=None,
        concept_col=None,
        item_difficulty_col=None,
        item_difficulty_map=None,
        concept_by_item=None,
        prerequisite_by_concept=None,
    )
    def rank(self, user_id, candidate_item_ids, *, top_k=5) -> list[AdaptiveLearningRecommendation]
    def observe(self, user_id, item_id, correct)
```

`policy="auto"` resolves to `ProgressionValuePolicy`, the stable default for
adaptive-learning serving. `DelayedGainValuePolicy` and
`SupportConstrainedDelayedGainPolicy` remain explicit opt-ins because they need
stronger logged-support and reward-model calibration evidence. The policy state
is warm-started from historical outcomes, so prerequisite gating and competence
estimates reflect the learner's prior history before the first live `observe`.

```python
from orchid_ranker import AdaptiveLearningEngine

rec = AdaptiveLearningEngine(
    tracer_model="akt",
    policy="auto",
    epochs=2,
    d_model=32,
).fit(
    events,
    timestamp_col="timestamp",
    concept_col="skill_id",
    item_difficulty_col="difficulty",
)

ranked = rec.rank("learner-7", [10, 20, 30], top_k=3)
rec.observe("learner-7", ranked[0].item_id, correct=True)
```

### AdaptiveLearningRecommendation

```python
@dataclass
class AdaptiveLearningRecommendation:
    item_id: Any
    score: float
    p_correct: float
    policy: str
    difficulty: float | None = None
    concept_id: Any | None = None
    competence: float | None = None
    expected_reward: float | None = None
    delayed_gain_prior: float | None = None
    model_prediction: float | None = None
    support_penalty: float = 0.0
    prerequisites_met: bool = True
```

Use `diagnostics()` to log the resolved tracer/policy, concept and item counts,
delayed-gain prior coverage, and reward-model report.

`AdaptiveLearningEngine` is a top-level alias for
`AdaptiveLearningRecommender`; use whichever name is clearer in your codebase.

---

## orchid_ranker.recommender

### OrchidRecommender

Batch/generic recommender supporting all built-in strategies. This is the
fallback path for ordinary user-item interactions without learning concepts,
difficulty, prerequisites, or live outcome state.

```python
class OrchidRecommender:
    def __init__(
        self,
        strategy: str = "als",
        device: str | None = None,
        validate_inputs: bool = True,
        **strategy_kwargs,
    )
```

**Parameters:**

- `strategy` — One of: `"auto"`, `"als"`, `"explicit_mf"`, `"neural_mf"`, `"linucb"`, `"user_knn"`, `"popularity"`, `"random"`, `"implicit_als"`, `"implicit_bpr"`. The historical `"als"` strategy is a binary-MF/BCE baseline; use `"implicit_als"` for true alternating least squares. Typos produce a helpful "did you mean?" suggestion.
- `device` — `"cpu"` or `"cuda"`. Defaults to auto-detect.
- `validate_inputs` — When `True`, validates DataFrame schema before fitting. Set `False` for legacy pipeline integration.
- `**strategy_kwargs` — Forwarded to the underlying strategy (e.g., `epochs=10`, `emb_dim=64`, `alpha=1.5`).

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(interactions, user_col="user_id", item_col="item_id", rating_col=None, item_features=None)` | `self` | Fit the model on interaction data. |
| `predict` | `(user_id, item_id)` | `float` | Predict a single score. |
| `predict_many` | `(user_ids, item_ids)` | `np.ndarray` | Batch prediction for user-item pairs. |
| `recommend` | `(user_id, top_k=10, filter_seen=True, candidate_item_ids=None)` | `List[Recommendation]` | Top-K recommendations for a user, optionally restricted to a candidate pool. |
| `baseline_rank` | `(user_id, top_k=10, candidate_item_ids=None)` | `List[Recommendation]` | Frozen fallback ranking for guardrail and safe-mode flows. |
| `as_streaming` | `(monitor=None, guardrail=None, lr=0.05, l2=1e-3, scaling_config=None)` | `StreamingAdaptiveRanker` | Promote a fitted `neural_mf` model into the streaming adapter. |
| `save` | `(path)` | `None` | Save fitted model to disk. |
| `load` | `(path)` | `OrchidRecommender` | Class method. Load a saved model. |
| `available_strategies` | `()` | `str` | Class method. Return all strategies with descriptions. |

`candidate_item_ids` accepts original item IDs from your DataFrame. Unknown
candidate IDs are ignored, which makes it safe to pass a shared catalog pool
while rolling models forward.

### Recommendation

```python
@dataclass
class Recommendation:
    item_id: int
    score: float
```

### STRATEGY_GUIDE

```python
STRATEGY_GUIDE: Dict[str, str]
# Maps strategy name to human-readable description
```

---

## orchid_ranker.scenarios

Scenario-selection helpers for choosing the right Orchid workflow before
instantiating a model.

```python
from orchid_ranker import available_scenarios, recommend_scenarios

matches = recommend_scenarios(
    has_outcomes=True,
    has_concepts=True,
    has_difficulty=True,
    has_prerequisites=True,
    needs_live_adaptation=True,
    use_case="adaptive math practice",
)

top = matches[0]
print(top.scenario.id)
print(top.scenario.entrypoints)
```

```python
@dataclass(frozen=True)
class ScenarioRecipe:
    id: str
    name: str
    summary: str
    use_when: str
    signals: tuple[str, ...]
    algorithms: tuple[str, ...]
    entrypoints: tuple[str, ...]
    docs_anchor: str
    tags: tuple[str, ...] = ()

@dataclass(frozen=True)
class ScenarioFit:
    scenario: ScenarioRecipe
    score: float
    reasons: tuple[str, ...]
```

`available_scenarios()` returns the stable catalog. `recommend_scenarios()`
scores the catalog from product-level booleans such as `has_outcomes`,
`has_concepts`, `needs_live_adaptation`, `needs_safe_rollout`,
`has_new_users`, and `is_regulated`.

---

## orchid_ranker.ope

Offline policy evaluation utilities for adaptive rollout decisions.

```python
def evaluate_logged_policy(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    target_value_col: str | None = None,
    logged_action_value_col: str | None = None,
    max_weight: float | None = None,
) -> LoggedPolicyReport
```

```python
def compare_logged_policies(
    events: pd.DataFrame,
    *,
    reward_col: str,
    propensity_col: str,
    target_probability_col: str,
    baseline_probability_col: str,
    target_value_col: str | None = None,
    baseline_value_col: str | None = None,
    logged_action_value_col: str | None = None,
    max_weight: float | None = None,
) -> PolicyComparisonReport
```

Reports include IPS, SNIPS, direct-method and doubly robust estimates when the
required columns are supplied, plus coverage, effective sample size, weight
diagnostics, and confidence intervals.

---

## orchid_ranker.progression_reward / learning_policy

Progression-aware policy scoring for adaptive sequencing.

```python
class ProgressionRewardConfig:
    target_correct: float = 0.70
    stretch_margin: float = 0.15
    stretch_width: float = 0.30
    default_competence: float = 0.50
    correctness_weight: float = 0.25
    mastery_gain_weight: float = 1.00
    stretch_weight: float = 0.75
    difficulty_weight: float = 0.20
    easy_penalty_weight: float = 0.35
    hard_penalty_weight: float = 0.15
    repetition_penalty_weight: float = 0.25
    easy_correct_threshold: float = 0.88
    hard_correct_threshold: float = 0.25
    incorrect_attempt_credit: float = 0.10
    repetition_window: int = 3
    clip_reward: bool = True
```

```python
class ProgressionValuePolicy:
    def rank(self, user_id, candidate_item_ids, *, top_k=5) -> list
    def observe(self, user_id, item_id, correct)

class DelayedGainValuePolicy:
    def rank(self, user_id, candidate_item_ids, *, top_k=5) -> list
    def observe(self, user_id, item_id, correct)

class SupportConstrainedDelayedGainPolicy:
    def rank(self, user_id, candidate_item_ids, *, top_k=5) -> list
    def observe(self, user_id, item_id, correct)

class DelayedGainRewardModel:
    def predict_one(self, features) -> float
    def predict_many(self, feature_rows) -> list[float]

def build_delayed_gain_training_frame(split, *, concept_col, **kwargs) -> pd.DataFrame
def fit_delayed_gain_reward_model_from_frame(
    examples,
    *,
    example_weighting="uniform",
    sample_weight_col=None,
    cross_fit_folds=1,
) -> DelayedGainRewardModel
def diagnose_delayed_gain_predictions(labels, predictions, *, n_bins=10) -> dict
```

Use `ProgressionValuePolicy` when correctness alone is too shallow. It ranks
items using expected progression reward, including target-correctness fit,
stretch-zone fit, mastery-gain potential, difficulty, and repetition/easy-item
penalties.

Use `DelayedGainValuePolicy` when you have concept labels and want ranking to
prefer items that historically preceded future same-concept improvement. It
combines training-only delayed-gain priors with the progression reward.

Use `SupportConstrainedDelayedGainPolicy` when you also have a fitted
`DelayedGainRewardModel` and want to penalize items with weak logged support.
The benchmark path exposes this as `--policy support_delayed_gain`.

For direct reward-model work, `build_delayed_gain_training_frame` exposes the
same feature frame used by the benchmark. `fit_delayed_gain_reward_model_from_frame`
supports uniform weighting, support-inverse weighting, and MRDR-style weighting
when `target_probability` and `logging_propensity` columns are available.
`diagnose_delayed_gain_predictions` reports RMSE, MAE, calibration error, and
decile lift for validation or target-policy action slices.

---

## orchid_ranker.pykt_bridge

Interoperability helpers for pyKT-style knowledge-tracing workflows.

```python
def export_pykt_sequences(
    interactions: pd.DataFrame,
    output_path: str | Path,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    concept_col: str | None = None,
    timestamp_col: str | None = None,
    duration_col: str | None = None,
) -> list[PyKTSequence]
```

```python
class PyKTPredictionAdapter:
    def predict_many(self, user_id, item_ids) -> dict
    def predict_correct(self, user_id, item_id) -> float
```

Use `export_pykt_sequences` to write Orchid interactions into pyKT's six-line
learner-sequence format. Use `PyKTPredictionAdapter` to feed exported pyKT
probability tables into `KTValuePolicy` and OPE.

---

## orchid_ranker.knowledge_tracing

### BayesianKnowledgeTracing

Hidden Markov Model for estimating category competence from response sequences.

```python
class BayesianKnowledgeTracing:
    def __init__(
        self,
        p_init: float = 0.1,
        p_transit: float = 0.1,
        p_slip: float = 0.1,
        p_guess: float = 0.2,
        mastery_threshold: float = 0.95,
    )
```

**Parameters:**

- `p_init` — Prior probability of knowing the category before any observations.
- `p_transit` — Probability of transitioning from unlearned to learned after each attempt.
- `p_slip` — Probability of an incorrect response despite knowing the category.
- `p_guess` — Probability of a correct response despite not knowing the category.
- `mastery_threshold` — BKT compatibility name for the `p_known()` value above which the category is considered completed.

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update` | `(correct: bool)` | `float` | Process one observation, return updated p_known. |
| `p_known` | `()` | `float` | Current probability of competence. |
| `is_mastered` | `()` | `bool` | BKT compatibility method; whether p_known exceeds the competence threshold. |
| `reset` | `()` | `None` | Reset to initial prior. |

### CompetencyTracker

Tracks competence across multiple categories simultaneously, each with its own BKT instance.

```python
class CompetencyTracker:
    def __init__(
        self,
        competencies: List[str],
        bkt_params: Optional[Dict[str, Dict[str, float]]] = None,
        default_params: Optional[Dict[str, float]] = None,
        success_threshold: float = 0.95,
    )
```

`ProficiencyTracker` is a non-deprecated alias for `CompetencyTracker`. The old `MasteryTracker` name and `skills` / `mastery_threshold` parameters remain as deprecated compatibility aliases.

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update` | `(competency, correct)` | `float` | Record an outcome and return updated competence. |
| `proficiency` | `(competency)` | `float` | Current competence probability for one category. |
| `get_mastery` | `()` | `Dict[str, float]` | Compatibility method returning all competence estimates. |
| `succeeded` | `()` | `List[str]` | Categories above competence threshold. |
| `remaining` | `()` | `List[str]` | Categories below competence threshold. |
| `recommend_next` | `()` | `List[str]` | Suggest the next categories to engage with. |
| `ready_for` | `(competency)` | `bool` | Whether prerequisites are met. |

### ForgettingCurve

Ebbinghaus exponential decay model for spaced repetition scheduling.

```python
class ForgettingCurve:
    def __init__(
        self,
        initial_strength: float = 1.0,
        strength_gain_on_review: float = 0.5,
    )
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `review` | `()` | `None` | Record a review, increasing memory strength. |
| `retention_at` | `(time_since_last_review: float)` | `float` | Predicted retention (0-1) after elapsed time. |
| `should_review` | `(threshold: float = 0.5)` | `bool` | Whether retention has dropped below threshold. |

---

## orchid_ranker.curriculum

### DependencyGraph

Directed acyclic graph for modeling category dependencies with automatic cycle detection.

```python
class DependencyGraph:
    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None)
```

The module name `curriculum` is retained for compatibility. `PrerequisiteGraph` and `SkillGraph` remain as deprecated aliases for `DependencyGraph`.

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_edge` | `(prerequisite, dependent)` | `None` | Add a dependency. Raises `ValueError` on cycles or self-loops. |
| `add_edges` | `(edges: List[Tuple[str, str]])` | `None` | Batch add with cumulative cycle detection. |
| `prerequisites_for` | `(node)` | `Set[str]` | Direct prerequisites of a category/item. |
| `all_prerequisites_for` | `(node)` | `Set[str]` | All transitive prerequisites. |
| `dependents_of` | `(node)` | `Set[str]` | Categories/items that depend on this node. |
| `topological_order` | `()` | `List[str]` | Valid progression sequence (Kahn's algorithm). |
| `path_to` | `(target, completed=None)` | `List[str]` | Shortest path to target from completed nodes. |
| `available` | `(completed: Set[str])` | `List[str]` | Categories/items whose prerequisites are all met. |
| `prerequisites_met` | `(node, completed: Set[str])` | `bool` | Whether all prerequisites are met. |
| `validate` | `()` | `None` | Validate graph integrity. |
| `to_dict` | `()` | `Dict` | Serialize to dictionary. |
| `from_dict` | `(data: Dict)` | `DependencyGraph` | Class method. Deserialize from dictionary. |
| `summary` | `()` | `str` | Human-readable graph summary. |

### ProgressionRecommender

Stretch-zone-aware recommendations that respect prerequisite ordering.

```python
class ProgressionRecommender:
    def __init__(
        self,
        graph: DependencyGraph,
        difficulty_map: Optional[Dict[str, float]] = None,
    )
```

`CurriculumRecommender` remains as a deprecated compatibility alias.

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `recommend` | `(completed: Set[str], n: int = 5)` | `List[str]` | Recommend next items respecting prerequisites and stretch zone. |
| `filter_candidates` | `(candidates, completed)` | `List[str]` | Filter items to those with satisfied prerequisites. |

---

## orchid_ranker.model_selection

### train_test_split

```python
def train_test_split(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    by_user: bool = True,
    random_state: int = 42,
    time_col: Optional[str] = "timestamp",
    chronological: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Split interactions into train/test. When `by_user=True` and a timestamp column is present, each user's future events are held out chronologically. Set `chronological=False` for the legacy random within-user split.

### chronological_user_split

```python
def chronological_user_split(
    interactions: pd.DataFrame,
    *,
    test_size: float = 0.2,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Leak-free per-user split for sequential KT/adaptive-learning evaluation.

### cross_validate

```python
def cross_validate(
    interactions: pd.DataFrame,
    strategy: str,
    k: int = 5,
    metrics: Optional[List[str]] = None,
    strategy_kwargs: Optional[Dict] = None,
    random_state: int = 42,
    time_col: Optional[str] = "timestamp",
    chronological: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]
```

K-fold cross-validation with per-user stratification. If timestamp data is present, folds use train-before-test chronological cutoffs per user. Returns dict of `{metric_name: {"mean": float, "std": float}}`.

### compare_models

```python
def compare_models(
    interactions: pd.DataFrame,
    strategies: List[str],
    k: int = 5,
    **kwargs,
) -> pd.DataFrame
```

Side-by-side comparison of multiple strategies via cross-validation.

### evaluate_on_holdout

```python
def evaluate_on_holdout(
    model: OrchidRecommender,
    test_interactions: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    k: int = 10,
) -> Dict[str, float]
```

Evaluate a fitted model on held-out test data.

---

## orchid_ranker.tuning

### GridSearchCV

```python
class GridSearchCV:
    def __init__(
        self,
        strategy: str,
        param_grid: Dict[str, List[Any]],
        cv: int = 3,
        scoring: str = "ndcg@10",
        random_state: int = 42,
        verbose: int = 0,
    )
```

Exhaustive search over all combinations in `param_grid`.

**Attributes (after fit):** `best_params_`, `best_score_`, `results_` (DataFrame), `best_model_`, `n_iter_`.

### RandomSearchCV

```python
class RandomSearchCV:
    def __init__(
        self,
        strategy: str,
        param_distributions: Dict[str, List[Any]],
        n_iter: int = 10,
        cv: int = 3,
        scoring: str = "ndcg@10",
        random_state: int = 42,
        verbose: int = 0,
    )
```

Samples `n_iter` random combinations from `param_distributions`. Same attributes as `GridSearchCV` after fit.

---

## orchid_ranker.evaluation

### Ranking Metrics

```python
def precision_at_k(recommended: List, relevant: Set, k: int) -> float
def recall_at_k(recommended: List, relevant: Set, k: int) -> float
def ndcg_at_k(recommended: List, relevant_scores: Dict, k: int) -> float
def average_precision(recommended: List, relevant: Set, k: int) -> float
```

### Progression Metrics

```python
def progression_gain(pre_score: float, post_score: float) -> float
    # Normalized gain: (post - pre) / (1 - pre)

def category_coverage(successful_categories: set, total_categories: set) -> float
    # Fraction of total categories where competence is achieved

def sequence_adherence(recommended_items: List, prerequisite_graph: DependencyGraph, completed: Set) -> float
    # Fraction of recommendations with satisfied prerequisites

def stretch_fit(recommended_difficulties: Sequence[float], user_competence: float, stretch_width: float = 0.25) -> float
    # Fraction of items within the user's stretch zone

def engagement_score(interactions: Sequence, total_available: int) -> float
    # Ratio of interactions to available items
```

### ProgressionReport

```python
@dataclass
class ProgressionReport:
    metric_name: str
    value: float
    ci_lower: float
    ci_upper: float
    timestamp: float
```

---

## orchid_ranker.serialization

```python
def save_model(model: Any, path: str | Path) -> None
    # Save OrchidRecommender or TwoTowerRecommender to a versioned checkpoint.
    # Raises ValueError if the model hasn't been fitted.

def load_model(path: str | Path) -> Any
    # Load and restore a saved model.
    # Raises FileNotFoundError, ValueError (corrupted), RuntimeError.
```

---

## orchid_ranker.dp

### get_dp_config

```python
def get_dp_config(preset: str) -> Dict[str, Any]
```

Returns a differential privacy configuration dictionary.

**Presets:**

| Preset | Epsilon | Description |
|--------|---------|-------------|
| `"off"` | -- | DP disabled |
| `"eps_2"` | 2.0 | Light privacy |
| `"eps_1"` | 1.0 | Standard privacy |
| `"eps_05"` | 0.5 | Strong privacy |
| `"eps_02"` | 0.2 | Very strong privacy |

**Returned dict keys:** `enabled`, `noise_multiplier`, `sample_rate`, `delta`, `max_grad_norm`.

---

## orchid_ranker.agents

### AdaptiveAgent

Simulates a user with knowledge, fatigue, trust, and engagement dynamics.

```python
class AdaptiveAgent:
    def __init__(
        self,
        user_id: int,
        knowledge_dim: int = 10,
        knowledge_mode: str = "scalar",
        lr: float = 0.2,
        decay: float = 0.1,
        trust_influence: bool = True,
        fatigue_growth: float = 0.05,
        fatigue_recovery: float = 0.02,
        seed: int = 42,
        zpd_delta: float = 0.10,  # legacy internal name for stretch-zone offset
        zpd_width: float = 0.25,  # legacy internal name for stretch-zone width
        pos_eta: float = 0.85,
    )
```

`StudentAgent` remains as a deprecated compatibility alias while the simulation internals are migrated.

**Key methods:** `accept(item_id, difficulty, correct, dwell_time, feedback)`, `get_knowledge()`, `get_fatigue()`, `get_engagement()`, `get_trust()`.

### TwoTowerRecommender

Neural two-tower recommender with FiLM gating and optional DP-SGD.

```python
class TwoTowerRecommender:
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dim: int = 64,
        device: str = "cpu",
        dp_cfg: Optional[Dict] = None,
        learning_rate: float = 0.001,
    )
```

**Key methods:** `fit(user_matrix, item_matrix, interactions_df)`, `infer(user_ids, item_ids)`, `get_candidates(user_idx, top_k)`.

### MultiUserOrchestrator

Runs multi-user adaptive experiments with online policy optimization.

### DualRecommender

Combines a fixed teacher recommender with an adaptive recommender for knowledge distillation.

---

## orchid_ranker.security

### AccessControl

```python
class AccessControl:
    def __init__(self, policy: Dict)
    def check_permission(self, role: str, action: str) -> bool
```

### AuditLogger

```python
class AuditLogger:
    def __init__(self, log_path: str)
    def log(self, event: str, user: str, details: Dict) -> None
```

---

## orchid_ranker.connectors

### SnowflakeConnector

```python
class SnowflakeConnector:
    def __init__(self, account, user, password, warehouse, database)
```

### BigQueryConnector

```python
class BigQueryConnector:
    def __init__(self, project_id, credentials_path=None)
```

### S3StreamConnector

```python
class S3StreamConnector:
    def __init__(self, bucket, prefix=None, aws_key=None, aws_secret=None)
```

### MLflowTracker

```python
class MLflowTracker:
    def __init__(self, tracking_uri, experiment_name)
```

---

## orchid_ranker.observability

```python
def start_metrics_server(port: int = 8000) -> None
def record_training(strategy: str, dataset: str, metric_name: str, value: float) -> None
def export_metrics() -> str
```

---

## orchid_ranker.visualization

```python
def plot_user_activity(interactions_df, top_n=25)
def plot_item_difficulty(items_df)
def plot_learning_curve(round_summary_df, metric)
def plot_acceptance_heatmap(...)
def plot_round_comparison(...)
def plot_knowledge_trajectory(...)
def plot_metric_trajectory(...)
def plot_metric_grid(...)
```

Each function returns a Matplotlib `Axes` object for further customization.
