# API Reference

Complete reference for all public classes and functions in Orchid Ranker v0.2.1.

---

## orchid_ranker.recommender

### OrchidRecommender

High-level, Surprise-style recommender supporting all built-in strategies.

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

- `strategy` — One of: `"als"`, `"explicit_mf"`, `"neural_mf"`, `"linucb"`, `"user_knn"`, `"popularity"`, `"random"`, `"implicit_als"`, `"implicit_bpr"`. Typos produce a helpful "did you mean?" suggestion.
- `device` — `"cpu"` or `"cuda"`. Defaults to auto-detect.
- `validate_inputs` — When `True`, validates DataFrame schema before fitting. Set `False` for legacy pipeline integration.
- `**strategy_kwargs` — Forwarded to the underlying strategy (e.g., `epochs=10`, `emb_dim=64`, `alpha=1.5`).

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(interactions, user_col="user_id", item_col="item_id", rating_col=None, item_features=None)` | `self` | Fit the model on interaction data. |
| `predict` | `(user_id, item_id)` | `float` | Predict a single score. |
| `predict_many` | `(user_ids, item_ids)` | `np.ndarray` | Batch prediction for user-item pairs. |
| `recommend` | `(user_id, top_k=10, filter_seen=True)` | `List[Recommendation]` | Top-K recommendations for a user. |
| `save` | `(path)` | `None` | Save fitted model to disk. |
| `load` | `(path)` | `OrchidRecommender` | Class method. Load a saved model. |
| `available_strategies` | `()` | `None` | Class method. Print all strategies with descriptions. |

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

## orchid_ranker.knowledge_tracing

### BayesianKnowledgeTracing

Hidden Markov Model for estimating skill mastery from response sequences.

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

- `p_init` — Prior probability of knowing the skill before any observations.
- `p_transit` — Probability of transitioning from unlearned to learned after each attempt.
- `p_slip` — Probability of an incorrect response despite knowing the skill.
- `p_guess` — Probability of a correct response despite not knowing the skill.
- `mastery_threshold` — `p_known()` value above which the skill is considered mastered.

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update` | `(correct: bool)` | `float` | Process one observation, return updated p_known. |
| `p_known` | `()` | `float` | Current probability of mastery. |
| `is_mastered` | `()` | `bool` | Whether p_known exceeds the mastery threshold. |
| `reset` | `()` | `None` | Reset to initial prior. |

### MasteryTracker

Tracks mastery across multiple skills simultaneously, each with its own BKT instance.

```python
class MasteryTracker:
    def __init__(
        self,
        skills: List[str],
        bkt_params: Optional[Dict[str, Dict[str, float]]] = None,
        default_params: Optional[Dict[str, float]] = None,
        mastery_threshold: float = 0.95,
    )
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update` | `(skill, correct)` | `None` | Record an attempt on a skill. |
| `get_mastery` | `(skill)` | `float` | Current mastery probability for a skill. |
| `mastered_skills` | `()` | `Set[str]` | All skills above mastery threshold. |
| `unmastered_skills` | `()` | `Set[str]` | All skills below mastery threshold. |
| `recommend_next` | `()` | `Optional[str]` | Suggest the next skill to practice. |
| `ready_for` | `(skill)` | `bool` | Whether prerequisites are met. |

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

### PrerequisiteGraph

Directed acyclic graph for modeling skill dependencies with automatic cycle detection.

```python
class PrerequisiteGraph:
    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_edge` | `(prerequisite, dependent)` | `None` | Add a dependency. Raises `ValueError` on cycles or self-loops. |
| `add_edges` | `(edges: List[Tuple[str, str]])` | `None` | Batch add with cumulative cycle detection. |
| `prerequisites_for` | `(skill)` | `Set[str]` | Direct prerequisites of a skill. |
| `all_prerequisites_for` | `(skill)` | `Set[str]` | All transitive prerequisites. |
| `dependents_of` | `(skill)` | `Set[str]` | Skills that depend on this skill. |
| `topological_order` | `()` | `List[str]` | Valid learning sequence (Kahn's algorithm). |
| `learning_path` | `(target_skill, mastered=None)` | `List[str]` | Shortest path to target from current mastery. |
| `available_skills` | `(mastered: Set[str])` | `List[str]` | Skills whose prerequisites are all mastered. |
| `is_ready` | `(skill, mastered: Set[str])` | `bool` | Whether all prerequisites are met. |
| `validate` | `()` | `None` | Validate graph integrity. |
| `to_dict` | `()` | `Dict` | Serialize to dictionary. |
| `from_dict` | `(data: Dict)` | `PrerequisiteGraph` | Class method. Deserialize from dictionary. |
| `summary` | `()` | `str` | Human-readable graph summary. |

### CurriculumRecommender

ZPD-aware recommendations that respect prerequisite ordering.

```python
class CurriculumRecommender:
    def __init__(
        self,
        graph: PrerequisiteGraph,
        difficulty_map: Optional[Dict[str, float]] = None,
    )
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `recommend` | `(student_mastery: Set[str], n: int = 5)` | `List[str]` | Recommend next items respecting prerequisites and ZPD. |
| `filter_candidates` | `(candidates, mastered)` | `List[str]` | Filter items to those with satisfied prerequisites. |

---

## orchid_ranker.model_selection

### train_test_split

```python
def train_test_split(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    by_user: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Split interactions into train/test. When `by_user=True`, each user's interactions are split proportionally (stratified).

### cross_validate

```python
def cross_validate(
    interactions: pd.DataFrame,
    strategy: str,
    k: int = 5,
    metrics: Optional[List[str]] = None,
    strategy_kwargs: Optional[Dict] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]
```

K-fold cross-validation with per-user stratification. Returns dict of `{metric_name: {"mean": float, "std": float}}`.

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

### Educational Metrics

```python
def learning_gain(pre_score: float, post_score: float) -> float
    # Normalized gain: (post - pre) / (1 - pre)

def knowledge_coverage(mastered_skills: set, total_skills: set) -> float
    # Fraction of total skills mastered

def curriculum_adherence(recommended_items: List, prerequisite_graph: PrerequisiteGraph, mastered: Set) -> float
    # Fraction of recommendations with satisfied prerequisites

def difficulty_appropriateness(recommended_difficulties: Sequence[float], student_ability: float, zpd_width: float = 0.25) -> float
    # Fraction of items within the student's ZPD

def engagement_score(interactions: Sequence, total_available: int) -> float
    # Ratio of interactions to available items
```

### EducationalReport

```python
@dataclass
class EducationalReport:
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

### StudentAgent

Simulates a learner with knowledge, fatigue, trust, and engagement dynamics.

```python
class StudentAgent:
    def __init__(
        self,
        user_id: int,
        knowledge_dim: int = 10,
        knowledge_mode: str = "scalar",  # "scalar", "IRT", "MIRT", "ZPD", "ContextualZPD"
        lr: float = 0.2,
        decay: float = 0.1,
        trust_influence: bool = True,
        fatigue_growth: float = 0.05,
        fatigue_recovery: float = 0.02,
        act_mode: str = "ZPD",
        seed: int = 42,
        zpd_delta: float = 0.10,
        zpd_width: float = 0.25,
        pos_eta: float = 0.85,
    )
```

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

Combines a fixed teacher recommender with an adaptive student recommender for knowledge distillation.

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
