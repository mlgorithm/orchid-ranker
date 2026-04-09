# Migrating from Legacy Orchestrator

The `orchid_ranker.contrib.legacy_orchestrator` module is deprecated as of v0.3.0 and will be removed in v0.5.0. This guide walks you through migrating to the supported `MultiUserOrchestrator` API.

## What changed

The legacy orchestrator was a single-file prototype that combined configuration, state management, and the orchestration loop in one monolithic class. The new implementation splits these concerns:

| Legacy | New location | Purpose |
|---|---|---|
| `LegacyOrchestrator` | `orchid_ranker.agents.orchestrator.MultiUserOrchestrator` | Main orchestration loop |
| Inline config dicts | `orchid_ranker.agents.config.MultiConfig` | Typed configuration dataclass |
| Inline user context | `orchid_ranker.agents.config.UserCtx` | Per-user context container |
| Inline policy state | `orchid_ranker.agents.config.PolicyState` | Adaptive policy parameters |

## Import changes

**Before (deprecated):**

```python
from orchid_ranker.contrib.legacy_orchestrator import LegacyOrchestrator
```

**After:**

```python
from orchid_ranker import MultiUserOrchestrator, MultiConfig, UserCtx
# or from the specific modules:
from orchid_ranker.agents.orchestrator import MultiUserOrchestrator
from orchid_ranker.agents.config import MultiConfig, UserCtx
```

## Configuration changes

**Before:** The legacy orchestrator accepted a flat dictionary of configuration options.

```python
orch = LegacyOrchestrator(
    model=model,
    users=users,
    config={"rounds": 50, "warmup_rounds": 5, "top_k": 10, ...},
)
```

**After:** Use the `MultiConfig` dataclass for type-safe, documented configuration.

```python
config = MultiConfig(
    rounds=50,
    warmup_rounds=5,
    top_k=10,
    # ... see MultiConfig docstring for all fields
)
orch = MultiUserOrchestrator(
    rec=model,
    users=users,
    cfg=config,
    device=torch.device("cpu"),
)
```

## Running the experiment

**Before:**

```python
results = orch.run_experiment()
```

**After:**

```python
results = orch.run()
```

The return format is largely the same: per-round and per-user metrics dictionaries. See the `MultiUserOrchestrator.run()` docstring for details on the output schema.

## Key behavioral differences

1. **Differential privacy**: The new orchestrator has first-class DP support via `MultiConfig.dp_preset`. The legacy orchestrator required manual DP configuration.

2. **Adaptive policy**: The new orchestrator supports adaptive policy tuning (exploration rate, diversity lambda, top-k) via `PolicyState`. The legacy orchestrator used fixed hyperparameters.

3. **Batched inference**: The new orchestrator can use `TwoTowerRecommender.infer_batch()` for significantly faster multi-user inference. The legacy orchestrator always called `infer()` per-user.

4. **Logging**: The new orchestrator uses `JSONLLogger` for structured event logging. Attach via the `logger` parameter in `MultiConfig`.

## Timeline

- **v0.3.0**: Legacy orchestrator emits `DeprecationWarning` on import.
- **v0.4.0**: Legacy orchestrator will raise `FutureWarning` (louder).
- **v0.5.0**: Legacy orchestrator module will be removed entirely.

## Need help?

Open an issue on the [GitHub repository](https://github.com/uib-slate/orchid-ranker/issues) if you encounter migration difficulties.
