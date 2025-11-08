# Tutorial: Pareto-Constrained Expert Bandit (PC-EB)

This tutorial shows how to configure the Pareto-aware controller that selects among teacher/adaptive experts while enforcing constraints.

## 1. Configuring experts

Create a YAML snippet (or inline dict) describing each expert's knobs:

```python
from pc_eb.types import ExpertConfig
experts = [
    ExpertConfig("teacher", delta_max=0.0, mmr_lambda=0.0, novelty_lambda=0.0),
    ExpertConfig("adapt_safe", delta_max=0.2, mmr_lambda=0.0, novelty_lambda=0.0),
    ExpertConfig("adapt_diverse", delta_max=0.2, mmr_lambda=0.2, novelty_lambda=0.1),
]
```

## 2. Metrics and constraints

```python
from pc_eb.multiobj import MetricSpec, ConstraintSpec
metrics = [MetricSpec("utility", weight=1.0, direction="max", z=1.64)]
constraints = [ConstraintSpec("accept", threshold=2.0, direction="min", z=1.64)]
```

## 3. Instantiate the orchestrator

```python
from pc_eb.orchestrator import SafeAgenticOrchestrator
from pc_eb.controller import BanditConfig, PCConfig

orch = SafeAgenticOrchestrator(
    ranking_api=my_api,  # see docs for implementing slate scoring API
    experts=experts,
    bandit_cfg=BanditConfig(gamma=0.05, eta=0.5),
    pc_cfg=PCConfig(
        acceptance_floor=2.0,
        dual_eta=0.05,
        pareto_z=1.64,
        selector="hv",  # hypervolume
    ),
    metrics=metrics,
    constraints=constraints,
)
```

## 4. Running a scenario

```python
summary = orch.run_rounds(num_rounds=20)
print(summary.metrics)
```

## 5. Interpreting output

- `meta_regret`: cumulative regret of the bandit vs best feasible expert.
- `constraint_violation`: how often constraints were broken before dual adjustments.
- Hypervolume score indicates joint performance across metrics.

## 6. Visualization

Use the included notebook `docs/tutorials/pc_eb.ipynb` to visualize mix probabilities, dual variables, and constraint satisfaction.
