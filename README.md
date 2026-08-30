# Orchid Ranker

[![PyPI version](https://img.shields.io/pypi/v/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![CI](https://github.com/mlgorithm/orchid-ranker/actions/workflows/ci.yaml/badge.svg)](https://github.com/mlgorithm/orchid-ranker/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Orchid Ranker is an adaptive-practice engine for learning products.

It chooses the next eligible exercise, observes the learner's result, and
adapts the next recommendation. It is designed for assessed practice,
technical-skills training, test preparation, and professional certification—
not generic feed or product recommendation.

Your platform owns the curriculum, content, safety rules, and learner
experience. Orchid owns the adaptive decision loop and the evidence needed to
test whether it improves retained mastery.

## Install

```bash
pip install orchid-ranker
```

Python 3.11–3.13 is supported.

## Use it

You need four columns: learner, exercise, outcome, and timestamp. In the API
these stay domain-neutral as `user_id`, `item_id`, `outcome`, and `timestamp`.

```python
import pandas as pd
from orchid_ranker import AdaptiveRanker

history = pd.DataFrame({
    "user_id":   ["learner-a", "learner-a", "learner-a", "learner-b", "learner-b", "learner-b"],
    "item_id":   [101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0],  # correct / not yet correct
    "timestamp": [1,   2,   3,   1,   2,   3],
})

ranker = AdaptiveRanker().fit(history)

# Inspect whether this pilot has enough support for knowledge tracing.
print(ranker.learning_readiness())

ranked = ranker.recommend(
    user_id="a",
    candidate_item_ids=[101, 102, 201],
    top_k=2,
)

ranker.observe(
    user_id="a",
    item_id=ranked[0].item_id,
    outcome=1,
    timestamp=4,
)
```

This is the complete loop. Small pilots automatically use a transparent
empirical learner; Orchid uses knowledge tracing only when basic support checks
are met. `outcome` is binary: `1` for a completed/correct practice result and
`0` otherwise. Do not use clicks as a learning outcome.

Your application supplies only pedagogically eligible items. It should enforce
availability, prerequisites, assessment holdouts, accommodations, and any
other hard curriculum rules; Orchid only orders that set.

## Learn more

- [Quickstart](docs/quickstart.md)
- [How Orchid works](docs/overview.md)
- [Adaptive-practice data readiness](docs/guides/00-adaptive-practice.md)
- [API](docs/api_reference.md)
- [Production serving and decision logging](docs/guides/02-serve-streaming.md)
- [Run a learning-efficacy pilot](docs/guides/03-learning-pilot.md)
- [Pilot integration contract](docs/guides/04-pilot-integration.md)
- [End-to-end reference-pilot workflow](docs/guides/05-pilot-workflow.md)
- [1.x API support policy](docs/api-support-policy.md)
- [Validate an adaptive rollout](docs/benchmarks/credibility.md)

## Development

```bash
python -m pip install -e '.[dev]'
./scripts/run_full_tests.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md),
[docs/coding-standards.md](docs/coding-standards.md), and
[RELEASING.md](RELEASING.md).

## Community and support

Use [SUPPORT.md](SUPPORT.md) for usage and contribution guidance,
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for participation expectations, and
[SECURITY.md](SECURITY.md) for private vulnerability reporting. The canonical
software-citation metadata is in [CITATION.cff](CITATION.cff).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{orchid_ranker,
  title={Orchid Ranker: Outcome-Driven Adaptive Recommendation},
  author={Sam Urmian},
  version={1.0.0},
  year={2026},
  url={https://github.com/mlgorithm/orchid-ranker}
}
```
