# Orchid Ranker

[![PyPI version](https://img.shields.io/pypi/v/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![CI](https://github.com/mlgorithm/orchid-ranker/actions/workflows/ci.yaml/badge.svg)](https://github.com/mlgorithm/orchid-ranker/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Orchid Ranker is an outcome-driven adaptive recommender.

It does one thing: choose the next item from a candidate set, observe what
happened, and adapt the next recommendation.

Use it for exercises, onboarding steps, training modules, practice tasks,
gameplay challenges, or any other sequence with a measurable positive outcome.

## Install

```bash
pip install orchid-ranker
```

Python 3.11–3.13 is supported.

## Use it

You need four columns: user, item, outcome, and timestamp.

```python
import pandas as pd
from orchid_ranker import AdaptiveRanker

history = pd.DataFrame({
    "user_id":   ["a", "a", "a", "b", "b", "b"],
    "item_id":   [101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0],
    "timestamp": [1,   2,   3,   1,   2,   3],
})

ranker = AdaptiveRanker().fit(history)

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

This is the complete loop. Orchid selects its internal policy; you do not
choose a model. `outcome` is binary: `1` for the result you want and `0` for
everything else.

Your application supplies only eligible items. Orchid orders them; it does not
override availability, safety, licensing, or business rules.

## Learn more

- [Quickstart](docs/quickstart.md)
- [How Orchid works](docs/overview.md)
- [API](docs/api_reference.md)
- [Production serving and decision logging](docs/guides/02-serve-streaming.md)
- [Reference pilots](docs/examples.md)
- [Validate an adaptive rollout](docs/benchmarks/credibility.md)

## Development

```bash
python -m pip install -e '.[dev]'
./scripts/run_full_tests.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md),
[docs/coding-standards.md](docs/coding-standards.md), and
[RELEASING.md](RELEASING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{orchid_ranker,
  title={Orchid Ranker: Outcome-Driven Adaptive Recommendation},
  author={Sam Urmian},
  year={2024},
  url={https://github.com/mlgorithm/orchid-ranker}
}
```
