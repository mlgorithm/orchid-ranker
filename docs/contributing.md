# Contributing

Use the repository-level [CONTRIBUTING.md](https://github.com/mlgorithm/orchid-ranker/blob/main/CONTRIBUTING.md) for the full contribution workflow.

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_recommender.py
python -m ruff check src/
```

Before opening a pull request, run the focused tests for the feature area you
changed and include the commands in the PR description.
