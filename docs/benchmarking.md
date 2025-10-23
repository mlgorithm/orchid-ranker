# Benchmarking Guide

Use the provided helpers to keep regression checks reproducible:

## Pytest smoke suite

```bash
python -m pytest tests/
```

Covers popular recommender strategies (ALS, LinUCB), cold-start behaviour, and
basic DP-accountant sanity checks.

## Agentic smoke run

```bash
python benchmarks/run_agentic_smoke.py --rounds 2 --users 3 --items 12
```

Creates synthetic users/items, wires them through `MultiUserOrchestrator`, and
emits JSONL logs under `runs/`.

## Surprise comparison (optional)

```bash
pip install surprise
python benchmarks/compare_surprise.py \
    --train data/train.csv \
    --test data/test.csv \
    --rating-col label
```

Prints RMSE for Orchid Ranker's ALS baseline alongside Surprise's SVD.

## Implicit comparison

```bash
pip install implicit
python benchmarks/compare_implicit.py \
    --train data/train.csv \
    --test data/test.csv \
    --rating-col label
```

Reports RMSE for Orchid ALS and the `implicit` ALS implementation on the same
splits.

## ReCLaB dense benchmark

```bash
pip install reclab
python benchmarks/compare_reclab.py --env topics-static-v1-small
```

Generates train/test splits from the selected ReCLaB environment and compares
Orchid ALS with ReCLaB's TopPop recommender.

## Orchid evaluation CLI

```bash
orchid-evaluate \
    --train data/train.csv \
    --test data/test.csv \
    --strategy "als,epochs=5" \
    --strategy "implicit_als,factors=64"
```

Runs the built-in metrics suite (Precision@5, Recall@5, MAP@10, NDCG@10) and
supports basic hyper-parameter sweeps via repeated `--strategy` flags.
