# Dataset Ingestion & Schema Guide

This guide walks through mapping any custom dataset into the CSV + YAML layout expected by Orchid Ranker.

## 1. Required artefacts

Every dataset must provide:

1. `train.csv`, `val.csv`, `test.csv` — interaction logs with at least columns `u` (user id), `i` (item id), and `label`. Optional fields include timestamps, acceptance flags, dwell time, etc.
2. `side_information_users.csv` — per-user features.
3. `side_information_items.csv` — per-item features.
4. Config YAML describing paths and feature types.

## 2. Example schema

```yaml
run:
  dataset: my_dataset

datasets:
  my_dataset:
    paths:
      base_dir: data/my_dataset
      train: train.csv
      val: val.csv
      test: test.csv
      side_information_users: side_information_users.csv
      side_information_items: side_information_items.csv
    interactions:
      timestamp: timestamp  # optional column name
      accept_col: accept    # optional acceptance flag
    users:
      categorical: [segment, locale]
      numeric: [days_since_signup, avg_session_minutes]
    items:
      categorical: [topic, difficulty_band]
      numeric: [avg_rating, recency_weight]
```

## 3. Validation checklist

- Ensure CSVs are UTF-8 with headers.
- User/item ids must be integers; if not, map to ints via pandas `factorize`.
- Numeric features should be floats; categorical features left as strings (the loader will one-hot encode).
- Empty values: prefer `NaN` for numeric columns and empty strings for categorical ones.

## 4. Example conversion snippet

```python
import pandas as pd

raw = pd.read_parquet("logs.parquet")

# Train/val/test split
train = raw.sample(frac=0.7, random_state=42)
val = raw.drop(train.index).sample(frac=0.5, random_state=42)
test = raw.drop(train.index.union(val.index))

def export_split(df, path):
    df = df.copy()
    df["u"] = df["user_guid"].astype("category").cat.codes
    df["i"] = df["content_id"].astype("category").cat.codes
    df[["u", "i", "label", "timestamp"]].to_csv(path, index=False)

export_split(train, "data/my_dataset/train.csv")
export_split(val, "data/my_dataset/val.csv")
export_split(test, "data/my_dataset/test.csv")
```

Generate side-information tables in a similar fashion (groupby aggregations for users, join metadata for items).

## 5. Loading the dataset

```python
from orchid_ranker.data import DatasetLoader
loader = DatasetLoader(config_path="configs/my_dataset.yaml")
train_df = loader.train_df
```

## 6. Schema validation helper

```python
from orchid_ranker.data import validate_schema

validate_schema("configs/my_dataset.yaml")
```

This checks file paths, required columns, and feature types before running experiments.

## 7. Common issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: column 'u' missing` | CSV headers mis-specified | Rename columns / update config |
| `KeyError: side_information_users` | Path missing in YAML | Provide file path or remove section |
| `TypeError: cannot convert string to float` | Non-numeric values in numeric column | Clean data prior to export |
