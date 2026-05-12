from __future__ import annotations

import pandas as pd
import yaml

from orchid_ranker.data import DatasetLoader, load_dataset


def _write_bundle(root):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    interactions = pd.DataFrame({"u": [1, 2], "i": [10, 11], "label": [1, 0]})
    users = pd.DataFrame({"user_id": [1, 2], "segment": ["a", "b"]})
    items = pd.DataFrame({"item_id": [10, 11], "kind": ["x", "y"]})
    for split in ("train", "val", "test"):
        interactions.to_csv(data_dir / f"{split}.csv", index=False)
    users.to_csv(data_dir / "users.csv", index=False)
    items.to_csv(data_dir / "items.csv", index=False)
    config = {
        "run": {"dataset": "demo"},
        "datasets": {
            "demo": {
                "paths": {
                    "base_dir": "data",
                    "train": "train.csv",
                    "val": "val.csv",
                    "test": "test.csv",
                    "side_information_users": "users.csv",
                    "side_information_items": "items.csv",
                }
            }
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_from_yaml_resolves_paths_relative_to_config_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "nested"
    config_dir.mkdir()
    config_path = _write_bundle(config_dir)
    monkeypatch.chdir(tmp_path)

    data, meta = DatasetLoader().load_from_yaml(str(config_path), encode_side_info=False)

    assert len(data["train"]) == 2
    assert meta["paths"]["train"].endswith("nested/data/train.csv")


def test_load_dataset_preserves_config_relative_paths(tmp_path, monkeypatch):
    config_dir = tmp_path / "nested"
    config_dir.mkdir()
    config_path = _write_bundle(config_dir)
    monkeypatch.chdir(tmp_path)

    result = load_dataset(None, None, str(config_path), encode_side_info=False)

    assert len(result["train"]) == 2
