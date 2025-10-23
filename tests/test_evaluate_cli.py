import pandas as pd
import pytest

from orchid_ranker.cli.evaluate import main as eval_main


def _write_dataset(tmp_path):
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 11, 10, 12, 11, 13],
            "label": [1, 0, 1, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [11, 12, 13],
            "label": [0, 1, 1],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def test_cli_evaluate_runs(tmp_path, capsys):
    train_path, test_path = _write_dataset(tmp_path)
    args = [
        "--train",
        str(train_path),
        "--test",
        str(test_path),
        "--strategy",
        "als,epochs=1",
    ]
    assert eval_main(args) == 0
    captured = capsys.readouterr()
    assert "Strategy" in captured.out
