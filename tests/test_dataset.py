from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "behavior_data.csv"


def test_dataset_has_expected_columns_and_binary_labels():
    data = pd.read_csv(DATA_PATH)
    assert {"Distance", "Time", "Jitter", "Is_Human"}.issubset(data.columns)
    assert set(data["Is_Human"].dropna().unique()) == {0, 1}


def test_dataset_features_are_non_negative_and_time_is_positive():
    data = pd.read_csv(DATA_PATH)
    assert (data[["Distance", "Jitter"]] >= 0).all().all()
    assert (data["Time"] > 0).all()
