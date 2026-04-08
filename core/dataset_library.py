"""
Dataset Library for NeuralArch-Bench.

Tabular CSV datasets — no image processing, trains in seconds.
Each entry has the csv_path, num_features, and num_classes needed by trainer.py.
"""

from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_LIBRARY: dict[str, dict] = {
    "iris": {
        "csv_path": str(_DATA_DIR / "iris.csv"),
        "num_features": 4,
        "num_classes": 3,
    },
    "wine": {
        "csv_path": str(_DATA_DIR / "wine.csv"),
        "num_features": 13,
        "num_classes": 3,
    },
    "breast_cancer": {
        "csv_path": str(_DATA_DIR / "breast_cancer.csv"),
        "num_features": 30,
        "num_classes": 2,
    },
}


def get_dataset_names() -> list[str]:
    return list(DATASET_LIBRARY.keys())
