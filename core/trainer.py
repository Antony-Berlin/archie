"""
Isolated training subprocess for NeuralArch-Bench (tabular data edition).

Invoked by the environment via subprocess.run(); writes results.json on completion.
Never import this module directly — always run as a subprocess.

Usage:
    python trainer.py --model-file model_to_edit.py --results-file results.json [--dataset iris]
"""

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Support both direct invocation and package-context imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset_library import DATASET_LIBRARY

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_tabular_constants(model_file: str, num_features: int, num_classes: int) -> None:
    """Prepend NUM_FEATURES / NUM_CLASSES constants to model_file if not already present."""
    path = Path(model_file)
    original = path.read_text()
    if not original.startswith("NUM_FEATURES"):
        header = f"NUM_FEATURES = {num_features}\nNUM_CLASSES = {num_classes}\n\n"
        path.write_text(header + original)


def load_model_from_file(model_file: str) -> nn.Module:
    spec = importlib.util.spec_from_file_location("dynamic_model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "ArchModel")()


def _write_results(results: dict, results_file: str) -> None:
    with open(results_file, "w") as f:
        json.dump(results, f)


def train(model_file: str, results_file: str, dataset_name: str = "iris") -> None:
    results: dict = {"accuracy": 0.0, "param_count": 0, "loss_curve": [], "error": None}

    cfg = DATASET_LIBRARY.get(dataset_name, DATASET_LIBRARY["iris"])

    try:
        inject_tabular_constants(model_file, cfg["num_features"], cfg["num_classes"])
    except Exception:
        results["error"] = traceback.format_exc()
        _write_results(results, results_file)
        sys.exit(0)

    try:
        model = load_model_from_file(model_file)
    except Exception:
        results["error"] = traceback.format_exc()
        _write_results(results, results_file)
        sys.exit(0)

    results["param_count"] = count_parameters(model)

    try:
        df = pd.read_csv(cfg["csv_path"])
        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].values.astype("int64")

        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype("float32")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=BATCH_SIZE,
        )
    except Exception:
        results["error"] = traceback.format_exc()
        _write_results(results, results_file)
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_curve: list[float] = []

    try:
        for _ in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            loss_curve.append(round(epoch_loss / max(batches, 1), 4))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, predicted = model(X_batch).max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)

        results["accuracy"] = round(correct / max(total, 1), 4)
        results["loss_curve"] = loss_curve[-5:]

    except RuntimeError:
        results["error"] = traceback.format_exc()

    _write_results(results, results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="model_to_edit.py")
    parser.add_argument("--results-file", default="results.json")
    parser.add_argument("--dataset", default="iris",
                        help="Dataset name from dataset_library.py (default: iris)")
    args = parser.parse_args()
    train(args.model_file, args.results_file, args.dataset)
