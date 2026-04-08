"""
Isolated training subprocess for NeuralArch-Bench.

Invoked by the environment via subprocess.run(); writes results.json on completion.
Never import this module directly — always run as a subprocess.

Usage:
    python trainer.py --model-file model_to_edit.py --results-file results.json [--dataset fashion_mnist]
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Support both direct invocation and package-context imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset_library import DATASET_LIBRARY

EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_input_constants(model_file: str, input_channels: int, input_size: int) -> None:
    """Prepend INPUT_CHANNELS / INPUT_SIZE constants to model_file if not already present."""
    path = Path(model_file)
    original = path.read_text()
    if not original.startswith("INPUT_CHANNELS"):
        header = f"INPUT_CHANNELS = {input_channels}\nINPUT_SIZE = {input_size}\n\n"
        path.write_text(header + original)


def load_model_from_file(model_file: str) -> nn.Module:
    spec = importlib.util.spec_from_file_location("dynamic_model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_cls = getattr(mod, "ArchModel")
    return model_cls()


def train(model_file: str, results_file: str, dataset_name: str = "fashion_mnist") -> None:
    results: dict = {
        "accuracy": 0.0,
        "param_count": 0,
        "loss_curve": [],
        "error": None,
    }

    cfg = DATASET_LIBRARY.get(dataset_name, DATASET_LIBRARY["fashion_mnist"])

    # Inject dataset constants so architectures can reference INPUT_CHANNELS / INPUT_SIZE
    try:
        inject_input_constants(model_file, cfg["input_channels"], cfg["input_size"])
    except Exception:
        results["error"] = traceback.format_exc()
        with open(results_file, "w") as f:
            json.dump(results, f)
        sys.exit(0)

    try:
        model = load_model_from_file(model_file)
    except Exception:
        results["error"] = traceback.format_exc()
        with open(results_file, "w") as f:
            json.dump(results, f)
        sys.exit(0)

    results["param_count"] = count_parameters(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
    ])

    try:
        dataset_cls = getattr(datasets, cfg["torchvision_class"])
        train_set = dataset_cls(cfg["data_dir"], train=True, download=True, transform=transform)
        test_set = dataset_cls(cfg["data_dir"], train=False, download=True, transform=transform)
    except Exception:
        results["error"] = traceback.format_exc()
        with open(results_file, "w") as f:
            json.dump(results, f)
        sys.exit(0)

    train_loader = DataLoader(
        torch.utils.data.Subset(train_set, range(cfg["train_subset"])),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(test_set, range(cfg["test_subset"])),
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_curve: list[float] = []

    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            batches = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            loss_curve.append(round(epoch_loss / max(batches, 1), 4))

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        results["accuracy"] = round(correct / max(total, 1), 4)
        results["loss_curve"] = loss_curve[-5:]

    except RuntimeError:
        results["error"] = traceback.format_exc()

    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="model_to_edit.py")
    parser.add_argument("--results-file", default="results.json")
    parser.add_argument("--dataset", default="fashion_mnist",
                        help="Dataset name from dataset_library.py (default: fashion_mnist)")
    args = parser.parse_args()
    train(args.model_file, args.results_file, args.dataset)
