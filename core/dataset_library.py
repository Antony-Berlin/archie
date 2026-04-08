"""
Dataset Library for NeuralArch-Bench.

DATASET_LIBRARY maps dataset names to config dicts used by trainer.py.
Each config is plain JSON-serializable data — no callables.
"""

import os

DATASET_LIBRARY: dict[str, dict] = {
    "fashion_mnist": {
        "torchvision_class": "FashionMNIST",
        "data_dir": os.environ.get("FASHION_MNIST_DIR", "/tmp/fashion_mnist"),
        "input_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "normalize_mean": [0.5],
        "normalize_std": [0.5],
        "train_subset": 8000,
        "test_subset": 2000,
    },
    "mnist": {
        "torchvision_class": "MNIST",
        "data_dir": os.environ.get("MNIST_DIR", "/tmp/mnist"),
        "input_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "normalize_mean": [0.5],
        "normalize_std": [0.5],
        "train_subset": 8000,
        "test_subset": 2000,
    },
    "cifar10": {
        "torchvision_class": "CIFAR10",
        "data_dir": os.environ.get("CIFAR10_DIR", "/tmp/cifar10"),
        "input_channels": 3,
        "input_size": 32,
        "num_classes": 10,
        "normalize_mean": [0.4914, 0.4822, 0.4465],
        "normalize_std": [0.2023, 0.1994, 0.2010],
        "train_subset": 8000,
        "test_subset": 2000,
    },
}


def get_dataset_names() -> list[str]:
    return list(DATASET_LIBRARY.keys())
