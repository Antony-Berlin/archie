"""
Architecture Library for NeuralArch-Bench (tabular data edition).

Each entry maps a name to a complete Python source string defining
ArchModel(nn.Module) for tabular classification tasks.

All architectures reference two module-level constants injected by trainer.py:
    NUM_FEATURES  — number of input features for the dataset
    NUM_CLASSES   — number of output classes
"""

_TABULAR_SIMPLE_MLP = '''\
import torch.nn as nn

NUM_FEATURES = 4
NUM_CLASSES = 3

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
'''

_TABULAR_BATCH_NORM_MLP = '''\
import torch.nn as nn

NUM_FEATURES = 4
NUM_CLASSES = 3

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
'''

_TABULAR_DROPOUT_MLP = '''\
import torch.nn as nn

NUM_FEATURES = 4
NUM_CLASSES = 3

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.drop1(self.fc1(x)))
        x = self.relu(self.drop2(self.fc2(x)))
        return self.fc3(x)
'''

_TABULAR_DEEP_MLP = '''\
import torch.nn as nn

NUM_FEATURES = 4
NUM_CLASSES = 3

class ArchModel(nn.Module):
    """Deep 4-layer MLP — may overfit on small datasets, inviting diagnosis."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.out(x)
'''

_TABULAR_MINIMAL = '''\
import torch.nn as nn

NUM_FEATURES = 4
NUM_CLASSES = 3

class ArchModel(nn.Module):
    """Intentionally under-parameterized — invites underfitting diagnosis."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 8)
        self.fc2 = nn.Linear(8, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
'''

ARCH_LIBRARY: dict[str, str] = {
    "tabular_simple_mlp":     _TABULAR_SIMPLE_MLP,
    "tabular_batch_norm_mlp": _TABULAR_BATCH_NORM_MLP,
    "tabular_dropout_mlp":    _TABULAR_DROPOUT_MLP,
    "tabular_deep_mlp":       _TABULAR_DEEP_MLP,
    "tabular_minimal":        _TABULAR_MINIMAL,
}
