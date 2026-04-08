"""
Architecture Library for NeuralArch-Bench.

Each entry in ARCH_LIBRARY maps a name to a complete Python source string
defining an ArchModel(nn.Module) class.

All architectures reference two module-level constants that trainer.py injects
before loading the file:
    INPUT_CHANNELS  — 1 (grayscale) or 3 (RGB)
    INPUT_SIZE      — side length in pixels (28 for MNIST/FashionMNIST, 32 for CIFAR-10)
"""

_SIMPLE_MLP = '''\
import torch
import torch.nn as nn

# INPUT_CHANNELS and INPUT_SIZE are injected by trainer.py
INPUT_CHANNELS = 1
INPUT_SIZE = 28

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

_BATCH_NORM_MLP = '''\
import torch
import torch.nn as nn

INPUT_CHANNELS = 1
INPUT_SIZE = 28

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
'''

_CONV_NET = '''\
import torch
import torch.nn as nn

INPUT_CHANNELS = 1
INPUT_SIZE = 28

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # After two pool ops: INPUT_SIZE // 4
        pooled = INPUT_SIZE // 4
        self.fc1 = nn.Linear(32 * pooled * pooled, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''

_DROPOUT_REGULARIZED = '''\
import torch
import torch.nn as nn

INPUT_CHANNELS = 1
INPUT_SIZE = 28

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        pooled = INPUT_SIZE // 4
        self.fc1 = nn.Linear(64 * pooled * pooled, 256)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.drop1(x.view(x.size(0), -1))
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x
'''

_RESNET_LIKE = '''\
import torch
import torch.nn as nn

INPUT_CHANNELS = 1
INPUT_SIZE = 28

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = BasicBlock(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = BasicBlock(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        pooled = INPUT_SIZE // 4
        self.classifier = nn.Linear(32 * pooled * pooled, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = x.view(x.size(0), -1)
        return self.classifier(x)
'''

ARCH_LIBRARY: dict[str, str] = {
    "simple_mlp":           _SIMPLE_MLP,
    "batch_norm_mlp":       _BATCH_NORM_MLP,
    "conv_net":             _CONV_NET,
    "dropout_regularized":  _DROPOUT_REGULARIZED,
    "resnet_like":          _RESNET_LIKE,
}
