"""
Initial "broken" model: linear layers with no activation functions.
The agent's first task (arch-foundations) is to add activations so accuracy > 85%.
"""

import torch
import torch.nn as nn


class ArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)   # no activation — intentionally broken
        x = self.fc2(x)
        x = self.fc3(x)
        return x
