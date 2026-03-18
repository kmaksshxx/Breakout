import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
