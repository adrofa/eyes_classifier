import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(version, weights=None):
    if version == 1:
        model = CustomNetV1()
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        raise Exception(f"Model version '{version}' is unknown!")
    return model


class CustomNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(3, 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=0)
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(self.global_pool(x), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
