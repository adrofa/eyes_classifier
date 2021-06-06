import torch
import torch.nn as nn
from collections import OrderedDict


def get_model(version, weights=None):
    if version == 1:
        model = CustomNetV1()
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu"))
    elif version == 2:
        model = CustomNetV2()
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        raise Exception(f"Model version '{version}' is unknown!")
    return model


class CustomNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=5, stride=1, padding=0)),
            ("relu1", nn.ReLU()),
            ("maxPool1", nn.MaxPool2d(3, 1)),

            ("conv2", nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=0)),
            ("relu2", nn.ReLU()),
            ("maxPool2", nn.MaxPool2d(3, 1)),
        ]))
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(32, 32)),
            ("relu1", nn.ReLU()),

            ("fc2", nn.Linear(32, 16)),
            ("relu2", nn.ReLU()),

            ("classifier", nn.Linear(16, 1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.squeeze()
        return x


class CustomNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=5, stride=1, padding=0)),
            ("relu1", nn.ReLU()),
            ("maxPool1", nn.MaxPool2d(3, 1)),

            ("batchNorm2", nn.BatchNorm2d(64, affine=True)),
            ("conv2", nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=0)),
            ("relu2", nn.ReLU()),
            ("maxPool2", nn.MaxPool2d(3, 1)),
        ]))
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(32, 32)),
            ("relu1", nn.ReLU()),

            ("fc2", nn.Linear(32, 16)),
            ("relu2", nn.ReLU()),

            ("classifier", nn.Linear(16, 1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.squeeze()
        return x
