import torch
import torch.nn as nn
from collections import OrderedDict


def get_model(version, weights=None):
    if version == 1:
        model = CustomNetV1()
    elif version == 2:
        model = CustomNetV2()
    elif version == 3:
        model = CustomNetV3()
    elif version == 4:
        model = CustomNetV4()
    else:
        raise Exception(f"Model version '{version}' is unknown!")
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
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


class CustomNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=5, stride=1, padding=0)),
            ("relu1", nn.ReLU()),
            ("maxPool1", nn.MaxPool2d(3, 1)),

            ("batchNorm2", nn.BatchNorm2d(128, affine=True)),
            ("conv2", nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=0)),
            ("relu2", nn.ReLU()),
            ("maxPool2", nn.MaxPool2d(3, 1)),

            ("batchNorm3", nn.BatchNorm2d(64, affine=True)),
            ("conv3", nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=0)),
            ("relu3", nn.ReLU()),
            ("maxPool3", nn.MaxPool2d(3, 1)),
        ]))
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(32, 32)),
            ("relu1", nn.ReLU()),

            ("fc2", nn.Linear(32, 32)),
            ("relu2", nn.ReLU()),

            ("fc3", nn.Linear(32, 16)),
            ("relu3", nn.ReLU()),

            ("classifier", nn.Linear(16, 1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.squeeze()
        return x


class CustomNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            # Conv-1
            ("conv1", nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=5, stride=1, padding=2)),
            ("relu1", nn.ReLU()),
            ("maxPool1", nn.MaxPool2d(3, 1)),

            # Conv-2
            ("batchNorm2", nn.BatchNorm2d(128, affine=True)),
            ("conv2", nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, stride=1, padding=1)),
            ("relu2", nn.ReLU()),
            ("maxPool2", nn.MaxPool2d(3, 1)),

            # Conv-3
            ("batchNorm3", nn.BatchNorm2d(128, affine=True)),
            ("conv3", nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1)),
            ("relu3", nn.ReLU()),
            ("maxPool3", nn.MaxPool2d(3, 1)),

            # Conv-4
            ("batchNorm4", nn.BatchNorm2d(64, affine=True)),
            ("conv4", nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=0)),
            ("relu4", nn.ReLU()),
            ("maxPool4", nn.MaxPool2d(3, 1)),
        ]))
        self.global_pool = nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.head = nn.Sequential(OrderedDict([
            # FC-1
            ("fc1", nn.Linear(64, 32)),
            ("relu1", nn.ReLU()),

            # FC-2
            ("fc2", nn.Linear(32, 32)),
            ("relu2", nn.ReLU()),

            # FC-3
            ("fc3", nn.Linear(32, 32)),
            ("relu3", nn.ReLU()),

            # FC-4
            ("fc4", nn.Linear(32, 16)),
            ("relu4", nn.ReLU()),

            # Classifier
            ("classifier", nn.Linear(16, 1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.squeeze()
        return x
