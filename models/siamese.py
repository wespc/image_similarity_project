import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSiameseNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(SimpleSiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Dynamically determine the output size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 28, 28) if in_channels == 1 else torch.zeros(1, in_channels, 32, 32)
            out = self.cnn(dummy_input)
            flatten_size = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)  # L2 normalization
        return output

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class EnhancedSiameseNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(EnhancedSiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Dynamically determine the output size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 28, 28) if in_channels == 1 else torch.zeros(1, in_channels, 32, 32)
            out = self.cnn(dummy_input)
            flatten_size = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)  # L2 normalization
        return output

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)