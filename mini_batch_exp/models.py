import torch
from torch import nn
from torch.nn import functional as F

class MnistClassifier(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxp1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.maxp2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)
    def forward(self, x):
        y = self.bn1(F.relu(self.maxp1(self.conv1(x))))
        y = self.bn2(F.relu(self.maxp2(self.conv2(y))))
        y = self.bn3(F.relu(self.conv3(y).view(-1, 64)))
        y = self.bn4(F.relu(self.fc1(y)))
        y = self.fc2(y)
        return y