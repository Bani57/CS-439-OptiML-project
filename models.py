from torch import nn
from torch.nn import functional as F


def create_circle_classifier():
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid())


class MnistClassifier(nn.Module):
    def __init__(self, mini_batch_size):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxp1 = nn.MaxPool2d(kernel_size=2)
        if mini_batch_size > 1:
            self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.maxp2 = nn.MaxPool2d(kernel_size=2)
        if mini_batch_size > 1:
            self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        if mini_batch_size > 1:
            self.bn3 = nn.BatchNorm1d(num_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=256)
        if mini_batch_size > 1:
            self.bn4 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        mini_batch_size = x.size(0)
        y = F.relu(self.maxp1(self.conv1(x)))
        if mini_batch_size > 1:
            y = self.bn1(y)
        y = F.relu(self.maxp2(self.conv2(y)))
        if mini_batch_size > 1:
            y = self.bn2(y)
        y = F.relu(self.conv3(y).view(-1, 64))
        if mini_batch_size > 1:
            y = self.bn3(y)
        y = F.relu(self.fc1(y))
        if mini_batch_size > 1:
            y = self.bn4(y)
        y = F.relu(self.fc2(y))
        return y
