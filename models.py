""" Module containing code required to instantiate the shallow and deep model """

from torch import nn
from torch.nn import functional as F


def create_circle_classifier():
    """
    Function to instantiate a three-layer MLP with 32 hidden units, to serve as the 2D circle classifier

    :returns: circle classifier, torch.nn.Sequential object
    """

    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid())


class MnistClassifier(nn.Module):
    """
    Class for the MNIST and FashionMNIST classifier, inherits torch.nn.Module
    Network architecture:
        1. 2D convolution of the input with a kernel size of 3x3 into 16 output channels
        2. 2D max-pooling of the convolution output with a kernel size of 2x2
        3. ReLU activation
        4. Batch normalization
        5. 2D convolution with a kernel size of 4x4 into 32 output channels
        6. 2D max-pooling of the convolution output with a kernel size of 2x2
        7. ReLU activation
        8. Batch normalization
        9. 2D convolution with a kernel size of 5x5 into 64 output channels
        10. ReLU activation
        11. Batch normalization
        12. Fully connected layer with 256 output units
        13. ReLU activation
        14. Batch normalization
        15. Fully connected layer with 10 output units
        16. ReLU activation
    """

    def __init__(self, mini_batch_size):
        nn.Module.__init__(self)
        self.mini_batch_size = mini_batch_size

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
        y = F.relu(self.maxp1(self.conv1(x)))
        if self.mini_batch_size > 1:
            y = self.bn1(y)
        y = F.relu(self.maxp2(self.conv2(y)))
        if self.mini_batch_size > 1:
            y = self.bn2(y)
        y = F.relu(self.conv3(y).view(-1, 64))
        if self.mini_batch_size > 1:
            y = self.bn3(y)
        y = F.relu(self.fc1(y))
        if self.mini_batch_size > 1:
            y = self.bn4(y)
        y = F.relu(self.fc2(y))
        return y
