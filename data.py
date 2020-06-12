""" Module containing the code for loading the datasets """

import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from settings import settings

seed = settings["seed"]
data_dir = settings["data_dir"]
reduced_data_size = True


def standardize_data(train_input, test_input):
    """
    Helper function to normalize input feature data using standard scaling
    Statistics are only computed from the training data

    :param train_input: training input data, torch.Tensor
    :param test_input: testing input data, torch.Tensor

    :returns: tuple of torch.Tensors, (normalized training input data, normalized testing input data)
    """

    mean, std = train_input.mean(), train_input.std()

    return (train_input - mean) / std, (test_input - mean) / std


def reduce_data_size(input_data, target_data, reduced_num_samples=1000):
    """
    Helper function to down-sample input and target data in order to reduce computational complexity
    Stratified sampling is performed in order to preserve the original class distribution

    :param input_data: input data, torch.Tensor of size (`num_samples`, `num_features`)
    :param target_data: target data, torch.Tensor of size `num_samples`
    :param reduced_num_samples: how many samples to preserve, positive int, optional, default value is 1000

    :returns: tuple of torch.Tensors with `reduced_num_samples` rows, (reduced input data, reduced target data)
    """

    reduced_input_data, _, reduced_target_data, _ = train_test_split(input_data, target_data,
                                                                     train_size=reduced_num_samples,
                                                                     random_state=seed,
                                                                     shuffle=True,
                                                                     stratify=target_data)
    return reduced_input_data, reduced_target_data


def generate_circle_classification_dataset(num_samples=1000, num_features=2):
    """
    Function to generate a simple circle classification dataset
    Uniformly-sampled points from [-1, 1]^2 are labeled as positive if they lie within a circle of radius sqrt(2 / pi)

    :param num_samples: number of samples to generate, positive int, optional, default value is 1000
    :param num_features: number of features to generate, positive int, optional, default value is 2

    :returns: tuple of 4 torch.Tensors, (train input, train targets, test input, test targets)
    """

    circle_radius = np.sqrt(2 / np.pi)
    input_data = torch.empty(num_samples * 2, num_features).uniform_(-1, 1)
    target_data = (input_data.pow(2).sum(dim=1) < circle_radius ** 2).long()

    circle_train_input, circle_train_target = input_data[:num_samples], target_data[:num_samples]
    circle_test_input, circle_test_target = input_data[num_samples:], target_data[num_samples:]
    return circle_train_input, circle_train_target, circle_test_input, circle_test_target


def load_mnist_data():
    """
    Function to load the MNIST classification dataset
    Samples are 28x28 grayscale images of handwritten records of the 10 digits 0-9
    The original number of training and testing samples is reduced to 1000 and the data is normalized

    :returns: tuple of 4 torch.Tensors, (train input, train targets, test input, test targets)
    """

    mnist_train_set = datasets.MNIST(data_dir + 'mnist/', train=True, download=True)
    mnist_test_set = datasets.MNIST(data_dir + 'mnist/', train=False, download=True)
    mnist_train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
    mnist_train_target = mnist_train_set.targets
    mnist_test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
    mnist_test_target = mnist_test_set.targets

    # Reduce the number of samples for computational efficiency
    mnist_train_input, mnist_train_target = reduce_data_size(mnist_train_input, mnist_train_target)
    mnist_test_input, mnist_test_target = reduce_data_size(mnist_test_input, mnist_test_target)

    # Standardize the data
    mnist_train_input, mnist_test_input = standardize_data(mnist_train_input, mnist_test_input)

    return mnist_train_input, mnist_train_target, mnist_test_input, mnist_test_target


def load_fashion_mnist_data():
    """
    Function to load the FashionMNIST classification dataset
    Samples are 28x28 grayscale images of 10 types of clothing items
    The original number of training and testing samples is reduced to 1000 and the data is normalized

    :returns: tuple of 4 torch.Tensors, (train input, train targets, test input, test targets)
    """

    fashion_mnist_train_set = datasets.FashionMNIST(data_dir + 'fashion_mnist/', train=True, download=True)
    fashion_mnist_test_set = datasets.FashionMNIST(data_dir + 'fashion_mnist/', train=False, download=True)
    fashion_mnist_train_input = fashion_mnist_train_set.data.view(-1, 1, 28, 28).float()
    fashion_mnist_train_target = fashion_mnist_train_set.targets
    fashion_mnist_test_input = fashion_mnist_test_set.data.view(-1, 1, 28, 28).float()
    fashion_mnist_test_target = fashion_mnist_test_set.targets

    # Reduce the number of samples for computational efficiency
    fashion_mnist_train_input, fashion_mnist_train_target = reduce_data_size(fashion_mnist_train_input,
                                                                             fashion_mnist_train_target)
    fashion_mnist_test_input, fashion_mnist_test_target = reduce_data_size(fashion_mnist_test_input,
                                                                           fashion_mnist_test_target)

    # Standardize the data
    fashion_mnist_train_input, fashion_mnist_test_input = standardize_data(fashion_mnist_train_input,
                                                                           fashion_mnist_test_input)

    return fashion_mnist_train_input, fashion_mnist_train_target, fashion_mnist_test_input, fashion_mnist_test_target
