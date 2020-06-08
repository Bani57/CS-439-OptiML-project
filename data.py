import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from config import config

seed = config["seed"]
data_dir = config["data_dir"]
reduced_data_size = True


def standardize_data(train_input, test_input):
    mean, std = train_input.mean(), train_input.std()

    return (train_input - mean) / std, (test_input - mean) / std


def reduce_data_size(input_data, target_data, reduced_num_samples=1000):
    reduced_input_data, _, reduced_target_data, _ = train_test_split(input_data, target_data,
                                                                     train_size=reduced_num_samples,
                                                                     random_state=seed,
                                                                     shuffle=True,
                                                                     stratify=target_data)
    return reduced_input_data, reduced_target_data


def generate_circle_classification_dataset(num_samples=1000, num_features=2):
    circle_radius = np.sqrt(2 / np.pi)
    input_data = torch.empty(num_samples * 2, num_features).uniform_(-1, 1)
    target_data = (input_data.pow(2).sum(dim=1) < circle_radius ** 2).long()

    circle_train_input, circle_train_target = input_data[:num_samples], target_data[:num_samples]
    circle_test_input, circle_test_target = input_data[num_samples:], target_data[num_samples:]
    return circle_train_input, circle_train_target, circle_test_input, circle_test_target


def load_mnist_data():
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
