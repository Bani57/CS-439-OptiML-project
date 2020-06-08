import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchvision import datasets
import time
import os
from config import config
from mini_batch_exp import run_mini_batch_experiment

def print_settings(params):
    for k, v in params.items():
        print(f"{k} = {v}")

def main():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(f"cuda:{config['gpu_id']}")

    if not config['hide_settings']:
        print('Settings:')
        print_settings(config)
        print()

    mnist_dir = os.path.join(config['data_dir'], 'mnist')

    mnist_train_raw = datasets.MNIST(mnist_dir, train=True, download=True)
    mnist_test_raw = datasets.MNIST(mnist_dir, train=False, download=True)

    mnist_train_input = mnist_train_raw.data.view(-1, 1, 28, 28).float()
    mnist_train_target = mnist_train_raw.targets
    mnist_test_input = mnist_test_raw.data.view(-1, 1, 28, 28).float()
    mnist_test_target = mnist_test_raw.targets

    mnist_train = (mnist_train_input, mnist_train_target)
    mnist_test = (mnist_test_input, mnist_test_target) 
    run_mini_batch_experiment(mnist_train, mnist_test)

if __name__ == '__main__':
    main()
