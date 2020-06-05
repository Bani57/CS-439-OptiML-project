import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from models import create_circle_classifier, MnistClassifier
from settings import settings

gpu_available = torch.cuda.is_available()
seed = settings["seed"]


def model_params_and_loss_gradients_to_flat_vector(model_params):
    params_vector = torch.Tensor()
    params_grad_vector = torch.Tensor()
    for param in model_params:
        params_layer_vector = param.data.view(-1)
        params_layer_grad_vector = param.grad.view(-1)
        if gpu_available:
            # If a GPU was available the model was transferred to it for training,
            # but the model parameter values and gradients are required to be on the CPU, so they need to be moved
            params_layer_vector, params_layer_grad_vector = params_layer_vector.cpu(), params_layer_grad_vector.cpu()
        params_vector = torch.cat((params_vector, params_layer_vector))
        params_grad_vector = torch.cat((params_grad_vector, params_layer_grad_vector))

    if params_vector.size(0) > 162:
        # The MNIST deep model has much more parameters than the circle classifier (78000+ vs. 162),
        # so the comparison to be equal and to reduce memory complexity,
        # only keep the values and gradients for the same number of parameters
        params_vector = params_vector[:162]
        params_grad_vector = params_grad_vector[:162]
    return np.array(params_vector), np.array(params_grad_vector)


class SgdToHalf(torch.optim.Optimizer):
    def __init__(self, params, lr, burn_in=1):
        self.params = params
        self.lr = lr
        self.burn_in = burn_in
        self.n = 1
        self.s = [0, ]
        self.tau = 0
        self.prev_grad = None
        defaults = {"lr": lr, "burn_in": burn_in, "n": 1, "s": [0, ], "tau": 0, "prev_grad": None}
        super(SgdToHalf, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SgdToHalf, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        burn_in = group["burn_in"]
        n = group["n"]
        s = float(group["s"][-1])
        tau = group["tau"]

        params = group["params"]
        for p in params:
            if p.grad is None:
                continue
            g_p = p.grad
            p.add_(g_p, alpha=-lr)

        _, g = model_params_and_loss_gradients_to_flat_vector(params)
        if n == 1:
            self.prev_grad = group["prev_grad"] = torch.clone(g).detach()
        else:
            g_prev = group["prev_grad"]
            s += torch.dot(g.view(-1), g_prev.view(-1))
            if n > (tau + burn_in) and s < 0:
                self.tau = group["tau"] = n
                s = 0
                self.lr = group["lr"] = lr / 2
            self.s.append(s)
            group["s"].append(s)
            self.prev_grad = group["prev_grad"] = torch.clone(g).detach()

        self.n = group["n"] = n + 1

        return loss

    def converged(self):
        return self.n > (self.tau + self.burn_in) and self.s[-1] < 0 and self.lr < 1e-10


def train_model(dataset_name, train_input, train_target,
                num_epochs=100, lr=1e-1, mini_batch_size=100,
                optimizer_algorithm="sgd", loss_function="mse", verbose=False):
    if dataset_name == "circle":
        model = create_circle_classifier()
    elif dataset_name in ("mnist", "fashion_mnist"):
        model = MnistClassifier(mini_batch_size=mini_batch_size)
    else:
        raise ValueError("Invalid value for the dataset name! " +
                         "Supported values are `circle`, `mnist` and `fashion_mnist`.")

    if optimizer_algorithm == "sgd":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_algorithm == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_algorithm == "sgd_to_half":
        optimizer = SgdToHalf(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid value for the optimization algorithm name! " +
                         "Supported values are `sgd`, `adam` and `sgd_to_half`.")

    if loss_function == "mse":
        criterion = nn.MSELoss()
        # For MSE loss the targets have to be one-hot encoded vectors
        train_target = F.one_hot(train_target, num_classes=2 if dataset_name == "circle" else 10).float()
    elif loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid value for the loss function name! " +
                         "Supported values are `mse` and `cross_entropy`.")

    # Separate 10% of the training data to be used for validation
    train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,
                                                                        test_size=0.1,
                                                                        random_state=seed, shuffle=True,
                                                                        stratify=train_target)

    if gpu_available:
        # If a GPU and CUDA are available, transfer the data, model and criterion to it, for efficient computation
        # If more than one GPU is available, wrap the model in the nn.DataParallel context,
        # in order to utilize all of them for training
        train_input, train_target = train_input.cuda(), train_target.cuda()
        val_input, val_target = val_input.cuda(), val_target.cuda()
        model = model.cuda()
        if settings["num_gpus"] > 1 and settings["split_task"]:
            model = nn.DataParallel(model)
        criterion = criterion.cuda()

    num_samples = train_input.size(0)
    param_values_series = []
    loss_gradients_series = []
    converged_early = False
    tic = time.time()
    training_log = []
    for e in range(num_epochs):
        # Epoch loop
        if converged_early:
            break
        sum_loss = 0
        mini_batch_count = 0

        for b in range(0, num_samples, mini_batch_size):
            # Mini-batch loop
            train_input_mini_batch = train_input[b:min(b + mini_batch_size, num_samples)]
            train_target_mini_batch = train_target[b:min(b + mini_batch_size, num_samples)]
            optimizer.zero_grad()
            prediction_mini_batch = model(train_input_mini_batch)
            loss = criterion(prediction_mini_batch, train_target_mini_batch)
            sum_loss += loss.item()
            loss.backward()
            with torch.no_grad():
                param_values, loss_gradients = model_params_and_loss_gradients_to_flat_vector(model.parameters())
                param_values_series.append(param_values)
                loss_gradients_series.append(loss_gradients)
            optimizer.step()
            torch.cuda.empty_cache()
            mini_batch_count += 1
            if optimizer_algorithm == "sgd_to_half" and optimizer.converged():
                converged_early = True
                break

        # Calculate current model evaluation metrics
        average_training_loss = sum_loss / mini_batch_count
        _, train_accuracy, train_f1 = test_model(model, train_input, train_target, loss_function)

        validation_loss, validation_accuracy, validation_f1 = test_model(model, val_input, val_target, loss_function)

        toc = time.time()
        elapsed_time = toc - tic

        training_log.append((dataset_name, optimizer_algorithm, loss_function, mini_batch_size, lr, e + 1,
                             average_training_loss, train_accuracy, train_f1,
                             validation_loss, validation_accuracy, validation_f1, elapsed_time))

        if verbose and ((e + 1) % settings["verbosity_mod"] == 0 or e in (0, num_epochs - 1)):
            print("Epoch {}, Loss: {}, Elapsed Time: {} sec".format(e + 1, average_training_loss, elapsed_time))

    training_log = pd.DataFrame(training_log,
                                columns=["DATASET", "OPTIMIZER", "LOSS", "MINI-BATCH SIZE", "LEARNING RATE", "EPOCH",
                                         "AVERAGE TRAINING LOSS", "TRAINING ACCURACY", "TRAINING F1",
                                         "VALIDATION LOSS", "VALIDATION ACCURACY", "VALIDATION F1", "ELAPSED TIME"])
    param_values_series = np.stack(param_values_series, axis=0)
    loss_gradients_series = np.stack(loss_gradients_series, axis=0)
    if optimizer_algorithm == "sgd_to_half":
        if verbose:
            print("Final convergence reached at iteration:", optimizer.tau)
            print("Final learning rate value:", optimizer.lr)
        return model, training_log, param_values_series, loss_gradients_series, optimizer.tau
    else:
        return model, training_log, param_values_series, loss_gradients_series, None


def test_model(model, test_input, test_target, loss_function):
    prediction = model(test_input)
    if loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction, test_target).item()
    predicted_labels = torch.argmax(prediction, dim=1).cpu()
    if len(test_target.size()) > 1:
        # If target data is in one-hot vector form, revert it back to class labels
        test_target = torch.argmax(test_target, dim=1)
    test_target = test_target.cpu()
    accuracy = accuracy_score(test_target, predicted_labels)
    f1 = f1_score(test_target, predicted_labels, average="weighted")
    return loss, accuracy, f1


def estimate_convergence_threshold_phlug_diagnostic(loss_gradients_series, burn_in=1):
    num_iterations_performed = loss_gradients_series.shape[0]
    s_series = [0, ]
    convergence_threshold_set = False
    convergence_threshold = num_iterations_performed
    for n in range(1, num_iterations_performed):
        s_n = s_series[n - 1] + np.dot(loss_gradients_series[n], loss_gradients_series[n - 1])
        s_series.append(s_n)
        if not convergence_threshold_set and n > burn_in and s_n < 0:
            convergence_threshold = n
            convergence_threshold_set = True
    s_series = np.array(s_series)
    return convergence_threshold, s_series


def estimate_convergence_region(simulations_param_value_data, simulations_diagnostic_data):
    simulations_param_value_data_two_params = simulations_param_value_data[:, :2]
    simulations_diagnostic_converged_data = (simulations_diagnostic_data < 0).reshape(-1, 1)
    data = np.hstack((simulations_param_value_data_two_params, simulations_diagnostic_converged_data))
    data = pd.DataFrame(data, columns=["param1", "param2", "converged"])

    conv_rect_x1, conv_rect_y1 = data.groupby("converged").quantile(0.025).loc[1, :]
    conv_rect_x2, conv_rect_y2 = data.groupby("converged").quantile(0.975).loc[1, :]
    conv_ell_x, conv_ell_y = data.groupby("converged").mean().loc[1, :]
    conv_ell_dx = conv_rect_x2 - conv_rect_x1
    conv_ell_dy = conv_rect_y2 - conv_rect_y1
    return conv_ell_x, conv_ell_y, conv_ell_dx, conv_ell_dy
