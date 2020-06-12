""" Module containing the main code for executing the experiments """

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment_utils import train_model, test_model, \
    estimate_convergence_threshold_phlug_diagnostic, estimate_convergence_region
from plotting import visualize_convergence_region
from settings import settings

gpu_available = torch.cuda.is_available()


def run_mini_batch_size_experiment(dataset_name, train_input, train_target, test_input, test_target):
    """
    Function to run the first experiment procedure

    :param dataset_name: name of the dataset used for training, string
    :param train_input: training input data, torch.Tensor of size (1000, `num_features`)
    :param train_target: training target data, torch.Tensor of size 1000
    :param test_input: testing input data, torch.Tensor of size (1000, `num_features`)
    :param test_target: testing target data, torch.Tensor of size 1000

    :returns: experiment log, pandas.Dataframe
              + detailed training logs, pandas.Dataframe
    """

    if gpu_available:
        # If a GPU and CUDA are available, transfer the test data to it, for efficient computation
        test_input, test_target = test_input.cuda(), test_target.cuda()

    # For MSE loss the test targets have to be one-hot encoded vectors
    test_target_one_hot = F.one_hot(test_target, num_classes=2 if dataset_name == "circle" else 10).float()

    possible_optimizers = ["sgd", "adam", "sgd_to_half"]
    possible_loss_functions = ["mse", "cross_entropy"]
    num_train_samples = train_input.size(0)
    max_mini_batch_size = np.ceil(num_train_samples * 0.9).astype(int)
    possible_mini_batch_sizes = np.unique(np.ceil(np.geomspace(1, max_mini_batch_size, 50)).astype(int))
    possible_learning_rates = np.geomspace(0.01, 1, 10)
    optimizer_lr_factor = {"sgd": 1, "adam": 0.05, "sgd_to_half": 5}
    loss_lr_factor = {"mse": 5, "cross_entropy": 1}
    possible_experiment_conditions = [(dataset_name, optimizer_algorithm, loss_function, mini_batch_size,
                                       lr * optimizer_lr_factor[optimizer_algorithm] * loss_lr_factor[loss_function])
                                      for optimizer_algorithm in possible_optimizers
                                      for loss_function in possible_loss_functions
                                      for mini_batch_size in possible_mini_batch_sizes
                                      for lr in possible_learning_rates]

    training_logs = []
    experiment_log = []
    for experiment_condition in possible_experiment_conditions:
        dataset_name, optimizer_algorithm, loss_function, mini_batch_size, lr = experiment_condition
        trained_model, training_log, _, param_grad_data, converged_at_iteration = \
            train_model(dataset_name, train_input, train_target,
                        optimizer_algorithm=optimizer_algorithm, lr=lr,
                        loss_function=loss_function, mini_batch_size=mini_batch_size)

        training_loss, training_accuracy, training_f1, \
            validation_loss, validation_accuracy, validation_f1, total_training_time = training_log.iloc[-1, 6:]

        if loss_function == "mse":
            test_loss, test_accuracy, test_f1 = \
                test_model(trained_model, test_input, test_target_one_hot, loss_function)
        else:
            test_loss, test_accuracy, test_f1 = \
                test_model(trained_model, test_input, test_target, loss_function)

        if optimizer_algorithm != "sgd_to_half":
            converged_at_iteration, _ = estimate_convergence_threshold_phlug_diagnostic(param_grad_data)
        mini_batches_per_epoch = np.ceil(max_mini_batch_size / mini_batch_size)
        converged_at_epoch = converged_at_iteration / mini_batches_per_epoch

        training_logs.append(training_log)
        experiment_log.append((dataset_name, optimizer_algorithm, loss_function, mini_batch_size, lr,
                               training_loss, training_accuracy, training_f1,
                               validation_loss, validation_accuracy, validation_f1, total_training_time,
                               test_loss, test_accuracy, test_f1, converged_at_epoch))
        print("Done with {}!".format(experiment_condition))

    training_logs = pd.concat(training_logs, axis=0)
    experiment_log = pd.DataFrame(experiment_log,
                                  columns=["DATASET", "OPTIMIZER", "LOSS", "MINI-BATCH SIZE", "LEARNING RATE",
                                           "TRAINING LOSS", "TRAINING ACCURACY", "TRAINING F1",
                                           "VALIDATION LOSS", "VALIDATION ACCURACY", "VALIDATION F1",
                                           "TOTAL TRAINING TIME", "TEST LOSS", "TEST ACCURACY", "TEST F1",
                                           "CONVERGED AT EPOCH"])
    return experiment_log, training_logs


def run_convergence_region_experiment(dataset_name, train_input, train_target,
                                      mini_batch_sizes, lrs, num_simulations=1000):
    """
    Function to run the second experiment procedure

    :param dataset_name: name of the dataset used for training, string
    :param train_input: training input data, torch.Tensor of size (1000, `num_features`)
    :param train_target: training target data, torch.Tensor of size 1000
    :param mini_batch_sizes: dictionary {(dataset, optimizer, loss): optimal mini-batch size}
    :param lrs: dictionary {(dataset, optimizer, loss): optimal learning rate}
    :param num_simulations: number of training simulations to run, positive int, optional, default value is 1000

    :returns: experiment log, pandas.Dataframe
    """

    possible_optimizers = ["sgd", "adam", "sgd_to_half"]
    possible_loss_functions = ["mse", "cross_entropy"]
    possible_experiment_conditions = [(dataset_name, optimizer_algorithm, loss_function)
                                      for optimizer_algorithm in possible_optimizers
                                      for loss_function in possible_loss_functions]

    experiment_log = []
    for experiment_condition in possible_experiment_conditions:
        dataset_name, optimizer_algorithm, loss_function = experiment_condition
        if experiment_condition in mini_batch_sizes and experiment_condition in lrs:
            mini_batch_size = mini_batch_sizes[(dataset_name, optimizer_algorithm, loss_function)]
            lr = lrs[(dataset_name, optimizer_algorithm, loss_function)]
        else:
            continue

        simulations_param_value_data = []
        simulations_diagnostic_data = []
        for i in range(num_simulations):
            _, _, param_value_data, param_grad_data, _ = \
                train_model(dataset_name, train_input, train_target,
                            optimizer_algorithm=optimizer_algorithm, lr=lr,
                            loss_function=loss_function, mini_batch_size=mini_batch_size)
            _, diagnostic_data = estimate_convergence_threshold_phlug_diagnostic(param_grad_data, burn_in=1)
            simulations_param_value_data.append(param_value_data)
            simulations_diagnostic_data.append(diagnostic_data)
            if (i + 1) % settings["verbosity_mod"] == 0 or i in (0, num_simulations - 1):
                print("Done with {}% of simulations!".format(round(100 * (i + 1) / num_simulations, 4)))
        simulations_param_value_data = np.vstack(simulations_param_value_data)
        simulations_diagnostic_data = np.concatenate(simulations_diagnostic_data)

        convergence_region_params = estimate_convergence_region(simulations_param_value_data,
                                                                simulations_diagnostic_data)
        conv_ell_x, conv_ell_y, conv_ell_dx, conv_ell_dy = convergence_region_params
        experiment_log.append((dataset_name, optimizer_algorithm, loss_function,
                               conv_ell_x, conv_ell_y, conv_ell_dx, conv_ell_dy))

        plot_filename = "convergence_region_{}_{}_{}.png".format(dataset_name, optimizer_algorithm, loss_function)
        visualize_convergence_region(simulations_param_value_data, simulations_diagnostic_data,
                                     convergence_region_params,
                                     plot_path=settings["plots_dir"] + "convergence_regions/" + plot_filename)

        print("Done with {}!".format(experiment_condition))

    experiment_log = pd.DataFrame(experiment_log, columns=["DATASET", "OPTIMIZER", "LOSS",
                                                           "ELLIPSE CENTER X", "ELLIPSE CENTER Y",
                                                           "ELLIPSE DIAMETER X", "ELLIPSE DIAMETER Y"])
    return experiment_log
