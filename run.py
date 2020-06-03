import random
import numpy as np
import pandas as pd
import torch

from data import generate_circle_classification_dataset, load_mnist_data, load_fashion_mnist_data
from experiments import run_mini_batch_size_experiment, run_convergence_region_experiment
from settings import settings


def main():
    # Reproducibility
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading datasets...")
    circle_train_input, circle_train_target, \
        circle_test_input, circle_test_target = generate_circle_classification_dataset()
    mnist_train_input, mnist_train_target, mnist_test_input, mnist_test_target = load_mnist_data()
    fashion_mnist_train_input, fashion_mnist_train_target, \
        fashion_mnist_test_input, fashion_mnist_test_target = load_fashion_mnist_data()
    print("Done!")

    results_dir = settings["results_dir"]
    plots_dir = settings["plots_dir"]

    # Boolean variable, whether to run the mini-batch size and learning rate experiment or was it already completed
    run_mini_batch_size_lr_experiments = True

    if run_mini_batch_size_lr_experiments:
        print("Running the mini-batch size and learning rate experiments...")
        circle_experiment_log, circle_training_logs = \
            run_mini_batch_size_experiment("circle", circle_train_input, circle_train_target,
                                           circle_test_input, circle_test_target)
        circle_experiment_log.to_csv(results_dir + "circle_mini_batch_size_lr_experiment_log.csv",
                                     sep=",", header=True, index=False, encoding="utf-8")
        circle_training_logs.to_csv(results_dir + "circle_mini_batch_size_lr_training_logs.csv",
                                    sep=",", header=True, index=False, encoding="utf-8")

        mnist_experiment_log, mnist_training_logs = \
            run_mini_batch_size_experiment("mnist", mnist_train_input, mnist_train_target,
                                           mnist_test_input, mnist_test_target)
        mnist_experiment_log.to_csv(results_dir + "mnist_mini_batch_size_lr_experiment_log.csv",
                                    sep=",", header=True, index=False, encoding="utf-8")
        mnist_training_logs.to_csv(results_dir + "mnist_mini_batch_size_lr_training_logs.csv",
                                   sep=",", header=True, index=False, encoding="utf-8")

        fashion_mnist_experiment_log, fashion_mnist_training_logs = \
            run_mini_batch_size_experiment("fashion_mnist", fashion_mnist_train_input, fashion_mnist_train_target,
                                           fashion_mnist_test_input, fashion_mnist_test_target)
        fashion_mnist_experiment_log.to_csv(results_dir + "fashion_mnist_mini_batch_size_lr_experiment_log.csv",
                                            sep=",", header=True, index=False, encoding="utf-8")
        fashion_mnist_training_logs.to_csv(results_dir + "fashion_mnist_mini_batch_size_lr_training_logs.csv",
                                           sep=",", header=True, index=False, encoding="utf-8")
        print("Done!")

    else:
        circle_experiment_log = pd.read_csv(results_dir + "circle_mini_batch_size_lr_experiment_log.csv",
                                            sep=",", header=0, index_col=None, encoding="utf-8")
        mnist_experiment_log = pd.read_csv(results_dir + "mnist_mini_batch_size_lr_experiment_log.csv",
                                           sep=",", header=0, index_col=None, encoding="utf-8")
        fashion_mnist_experiment_log = pd.read_csv(results_dir + "fashion_mnist_mini_batch_size_lr_experiment_log.csv",
                                                   sep=",", header=0, index_col=None, encoding="utf-8")

    def extract_optimal_parameters_from_experiment_log(experiment_log):
        optimal_values_indices = experiment_log[experiment_log["CONVERGED AT EPOCH"] < 100] \
            .groupby(["DATASET", "OPTIMIZER", "LOSS"])["TEST F1"].idxmax()
        optimal_values = experiment_log.loc[optimal_values_indices, ["MINI-BATCH SIZE", "LEARNING RATE"]]
        optimal_values.index = optimal_values_indices.index
        optimal_values = optimal_values.to_dict()
        best_mini_batch_sizes, best_lrs = optimal_values["MINI-BATCH SIZE"], optimal_values["LEARNING RATE"]
        return best_mini_batch_sizes, best_lrs

    print("Finding optimal mini-batch sizes and learning rates...")
    circle_best_mini_batch_sizes, circle_best_lrs = \
        extract_optimal_parameters_from_experiment_log(circle_experiment_log)
    mnist_best_mini_batch_sizes, mnist_best_lrs = \
        extract_optimal_parameters_from_experiment_log(mnist_experiment_log)
    fashion_mnist_best_mini_batch_sizes, fashion_mnist_best_lrs = \
        extract_optimal_parameters_from_experiment_log(fashion_mnist_experiment_log)
    print("Done!")

    # Boolean variable, whether to run the convergence region experiment or was it already completed
    run_convergence_region_experiments = True

    if run_convergence_region_experiments:
        print("Running the convergence region experiments...")
        circle_experiment_log = run_convergence_region_experiment("circle", circle_train_input, circle_train_target,
                                                                  circle_best_mini_batch_sizes, circle_best_lrs)
        circle_experiment_log.to_csv(results_dir + "circle_convergence_region_experiment_log.csv",
                                     sep=",", header=True, index=False, encoding="utf-8")

        mnist_experiment_log = run_convergence_region_experiment("mnist", mnist_train_input, mnist_train_target,
                                                                 mnist_best_mini_batch_sizes, mnist_best_lrs)
        mnist_experiment_log.to_csv(results_dir + "mnist_convergence_region_experiment_log.csv",
                                    sep=",", header=True, index=False, encoding="utf-8")

        fashion_mnist_experiment_log = \
            run_convergence_region_experiment("fashion_mnist", fashion_mnist_train_input, fashion_mnist_train_target,
                                              fashion_mnist_best_mini_batch_sizes, fashion_mnist_best_lrs)
        fashion_mnist_experiment_log.to_csv(results_dir + "fashion_mnist_convergence_region_experiment_log.csv",
                                            sep=",", header=True, index=False, encoding="utf-8")
    else:
        circle_experiment_log = pd.read_csv(results_dir + "circle_convergence_region_experiment_log.csv",
                                            sep=",", header=0, index_col=None, encoding="utf-8")
        mnist_experiment_log = pd.read_csv(results_dir + "mnist_convergence_region_experiment_log.csv",
                                           sep=",", header=0, index_col=None, encoding="utf-8")
        fashion_mnist_experiment_log = pd.read_csv(results_dir + "fashion_mnist__convergence_region_experiment_log.csv",
                                                   sep=",", header=0, index_col=None, encoding="utf-8")


if __name__ == "__main__":
    main()
