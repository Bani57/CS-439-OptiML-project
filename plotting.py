""" Module containing functions for generating the visualizations of the experiment results """

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from settings import settings

sns.set(style="darkgrid", context="paper")
plots_dir = settings["plots_dir"]
colormap = sns.color_palette("Set1", n_colors=3)
colormap[0], colormap[1] = colormap[1], colormap[0]


def visualize_convergence_region(simulations_param_data, simulations_diagnostic_data,
                                 convergence_region_params, plot_path):
    """
    Function to generate the plot of a convergence region using the data from the second experiment

    :param simulations_param_data: parameter value data from 1000 simulations, torch.Tensor
    :param simulations_diagnostic_data: diagnostic value data from 1000 simulations, torch.Tensor
    :param convergence_region_params: ellipse parameters, tuple of 4 floats (x-coordinate of center,
                                                                             y-coordinate of center,
                                                                             horizontal diameter length,
                                                                             vertical diameter length)
    :param plot_path: local filepath where to save the plot, string
    """

    simulations_diagnostic_converged_data = simulations_diagnostic_data < 0
    conv_ell_x, conv_ell_y, conv_ell_dx, conv_ell_dy = convergence_region_params

    plt.figure(figsize=(10, 5))
    conv_ell = Ellipse(xy=(conv_ell_x, conv_ell_y), width=conv_ell_dx, height=conv_ell_dy,
                       fill=True, facecolor="darkblue", alpha=0.25)
    sns.scatterplot(x=simulations_param_data[:, 0], y=simulations_param_data[:, 1], s=50,
                    hue=simulations_diagnostic_converged_data, edgecolor="none",
                    palette=sns.color_palette("Greys", n_colors=2)).add_patch(conv_ell)
    plt.legend(loc="upper right", title="Diagnostic converged", title_fontsize=12, fontsize=10)
    plt.xlabel("Parameter 1 value", fontsize=14)
    plt.ylabel("Parameter 2 value", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(fname=plot_path, dpi="figure", format="png")


verbose_labels_map = {"TEST F1": "Test F1", "TOTAL TRAINING TIME": "Total training time (sec)",
                      "COMBINED SCORE": "Combined score", "CONVERGED AT EPOCH": "Epoch at which training converged",
                      "MINI-BATCH SIZE": "Mini-batch size", "LEARNING RATE": "Learning rate",
                      "DATASET": "Dataset", "OPTIMIZER": "Optimization algorithm", "LOSS": "Loss function",
                      "circle": "2D Circle", "mnist": "MNIST", "fashion_mnist": "FashionMNIST",
                      "sgd": "SGD", "adam": "Adam", "sgd_to_half": "SGD^(1/2)",
                      "mse": "MSE", "cross_entropy": "Cross-entropy", "MEASURE": "Measure",
                      "TRAINING ACCURACY": "Training accuracy", "VALIDATION ACCURACY": "Validation accuracy"}


def visualize_results_first_experiment(experiment_log, training_logs):
    """
    Function to generate the plots using the experiment results from the first experiment

    :param experiment_log: first experiment log, pandas.Dataframe
    :param training_logs: first experiment training log, pandas.Dataframe
    """

    target_variables = ["TEST F1", "TOTAL TRAINING TIME", "COMBINED SCORE"]
    optimizers = ["sgd", "adam", "sgd_to_half"]

    # Visualize influence of mini-batch size on prediction performance (test F1) and training time
    fig, ax = plt.subplots(len(target_variables), len(optimizers), sharex="all", sharey="row", figsize=(50, 21))
    for i, target_variable in enumerate(target_variables):
        for j, optimizer_algorithm in enumerate(optimizers):
            plot_data = experiment_log[experiment_log["OPTIMIZER"] == optimizer_algorithm]
            plot_data = plot_data.replace(verbose_labels_map).rename(columns=verbose_labels_map)

            if i == 0:
                ax[i, j].set_title(verbose_labels_map[optimizer_algorithm], fontsize=48)
            sns.lineplot(x=verbose_labels_map["MINI-BATCH SIZE"], y=verbose_labels_map[target_variable],
                         hue=verbose_labels_map["DATASET"], style=verbose_labels_map["LOSS"],
                         palette=colormap, ci=90, linewidth=3, data=plot_data, ax=ax[i, j])
            ax[i, j].set_xscale("log")
            ax[i, j].set_xlabel(verbose_labels_map["MINI-BATCH SIZE"], fontsize=36)
            ax[i, j].set_ylabel(verbose_labels_map[target_variable], fontsize=36)
            ax[i, j].tick_params(axis="x", labelsize=28)
            ax[i, j].tick_params(axis="y", labelsize=28)
            ax[i, j].legend([], [])
    legend = ax[0, 2].legend(bbox_to_anchor=(1, 1), title="Experiment condition", title_fontsize=48, fontsize=32)
    for i, handle in enumerate(legend.legendHandles):
        if i not in (0, 4):
            handle.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(fname=plots_dir + "experiment_1_mini_batch_size_vs_target_quantities.png", dpi="figure", format="png")
    plt.savefig(fname=plots_dir + "experiment_1_mini_batch_size_vs_target_quantities.pdf", dpi="figure", format="pdf")
    plt.close()

    # Normalize learning rates across optimizers and loss functions (revert the scaling)
    plot_data = experiment_log.copy()
    scaler = MinMaxScaler(feature_range=(0.01, 1))
    for (optimizer_algorithm, loss_function), records in experiment_log.groupby(["OPTIMIZER", "LOSS"]):
        plot_data.loc[(plot_data["OPTIMIZER"] == optimizer_algorithm)
                      & (plot_data["LOSS"] == loss_function), "LEARNING RATE"] = \
            scaler.fit_transform(records["LEARNING RATE"].to_numpy().reshape(-1, 1))
    plot_data = plot_data.replace(verbose_labels_map).rename(columns=verbose_labels_map)

    # Visualize influence of learning rate on convergence time
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=verbose_labels_map["LEARNING RATE"], y=verbose_labels_map["CONVERGED AT EPOCH"],
                 hue=verbose_labels_map["OPTIMIZER"], style=verbose_labels_map["LOSS"],
                 palette=colormap, ci=90, linewidth=3, data=plot_data)
    plt.xscale("log")
    plt.ylim((0, 71))
    plt.xlabel(verbose_labels_map["LEARNING RATE"], fontsize=18)
    plt.ylabel(verbose_labels_map["CONVERGED AT EPOCH"], fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    legend = plt.legend(loc="upper right", title="Experiment condition", title_fontsize=14, fontsize=12)
    for i, handle in enumerate(legend.legendHandles):
        if i not in (0, 4):
            handle.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(fname=plots_dir + "experiment_1_learning_rate_vs_convergence_time.png", dpi="figure", format="png")
    plt.savefig(fname=plots_dir + "experiment_1_learning_rate_vs_convergence_time.pdf", dpi="figure", format="pdf")
    plt.close()

    # Visualize relationship between training and validation accuracy
    group_by_vars = ["EPOCH", "DATASET", "OPTIMIZER", "LOSS"]
    accuracy_vars = ["TRAINING ACCURACY", "VALIDATION ACCURACY"]
    plot_data = training_logs.groupby(group_by_vars)[accuracy_vars].quantile(0.9).reset_index()
    plot_data = plot_data.melt(id_vars=group_by_vars,
                               value_vars=accuracy_vars,
                               var_name="MEASURE", value_name="ACCURACY")
    plot_data = plot_data.replace(verbose_labels_map).rename(columns=verbose_labels_map)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="EPOCH", y="ACCURACY",
                 hue=verbose_labels_map["DATASET"], style=verbose_labels_map["MEASURE"],
                 palette=colormap, ci=90, linewidth=3, data=plot_data)
    plt.xlabel("Training epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    legend = plt.legend(loc="upper right", title="Data source", title_fontsize=14, fontsize=12)
    for i, handle in enumerate(legend.legendHandles):
        if i not in (0, 4):
            handle.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(fname=plots_dir + "experiment_1_training_vs_validation_accuracy.png", dpi="figure", format="png")
    plt.savefig(fname=plots_dir + "experiment_1_training_vs_validation_accuracy.pdf", dpi="figure", format="pdf")
    plt.close()


def visualize_results_second_experiment(experiment_log):
    """
    Function to generate the plots using the experiment results from the second experiment

    :param experiment_log: second experiment log, pandas.Dataframe
    """

    datasets = ["circle", "mnist", "fashion_mnist"]
    optimizer_color_map = {"SGD": colormap[0], "Adam": colormap[1], "SGD^(1/2)": colormap[2]}
    loss_line_style_map = {"MSE": "solid", "Cross-entropy": "dashed"}

    diameter_ratio = experiment_log["ELLIPSE DIAMETER X"] / experiment_log["ELLIPSE DIAMETER Y"]
    print("Average ellipse diameter ratio: {} (+/- {})".format(diameter_ratio.mean(), diameter_ratio.std()))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, dataset_name in enumerate(datasets):
        plot_data = experiment_log[experiment_log["DATASET"] == dataset_name]
        plot_data = plot_data.replace(verbose_labels_map).rename(columns=verbose_labels_map)
        subplot = sns.lineplot(x="ELLIPSE CENTER X", y="ELLIPSE CENTER Y",
                               hue=verbose_labels_map["OPTIMIZER"], style=verbose_labels_map["LOSS"],
                               palette=colormap, data=plot_data, ax=ax[i])
        ax[i].set_title(verbose_labels_map[dataset_name], fontsize=32)
        for j, record in plot_data.iterrows():
            conv_ell = Ellipse(xy=(record["ELLIPSE CENTER X"], record["ELLIPSE CENTER Y"]),
                               width=record["ELLIPSE DIAMETER X"], height=record["ELLIPSE DIAMETER Y"],
                               fill=True, alpha=0.25,
                               facecolor=optimizer_color_map[record[verbose_labels_map["OPTIMIZER"]]],
                               linestyle=loss_line_style_map[record[verbose_labels_map["LOSS"]]],
                               linewidth=3, edgecolor="black")
            subplot.add_patch(conv_ell)
        if i == 0:
            ax[i].set_xlim((-4, 4))
            ax[i].set_ylim((-4, 4))
        else:
            ax[i].set_xlim((-0.5, 0.5))
            ax[i].set_ylim((-0.5, 0.5))
        ax[i].set_xlabel("Parameter 1 value", fontsize=24)
        ax[i].set_ylabel("Parameter 2 value", fontsize=24)
        ax[i].tick_params(axis="x", labelsize=18)
        ax[i].tick_params(axis="y", labelsize=18)
        ax[i].legend([], [])
    legend = ax[2].legend(bbox_to_anchor=(1, 1), title="Experiment condition", title_fontsize=28, fontsize=18)
    for i, handle in enumerate(legend.legendHandles):
        if i not in (0, 4):
            handle.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(fname=plots_dir + "experiment_2_convergence_region_comparison.png", dpi="figure", format="png")
    plt.savefig(fname=plots_dir + "experiment_2_convergence_region_comparison.pdf", dpi="figure", format="pdf")
    plt.close()
