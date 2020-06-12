from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

sns.set(style="darkgrid", context="paper")


def visualize_convergence_region(simulations_param_data, simulations_diagnostic_data,
                                 convergence_region_params, plot_path):
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
