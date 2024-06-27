"""
Plotting functions for GPyTorch implementation.
"""

import os
import csv
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tabulate import tabulate

if shutil.which("latex"):
    plt.style.use(
        "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
    )
else:
    relative_style_path = "../../dissertation.mplstyle"
    absolute_style_path = os.path.join(os.path.dirname(__file__), relative_style_path)
    plt.style.use(absolute_style_path)

colors = rcParams["axes.prop_cycle"].by_key()["color"]


def plot_lf(gp, timepoints, stddev=2, scatter=None, save=True):
    """
    Plot latent force model. (fig. 1a in Lawrence et al. 2007)

    Parameters
    ----------
    gp: gpytorch.distributions.MultivariateNormal
        Trained GP model.
    timepoints: torch.Tensor
        Timepoints.
    stddev: int, optional
        Number of standard deviations to plot around the mean. Default is 2.
    scatter: torch.Tensor, optional
        Scatter points to plot. Default is None.
    save: bool, optional
        Save the plot. Default is True.
    """
    mean = gp.mean.detach().squeeze()
    std = gp.variance.detach().sqrt().squeeze()

    fig, ax = plt.subplots(figsize=(7.5, 2.5), dpi=300)

    ax.fill_between(
        timepoints,
        mean - stddev * std,
        mean + stddev * std,
        alpha=0.2,
        label=f"{stddev} sigma",
        color=colors[1],
    )
    ax.plot(
        timepoints,
        mean - stddev * std,
        linestyle="--",
        linewidth=1,
        color=colors[1],
    )
    ax.plot(
        timepoints,
        mean + stddev * std,
        linestyle="--",
        linewidth=1,
        color=colors[1],
    )

    ax.plot(timepoints, mean, color=colors[1], label="Predictive mean")

    if scatter is not None:
        times = np.linspace(0, 12, len(scatter))
        ax.plot(times, scatter, "x", label="True values", color=colors[0])

    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA Expression")
    ax.set_title("Latent Force Model (GPyTorch)")
    ax = clean_legend(ax)

    if save:
        save_plot("gpytorch_lf.png")
    else:
        plt.show()
    plt.clf()


def plot_gxpred(gp, timepoints, dataset, stddev=2, scatter=None, save=True):
    """
    Plot gene expression predictions (Kxx).

    Parameters
    ----------
    gp: gpytorch.distributions.MultivariateNormal
        Trained GP model.
    timepoints: torch.Tensor
        Timepoints.
    dataset: PyTorchDataset
        Dataset used for training.
    stddev: int, optional
        Number of standard deviations to plot around the mean. Default is 2.
    scatter: torch.Tensor, optional
        Scatter points to plot. Default is None.
    save: bool, optional
        Save the plot. Default is True.
    """
    mean = gp.mean.detach()
    std = gp.variance.detach().sqrt()
    num_genes = mean.shape[1]

    if len(dataset.gene_names) == num_genes:
        gene_names = dataset.gene_names
    else:
        gene_names = [f"Gene {i}" for i in range(num_genes)]

    fig = plt.figure(figsize=(7.5, 5.5 * np.ceil(num_genes / 3)), dpi=300)

    for i in range(num_genes):
        ax = fig.add_subplot(num_genes, min(num_genes, 1), i + 1)
        mean_i = mean[:, i].numpy()
        std_i = std[:, i].numpy()

        ax.fill_between(
            timepoints,
            mean_i - stddev * std_i,
            mean_i + stddev * std_i,
            alpha=0.2,
            label=f"{stddev} sigma",
            color=colors[1],
        )
        ax.plot(
            timepoints,
            mean_i - stddev * std_i,
            linestyle="--",
            linewidth=1,
            color=colors[1],
        )
        ax.plot(
            timepoints,
            mean_i + stddev * std_i,
            linestyle="--",
            linewidth=1,
            color=colors[1],
        )

        ax.plot(timepoints, mean_i, color=colors[1], label="Predictive mean")

        ax.scatter(
            dataset.timepoints,
            dataset.gene_expressions[:, i],
            color=colors[0],
            label="True values",
        )

        ax.set_title(f"{gene_names[i]} Expression Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Expression Level")
        ax.legend()
        ax = clean_legend(ax)
    if save:
        save_plot("gpytorch_gxpr.png")
    else:
        plt.show()
    plt.clf()


def plot_comparison_torch(model, dataset, trainer, save=True):
    """
    Plot double bar chart showing inference results on the hyperparameters for p53 dataset.

    Parameters
    ----------
    model: gpytorch.models.ExactGP
        Trained model.
    dataset: PyTorchDataset
        Dataset used for training (contains ground truth parameters).
    trainer: TorchTrainer
        Trainer used for training the model (contains learned parameters).
    save: bool, optional
        Save the plot. Default is True.
    """

    # Extract ground truth parameters from dataset
    basal_true, sensitivity_true, decay_true = dataset.params_ground_truth()

    # Extract estimated parameters from model
    kinetics = list()
    constraints = dict(model.named_constraints())
    for key in [
        "mean_module.raw_basal",
        "covar_module.raw_sensitivity",
        "covar_module.raw_decay",
    ]:
        val = trainer.parameter_trace[key][-1].squeeze()
        if key + "_constraint" in constraints:
            val = constraints[key + "_constraint"].transform(val)
        kinetics.append(val.numpy())
    kinetics = np.array(kinetics)

    basal_learned = kinetics[0:1].squeeze()
    sensitivity_learned = kinetics[1:2].squeeze()
    decay_learned = kinetics[2:3].squeeze()

    # Get gene names from dataset
    gene_names = dataset.gene_names

    # Combine the data into a table
    data = zip(gene_names, basal_learned, sensitivity_learned, decay_learned)
    headers = ["Gene Name", "Basal", "Sensitivity", "Decay"]

    # Print the table
    print("\n")
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
    print("\n")

    data = zip(gene_names, basal_learned, sensitivity_learned, decay_learned)
    headers = ["Gene Name", "Basal", "Sensitivity", "Decay"]

    # Write the table to a CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_name = os.path.join(script_dir, "hyperparams.csv")

    with open(save_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), dpi=300)

    true_colour = colors[0]
    learned_colour = colors[1]

    # Plot basal rates
    x = np.arange(len(basal_true))
    axes[0].bar(
        x + 0.2, basal_learned, width=0.4, color=learned_colour, label="basal_rates"
    )
    axes[0].bar(x - 0.2, basal_true, width=0.4, color=true_colour, label="B_exact")
    axes[0].set_title("Basal rates")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gene_names, rotation=45, ha="right")

    # Plot kxx_sensitivities
    axes[1].bar(
        x + 0.2,
        sensitivity_learned,
        width=0.4,
        color=learned_colour,
        label="kxx_sensitivities",
    )
    axes[1].bar(
        x - 0.2, sensitivity_true, width=0.4, color=true_colour, label="S_exact"
    )
    axes[1].set_title("Sensitivities")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gene_names, rotation=45, ha="right")

    # Plot kxx_degradations
    axes[2].bar(
        x + 0.2, decay_learned, width=0.4, color=learned_colour, label="Calculated"
    )
    axes[2].bar(x - 0.2, decay_true, width=0.4, color=true_colour, label="Measured")
    axes[2].set_title("Decay rates")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(gene_names, rotation=45, ha="right")

    # Single legend at the bottom
    handles, labels = axes[2].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center', ncol=2, fontsize='medium')

    # plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if save:
        save_plot("gpytorch_comparison.png")
    else:
        plt.show()
    plt.clf()


def clean_legend(ax):
    """
    Cleans up the legend of a matplotlib Axes object by removing duplicate entries.

    Parameters
    ----------
    gp: gpytorch.distributions.MultivariateNormal
        Trained GP model.
    timepoints: torch.Tensor
        Timepoints.
    stddev: int, optional
        Number of standard deviations to plot around the mean. Default is 2.
    scatter: torch.Tensor, optional
        Scatter points to plot. Default is None.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def save_plot(plot_name):
    """
    Save plots to gpytorch/plots regardless of where script is run from.

    Parameters
    ----------
    plot_name: str
        Name of the plot to save.
    """

    # Get the directory of the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    save_name = os.path.join(plots_dir, plot_name)
    print(f"Saving plot to {save_name}")
    plt.savefig(save_name, format="png", facecolor="white")
