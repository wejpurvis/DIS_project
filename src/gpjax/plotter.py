"""
Plotting functions for GPJax implementation.
"""

import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib import rcParams
from gpjax.typing import Array
from jaxtyping import Float, Num
from beartype.typing import Optional
from gpjax.distributions import GaussianDistribution
from model import ExactLFM
from dataset import JaxP53Data
import beartype.typing as tp

CustomModel = tp.TypeVar("CustomModel", bound="ExactLFM")

if shutil.which("latex"):
    plt.style.use(
        "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
    )
else:
    relative_style_path = "../../dissertation.mplstyle"
    absolute_style_path = os.path.join(os.path.dirname(__file__), relative_style_path)
    plt.style.use(absolute_style_path)

colors = rcParams["axes.prop_cycle"].by_key()["color"]


def plot_lf(
    testing_times: Float[Array, "1 3"],
    predictive_dist: GaussianDistribution,
    stddev: Optional[int] = 2,
    y_scatter: Optional[Float[Array, "7 "]] = None,
    save: Optional[bool] = True,
):
    """
    Plot latent force model. (fig. 1a in Lawrence et al. 2007)

    Parameters
    ----------
    testing_times : jnp.ndarray
        Array of testing times of shape (t, 3) where t is the number of testing times.
    predictive_dist : jax mvn
        Trained GP model.
    stddev: int, optional
        Number of standard deviations to plot around the mean. Default is 2.
    scatter: torch.Tensor, optional
        Scatter points to plot. Default is None.
    save: bool, optional
        Save the plot. Default is True.

    """
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    # (100,)
    timepoints = testing_times[:, 0]

    fig, ax = plt.subplots(figsize=(7.5, 2.5), dpi=300)

    ax.fill_between(
        timepoints,
        predictive_mean - stddev * predictive_std,
        predictive_mean + stddev * predictive_std,
        alpha=0.2,
        label=f"{stddev} sigma",
        color=colors[1],
    )
    ax.plot(
        timepoints,
        predictive_mean - stddev * predictive_std,
        linestyle="--",
        linewidth=1,
        color=colors[1],
    )
    ax.plot(
        timepoints,
        predictive_mean + stddev * predictive_std,
        linestyle="--",
        linewidth=1,
        color=colors[1],
    )

    ax.plot(timepoints, predictive_mean, label="Predictive mean", color=colors[1])

    if y_scatter is not None:
        times = jnp.linspace(0, 12, len(y_scatter))
        ax.plot(times, y_scatter, "x", label="True values", color=colors[0])

    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA Expression")
    ax.set_title("Latent Force Model (GPJax)")
    ax = clean_legend(ax)

    if save:
        save_plot("gpjax_lf.png")
    else:
        plt.show()
    plt.clf()


def plot_gxpred(
    testing_times: Float[Array, "1 3"],
    predictive_dist: GaussianDistribution,
    stddev: Optional[int] = 2,
    y_scatter: Optional[Float[Array, "7 "]] = None,
    save: Optional[bool] = True,
):
    """
    Plot gene expression predictions (Kxx).
    """

    mean = predictive_dist.mean()
    std = predictive_dist.stddev()

    # TODO look into tensorflow_probability.substrates.jax.distributions.MultivariateNormalFullCovariance

    raise NotImplementedError("plot_gxpred not implemented yet")


def plot_comparison_gpjax(
    model: CustomModel, dataset: JaxP53Data, save: Optional[bool] = True
):
    """
    Plot double bar chart showing inference results on the hyperparameters for p53 dataset.

    Parameters
    ----------
    model : CustomModel
        The trained model containing the learned parameters.
    dataset : JaxP53Data
        The dataset containing the true parameters.
    save : bool, optional
        Save the plot. Default is True.
    """
    # Extract ground truth parameters from dataset
    basal_true, sensitivity_true, decay_true = dataset.params_ground_truth()

    # Extract learned parameters from model
    basal_learned = model.true_b
    sensitivity_learned = model.true_s
    decay_learned = model.true_d

    # Get gene names from dataset
    gene_names = dataset.gene_names

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), dpi=300)

    true_colour = colors[0]
    learned_colour = colors[1]

    # Plot basal rates
    x = jnp.arange(len(basal_true))
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
        save_plot("gpjax_comparison.png")
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
