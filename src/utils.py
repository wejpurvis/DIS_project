"""
Utility functions used throughout project.
Includes the prediction class for gene expressions
"""

import jax
import shutil
import os
from model import ExactLFM
from dataset import JaxP53Data
from tabulate import tabulate
import beartype.typing as tp
import jax.numpy as jnp
from beartype.typing import Optional
from plotter import save_plot, clean_legend
import matplotlib.pyplot as plt
from matplotlib import rcParams

if shutil.which("latex"):
    plt.style.use(
        "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
    )
else:
    relative_style_path = "../dissertation.mplstyle"
    absolute_style_path = os.path.join(os.path.dirname(__file__), relative_style_path)
    plt.style.use(absolute_style_path)

colors = rcParams["axes.prop_cycle"].by_key()["color"]

CustomModel = tp.TypeVar("CustomModel", bound="ExactLFM")
key = jax.random.PRNGKey(42)


class GeneExpressionPredictor:
    """
    A class for predicting gene expressions using a given model and p53 data.

    To initialise the class, provide the trained model and corresponding data:

    .. code-block:: python
        gene_predictions = GeneExpressionPredictor(trained_model, p53_data)

    Parameters
    ----------
    model : Model
        The model used for gene expression prediction.
    p53_data : P53Data
        The p53 data containing gene expressions.
    t : int, optional
        Number of testing times. Default is 100.

    Attributes
    ----------
    model : Model
        The model used for gene expression prediction.
    p53_data : P53Data
        The p53 data containing gene expressions.
    num_genes : int
        The number of genes in the p53 data.
    gene_names : list
        The names of the genes in the p53 data.
    t : int
        Number of testing times.
    """

    def __init__(self, model, p53_data, t=100):
        self.model = model
        self.p53_data = p53_data
        self.num_genes = p53_data.num_genes
        self.gene_names = p53_data.gene_names
        self.t = t

    def generate_test_times_pred(self):
        """
        Generate testing times for the GP model to predict gene expressions.

        Returns
        -------
        testing_times : jnp.ndarray
            Array of testing times of shape (t, 3) where t is the number of testing times.
        """
        times = jnp.linspace(0, 13, self.t)
        times_repeated = jnp.tile(times, self.num_genes)
        gene_indices = jnp.repeat(jnp.arange(1, self.num_genes + 1), self.t)

        testing_times = jnp.stack(
            (times_repeated, gene_indices, jnp.repeat(1, times_repeated.shape[0])),
            axis=1,
        )
        return testing_times

    def decompose_predictions(self, pred):
        """
        Given a mutli-variate Gaussian distribution mean or standard deviation, decompose the predictions into individual gene expressions (for a variable number of genes).

        Parameters
        ----------
        pred : jnp.ndarray
            Array of predictions of shape (t * num_genes, num_genes).

        Returns
        -------
        gene_expressions : tuple
            Tuple of gene expressions of shape (t, num_genes) for each gene.
        """
        test_size = self.t
        return tuple(
            pred[i * test_size : (i + 1) * test_size] for i in range(self.num_genes)
        )

    def decompose_predictions2(self, pred):
        """
        Given a mutli-variate Gaussian distribution mean or standard deviation, decompose the predictions into individual gene expressions (for all five genes).

        Parameters
        ----------
        pred : jnp.ndarray
            Array of predictions of shape (t * num_genes, num_genes).

        Returns
        -------
        gene_expressions : tuple
            Tuple of gene expressions of shape (t, num_genes) for each gene.
        """
        test_size = self.t

        # TODO: genes 3 and 4 are swapped for some reason
        gene_1 = pred[:test_size]
        gene_2 = pred[test_size : test_size * 2]
        gene_4 = pred[test_size * 2 : test_size * 3]
        gene_3 = pred[test_size * 3 : test_size * 4]
        gene_5 = pred[test_size * 4 :]

        return gene_1, gene_2, gene_3, gene_4, gene_5

    def plot_predictions(self, p53_data, stddev=2, save=True):
        """
        Plot gene expression predictions (Kxx).

        To plot the gene expressions after initialising the class:

        .. code-block:: python
            gene_predictions = GeneExpressionPredictor(trained_model, p53_data)
            gene_predictions.plot_predictions(p53_data)

        Parameters
        ----------
        p53_data : JaxP53Data
            P53 data.
        stddev : int, optional
            Number of standard deviations to plot around the mean. Default is 2.
        save : bool, optional
            Save the plot. Default is True.
        """
        xpr_times = self.generate_test_times_pred()
        all_gene_dists = self.model.multi_gene_predict(xpr_times, p53_data)

        if self.num_genes == 5:
            means = self.decompose_predictions2(all_gene_dists.mean())
            stds = self.decompose_predictions2(all_gene_dists.stddev())
        else:
            means = self.decompose_predictions(all_gene_dists.mean())
            stds = self.decompose_predictions(all_gene_dists.stddev())

        gene_names = self.gene_names
        timepoints = xpr_times[:100, 0]
        fig = plt.figure(figsize=(7.5, 5.5 * jnp.ceil(self.num_genes / 3)), dpi=300)

        for i in range(self.num_genes):
            ax = fig.add_subplot(self.num_genes, min(self.num_genes, 1), i + 1)
            mean_i = means[i]
            std_i = stds[i]

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
                p53_data.timepoints,
                p53_data.gene_expressions[:, i].flatten(),
                color=colors[0],
                label="True values",
            )

            ax.set_title(f"{gene_names[i]} Expression Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Expression Level")
            ax = clean_legend(ax)

        if save:
            save_plot("gpjax_gxpr.png")
        else:
            plt.show()
        plt.clf()


def print_hyperparams(model: CustomModel, dataset: JaxP53Data):
    # Extract learned parameters from model
    basal_learned = jnp.array(model.true_b, dtype=jnp.float64)
    sensitivity_learned = jnp.array(model.true_s, dtype=jnp.float64)
    decay_learned = jnp.array(model.true_d, dtype=jnp.float64)

    # Get gene names from dataset
    gene_names = dataset.gene_names

    # Combine the data into a table
    data = zip(gene_names, basal_learned, sensitivity_learned, decay_learned)
    headers = ["Gene Name", "Basal", "Sensitivity", "Decay"]

    # Print the table
    print(tabulate(data, headers=headers, tablefmt="grid"))


def generate_test_times(t: Optional[int] = 100):
    """
    Generate testing times for the GP model to predict the latent force function.

    Parameters
    ----------
    t : int, optional
        Number of testing times. Default is 80.

    Returns
    -------
    testing_times : jnp.ndarray
        Array of testing times of shape (t, 3) where t is the number of testing times.
    """

    times = jnp.linspace(0, 13, t)
    # Gene indices shouldn't matter
    gene_indices = jnp.repeat(-1, t)
    testing_times = jnp.stack((times, gene_indices, jnp.repeat(0, t)), axis=-1)
    return testing_times


def generate_test_times_pred(t: Optional[int] = 100):
    """
    Generate testing times for the GP model to predict gene expressions.

    Parameters
    ----------
    t : int, optional
        Number of testing times. Default is 80.

    Returns
    -------
    testing_times : jnp.ndarray
        Array of testing times of shape (t, 3) where t is the number of testing times.
    """
    # Placeholder
    num_genes = 5
    times = jnp.linspace(0, 13, t)
    times_repeated = jnp.tile(times, num_genes)
    gene_indices = jnp.repeat(jnp.arange(1, num_genes + 1), t)

    testing_times = jnp.stack(
        (times_repeated, gene_indices, jnp.repeat(1, times_repeated.shape[0])), axis=1
    )

    return testing_times
