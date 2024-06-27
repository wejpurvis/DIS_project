"""Utility functions used throughout project."""

from model import ExactLFM
from dataset import JaxP53Data
from tabulate import tabulate
import beartype.typing as tp
import jax.numpy as jnp

CustomModel = tp.TypeVar("CustomModel", bound="ExactLFM")


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
