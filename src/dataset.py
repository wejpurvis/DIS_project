"""
This module loads the necessary data for the SIMM latent force model.

The dataset required is small and is available preprocessed here:

- https://drive.google.com/drive/folders/1Tg_3SlKbdv0pDog6k2ys0J79e1-vgRyd?usp=sharing


For `load_barenco_data` function to work, download the above data and place it in the `data` directory in the root of the repository (i.e. `DIS_project/data`).
"""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import os
import jax

jax.config.update("jax_enable_x64", True)


class JaxP53Data:
    """
    Custom data handling class for managing gene expression data from the Barenco et al. paper. Provides structured access to gene expressions, their variances, and associated timepoints, optionally filtered by replicates.

    Attributes
    ----------
    gene_names : list
        List of gene names.
    gene_expressions : jnp.ndarray
        Array of gene expressions of shape (num_replicates, num_genes, num_timepoints).
    gene_variances_raw : jnp.ndarray
        Array of gene expression variances of shape (num_replicates, num_genes, num_timepoints).
    num_genes : int
        Number of genes in the dataset.
    timepoints : jnp.ndarray
        Array of timepoints.
    f_observed : jnp.ndarray
        Array of observed latent forces reported in the Barenco paper.
    data : list
        List of tuples containing timepoints and gene expressions for each gene and replicate.
    gene_variances : jnp.ndarray
        Array of gene expression variances for the selected replicate.
    """

    def __init__(self, replicate=None, data_dir="data", selected_genes=None):
        """
        Initialises JAXP53_Data object by loading data from specified directory and filtering by replicate if needed.

        Parameters
        ----------
        replicate : int, optional
            Replicate number to filter the data by. If None, all replicates are used. Default is None. Error is raised if replicate number is out of range.
        data_dir : str, optional
            Path to directory containing the data files. Default is '../data/'.
        selected_genes : list of str, optional
            List of gene names to load. If None, all genes are loaded. Default is None.
        """
        gene_data = load_barenco_data(data_dir)
        all_genes = gene_data["gene_names"]

        # Error trap replicate
        assert replicate is None or 0 <= replicate < 3, "Invalid replicate number"

        if selected_genes is not None:
            # Error handling for provided gene names
            valid_genes = set(all_genes)
            selected_genes_set = set(selected_genes)

            # Check if genes are valid
            if not selected_genes_set.issubset(valid_genes):
                missing_genes = selected_genes_set - valid_genes
                raise ValueError(
                    f"Invalid gene names provided: {', '.join(missing_genes)}"
                )
            # Check for duplicates by comparing list and set lengths
            if len(selected_genes) != len(selected_genes_set):
                duplicates = [
                    gene for gene in selected_genes if selected_genes.count(gene) > 1
                ]
                raise ValueError(
                    f"Duplicate genes provided: {', '.join(set(duplicates))}"
                )

            # Check if genes are empty
            if len(selected_genes) == 0:
                raise ValueError(
                    "Empty list of genes selected, set 'selected_genes' to None"
                )

            indices = [
                i
                for i, gene in enumerate(gene_data["gene_names"])
                if gene in selected_genes
            ]
            self.selected_indices = [
                all_genes.index(gene) for gene in selected_genes if gene in all_genes
            ]
            self.gene_names = selected_genes
            self.gene_expressions = gene_data["gene_expressions"][:, indices]
            self.gene_variances_raw = gene_data["gene_variances"][:, indices]
        else:
            self.selected_indices = list(range(len(all_genes)))
            self.gene_names = gene_data["gene_names"]
            self.gene_expressions = gene_data["gene_expressions"]
            self.gene_variances_raw = gene_data["gene_variances"]

        self.num_genes = len(self.gene_names)
        self.timepoints = jnp.linspace(0, 12, 7)

        # Latent force reported in Barenco paper
        f_barenco = jnp.array(
            [0.1845, 1.1785, 1.6160, 0.8156, 0.6862, -0.1828, 0.5131]
        ).reshape(1, 1, 7)
        self.f_observed = f_barenco

        # Handle gene expression and variances based on 'replicate' number
        if replicate is None:
            # Use all replicates (triplicates) of the data
            # Iterate over replicates first, then genes:
            # Gene 1, rep 1, ..., gene 5, rep 1, gene 1, rep 2, ..., gene 5, rep 3
            self.data = [
                (self.timepoints, self.gene_expressions[r, i])
                for r in range(self.gene_expressions.shape[0])
                for i in range(self.num_genes)
            ]
            self.gene_variances = jnp.array(
                [
                    self.gene_variances_raw[r, i]
                    for r in range(self.gene_expressions.shape[0])
                    for i in range(self.num_genes)
                ]
            )

        else:
            self.gene_expressions = jnp.array(
                self.gene_expressions[replicate : replicate + 1]
            )
            self.data = [
                (self.timepoints, self.gene_expressions[0, i])
                for i in range(self.num_genes)
            ]
            self.gene_variances = jnp.array(
                self.gene_variances_raw[replicate : replicate + 1]
            )

    def __getitem__(self, index):
        """
        Returns the tuple of timepoints and gene expressions for the dataset at the specified index.

        Parameters
        ----------
        index : int
            Index of the dataset.

        Returns
        -------
        tuple
            Tuple containing timepoints and gene expressions for the dataset at the specified index.
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[index]

    def __len__(self):
        """
        Returns the number of genes x number of replicates in the dataset.

        Returns
        -------
        int
            Number of genes x number of replicates in the dataset.
        """
        return len(self.data)

    @property
    def shape(self):
        """
        Returns the shape of the data array formed by converting the list of tuples
        into a JAX array.

        Returns
        -------
        tuple
            Shape of the data array.
        """
        jnp_data = jnp.array(self.data)
        return jnp_data.shape

    def params_ground_truth(self):
        """
        Ground truth parameters for the SIMM model, measured experimentally by Barenco et al. (2006).

        Returns
        -------
        tuple
            Tuple containing the following arrays:
            - B_exact: array of shape (5,) containing the basal expression levels
            - S_exact: array of shape (5,) containing the sensitivities
            - D_exact: array of shape (5,) containing the decay rates
        """
        B_exact = np.array([0.0649, 0.0069, 0.0181, 0.0033, 0.0869])
        D_exact = np.array([0.2829, 0.3720, 0.3617, 0.8000, 0.3573])
        S_exact = np.array([0.9075, 0.9748, 0.9785, 1.0000, 0.9680])

        # Use self.selected_indices to filter the data
        B_exact = B_exact[self.selected_indices]
        D_exact = D_exact[self.selected_indices]
        S_exact = S_exact[self.selected_indices]

        return B_exact, S_exact, D_exact


def load_barenco_data(dir_path):
    """
    Load gene expressions and associated uncertainties from Barenco et al. (2006) microarray measureents return log-normalised data.

    Parameters
    ----------
    dir_path : str
        Path to directory containing the data files.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - gene_names: list of gene names
        - gene_expressions: array of shape (3, 5, 7) containing gene expressions
        - gene_variances: array of shape (3, 5, 7) containing gene expression variances
        - p53_expressions: array of shape (3, 1, 7) containing p53 expressions
        - p53_variances: array of shape (3, 1, 7) containing p53 expression variances
    """

    # Load data from .csv files
    try:
        with open(os.path.join(dir_path, "barencoPUMA_exprs.csv"), "r") as f:
            gene_expressions = pd.read_csv(f, index_col=0)
        with open(os.path.join(dir_path, "barencoPUMA_se.csv"), "r") as f:
            gene_expressions_se = pd.read_csv(f, index_col=0)
    except FileNotFoundError:
        with open(os.path.join("../data", "barencoPUMA_exprs.csv"), "r") as f:
            gene_expressions = pd.read_csv(f, index_col=0)
        with open(os.path.join("../data", "barencoPUMA_se.csv"), "r") as f:
            gene_expressions_se = pd.read_csv(f, index_col=0)

    columns = [f"cARP{r}-{t}hrs.CEL" for r in range(1, 4) for t in np.arange(7) * 2]

    # Known genes from Barenco paper
    known_target_genes = [
        "203409_at",
        "202284_s_at",
        "218346_s_at",
        "205780_at",
        "209295_at",
        "211300_s_at",
    ]

    genes = gene_expressions[gene_expressions.index.isin(known_target_genes)][columns]
    genes_se = gene_expressions_se[gene_expressions_se.index.isin(known_target_genes)][
        columns
    ]

    index = {
        "203409_at": "DDB2",
        "202284_s_at": "p21",
        "218346_s_at": "SESN1",
        "205780_at": "BIK",
        "209295_at": "DR5",
        "211300_s_at": "p53",
    }

    genes.rename(index=index, inplace=True)
    genes_se.rename(index=index, inplace=True)

    # Reorder genes
    genes_df = genes.reindex(["DDB2", "BIK", "DR5", "p21", "SESN1", "p53"])
    genes_se = genes_se.reindex(["DDB2", "BIK", "DR5", "p21", "SESN1", "p53"])

    p53_df = genes_df.iloc[-1:]
    genes_df = genes_df.iloc[:-1]
    genes = genes_df.values
    p53 = p53_df.values

    # Get variance for each gene expression value
    p53_var = genes_se.iloc[-1:].values ** 2
    genes_var = genes_se.iloc[:-1].values ** 2

    # Log-normal transform
    p53_full = np.exp(p53 + p53_var / 2)
    genes_full = np.exp(genes + genes_var / 2)

    # Calculate full variance in transformed space
    p53_var_full = (np.exp(p53_var) - 1) * np.exp(2 * p53 + p53_var)
    genes_var_full = (np.exp(genes_var) - 1) * np.exp(2 * genes + genes_var)

    # Normalise and rescale the data
    p53_scale = np.sqrt(np.var(p53_full[:, :7], ddof=1))
    p53_scale = np.c_[[p53_scale for _ in range(7 * 3)]].T

    p53_expressions = np.float64(p53_full / p53_scale).reshape((3, 1, 7))
    p53_variances = np.float64(p53_var_full / p53_scale**2).reshape((3, 1, 7))

    genes_scale = np.sqrt(np.var(genes_full[:, :7], axis=1, ddof=1))
    genes_scale = np.c_[[genes_scale for _ in range(7 * 3)]].T

    genes_expressions = (
        np.float64(genes_full / genes_scale).reshape((5, 3, 7)).swapaxes(0, 1)
    )
    genes_variances = (
        np.float64(genes_var_full / genes_scale**2).reshape((5, 3, 7)).swapaxes(0, 1)
    )

    # Get gene names
    gene_names = list(genes_df.index)

    return {
        "gene_names": gene_names,
        "gene_expressions": genes_expressions,
        "gene_variances": genes_variances,
        "p53_expressions": p53_expressions,
        "p53_variances": p53_variances,
    }


def flatten_dataset_jax(dataset):
    """
    Flatten the dataset using JAX operations, modified to handle multiple replicates if needed.

    Parameters
    ----------
    data : JAXP53_Data
        Data object containing gene expression data.

    Returns
    -------
    train_t : jnp.ndarray
        Time points for the training data.
    train_y : jnp.ndarray
        Flattened gene expression data.
    """
    # Total number of entries in the dataset (depends on genes and replicates)
    num_entries = len(dataset)

    # Extract time points from the first entry
    train_t = dataset[0][0]

    m_observed = jnp.concatenate([dataset[i][1] for i in range(num_entries)])

    # Each entry in `dataset` corresponds to (time, values) for a gene in a replicate
    # Since each gene at each time point is an entry, we need to repeat `train_t` for each entry
    train_t = jnp.tile(train_t, num_entries)

    # `m_observed` is already in the correct shape, directly from concatenation
    train_y = m_observed.reshape(-1)

    return train_t, train_y


def dataset_3d(data):
    """
    Modify representation of dataset to include flag for training and testing data.

    Parameters
    ----------
    data : JAXP53_Data
        Data object containing gene expression data.

    Returns
    -------
    training_times: jnp.array
        Array of timepoints of shape (t, i, z) where t is the time of measurement, i is the gene index, and z is a flag (1 for training and 0 for testing).
    gene_expressions: jnp.array
        Array of gene expression data of shape (x,1) where x is the measured gene expression.
    """
    num_genes = data.num_genes
    replicates = data.shape[0] // num_genes

    # (num_genes*num replicates, dimension, num_timepoints)
    gene_data = jnp.array([data[i] for i in range(len(data))])

    time_points = gene_data[0, 0, :]
    time_points_repeated = jnp.tile(time_points, gene_data.shape[0])

    # Repeat gene index for each timepoint and replicate
    gene_indices = jnp.tile(
        jnp.repeat(jnp.arange(num_genes), len(time_points)), replicates
    )

    ones = jnp.ones(num_genes * len(time_points) * replicates, dtype=int)

    # Shape (t x j x r) x 3 where t is timepoints and j is genes and r is replicates
    training_times = jnp.stack((time_points_repeated, gene_indices, ones), axis=-1)

    # Shape (t x j x r) x 1 where t is timepoints and j is genes and r is replicates
    gene_expressions = gene_data[:, 1, :].flatten().reshape(-1, 1)

    # Shape (t x j x r) x 1 where t is timepoints and j is genes and r is replicates
    variances = data.gene_variances.flatten().reshape(-1, 1)

    return training_times, gene_expressions, variances
