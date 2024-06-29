"""
This module loads the necessary data for the SIMM latent force model.

The dataset required is small and is available preprocessed here:

- https://drive.google.com/drive/folders/1Tg_3SlKbdv0pDog6k2ys0J79e1-vgRyd?usp=sharing


For `load_barenco_data` function to work, download the above data and place it in the `data` directory in the root of the repository (i.e. `DIS_project/data`).
"""

import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from beartype.typing import Optional


class PyTorchDataset(Dataset):
    """
    Custom PyTorch dataset class for gene expression data.

    Parameters
    ----------
    replicate : int, optional
        Index of the replicate to use. If None, all replicates are used. Default is None.
    data_dir : str, optional
        Path to the directory containing the data files. Default is "../data/".

    Attributes
    ----------
    gene_names : list
        List of gene names.
    gene_expressions : torch.Tensor
        Tensor of shape (num_replicates, num_genes, num_timepoints) containing gene expressions.
    gene_variances_raw : torch.Tensor
        Tensor of shape (num_replicates, num_genes, num_timepoints) containing gene expression variances.
    num_outputs : int
        Number of outputs (genes).
    num_genes : int
        Number of genes.
    timepoints : torch.Tensor
        Tensor of timepoints (torch.linspace(0, 12, 7)).
    f_observed : torch.Tensor
        Tensor of shape (1, 1, 7) containing the observed latent force from Barenco paper.
    data : list
        List of tuples containing timepoints and gene expressions for each replicate and gene.
    gene_variances : np.array
        Array of gene expression variances.

    Examples
    --------
    Load the dataset:

    .. code-block:: python

        >>> dataset = PyTorchDataset(replicate=0, data_dir="../data/")
    """

    def __init__(
        self, replicate: Optional[int] = None, data_dir: Optional[str] = "../data/"
    ):
        gene_data = load_barenco_data(data_dir)

        self.gene_names = gene_data["gene_names"]
        self.gene_expressions = torch.tensor(
            gene_data["gene_expressions"], dtype=torch.float64
        )
        self.gene_variances_raw = torch.tensor(
            gene_data["gene_variances"], dtype=torch.float64
        )
        self.num_outputs = len(self.gene_names)
        self.num_genes = self.num_outputs
        self.timepoints = torch.linspace(0, 12, 7)

        # Latent force reported in Barenco paper
        f_observed = torch.tensor(
            [0.1845, 1.1785, 1.6160, 0.8156, 0.6862, -0.1828, 0.5131]
        ).view(1, 1, 7)
        self.f_observed = f_observed

        # Handle gene expression and variances based on 'replicate' number
        if replicate is None:
            # Use all replicates (triplicates) of the data
            # Iterate over replicates first, then genes:
            # Gene 1, rep 1, ..., gene 5, rep 1, gene 1, rep 2, ..., gene 5, rep 3
            self.data = [
                (
                    self.timepoints,
                    torch.tensor(self.gene_expressions[r, i], dtype=torch.float64),
                )
                for r in range(self.gene_expressions.shape[0])
                for i in range(self.num_genes)
            ]
            self.gene_variances = np.array(
                [
                    self.gene_variances_raw[r, i]
                    for r in range(self.gene_expressions.shape[0])
                    for i in range(self.num_genes)
                ]
            )

        else:
            self.gene_expressions = np.array(
                self.gene_expressions[replicate : replicate + 1]
            )
            self.data = [
                (
                    self.timepoints,
                    torch.tensor(self.gene_expressions[0, i], dtype=torch.float64),
                )
                for i in range(self.num_genes)
            ]
            self.gene_variances = np.array(
                self.gene_variances_raw[replicate : replicate + 1]
            )

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def params_ground_truth():
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
        return B_exact, S_exact, D_exact


def load_barenco_data(dir_path: str):
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
    except FileNotFoundError:
        print(
            "Please download the gene expression data and place it in the appropriate directory."
        )
    try:
        with open(os.path.join(dir_path, "barencoPUMA_se.csv"), "r") as f:
            gene_expressions_se = pd.read_csv(f, index_col=0)
    except FileNotFoundError:
        print(
            "Please download the gene expression variance data and place it in the appropriate directory."
        )

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
