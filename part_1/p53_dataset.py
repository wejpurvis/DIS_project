import numpy as np
import jax.numpy as jnp
import pandas as pd
import os
import jax

jax.config.update("jax_enable_x64", True)


class JAXP53_Data:
    """
    Class to handle the p53 dataset from Barenco et al. (2006)
    """

    def __init__(self, replicate=None, data_dir="../data/"):
        replicate -= 1
        m_observed, f_observed, gene_var, tf_var, t = load_barenco_data(data_dir)

        # m_observed is a tuple containing a data frame of gene expression levels and log-normalised gene expression levels (size: (replicates, genes, times))
        m_df, actual_m_observed = m_observed

        self.gene_names = np.array(m_df.index)
        self._num_outputs = actual_m_observed.shape[1]

        self.m_observed = jnp.array(actual_m_observed)
        self.t_observed = jnp.linspace(0, 12, 7)

        # transcription rates from Barenco paper
        f_barenco = np.array(
            [0.1845, 1.1785, 1.6160, 0.8156, 0.6862, -0.1828, 0.5131]
        ).reshape(1, 1, 7)
        self.f_observed = jnp.array(f_barenco)

        # handle variance and data based on 'replicate' number
        if replicate is None:
            # self.m_observed = jnp.array(self.m_observed)
            self._data = [
                (self.t_observed, self.m_observed[r, i])
                for r in range(actual_m_observed.shape[0])
                for i in range(self._num_outputs)
            ]
            self.variance = jnp.array(
                [
                    gene_var[r, i]
                    for r in range(actual_m_observed.shape[0])
                    for i in range(self._num_outputs)
                ]
            )
        else:
            self.m_observed = self.m_observed[replicate : replicate + 1]
            self.f_observed = self.f_observed[0:1]
            self._data = [
                (self.t_observed, self.m_observed[0, i])
                for i in range(self._num_outputs)
            ]
            self.variance = jnp.array(
                [gene_var[replicate, i] for i in range(self._num_outputs)]
            )

    def __getitem__(self, index):
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of range")
        return self._data[index]

    @property
    def num_outputs(self):
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, value):
        self._num_outputs = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __len__(self):
        return len(self._data)


# load data from .csv files
def load_barenco_data(dir_path):
    """
    Load data from Barenco et al. (2006) dataset and log-normalise it
    """
    with open(os.path.join(dir_path, "barencoPUMA_exprs.csv"), "r") as f:
        gene_expressions = pd.read_csv(f, index_col=0)
    with open(os.path.join(dir_path, "barencoPUMA_se.csv"), "r") as f:
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

    assert gene_expressions[gene_expressions.duplicated()].size == 0

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

    # Log-normal transormation
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

    return (
        (genes_df, genes_expressions),
        (p53_df, np.float64(p53_expressions)),
        genes_variances,
        p53_variances,
        np.arange(7) * 2,
    )


def flatten_dataset_jax(dataset):
    """
    Flatten the dataset using JAX operations.

    Args:
        dataset: The dataset containing gene data.

    Returns:
        train_t: The flattened time values.
        train_y: The flattened observed values across genes.
    """
    num_genes = dataset.num_outputs
    train_t = dataset[0][0]
    num_times = train_t.shape[0]

    # Collect all observed values across genes, now using JAX operations
    m_observed = jnp.stack([dataset[i][1] for i in range(num_genes)]).reshape(
        num_genes, num_times
    )

    # Repeat 'train_t' for each gene (tile is eqv to repeat in pytorch)
    train_t = jnp.tile(train_t, num_genes)

    # Flatten 'm_observed' to match the repeated 'train_t'
    train_y = m_observed.reshape(-1)

    return train_t, train_y
