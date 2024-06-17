"""
Custom GP Posteriors for sampling latent distribution and gene expressions from
"""

import gpjax as gpx
import jax.numpy as jnp
import cola
from p53_data import JAXP53_Data, dataset_3d
from gpjax.distributions import GaussianDistribution
from torch.distributions import MultivariateNormal
from jaxtyping import Num
from gpjax.typing import (
    Array,
)


class p53_posterior(gpx.gps.ConjugatePosterior):

    def latent_predict(
        self, test_inputs: Num[Array, "N D"], train_data: JAXP53_Data
    ) -> GaussianDistribution:
        """
        Predicts the latent function values at test inputs given the training data.

        Parameters
        ----------
        test_inputs : Num[Array, "N D"]
            A Jax array of test inputs at which the predictive distribution is evaluated.
        train_data : JAXP53_Data
            A custom dataclass object that contains the input, output, and the associated uncertainties of the inputs used for the training dataset.

        Returns
        -------
        GaussianDistribution
            A function that accepts an input array and returns the predictive distribution as a `GaussianDistribution`.
        """
        # x = training_times, y = gene expressions
        x, y, variances = dataset_3d(train_data)
        t = test_inputs

        # mean_x = self.prior.mean_function(x)
        # mean_t = self.prior.mean_function(t)

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.prior.kernel.gram(x)

        # Σ = Kxx + Io²
        Sigma = Kxx + diag_variances
        Sigma += cola.ops.I_like(Sigma) * self.prior.jitter  # + 1e-4
        Sigma = cola.PSD(Sigma)

        Kff = self.prior.kernel.gram(t)
        Kxf = self.prior.kernel.cross_covariance(x, t)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxf)

        # Kfx (Kxx + Io²)⁻¹ (y)
        mean = jnp.matmul(Sigma_inv_Kxt.T, y)
        # mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mean_x)

        covariance = Kff - jnp.matmul(Kxf.T, Sigma_inv_Kxt)
        # Covariance matrix is not PSD - take diagonal
        covariance = jnp.diag(jnp.diag(covariance.to_dense()))
        covariance += cola.ops.I_like(covariance) * self.prior.jitter  # + 1e-3
        covariance = cola.PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

    def expression_predict(
        self, test_inputs: Num[Array, "N D"], train_data: JAXP53_Data
    ) -> GaussianDistribution:
        """
        Predicts the gene expression values at test inputs given the training data.
        TODO: make this more jax-like and ensure means are calculated correctly

        """

        # x = training_times, y = gene expressions
        x, y, variances = dataset_3d(train_data)
        num_outputs = len(jnp.unique(x[:, 1]))
        t = test_inputs[:, 0]

        t_tiled = jnp.tile(t, (num_outputs))
        tile_len = t_tiled.shape[0]
        gene_indices = jnp.repeat(-1, tile_len)
        t_tiled = jnp.stack((t_tiled, gene_indices, jnp.repeat(0, tile_len)), axis=1)
        print(f" t_tiled shape: {t_tiled.shape}")

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.prior.kernel.gram(x)
        Kxx += diag_variances
        Kxx += cola.ops.I_like(Kxx) * self.prior.jitter  # + 1e-4
        K_inv = cola.inv(Kxx)

        Kxt = self.prior.kernel.cross_covariance(x, t_tiled)
        K_tx = Kxt.T

        KtxK_inv = jnp.matmul(K_tx, K_inv.to_dense())

        KtxK_invY = jnp.matmul(KtxK_inv, y)

        # Reshape for multiple outputs
        print(f"t shape 0 {t.shape[0]}, shape: {t.shape}")
        print(f"t tiled shape 0 {t_tiled.shape[0]}, shape: {t_tiled.shape}")
        mean = KtxK_invY.reshape(num_outputs, t.shape[0])
        print(f"mean shape: {mean.shape}")

        K_xstarxstar = self.prior.kernel.gram(t_tiled)
        var = K_xstarxstar - jnp.matmul(KtxK_inv, K_tx.T)
        var = jnp.diagonal(var.to_dense()).reshape(num_outputs, t.shape[0])

        # Final reshaping and adjustmenst for mean and covariance
        mean = mean.T
        var = var.T
        var = self.diag_embed_jax(var)
        var += jnp.eye(var.shape[-1]) * 1e-3

        import tensorflow_probability.substrates.jax.distributions as tfd

        mvn = tfd.MultivariateNormalFullCovariance(mean, var)
        return mvn, t_tiled

    def predict_gene1(
        self, test_inputs: Num[Array, "N D"], train_data: JAXP53_Data
    ) -> GaussianDistribution:
        # Predict gene expression for 1st gene

        # x = training_times, y = gene expressions
        x, y, variances = dataset_3d(train_data)

        t = test_inputs

        # Slice traning data for 1st gene
        x = x[:7]
        y = y[:7]
        variances = variances[:7]

        mean_x = self.prior.mean_function(x)
        mean_t = self.prior.mean_function(t)

        print(f" mean x shape: {mean_x.shape}, mean t shape: {mean_t.shape}")

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.prior.kernel.gram(x)

        # Σ = Kxx + Io²
        Sigma = Kxx + diag_variances
        Sigma += cola.ops.I_like(Sigma) * self.prior.jitter  # + 1e-4
        Sigma = cola.PSD(Sigma)

        Kff = self.prior.kernel.gram(t)
        Kxf = self.prior.kernel.cross_covariance(x, t)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxf)

        # Kfx (Kxx + Io²)⁻¹ (y)
        # mean = jnp.matmul(Sigma_inv_Kxt.T, y)
        print(f"y shape: {y.shape}, Sigma_inv_Kxt shape: {Sigma_inv_Kxt.shape}")
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mean_x)

        var = Kff - jnp.matmul(Kxf.T, Sigma_inv_Kxt)
        var = jnp.diag(jnp.diag(var.to_dense()))
        var += cola.ops.I_like(var) * self.prior.jitter

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), var)

    def diag_embed_jax(self, var):
        # Diagonally embed var into a matrix of shape (80, 5, 5)
        # Assuming var has shape (80, 5)
        n, d = var.shape
        # Create an identity matrix of shape (d, d)
        eye = jnp.eye(d)
        # Expand dimensions of var to (80, 1, 5)
        var_expanded = var[:, None, :]
        # Multiply each diagonal matrix by the corresponding expanded var
        result = var_expanded * eye
        return result
