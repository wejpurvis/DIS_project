"""
Custom GP Posteriors for sampling latent distribution and gene expressions from
"""

import gpjax as gpx
import jax.numpy as jnp
import cola
from p53_data import JAXP53_Data, dataset_3d
from gpjax.distributions import GaussianDistribution
from jaxtyping import Num
from gpjax.typing import (
    Array,
    ScalarFloat,
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

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.prior.kernel.gram(x)
        Kxx += diag_variances
        Kxx += cola.ops.I_like(Kxx) * self.prior.jitter  # + 1e-4
        # Kxx += cola.ops.I_like(Kxx) * jitter
        K_inv = cola.inv(Kxx)

        Kxf = self.prior.kernel.cross_covariance(x, t)
        KfxKxx = jnp.matmul(Kxf.T, K_inv.to_dense())
        mean = jnp.matmul(KfxKxx, y)

        Kff = self.prior.kernel.gram(t)
        Kff += cola.ops.I_like(Kff) * self.prior.jitter * 10  # + 1e-3 in pytorch

        var = Kff - jnp.matmul(KfxKxx, Kxf)
        var = jnp.diag(jnp.diag(var.to_dense()))
        var += cola.ops.I_like(var) * self.prior.jitter * 10  # + 1e-3 in pytorch

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), var)
