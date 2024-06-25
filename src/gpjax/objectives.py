"""
This module defines a custom objective function for the SIMM latent force model. The objective function is a slight modification of `gpx.objectives.ConjugateMLL` to allow passing of a custom model (instead of using a `gpx.gps.ConjugatePosterior`). The objective function is used in the training of the model.
"""

import cola
import jax.numpy as jnp
import gpjax as gpx

from dataclasses import dataclass
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.typing import ScalarFloat
from model import ExactLFM
import beartype.typing as tp

CustomModel = tp.TypeVar("CustomModel", bound="ExactLFM")


@dataclass
class CustomConjMLL(gpx.objectives.AbstractObjective):
    def step(self, model: CustomModel, train_data: Dataset) -> ScalarFloat:
        r"""Evaluate the marginal log-likelihood of the Gaussian process.

        Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.

        For a training dataset $`\{x_n, y_n\}_{n=1}^N`$, set of test inputs
        $`\mathbf{x}^{\star}`$ the corresponding latent function evaluations are given
        by $`\mathbf{f}=f(\mathbf{x})`$ and $`\mathbf{f}^{\star}f(\mathbf{x}^{\star})`$,
        the marginal log-likelihood is given by:
        ```math
        \begin{align}
            \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})p(\mathbf{f}, \mathbf{f}^{\star}\mathrm{d}\mathbf{f}^{\star}\\
            &=0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}') +\sigma^2\mathbf{I}_N  \right)^{-1}\mathbf{y}-\log\lvert k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
        \end{align}
        ```

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising the model's parameters with respect to the parameters, we use the negative marginal log-likelihood. This can be realised through

        ```python
            mll = gpx.objectives.ConjugateMLL(negative=True)
        ```

        For optimal performance, the marginal log-likelihood should be ``jax.jit``
        compiled.
        ```python
            mll = jit(gpx.objectives.ConjugateMLL(negative=True))
        ```

        Parameters
        ----------
        model : ExactLFM
            The model to evaluate the marginal log-likelihood of.
        train_data : Dataset
            The training dataset.

        Returns
        -------
        ScalarFloat
            The marginal log-likelihood of the Gaussian process for the current parameter set.
        """
        x, y = train_data.X, train_data.y

        obs_noise = model.obs_stddev**2
        mx = model.mean_function(x)

        # Σ = (Kxx + Io²) = LLᵀ
        Kxx = model.gram(model.kernel, x)
        Kxx += cola.ops.I_like(Kxx) * model.jitter
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        # p(y | x, θ), where θ are the model hyperparameters:
        mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)

        return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze())
