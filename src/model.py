"""
This module defines the GPJax implementation of the SIMM latent force model. As GPJax does not inherently support the sharing of parameters across mean functions and kernels, a custom model is defined (instead of using a `gpx.gps.ConjugatePosterior`)
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

from beartype.typing import Union
from dataclasses import dataclass, field

import beartype.typing as tp
import cola
import gpjax as gpx

from cola.ops import Dense, LinearOperator
from dataset import JaxP53Data, dataset_3d
from gpjax.base import param_field, static_field
from gpjax.distributions import GaussianDistribution
from gpjax.typing import Array, ScalarFloat
from jaxtyping import Float, Num
from dataset import JaxP53Data


Kernel = tp.TypeVar("Kernel", bound="gpx.kernels.base.AbstractKernel")


# TODO: docstrings !
@dataclass
class ExactLFM(gpx.base.Module):
    """
    GPJax implementation of the Single Input Motif Model (SIMM) latent force model from
    Lawrence et al. (2006) using the p53 dataset. This class extends `gpx.base.Module`
    to support shared parameters across mean functions and kernels, which is not
    inherently supported by GPJax.

    Examples
    --------

    By default, the model will work with an input of five genes. An example of how to use this model is shown below for predicting the latent function:

    .. code-block:: python

        >>> # Load dataset
        >>> data = JaxP53Data()
        >>> # Instantiate the model
        >>> model = ExactLFM()
        >>> # Define trainer an optimiser to train model
        >>> trainer = JaxTrainer(model, CustomConjMLL(negative=True), dataset_train, optimiser, key, num_iters=150)
        >>> # Train the model
        >>> trained_model, training_history = trainer.fit(num_steps_per_epoch=1000)
        >>> # Predict the latent function
        >>> predicted_distribution = model.latent_predict(test_inputs, data)

    For additional flexibility, the number of genes can be changed by setting the `num_genes` attribute of the model. The model can also be used to predict the gene expression values:

    .. code-block:: python

        >>> # Instantiate the model with 3 genes
        >>> model = ExactLFM(num_genes=3)
    """

    # Define jitter (prior) and noise (likelihood)
    jitter: float = static_field(1e-6)
    obs_stddev: Union[ScalarFloat, Float[Array, "#N"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )

    # Get data (number of genes) and initialise hyperparameters
    data: JaxP53Data = static_field(default_factory=lambda: JaxP53Data())
    num_genes: int = static_field(5)

    initial_decays: Float[Array, "1 N"] = static_field(init=False)
    initial_sensitivities: Float[Array, "1 N"] = static_field(init=False)
    initial_basals: Float[Array, "1 N"] = static_field(init=False)

    true_d: Float[Array, "1 N"] = param_field(
        init=False,
        bijector=tfb.Softplus(),
        metadata={"name": " kxx_degradations"},
        trainable=True,
    )

    true_s: Float[Array, "1 5"] = param_field(
        init=False,
        bijector=tfb.Softplus(),
        metadata={"name": " kxx_sensitivities"},
        trainable=True,
    )

    true_b: Float[Array, "1 5"] = param_field(
        init=False,
        bijector=tfb.Softplus(),
        metadata={"name": " basal_rates"},
        trainable=True,
    )

    # Set initial values for hyperparameters
    def __post_init__(self):
        self.initial_decays = jnp.array([0.4] * self.num_genes, dtype=jnp.float64)
        self.initial_sensitivities = jnp.array(
            [1.0] * self.num_genes, dtype=jnp.float64
        )
        self.initial_basals = jnp.array([0.05] * self.num_genes, dtype=jnp.float64)

        self.true_d = self.initial_decays
        self.true_s = self.initial_sensitivities
        self.true_b = self.initial_basals

    # Restrict lengthscale to be between 0.5 and 3.5
    l_bijector = tfb.Sigmoid(low=0.5, high=3.5)

    # l doesn't depend on the number of genes
    initial_lengthscale = jnp.array(2.5, dtype=jnp.float64)

    l: Float[Array, " O"] = param_field(
        initial_lengthscale,
        bijector=l_bijector,
        metadata={"name": "lengthscale"},
        trainable=True,
    )

    # Define mean function
    def mean_function(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        r"""
        Single Input Motif Model (SIMM) mean function.

        .. math::
            f(x_{j}) = \frac{B_{j}}{D_{j}}

        From equation 2 in Lawrence et al. (2006), the mean function is defined as the ratio of the basal rate to the degradation rate for the gene expressions, and assumed to be zero for the latent force. This function therefore uses the flag to determine whether the input is a gene expression or latent force function.

        Parameters
        ----------
        x : Num[Array, "N D"]
            Input data.

        Returns
        -------
        Float[Array, "N O"]
            Mean function (0 for latent force, B/D for gene expression)
        """
        f = jnp.array(x[:, 2:], dtype=int)

        block_size = x.shape[0] // self.num_genes
        mean = (self.true_b / self.true_d).reshape(-1, 1)
        mean = mean.repeat(block_size, jnp.newaxis).reshape(-1, 1)

        return mean * f

    # Define kernel
    def kernel(
        self, t: Float[Array, "1 3"], t_prime: Float[Array, "1 3"]
    ) -> ScalarFloat:
        """
        Compute kxx, kxf, or kff between a pair of arrays depending on the flags in the input.

        Parameters
        ----------
        t: Array of shape (1, 3)
            First timepoint input where the first dimension is the time, second is the gene index and the third is the flag.

        t_prime: Array of shape (1, 3)
            Second timepoint input where the first dimension is the time, second is the gene index and the third is the flag.

        Returns
        -------
        final_kernel: float
            The covariance between the input at times t and t'.

        Notes
        -----
        The flag is used to determine if the input is gene expression or latent force.
        1 = gene expression, 0 = latent force function.
        GPJAx does not support if statements in kernels, so we use switches to determine the kernel to use.
        """

        # Get flag from input (1 = gene expression, 0 = latent force function)
        f1 = jnp.array(t[2], dtype=int)
        f2 = jnp.array(t_prime[2], dtype=int)

        # Cannot use if statements in kernels -> use switches
        kxx_switch = f1 * f2
        kff_switch = (1 - f1) * (1 - f2)
        kxf_switch = f1 * (1 - f2)
        kxf_t_switch = (1 - f1) * f2

        final_kernel = (
            kxx_switch * self.kernel_xx(t, t_prime)
            + kff_switch * self.kernel_ff(t, t_prime)
            + kxf_switch * self.kernel_xf(t, t_prime)
            + kxf_t_switch * self.kernel_xf(t_prime, t)
        )

        return final_kernel

    def kernel_xx(
        self, t: Float[Array, "1 3"], t_prime: Float[Array, "1 3"]
    ) -> ScalarFloat:
        r"""
        Calculates the covariance between the expression leves of genes j and k at times t and t'. Equation 5 in Lawrence et al. (2006):

        .. math::
            k_{x_{j}x_{k}}(t,t') = S_{j}S_{k}\frac{\sqrt{\pi}l}{2}[h_{kj}(t,t') + h_{kj}(t',t)]

        As GPJax handles 1D inputs, the gene index is passed as an additional dimension in the input to this method.

        Parameters
        ----------
        t: Array of shape (1, 3)
            The input time points for the first gene expression where the first dimension is the time, second is the gene index and the third is the flag. (t, j, 1) where 1 is the flag for training data.

        t_prime: Array of shape (1, 3)
            The input time points for the second gene expression where the first dimension is the time, second is the gene index and the third is the flag. (t', k, 1) where 1 is the flag for training data.

        Returns
        -------
        kxx: float
            The covariance between the expression levels of genes j and k at times t and t'.
        """

        # Get gene indices
        j = t[1].astype(int)
        k = t_prime[1].astype(int)

        t = t[0]
        t_prime = t_prime[0]

        # Equation 5
        mult = self.true_s[j] * self.true_s[k] * self.l * jnp.sqrt(jnp.pi) * 0.5
        second_term = self.h(k, j, t_prime, t) + self.h(j, k, t, t_prime)

        kxx = mult * second_term

        return kxx.squeeze()

    def kernel_xf(
        self, t: Float[Array, "1 3"], t_prime: Float[Array, "1 3"]
    ) -> ScalarFloat:
        r"""
        Cross covariance term required to infer the latent force from gene expression data. Equation 6 in Lawrence et al. (2006):

        .. math::
            k_{x_{j}f}(t,t') = \frac{S_{rj}\sqrt{\pi}l}{2} \textrm{exp}(\gamma_{j})^{2}\textrm{exp}(-D_{j}(t'-t)) \biggl[ \textrm{erf} \left (\frac{t'-t}{l} - \gamma_{k}\right ) + \textrm{erf} \left (\frac{t}{l} +\gamma_{k} \right ) \biggr]

        This method uses the third flag dimension to determine which input is the gene expression and which is the latent force (as both kxf and it's transpose are used).

        Parameters
        ----------
        t: Array of shape (1, 3)
            The input time points for the gene expression and latent force where the first dimension is the time, second is the gene index and the third is the flag. (t, j, f) where f is the flag.

        t_prime: Array of shape (1, 3)
            The input time points for the gene expression and latent force where the first dimension is the time, second is the gene index and the third is the flag. (t', j, f) where 0 is the flag.

        Returns
        -------
        kxf: float
            The cross covariance between the expression levels of gene j and the latent force at times t and t'.
        """

        gene_xpr = jnp.where(t[2] == 0, t_prime, t)
        latent_force = jnp.where(t[2] == 0, t, t_prime)

        j = gene_xpr[1].astype(int)

        # Slice inputs
        gene_xpr = gene_xpr[0]
        latent_force = latent_force[0]

        t_dist = gene_xpr - latent_force

        first_term = 0.5 * self.l * jnp.sqrt(jnp.pi) * self.true_s[j]
        first_expon_term = jnp.exp(self.gamma(j) ** 2)
        second_expon_term = jnp.exp(-self.true_d[j] * t_dist)
        erf_terms = jax.scipy.special.erf(
            (t_dist / self.l) - self.gamma(j)
        ) + jax.scipy.special.erf(latent_force / self.l + self.gamma(j))

        kxf = first_term * first_expon_term * second_expon_term * erf_terms

        return kxf.squeeze()

    def kernel_ff(
        self, t: Float[Array, "1 3"], t_prime: Float[Array, "1 3"]
    ) -> ScalarFloat:
        """
        Custom RBF kernel taken as the process prior over f(t) (enables analaytical solution for kxf anf kxx).

        Parameters
        ----------
        t: Array of shape (1, 3)
            The input time points for the latent force at time t.

        t_prime: Array of shape (1, 3)
            The input time points for the latent force at time t'.

        Returns
        -------
        K: float
            The covariance between the latent force at times t and t'.
        """

        t = t[0].reshape(-1)
        t_prime = t_prime[0].reshape(-1)

        sq_dist = jnp.square(t.reshape(-1, 1) - t_prime)
        sq_dist = jnp.divide(sq_dist, 2 * self.l.reshape((-1, 1)))

        K = jnp.exp(-sq_dist)

        return K.squeeze()

    # Helper functions
    def h(
        self,
        j: Float[Array, " O"],
        k: Float[Array, " O"],
        t1: Float[Array, " O"],
        t2: Float[Array, " O"],
    ) -> ScalarFloat:
        """
        Analytical solution for the convolution of the exponential kernel with a step function.

        Parameters
        ----------
        j: int
            Gene index j.
        k: int
            Gene index k.
        t1: float
            Time point t1.
        t2: float
            Time point t2.


        Returns
        -------
        result: float
            The convolution of the exponential kernel with a step function.
        """

        t_dist = t2 - t1

        multiplier = jnp.exp(self.gamma(k) ** 2) / (self.true_d[j] + self.true_d[k])

        first_multiplier = jnp.exp(-self.true_d[k] * t_dist)

        first_erf_terms = jax.scipy.special.erf(
            (t_dist / self.l) - self.gamma(k)
        ) + jax.scipy.special.erf(t1 / self.l + self.gamma(k))

        second_multiplier = jnp.exp(-(self.true_d[k] * t2 + self.true_d[j] * t1))

        second_erf_terms = jax.scipy.special.erf(
            (t2 / self.l) - self.gamma(k)
        ) + jax.scipy.special.erf(self.gamma(k))

        result = multiplier * (
            jnp.multiply(first_multiplier, first_erf_terms)
            - jnp.multiply(second_multiplier, second_erf_terms)
        )

        # print(f"result: {result}")
        return result

    def gamma(self, k: Float[Array, " O"]) -> ScalarFloat:
        # Gamma term for h function
        return (self.true_d[k] * self.l) / 2

    # Calculate cross-covariance
    def cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """
        Calculate the cross-covariance between two sets of inputs for a given kernel.

        Parameters
        ----------
        kernel : Kernel
            The kernel function used to calculate the covariance.
        x : Float[Array, "N D"]
            The first set of inputs.
        y : Float[Array, "M D"]
            The second set of inputs.

        Returns
        -------
        Float[Array, "N M"]
            The cross-covariance between the two sets of inputs.
        """
        cross_cov = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(y))(x)

        return cross_cov

    def gram(self, kernel: Kernel, x: Float[Array, "N D"]) -> LinearOperator:
        """
        Calculate the Gram matrix for a given kernel and input data.

        Parameters
        ----------
        kernel : Kernel
            The kernel function used to calculate the covariance.
        x : Float[Array, "N D"]
            The input data.

        Returns
        -------
        LinearOperator
            The Gram matrix for the input data.
        """
        Kxx = self.cross_covariance(kernel, x, x)

        return cola.PSD(Dense(Kxx))

    ############################
    # Define predict methods
    ############################

    def latent_predict(
        self, test_inputs: Num[Array, "N D"], train_data: JaxP53Data
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

        x, y, variances = dataset_3d(train_data)
        t = test_inputs
        # jitter = 1e-3

        mean_x = self.mean_function(x)
        mean_t = self.mean_function(t)

        diag_variances = jnp.diag(variances.reshape(-1))
        Kxx = self.gram(self.kernel, x)
        Kxx += diag_variances
        Kxx += cola.ops.I_like(Kxx) * self.jitter
        K_inv = cola.inv(Kxx)

        Kxf = self.cross_covariance(self.kernel, x, t)
        KfxKxx = jnp.matmul(Kxf.T, K_inv.to_dense())
        mean = mean_t + jnp.matmul(KfxKxx, y - mean_x)

        Kff = self.gram(self.kernel, t)
        Kff += cola.ops.I_like(Kff) * self.jitter

        var = Kff - jnp.matmul(KfxKxx, Kxf)
        var = jnp.diag(jnp.diag(var.to_dense()))
        var += cola.ops.I_like(var) * self.jitter

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), var)

    def multi_gene_predict(
        self, test_inputs: Num[Array, "N D"], train_data: JaxP53Data
    ) -> GaussianDistribution:
        """
        Predict the gene expression values at test inputs given the training data.

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
        x, y, variances = dataset_3d(train_data)
        t = test_inputs

        t2 = t.at[:, 2].set(1)
        # t2 = t

        obs_noise = self.obs_stddev**2
        mean_x = self.mean_function(x)

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.gram(self.kernel, x)

        # Σ = Kxx + Io²
        Sigma = Kxx + diag_variances
        Sigma += cola.ops.I_like(Sigma) * obs_noise
        Sigma = cola.PSD(Sigma)

        mean_t = self.mean_function(t)
        Ktt = self.gram(self.kernel, t)
        Kxt = self.cross_covariance(self.kernel, x, t)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mean_x)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt
        var = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        var += cola.ops.I_like(var) * self.jitter
        var = cola.PSD(var)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), var)
