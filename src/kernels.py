import jax
import gpjax as gpx
from dataclasses import dataclass

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import param_field
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from jaxtyping import Int

jax.config.update("jax_enable_x64", True)


@dataclass
class latent_kernel(gpx.kernels.AbstractKernel):
    r"Combined kernel for latent force and gene expression predictions."

    name: str = "p53 kernel"

    ###############################
    # Define trainable parameters #
    ###############################

    # Sensitivities of the genes
    initial_sensitivities = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

    true_s: Float[Array, "1 5"] = param_field(
        initial_sensitivities,
        bijector=tfb.Softplus(),
        metadata={"name": " kxx_sensitivities"},
        trainable=True,
    )

    # Degradation rates of the genes
    initial_decays = jnp.array([0.4, 0.4, 0.4, 0.4, 0.4], dtype=jnp.float64)

    true_d: Float[Array, "1 5"] = param_field(
        initial_decays,
        bijector=tfb.Softplus(),
        metadata={"name": " kxx_degradations"},
        trainable=True,
    )

    # Lengthscale constrained to be between 0.5 and 3.5
    l_bijector = tfb.Sigmoid(low=0.5, high=3.5)

    initial_lengthscale = jnp.array(2.5, dtype=jnp.float64)

    l: Float[Array, " O"] = param_field(
        initial_lengthscale, bijector=l_bijector, metadata={"name": "lengthscale"}
    )

    def __call__(self, t: Int[Array, "1 3"], t_prime: Int[Array, "1 3"]) -> ScalarFloat:
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
        self, t: Int[Array, "1 3"], t_prime: Int[Array, "1 3"]
    ) -> ScalarFloat:
        r"""
        Calculates the covariance between the expression leves of genes j and k at times t and t'. Equation 5 in Lawrence et al. (2006):

        ```math
        k_{x_{j}x_{k}}(t,t') = S_{j}S_{k}\frac{\sqrt{\pi}l}{2}[h_{kj}(t,t') + h_{kj}(t',t)]
        ```
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

        """
        # Error trap (JAX friendly)
        def check_validity(condition):
            if condition:
                # raise ValueError("t or t' cannot be testing points (z=0)")
                return 0

        condition = jnp.logical_or(t[2] == 0, t_prime[2] == 0)
        jax.debug.callback(check_validity, condition)
        """

        # Get gene indices
        j = t[1].astype(int)
        k = t_prime[1].astype(int)
        # try jnp.array(X[...,[1]], dtype=int)

        t = t[0]
        t_prime = t_prime[0]

        # Equation 5
        mult = self.true_s[j] * self.true_s[k] * self.l * jnp.sqrt(jnp.pi) * 0.5
        second_term = self.h(k, j, t_prime, t) + self.h(j, k, t, t_prime)

        kxx = mult * second_term

        return kxx.squeeze()

    def kernel_xf(
        self, t: Int[Array, "1 3"], t_prime: Int[Array, "1 3"]
    ) -> ScalarFloat:
        r"""
        Cross covariance term required to infer the latent force from gene expression data. Equation 6 in Lawrence et al. (2006):

        ```math
        k_{x_{j}f}(t,t') = \frac{S_{rj}\sqrt{\pi}l}{2} \textrm{exp}(\gamma_{j})^{2}\textrm{exp}(-D_{j}(t'-t)) \biggl[ \textrm{erf} \left (\frac{t'-t}{l} - \gamma_{k}\right ) + \textrm{erf} \left (\frac{t}{l} +\gamma_{k} \right ) \biggr]
        ```

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
        # Get gene expression and latent force from flag (kxf anf kfx are transposes)
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
        self, t: Int[Array, "1 3"], t_prime: Int[Array, "1 3"]
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
        j: Int[Array, " O"],
        k: Int[Array, " O"],
        t1: Int[Array, " O"],
        t2: Int[Array, " O"],
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

        return result

    def gamma(self, k: Int[Array, " O"]) -> ScalarFloat:
        # Gamma term for h function
        return (self.true_d[k] * self.l) / 2
