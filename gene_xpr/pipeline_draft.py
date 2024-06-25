import jax
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
from gpjax.base import param_field, static_field
from dataclasses import dataclass
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.distributions import GaussianDistribution

from gpjax.typing import (
    Array,
    ScalarFloat,
)
from jaxtyping import (
    Float,
    Num,
    Int,
)

import cola
from cola.ops import (
    Dense,
    LinearOperator,
)

from beartype.typing import (
    Union,
)
import beartype.typing as tp

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")

key = jr.PRNGKey(42)


# define model
@dataclass
class p53_model(gpx.base.Module):
    """
    Implementation of p53 model from Lawrence et al. 2006
    """

    # Define jitter (prior) and noise (likelihood)
    jitter: float = static_field(1e-6)
    obs_stddev: Union[ScalarFloat, Float[Array, "#N"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )

    ####### Define parameters

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

    # Restrict lengthscale to be between 0.5 and 3.5
    l_bijector = tfb.Sigmoid(low=0.5, high=3.5)

    initial_lengthscale = jnp.array(2.5, dtype=jnp.float64)

    l: Float[Array, " O"] = param_field(
        initial_lengthscale,
        bijector=l_bijector,
        metadata={"name": "lengthscale"},
        trainable=True,
    )

    initial_constrained_b = jnp.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=jnp.float64)

    true_b: Float[Array, "1 5"] = param_field(
        initial_constrained_b,
        bijector=tfb.Softplus(),
        metadata={"name": " basal_rates"},
        trainable=True,
    )

    # Define mean function
    def mean_function(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        f = jnp.array(x[:, 2:], dtype=int)

        num_genes = 5
        block_size = x.shape[0] // num_genes
        mean = (self.true_b / self.true_d).reshape(-1, 1)
        mean = mean.repeat(block_size, jnp.newaxis).reshape(-1, 1)

        return mean * f

    # Define kernel
    def kernel(
        self, t: Float[Array, "1 3"], t_prime: Float[Array, "1 3"]
    ) -> ScalarFloat:

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
        cross_cov = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(y))(x)

        return cross_cov

    def gram(self, kernel: Kernel, x: Float[Array, "N D"]) -> LinearOperator:
        Kxx = self.cross_covariance(kernel, x, x)

        return cola.PSD(Dense(Kxx))

    ############################
    # Define predict methods
    ############################

    def latent_predict(
        self, test_inputs: Num[Array, "N D"], train_data: JAXP53_Data
    ) -> GaussianDistribution:

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

    def single_gene_predict(
        self,
        test_inputs: Num[Array, "N D"],
        gene: Int[Array, " O"],
        train_data: JAXP53_Data,
    ) -> GaussianDistribution:
        x, y, variances = dataset_3d(train_data)
        t = test_inputs
        jitter = 1e-3

        t2 = t.at[:, 2].set(1)
        # t2 = t

        mean_x = self.mean_function(x)
        mean_t = self.mean_function(t2)

        # Slice traning data for given gene
        start_slice = (gene - 1) * 7
        end_slice = start_slice + 7

        x = x[start_slice:end_slice]
        y = y[start_slice:end_slice]
        variances = variances[start_slice:end_slice]
        mean_x = mean_x[start_slice:end_slice]
        # mean_t = mean_t[start_slice:end_slice]

        diag_variances = jnp.diag(variances.reshape(-1))

        Kxx = self.gram(self.kernel, x)
        Kxx += diag_variances
        Kxx += cola.ops.I_like(Kxx) * jitter
        K_inv = cola.inv(Kxx)

        Kxf = self.cross_covariance(self.kernel, x, t2)
        KfxKxx = jnp.matmul(Kxf.T, K_inv.to_dense())
        mean = mean_t + jnp.matmul(KfxKxx, y - mean_x)

        Kff = self.gram(self.kernel, t2)
        Kff += cola.ops.I_like(Kff) * jitter

        var = Kff - jnp.matmul(KfxKxx, Kxf)
        var = jnp.diag(jnp.diag(var.to_dense()))
        var += cola.ops.I_like(var) * jitter * 1e-4

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), var)