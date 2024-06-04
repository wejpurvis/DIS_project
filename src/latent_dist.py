"""
Python implementation of work carried out in Jupyter notebooks in order to use pytest
"""

# Importing libraries
import pytest
import jax
import gpjax as gpx
import jax.numpy as jnp
from p53_data import JAXP53_Data, dataset_3d, generate_test_times
from kernels import latent_kernel
import optax as ox
import jax.random as jr

jax.config.update("jax_enable_x64", True)
key = jr.PRNGKey(42)


# Function definitions
def initialise_gp(kernel, mean, dataset):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array([1.0e-3], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior


def optimise_mll2(posterior, dataset, NIters=1000, key=key):
    # define the MLL using dataset_train
    objective = gpx.objectives.ConjugateMLL(negative=True)
    print(f"MLL before opt: {objective(posterior, dataset):.3f}")
    # Optimise to minimise the MLL
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=ox.adam(1e-1),
        num_iters=NIters,
        key=key,
        safe=False,
    )
    return opt_posterior, history


if __name__ == "__main__":

    p53_data = JAXP53_Data(replicate=0)
    training_times, gene_expressions, variances = dataset_3d(p53_data)

    dataset_train = gpx.Dataset(training_times, gene_expressions)

    testing_times = generate_test_times()

    meanf = gpx.mean_functions.Zero()
    p53_ker = latent_kernel()

    p53_ker.true_s

    # Obtain posterior distribution
    posterior = initialise_gp(p53_ker, meanf, dataset_train)

    # Optimisise MLL for GPR to obtain optimised posterior
    opt_posterior, history = optimise_mll2(posterior, dataset_train)

    # Predict latent function
    latent_dist = opt_posterior.predict(testing_times, train_data=dataset_train)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
