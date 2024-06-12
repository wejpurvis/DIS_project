"""
Script to run GPR for the prediction of the latent function.
"""

# Importing libraries
import jax
import gpjax as gpx
import jax.numpy as jnp
from p53_data import JAXP53_Data, dataset_3d, generate_test_times
from kernels import latent_kernel
from custom_gps import p53_posterior
from plotter import plot_gp
import optax as ox
import jax.random as jr

jax.config.update("jax_enable_x64", True)
key = jr.PRNGKey(42)


# Function definitions
def initialise_gp(kernel, mean, dataset):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel, jitter=1e-4)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array([1.0e-3], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior


def optimise_mll(posterior, dataset, NIters=1000, key=key):
    # define the MLL using dataset_train
    objective = gpx.objectives.ConjugateMLL(negative=True)
    print(f"MLL before opt: {objective(posterior, dataset):.3f}")
    # Optimise to minimise the MLL
    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=objective,
        train_data=dataset,
    )
    return opt_posterior, history


if __name__ == "__main__":

    p53_data = JAXP53_Data(replicate=0)
    training_times, gene_expressions, variances = dataset_3d(p53_data)

    dataset_train = gpx.Dataset(training_times, gene_expressions)

    testing_times = generate_test_times()

    meanf = gpx.mean_functions.Zero()
    p53_ker = latent_kernel()

    # Obtain posterior distribution
    posterior = initialise_gp(p53_ker, meanf, dataset_train)

    # Optimisise MLL for GPR to obtain optimised posterior
    opt_post, history = optimise_mll(posterior, dataset_train)

    # Predict latent function using custom posterior and prediction method
    p53_post = p53_posterior(prior=opt_post.prior, likelihood=opt_post.likelihood)
    latent_dist = p53_post.latent_predict(testing_times, p53_data)

    predictive_dist = p53_post.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    # Plot the GP
    f = p53_data.f_observed.squeeze()
    plot_gp(testing_times, predictive_dist, y_scatter=f)
