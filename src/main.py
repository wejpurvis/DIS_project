"""
This script is the entry point for calling the GPJax implementation of the SIMM latent force model.

| **Author:** William Purvis
| **Created:** 25/06/2024
| **Last updated:** 25/06/2024
"""

import jax
import gpjax as gpx
import jax.numpy as jnp
import optax as ox


from dataset import JaxP53Data, dataset_3d, generate_test_times
from plotter import plot_lf, plot_comparison_gpjax
from model import ExactLFM
from objectives import CustomConjMLL
from trainer import JaxTrainer
from utils import print_hyperparams

key = jax.random.PRNGKey(42)


if __name__ == "__main__":
    # Load the data
    p53_data = JaxP53Data(replicate=0, data_dir="data")

    # Artificially augment the data to 3D
    training_times, gene_expressions, variances = dataset_3d(p53_data)

    # Convert data to GPJax dataset
    dataset_train = gpx.Dataset(training_times, gene_expressions)

    # Define the model
    custom_posterior = ExactLFM(jitter=jnp.array(1e-4))

    # Define the optimiser and loss function
    loss = CustomConjMLL(negative=True)
    optimiser = ox.adam(0.01)

    # Initialise the trainer
    trainer = JaxTrainer(
        model=custom_posterior,
        objective=loss,
        training_data=dataset_train,
        optim=optimiser,
        key=key,
        num_iters=150,
    )

    # Train the model
    print("Training model...")
    trained_model, training_history = trainer.fit(num_steps_per_epoch=1000)

    # Print the learned hyperparameters
    print_hyperparams(trained_model, p53_data)

    # Make predictions
    print("Making predictions and plotting...")
    testing_times = generate_test_times()
    latent_dist = trained_model.latent_predict(testing_times, p53_data)
    # TODO: Add gene expression predictions

    # Plot latent force
    f = p53_data.f_observed.squeeze()
    plot_lf(testing_times, latent_dist, y_scatter=f, stddev=2)

    # Plot gene expression predictions
    # TODO

    # Plot hyperparameter comparison
    plot_comparison_gpjax(trained_model, p53_data)
