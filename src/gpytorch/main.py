"""
This script is the entry point for calling the GPyTorch implementation of the SIMM latent force model.

| **Author:** William Purvis
| **Created:** 25/06/2024
| **Last updated:** 25/06/2024
"""

import torch
import numpy as np

from dataset import PyTorchDataset
from model import ExactLFM
from trainer import TorchTrainer
from plotter import plot_lf, plot_gxpred, plot_comparison_torch

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


if __name__ == "__main__":
    # Load the dataset
    dataset = PyTorchDataset(replicate=0, data_dir="data")

    # Define the model
    model = ExactLFM(dataset, dataset.gene_variances.reshape(-1))

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = ExactMarginalLogLikelihood(model.likelihood, model)

    # Define parameters to track
    track_parameters = [
        "mean_module.raw_basal",
        "covar_module.raw_decay",
        "covar_module.raw_sensitivity",
        "covar_module.raw_lengthscale",
    ]

    # Initialize the trainer
    trainer = TorchTrainer(
        model, [optimizer], dataset, loss_fn=loss_fn, track_parameters=track_parameters
    )

    # Train the model
    print("Training model...")
    model.likelihood.train()
    a = trainer.train(epochs=150, report_interval=25)

    # Make predictions
    print("Making predictions and plotting...")
    t_predict = torch.linspace(0, 13, 80, dtype=torch.float64)
    p_f = model.predict_f(t_predict, jitter=1e-3)
    p_m = model.predict_m(t_predict, jitter=1e-3)

    # Plot latent force
    plot_lf(p_f, t_predict, scatter=dataset.f_observed[0, 0])

    # Plot gene expression predictions
    plot_gxpred(p_m, t_predict, dataset)

    # Plot hyperparameter comparison
    plot_comparison_torch(model, dataset, trainer)
