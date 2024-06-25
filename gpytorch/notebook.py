#!/usr/bin/env python
# coding: utf-8

# ## Pytorch implementation of LFM model
#
# This notebook is an interactive version of `main.py`, in which the latent force model is implemented in GPyTorch. The code is adapted from from the [Alfi: Approximate Latent Force Inference](https://github.com/mossjacob/alfi) package.

# In[1]:


import torch
import numpy as np

from dataset import PyTorchDataset
from model import ExactLFM
from trainer import TorchTrainer
from plotter import plot_lf, plot_gxpred, plot_comparison_torch

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


# First, load the data from the correct repository. `PyTorchDataset` log-normalises the gene expression data and the corresponding variances, additionally the replicate number can be specified (as the data collection was performed in triplicate).

# In[2]:


dataset = PyTorchDataset(replicate=0, data_dir="../data/")


# Next, the GP model (which uses a custom kernel and mean function) is defined, as well as an optimiser and loss function. In order to retrieve the learned hyperparameters, a dictionary for the parameters of interest must be defined.

# In[3]:


model = ExactLFM(dataset, dataset.gene_variances.reshape(-1))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = ExactMarginalLogLikelihood(model.likelihood, model)

track_parameters = [
    "mean_module.raw_basal",
    "covar_module.raw_decay",
    "covar_module.raw_sensitivity",
    "covar_module.raw_lengthscale",
]


trainer = TorchTrainer(
    model, [optimizer], dataset, loss_fn=loss_fn, track_parameters=track_parameters
)


# The model is trained for 150 epochs, and the loss is printed every 25 epochs

# In[4]:


model.likelihood.train()
a = trainer.train(epochs=150, report_interval=25)


# Once the model is trained, predictions for both gene expressions and the latent force function can be made. The `jitter` parameter can be modified to change the resulting output.

# In[5]:


t_predict = torch.linspace(0, 13, 80, dtype=torch.float64)
p_f = model.predict_f(t_predict, jitter=1e-2)
p_m = model.predict_m(t_predict, jitter=1e-3)


# In[21]:


plot_lf(p_f, t_predict, stddev=2, scatter=dataset.f_observed[0, 0], save=False)


# In[8]:


plot_gxpred(p_m, t_predict, dataset, save=False)


# Finally, the optimised parameters of the GP model are compared with the *ground truth* parameters measured experimentally by Barenco et al. (2006).

# In[14]:


plot_comparison_torch(model, dataset, trainer, save=False)
