#!/usr/bin/env python
# coding: utf-8

# # GPJax implementation of LFM
#
# This notebook is an interactive version of `main.py`, in which the latent force model is implemented in `GPJax`.

# In[502]:


import jax
import gpjax as gpx
import jax.numpy as jnp
import optax as ox


from dataset import JaxP53Data, dataset_3d, load_barenco_data
from plotter import plot_lf, plot_comparison_gpjax
from model import ExactLFM
from objectives import CustomConjMLL
from trainer import JaxTrainer
from utils import GeneExpressionPredictor, print_hyperparams, generate_test_times

key = jax.random.PRNGKey(42)


# First, load the data from the correct repository. `JaxP53Data` log-normalises the gene expression data and the corresponding variances, additionally the replicate number can be specified (as the data collection was performed in triplicate). To perform an ablation study, the selected genes can be modified.

# In[505]:


# Load the data
selected_genes = ["DDB2", "BIK", "DR5", "p21", "SESN1"]
selected_gene_names = "(" + ", ".join(selected_genes) + ")"
num_genes = len(selected_genes) if selected_genes is not None else 5
p53_data = JaxP53Data(replicate=None, data_dir="data", selected_genes=selected_genes)


# Next, the GP model (which uses a custom kernel and mean function) is defined, as well as an optimiser and loss function.

# In[508]:


# Artificially augment the data to 3D
training_times, gene_expressions, variances = dataset_3d(p53_data)

# Convert data to GPJax dataset
dataset_train = gpx.Dataset(training_times, gene_expressions)

# Define the model (CHANGE NUMBER OF GENES MANUALLY)
custom_posterior = ExactLFM(jitter=jnp.array(1e-4), num_genes=len(selected_genes))

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


# The model is then trained for 150 epochs

# In[510]:


trained_model, training_history = trainer.fit(
    num_steps_per_epoch=1000, fix_params=False
)


# The `print_hyperparams` method can be used to retrieve the learned gene hyperparameters.

# In[511]:


print_hyperparams(trained_model, p53_data)


# Now, to test the model's ability to predict the latent function, test times must be generated from the `generate_test_times()` method (which defaults to 100 test points) and the `latent_predict()` method can be called on the trained model.

# In[513]:


testing_times = generate_test_times()
latent_dist = trained_model.latent_predict(testing_times, p53_data)


# To view the predicted latent function, use the `plot_lf()` function. For visualising this, set `save=False`.

# In[514]:


f = p53_data.f_observed.squeeze()
plot_lf(
    testing_times, latent_dist, y_scatter=f, stddev=2, save=False, title=f"Replicate 3"
)


# In[383]:


gene_predictor = GeneExpressionPredictor(trained_model, p53_data)
gene_predictor.plot_predictions(p53_data, save=False)


# In[374]:


plot_comparison_gpjax(trained_model, p53_data, save=False)
