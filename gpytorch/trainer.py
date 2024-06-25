"""
This module is adapted from the [Alfi: Approximate Latent Force Inference](https://github.com/mossjacob/alfi) package and implements a custom trainer for the SIMM latent force model using GPyTorch.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


class TorchTrainer:
    """
    A class for training models using the SIMM latent force model approach in a PyTorch environment.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizers : list of torch.optim.Optimizer
        Optimizers used for training the model.
    dataset : torch.utils.data.Dataset
        The complete dataset to be used for training, validation, and testing.
    loss_fn : callable
        A loss function that takes model outputs and targets and returns a loss.
    batch_size : int, optional
        Batch size for training, by default 1.
    valid_size : float, optional
        Proportion of the dataset to use for validation, by default 0.
    test_size : float, optional
        Proportion of the dataset to use for testing, by default 0.
    track_parameters : list of str, optional
        List of parameter names to track across training epochs.

    Attributes
    ----------
    parameter_trace : dict or None
        Dictionary containing parameter traces if `track_parameters` is not None.
    num_epochs : int
        Counter for the number of completed training epochs.
    losses : numpy.ndarray
        Array to store loss values over epochs.
    """

    def __init__(
        self,
        model,
        optimizers,
        dataset,
        loss_fn,
        batch_size=1,
        valid_size=0.0,
        test_size=0.0,
        track_parameters=None,
    ):
        self.model = model
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.batch_size = batch_size
        self.parameter_trace = None
        self.num_epochs = 0
        self.losses = np.empty(0)

        # Handle dataset splitting
        dataset_size = len(dataset)
        indices = np.random.permutation(dataset_size)
        valid_split = int(np.floor(valid_size * dataset_size))
        test_split = int(np.floor(test_size * dataset_size))

        self.valid_indices = indices[:valid_split]
        self.test_indices = indices[valid_split : test_split + valid_split]
        self.train_indices = indices[test_split + valid_split :]
        self.set_loaders()

        if track_parameters:
            self.parameter_trace = {
                key: [value.detach()]
                for key, value in model.named_parameters()
                if key in track_parameters
            }

    def set_loaders(self):
        subsets = {
            "valid": self.valid_indices,
            "test": self.test_indices,
            "train": self.train_indices,
        }
        for key, indices in subsets.items():
            data_subset = Subset(self.dataset, indices)
            setattr(
                self,
                f"{key}_loader",
                DataLoader(
                    data_subset, batch_size=self.batch_size, shuffle=(key == "train")
                ),
            )

    def train(self, epochs=20, report_interval=1, reporter_callback=None):
        """
        Trains the model for a specified number of epochs, optionally reporting progress at intervals.

        Parameters
        ----------
        epochs : int, optional
            The number of epochs to train the model for, default is 20.
        report_interval : int, optional
            Interval between progress reports on training, default is 1 epoch.
        reporter_callback : callable, optional
            A callback function to be called after each report interval with the epoch and loss.

        Returns
        -------
        list of float
            A list of loss values at the end of each epoch.

        Notes
        -----
        The training involves repeated calls to `single_epoch()` which handles the
        optimization steps. If `reporter_callback` is provided, it is invoked after
        each reporting interval with parameters such as current epoch and loss.
        """
        end_epoch = self.num_epochs + epochs

        self.model.train()
        losses = []
        for epoch in range(epochs):
            loss = self.single_epoch()

            # Print progress
            if epoch % report_interval == 0:
                print(
                    f"Epoch {self.num_epochs + 1:03d}/{self.num_epochs + epochs:03d} - Loss: {loss:.2f}",
                    end="",
                )
                print(
                    f" (lengthscale: {self.model.covar_module.lengthscale.item():.2f}, noise: {self.model.likelihood.noise.item():.2f})"
                )
            self.after_epoch()
            self.num_epochs += 1
            losses.append(loss)

        self.losses = losses
        return losses

    def single_epoch(self, epoch=0):
        """
        Executes a single training epoch, computes loss, and updates model parameters.

        Returns
        -------
        float
            The loss computed for the epoch.

        Notes
        -----
        This method conducts a forward pass using the model's current configuration,
        calculates the loss using the designated loss function, and performs a backward
        pass to update model parameters. Optimizers are used to update weights based on gradients.
        Each optimizer must be zeroed out before the backward pass to prevent gradient accumulation.
        """

        epoch_loss = 0
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        output = self.model(self.model.train_t)
        loss = -self.loss_fn(output, self.model.train_y.squeeze())
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        epoch_loss += loss.item()
        return epoch_loss

    def after_epoch(self):
        """
        Fix the sensitivity of p21 to 1 and update the parameter trace if enabled.
        """
        if self.parameter_trace is not None:
            params = dict(self.model.named_parameters())
            for key in params:
                if key in self.parameter_trace:
                    self.parameter_trace[key].append(params[key].detach().clone())

        with torch.no_grad():
            # Fix sensitivities of P21 to 1 (as in the paper)
            sens = self.model.covar_module.sensitivity
            sens[3] = np.float64(1.0)
            deca = self.model.covar_module.decay
            deca[3] = np.float64(0.8)
            self.model.covar_module.sensitivity = sens
            self.model.covar_module.decay = deca
