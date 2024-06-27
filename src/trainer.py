"""
This module implements a custom trainer for the SIMM latent force model using GPJax.
"""

import jax
import jax.numpy as jnp
import optax as ox
import jax.random as jr

from model import ExactLFM
from beartype.typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from jaxtyping import Int
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
from gpjax.scan import vscan
from gpjax.objectives import AbstractObjective
from gpjax.dataset import Dataset

ModuleModel = TypeVar("ModuleModel", bound=ExactLFM)

# Set the random seed for reproducibility
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)


class JaxTrainer:
    """
    Custom trainer for the SIMM latent force model using GPJax which aims to fix the sensitivity of the third data point to 1.0 and the decay to 0.8.

    Parameters
    ----------
    model : ModuleModel
        The model to train.
    objective : Union[AbstractObjective, Callable[[ModuleModel, Dataset], ScalarFloat]]
        The objective function to minimize, either as a predefined GPJax objective or a custom callable.
    training_data : Dataset
        The dataset used for training.
    optim : ox.GradientTransformation
        The optimizer from Optax.
    key : KeyArray
        The random key for JAX operations.
    num_iters : int
        The number of iterations for training.
    track_parameters : Optional[bool], optional
        Flag to track parameters during training, by default None.

    Attributes
    ----------
    history : list
        Stores the history of training losses or parameter values if tracking is enabled.
    """

    def __init__(
        self,
        model: ModuleModel,
        objective: Union[
            AbstractObjective, Callable[[ModuleModel, Dataset], ScalarFloat]
        ],
        training_data: Dataset,
        optim: ox.GradientTransformation,
        key: KeyArray,
        num_iters: Int,
        track_parameters: Optional[bool] = None,
    ):
        self.model = model.unconstrain()
        self.objective = objective
        self.training_data = training_data
        self.optim = optim
        self.key = key
        self.num_iters = num_iters
        self.track_parameters = (
            {key: [] for key in track_parameters} if track_parameters else None
        )
        self.history = []

    def loss(self, model: ModuleModel, batch: Dataset) -> ScalarFloat:
        """
        Calculates the loss for a given model and batch of data.

        Parameters
        ----------
        model : ModuleModel
            The model with constrained parameters.
        batch : Dataset
            A batch of data.

        Returns
        -------
        ScalarFloat
            The computed loss value.
        """
        model = model.stop_gradient()
        return self.objective(model.constrain(), batch)

    def step(self, carry: tuple, key: KeyArray, step_count: int) -> tuple:
        """
        Performs a single training step.

        Parameters
        ----------
        carry : tuple
            Contains the current model and optimizer state.
        key : KeyArray
            Random key for this step.
        step_count : int
            The current step number.

        Returns
        -------
        tuple
            Updated carry and loss value.
        """
        model, opt_state = carry
        batch = self.training_data

        loss_val, loss_gradient = jax.value_and_grad(self.loss)(model, batch)
        updates, opt_state = self.optim.update(loss_gradient, opt_state, model)
        model = ox.apply_updates(model, updates)

        carry = model, opt_state
        return carry, loss_val

    def after_epoch_jax(
        self, model: ModuleModel, fix_params: Optional[bool]
    ) -> ModuleModel:
        """
        Adjusts model parameters after each epoch (fixed sensitivity and decay for the third data point)

        Parameters
        ----------
        model : ModuleModel
            The current model state.
        fix_params : Optional[bool], optional
            Flag to fix parameters after each epoch, by default True. (sensitivity of p21 to 1 and decay to 0.8)

        Returns
        -------
        ModuleModel
            The updated model.
        """
        if fix_params:
            new_sensitivities = model.true_s.at[3].set(jnp.array(1, dtype=jnp.float64))
            new_decays = model.true_d.at[3].set(jnp.array(0.8, dtype=jnp.float64))
        else:
            new_sensitivities = model.true_s
            new_decays = model.true_d

        updated_model = model.replace(true_s=new_sensitivities, true_d=new_decays)

        return updated_model

    def fit(
        self,
        fix_params: Optional[bool] = True,
        num_steps_per_epoch: Optional[int] = 1000,
    ) -> tuple:
        r"""Train a Module model with respect to a supplied Objective function for a given number of iterations. Optimisers used here should originate from Optax.

        Example:

        .. code-block:: python

            >>> # Define the model, objective, training data, and optimizer
            >>> model = ExactLFM(kernel, t, y)
            >>> objective = MarginalLikelihood()
            >>> training_data = Dataset(t, y)
            >>> optim = optax.adam(1e-3)
            >>> key = jax.random.PRNGKey(42)
            >>> num_iters = 1000
            >>> track_parameters = ["s", "d"]
            >>> # Create a trainer instance
            >>> trainer = JaxTrainer(model, objective, training_data, optim, key, num_iters, track_parameters)
            >>> # Train the model
            >>> model, history, track_parameters = trainer.fit(num_steps_per_epoch=100)

        Parameters
        ----------
        fix_params : Optional[bool], optional
            Flag to fix parameters after each epoch, by default True. (sensitivity of p21 to 1 and decay to 0.8)
        num_steps_per_epoch : Optional[int], optional
            The number of steps per epoch, by default 1000.

        Returns
        -------
        tuple
            The trained model and history.
        """
        iter_keys = jr.split(self.key, self.num_iters)
        state = self.optim.init(self.model)

        def step_fn(carry, inputs):
            key, step_count = inputs
            carry, loss_val = self.step(carry, key, step_count)
            model, opt_state = carry
            model = jax.lax.cond(
                step_count % num_steps_per_epoch == 0,
                lambda model: self.after_epoch_jax(model, fix_params),
                lambda model: model,
                model,
            )
            carry = model, opt_state
            return carry, loss_val

        (model, _), history = vscan(
            step_fn, (self.model, state), (iter_keys, jnp.arange(self.num_iters))
        )

        model = model.constrain()
        if fix_params:
            self.model = self.after_epoch_jax(model, fix_params)
        else:
            self.model = model

        self.history = history
        if self.track_parameters:
            return self.model, self.history, self.track_parameters
        else:
            return self.model, self.history
