"""
Plotting functions for the project
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
from matplotlib import rcParams

plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)

# Check if LaTeX is in notebook path
if os.environ.get("PATH") is not None:
    if "TeX" not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

colors = rcParams["axes.prop_cycle"].by_key()["color"]

# TODO: Add more plotting functions, edit docstrings


def plot_gp(x_test, predictive_dist, y_scatter=None):
    """
    Plot LF GP given test points and prediction

    Parameters
    ----------
    x_test: array of shape (n, 3)
    predictive_dist:  jax mvn
    y_scatter: array of shape (n,)
    """

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.variance()

    # (100,)
    x_test = x_test[:, 0]

    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7.5, 2.5))

    ax.fill_between(
        x_test.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.2,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(
        x_test,
        predictive_mean - 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(
        x_test,
        predictive_mean + 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )

    ax.plot(x_test, predictive_mean, label="Predictive mean", color=cols[1])

    if y_scatter is not None:
        times = jnp.linspace(0, 12, len(y_scatter))
        ax.plot(times, y_scatter, "x", label="True values", color=cols[0])

    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA Expression")

    plt.show()
