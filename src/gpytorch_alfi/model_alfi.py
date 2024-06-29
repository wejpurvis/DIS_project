"""
This module defines the kernel and mean functions for the SIMM latent force model, as well as the model itself.
"""

import torch
import gpytorch
import numpy as np

from gpytorch.constraints import Positive, Interval
from torch.utils.data import Dataset
from beartype.typing import Optional

PI = torch.tensor(np.pi, requires_grad=False)


class ExactLFM(gpytorch.models.ExactGP):
    """
    GPyTorch implementation of the single input motif from Lawrence et al., 2006. Refactored from the Alfi package.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing gene expression data for project
    variance : np.array
        Array of gene expression variances

    Attributes
    ----------
    num_outputs : int
        Number of outputs (genes)
    train_t : torch.Tensor
        Training timepoints
    train_y : torch.Tensor
        Training gene expressions
    covar_module : TorchKernel
        Kernel function
    mean_module : TorchMean
        Mean function
    """

    def __init__(self, dataset: Dataset, variance: torch.Tensor):
        train_t, train_y = flatten_dataset(dataset)
        super().__init__(
            train_t, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
        )

        self.num_outputs = dataset.num_outputs
        self.train_t = train_t.view(-1, 1)
        self.train_y = train_y.view(-1, 1)
        self.covar_module = TorchKernel(
            self.num_outputs, torch.tensor(variance, requires_grad=False)
        )
        self.mean_module = TorchMean(self.covar_module, self.num_outputs)

    @property
    def decay_rate(self):
        return self.covar_module.decay

    @decay_rate.setter
    def decay_rate(self, val):
        self.covar_module.decay = val

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_m(
        self, pred_t: torch.Tensor, jitter: Optional[float] = 1e-5
    ) -> torch.distributions.MultivariateNormal:
        """
        Predict the gene expression levels.

        Parameters
        ----------
        pred_t: torch.Tensor
            Prediction times
        jitter: float
            Jitter value for numerical stability, default is 1e-5

        Returns
        -------
        torch.distributions.MultivariateNormal
            Multivariate normal distribution of the predicted gene expression levels
        """

        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())
        pred_t_blocked = pred_t.repeat(self.num_outputs)

        Kxt = self.covar_module(self.train_t, pred_t_blocked).evaluate()

        Ktx = torch.transpose(Kxt, 0, 1).type(torch.float64)
        KtxK_inv = torch.matmul(Ktx, K_inv)
        KtxK_invY = torch.matmul(KtxK_inv, self.train_y)
        mean = KtxK_invY.view(self.num_outputs, pred_t.shape[0])

        Ktt = self.covar_module(pred_t_blocked, pred_t_blocked).evaluate()

        var = Ktt - torch.matmul(KtxK_inv, torch.transpose(Ktx, 0, 1))
        var = torch.diagonal(var, dim1=0, dim2=1)
        var = var.view(self.num_outputs, pred_t.shape[0])

        mean = mean.transpose(0, 1)
        var = var.transpose(0, 1)
        var1 = var
        var = torch.diag_embed(var)
        var += jitter * torch.eye(var.shape[-1])

        return torch.distributions.MultivariateNormal(mean, var)

    def predict_f(
        self, pred_t: torch.Tensor, jitter: Optional[float] = 1e-3
    ) -> torch.distributions.MultivariateNormal:
        """
        Predict the latent force function.

        Parameters
        ----------
        pred_t: torch.Tensor
            Prediction times
        jitter: float
            Jitter value for numerical stability, default is 1e-3

        Returns
        -------
        MultivariateNormal
            Multivariate normal distribution of the predicted latent force function
        """
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())

        Kxf = self.covar_module.K_xf(self.train_t, pred_t).type(torch.float64)
        KfxKxx = torch.matmul(torch.transpose(Kxf, 0, 1), K_inv)
        mean = torch.matmul(KfxKxx, self.train_y).view(-1).unsqueeze(0)

        # Kff-KfxKxxKxf
        Kff = self.covar_module.K_ff(pred_t, pred_t)  # (100, 500)
        var = Kff - torch.matmul(KfxKxx, Kxf)

        var = var.unsqueeze(0)
        # Full covariance matrix is not PSD (take diagonal)
        var = torch.diagonal(var, dim1=1, dim2=2)
        var = torch.diag_embed(var)
        var += jitter * torch.eye(var.shape[-1])

        batch_mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            batch_mvn, task_dim=0
        )


class TorchKernel(gpytorch.kernels.Kernel):
    """
    Combined kernel for latent force and gene expression predictions

    Parameters
    ----------
    num_genes : int
        Number of genes
    variance : np.array
        Array of gene expression variances
    dtype : torch.dtype
        Data type for the kernel parameters

    Attributes
    ----------
    num_genes : int
        Number of genes
    pos_constraint : gpytorch.constraints.Constraint
        Constraint for positive values
    lengthscale_constraint : gpytorch.constraints.Constraint
        Constraint for lengthscale values
    raw_lengthscale : torch.Tensor
        Lengthscale parameter (for RBF kernel)
    raw_decay : torch.Tensor
        Decay parameter
    raw_sensitivity : torch.Tensor
        Sensitivity parameter
    variance : torch.Tensor
        Array of gene expression variances
    """

    # Whether kernel is stationary
    is_stationary = True

    def __init__(
        self,
        num_genes: int,
        variance: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_genes = num_genes
        self.pos_constraint = Positive()
        self.lengthscale_constraint = Interval(0.5, 3.5)

        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(
                self.lengthscale_constraint.inverse_transform(
                    2.5 * torch.ones(1, 1, dtype=dtype)
                )
            ),
        )
        self.register_parameter(
            name="raw_decay",
            parameter=torch.nn.Parameter(
                self.pos_constraint.inverse_transform(
                    0.4 * torch.ones(self.num_genes, dtype=dtype)
                )
            ),
        )
        self.register_parameter(
            name="raw_sensitivity",
            parameter=torch.nn.Parameter(
                self.pos_constraint.inverse_transform(
                    1 * torch.ones(self.num_genes, dtype=dtype)
                )
            ),
        )

        # register the constraints
        self.register_constraint("raw_lengthscale", self.lengthscale_constraint)
        self.register_constraint("raw_sensitivity", Positive())
        self.register_constraint("raw_decay", Positive())

        self.variance = torch.diag(variance)

    @property
    def lengthscale(self):
        """
        Lengthscale parameter after applying constraint (for RBF kernel)
        """
        return self.lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value: torch.Tensor):
        self.initialize(
            raw_lengthscale=self.lengthscale_constraint.inverse_transform(value)
        )

    @property
    def decay(self):
        """
        Decay parameter after applying constraint (positive)
        """
        return self.pos_constraint.transform(self.raw_decay)

    @decay.setter
    def decay(self, value: torch.Tensor):
        self.initialize(raw_decay=self.pos_constraint.inverse_transform(value))

    @property
    def sensitivity(self):
        """
        Sensitivity parameter after applying constraint (positive)
        """
        return self.pos_constraint.transform(self.raw_sensitivity)

    @sensitivity.setter
    def sensitivity(self, value: torch.Tensor):
        self.initialize(raw_sensitivity=self.pos_constraint.inverse_transform(value))

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, **params) -> torch.Tensor:
        """
        Calculates the covariance matrix for the expression leves of genes j and k at times t and t'.

        Parameters
        ----------
        t1: tensor shape (T1,)
            Timepoints for the first gene
        t2: tensor shape (T2,)
            Timepoints for the second gene

        Returns
        -------
        K_xx: tensor shape (T1, T2)
            Covariance matrix for the expression levels of genes j and k across all timepoints
        """
        vert_block_size = int(t1.shape[0] / self.num_genes)
        hori_block_size = int(t2.shape[0] / self.num_genes)
        t1_block, t2_block = t1[:vert_block_size], t2[:hori_block_size]
        shape = [vert_block_size * self.num_genes, hori_block_size * self.num_genes]
        K_xx = torch.zeros(shape, dtype=torch.float64)
        for j in range(self.num_genes):
            for k in range(self.num_genes):
                kxx = self.k_xx(j, k, t1_block, t2_block)
                K_xx[
                    j * vert_block_size : (j + 1) * vert_block_size,
                    k * hori_block_size : (k + 1) * hori_block_size,
                ] = kxx

        if hori_block_size == vert_block_size:
            jitter = 1e-4 * torch.eye(K_xx.shape[0], dtype=torch.float64)
            K_xx += jitter
            if K_xx.shape[0] == self.variance.shape[0]:
                K_xx += self.variance
        return K_xx

    def k_xx(
        self, j: int, k: int, t1_block: torch.Tensor, t2_block: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Calculates the covariance between the expression leves of genes j and k at times t and t'. Equation 5 in Lawrence et al. (2006):

        ```math
        k_{x_{j}x_{k}}(t,t') = S_{j}S_{k}\frac{\sqrt{\pi}l}{2}[h_{kj}(t,t') + h_{kj}(t',t)]
        ```
        Parameters
        ----------
        j: int
            Gene index of first gene
        k: int
            Gene index of second gene
        t1_block: tensor shape (7,) (i.e., number of timepoints)
        t2_block: tensor shape (7,) (i.e., number of timepoints)

        Returns
        -------
        k_xx: tensor shape (7,7)
            The covariance between the expression levels of genes j and k across all timepoints
        """
        t1_block = t1_block.view(-1, 1)
        t2_block = t2_block.view(1, -1)

        mult = (
            self.sensitivity[j]
            * self.sensitivity[k]
            * self.lengthscale
            * 0.5
            * torch.sqrt(PI)
        )
        second_term = self.h(k, j, t2_block, t1_block) + self.h(
            j, k, t1_block, t2_block
        )

        k_xx = mult * second_term

        return k_xx

    def h(self, k: int, j: int, t2: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """
        Analytical solution for the convolution of the exponential kernel with a step function.

        Parameters
        ----------
        k: int
            Gene index of first gene
        j: int
            Gene index of second gene
        t2: tensor shape (1,7)
            Timepoints
        t1: tensor shape (1,7)
            Timepoints

        Returns
        -------
        result: tensor shape (1,7)
            Convolution of the exponential kernel with a step function
        """

        l = self.lengthscale
        t_dist = t2 - t1
        multiplier = torch.exp(self.gamma(k) ** 2) / (self.decay[j] + self.decay[k])
        first_erf_term = torch.erf(t_dist / l - self.gamma(k)) + torch.erf(
            t1 / l + self.gamma(k)
        )
        second_erf_term = torch.erf(t2 / l - self.gamma(k)) + torch.erf(self.gamma(k))
        result = multiplier * (
            torch.multiply(torch.exp(-self.decay[k] * t_dist), first_erf_term)
            - torch.multiply(
                torch.exp(-self.decay[k] * t2 - self.decay[j] * t1), second_erf_term
            )
        )

        return result

    def gamma(self, k: int) -> torch.Tensor:
        # Gamma term for h function
        return self.decay[k] * self.lengthscale / 2

    def K_xf(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Compute cross covariance matrix to infer the latent force (using kernel function k_xf).

        Parameters
        ----------
        x: tensor shape (35, 1)
            Gene expression data
        f: tensor shape (n,)
            n is the number of test points for the latent force

        Returns
        -------
        K_xf: tensor shape (35, n)
            Cross covariance between gene expression and latent force
        """

        shape = [x.shape[0], f.shape[0]]
        K_xf = torch.zeros(shape, dtype=torch.float32)
        self.block_size = int(x.shape[0] / self.num_genes)  # 7
        t1_block, t2_block = x[: self.block_size].view(-1, 1), f.view(1, -1)
        for j in range(self.num_genes):
            kxf = self.k_xf(j, t1_block, t2_block)
            K_xf[j * self.block_size : (j + 1) * self.block_size] = kxf

        # save k_xf as global variable
        self.k_xf = K_xf

        return K_xf

    def k_xf(self, j: int, x: torch.Tensor, t_f: torch.Tensor) -> torch.Tensor:
        r"""
        Cross covariance term required to infer the latent force from gene expression data. Equation 6 in Lawrence et al. (2006):

        ```math
        k_{x_{j}f}(t,t') = \frac{S_{rj}\sqrt{\pi}l}{2} \textrm{exp}(\gamma_{j})^{2}\textrm{exp}(-D_{j}(t'-t)) \biggl[ \textrm{erf} \left (\frac{t'-t}{l} - \gamma_{k}\right ) + \textrm{erf} \left (\frac{t}{l} +\gamma_{k} \right ) \biggr]
        ```

        Parameters
        ----------
        j: int
            Gene index
        x: tensor shape (7,1)
            Gene expression data
        t_f: tensor shape (1,n)
            Latent force test points

        Returns
        -------
        kxf: tensor shape (7,n)
            Cross covariance between the expression levels of gene j and the latent force at times t and t'
        """
        l = self.lengthscale
        t_dist = x - t_f

        first_term = 0.5 * l * torch.sqrt(PI) * self.sensitivity[j]
        first_expon_term = torch.exp(self.gamma(j) ** 2)
        second_expon_term = torch.exp(-self.decay[j] * t_dist)
        erf_terms = torch.erf((t_dist / l) - self.gamma(j)) + torch.erf(
            t_f / l + self.gamma(j)
        )

        kxf = first_term * first_expon_term * second_expon_term * erf_terms

        return kxf

    def K_ff(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Custom RBF kernel taken as the process prior over f(t) (enables analaytical solution for kxf anf kxx).

        Parameters
        ----------
        x1: tensor shape (n,1)
            Latent force test points
        x2: tensor shape (n,1)
            Latent force test points

        Returns
        -------
        K_ff: tensor shape (n,n)
            Covariance matrix for the latent force at times t and t'
        """
        x1 = x1.view(-1)
        x2 = x2.view(-1)

        sq_dist = torch.square(x1.view(-1, 1) - x2)
        sq_dist = torch.div(sq_dist, 2 * self.lengthscale.view((-1, 1)))
        K = torch.exp(-sq_dist)
        if K.shape[0] == K.shape[1]:
            jitter = 1e-3 * torch.eye(x1.shape[0])
            K += jitter

        return K


class TorchMean(gpytorch.means.Mean):
    r"""
    Simple Input Motif mean function

    .. math::
        f(x_{j}) = \frac{B_{j}}{D_{j}}

    From equation 2 in paper.

    :math:`B_{j}` represents the basal rate for gene :math:`j` and is a trainable parameter.

    Parameters
    ----------
    covar_module : gpytorch.kernels.Kernel
        Kernel function
    num_genes : int
        Number of genes

    Attributes
    ----------
    covar_module : gpytorch.kernels.Kernel
        Kernel function
    pos_contraint : gpytorch.constraints.Constraint
        Constraint for positive values
    num_genes : int
        Number of genes
    raw_basal : torch.Tensor
        Basal gene expression level before applying constraint

    """

    def __init__(self, covar_module: gpytorch.kernels.Kernel, num_genes: int):
        super().__init__()
        self.covar_module = covar_module
        self.pos_contraint = Positive()
        self.covar_module = covar_module
        self.num_genes = num_genes

        self.register_parameter(
            name="raw_basal",
            parameter=torch.nn.Parameter(
                self.pos_contraint.inverse_transform(0.05 * torch.ones(self.num_genes))
            ),
        )
        self.register_constraint("raw_basal", self.pos_contraint)

    @property
    def basal(self):
        """
        Basal gene expression level
        """
        return self.pos_contraint.transform(self.raw_basal)

    @basal.setter
    def basal(self, value: torch.Tensor):
        self.initialize(raw_basal=self.pos_contraint.inverse_transform(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_size = int(x.shape[0] / self.num_genes)
        # Define mean function
        mean = (self.basal / self.covar_module.decay).view(-1, 1)
        mean = mean.repeat(1, block_size).view(-1)

        return mean


def flatten_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens the given dataset into a format suitable for training.

    Parameters
    ----------
    dataset: LFMDataset
        Dataset containing gene expression data

    Returns
    -------
    train_t: torch.Tensor
        Training timepoints (shape (num_genes * num_timepoints, 1))
    train_y: torch.Tensor
        Training gene expressions (shape (num_genes * num_timepoints, 1))
    """
    num_genes = dataset.num_outputs
    train_t = dataset[0][0]
    num_times = train_t.shape[0]
    m_observed = torch.stack([dataset[i][1] for i in range(num_genes)]).view(
        num_genes, num_times
    )
    train_t = train_t.repeat(num_genes)
    train_y = m_observed.view(-1)
    return train_t, train_y
