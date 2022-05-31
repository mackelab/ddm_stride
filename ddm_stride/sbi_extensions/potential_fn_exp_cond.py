from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import *

from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.inference.snle.mnle import MixedDensityEstimator
from sbi.types import TorchTransform
from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


def mnle_potential_exp_cond(
    likelihood_estimator: MixedDensityEstimator,
    prior: Any,
    x_o: Optional[Tensor],
    exp_cond: Optional[Tensor] = None,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns $\log(p(x_o|\theta, \pi)p(\theta))$ for mixed likelihood-based methods.
    It also returns a transformation that can be used to transform the potential into
    unconstrained space.
    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.
        exp_cond: The experimental condition $\pi$.
    Returns:
        The potential function $p(x_o|\theta, \pi)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.discrete_net.parameters()).device)

    potential_fn = MixedLikelihoodBasedPotentialExpCond(
        likelihood_estimator, prior, x_o, exp_cond, device=device
    )
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


class MixedLikelihoodBasedPotentialExpCond(LikelihoodBasedPotential):
    r"""Returns the potential function for the MNLE given observations and optionally experimental conditions.

    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.
        exp_cond: The experimental conditions corresponding to the observed data.
        device: The device to which parameters and data are moved before evaluating
            the `likelihood_nn`.

    Returns:
        The potential function $p(x_o|\theta, \pi)p(\theta)$.
    """

    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Any,
        x_o: Optional[Tensor],
        exp_cond: Optional[Tensor],
        device: str = "cpu",
    ):
        super().__init__(likelihood_estimator, prior, None, device)
        self.set_exp_cond(exp_cond)
        self.set_x(x_o)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:

        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
            theta=atleast_2d(theta), x=atleast_2d(self._x_o)
        )
        assert (
            x_repeated.shape[0] == theta_repeated.shape[0]
        ), "x and theta must match in batch shape."

        # Repeat the experimental condition in the same way as the observations
        if self._exp_cond is not None:
            # If only one experimental condition is passed, repeat it for each observation
            if self._exp_cond.shape[0] == 1:
                exp_cond = self._exp_cond.repeat(self._x_o.shape[0], 1)
            else:
                exp_cond = self._exp_cond
            _, exp_cond_repeated = match_theta_and_x_batch_shapes(
                theta=atleast_2d(theta), x=atleast_2d(exp_cond)
            )
            assert (
                x_repeated.shape[0] == exp_cond_repeated.shape[0]
            ), "x and exp cond must match in batch shape."
            # Concatenate experimental condition to theta
            input = torch.cat((theta_repeated, exp_cond_repeated), dim=1)
        else:
            input = theta_repeated

        with torch.set_grad_enabled(track_gradients):

            # Call the log prob method of the mixed likelihood estimator.
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                x_repeated.to(self.device), input.to(self.device)
            )
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self._x_o.shape[0], -1
            ).sum(0)
            # print('llh', log_likelihood_trial_sum, log_likelihood_trial_sum.shape)

        return log_likelihood_trial_sum + self.prior.log_prob(theta)

    def set_exp_cond(self, exp_cond: Optional[Tensor]):
        """Check the shape of the experimental condition and, if valid, set it."""
        if exp_cond is not None and exp_cond.shape[-1] > 0:
            self._exp_cond = atleast_2d(exp_cond)
        else:
            self._exp_cond = None
