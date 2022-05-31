from typing import Any, Tuple

import torch
from torch import Tensor
from torch.distributions import *

from ddm_stride.ddm_models.base_simulator import Simulator
from sbi.simulators.simutils import simulate_in_batches


def simulate_for_sbi_iid(
    simulator: Simulator,
    proposal: Any,
    num_params: int,
    num_iid_simulations_per_param: int,
    num_workers: int = 1,
    simulation_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.
    This function performs two steps:
    - Sample parameters $\theta$ from the `proposal`.
    - Simulate multiple times from these parameters to obtain $x$.

    Parameters
    ----------
    simulator:
        A simulator takes parameters $\theta$ and inputs $\pi$ and maps them to simulations, or observations,
        `x`, $\text{sim}(\theta, \pi)\to x$.
    proposal:
        Probability distribution that the parameters $\theta$ are sampledfrom.
    num_params:
        Number of parameters sampled from the prior.
    num_iid_simulations_per_param:
        Number of observations sampled per parameter.
    num_workers:
        Number of parallel workers to use for simulations.
    simulation_batch_size:
        Number of parameter sets that the simulator maps to data x at once. If None, we simulate all
        parameter sets at the same time. If >= 1, the simulator has to process data of shape
        (simulation_batch_size, parameter_dimension).
    show_progress_bar:
        Whether to show a progress bar for simulating. This will not affect whether there will be a
        progressbar while drawing samples from the proposal.

    Returns
    -------
    theta:
        Sampled parameters $\theta$
    x:
        simulation-outputs $x$.
    """

    theta = proposal.sample((num_params,))
    theta = theta.repeat(num_iid_simulations_per_param, 1)

    x = simulate_in_batches(
        simulator, theta, simulation_batch_size, num_workers, show_progress_bar
    )

    return theta, x
