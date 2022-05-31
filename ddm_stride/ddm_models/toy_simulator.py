from typing import Dict, List, Union

import numpy as np
import torch
from pyro.distributions import InverseGamma
from torch.distributions.binomial import Binomial

from ddm_stride.ddm_models.base_simulator import Simulator
from sbi.utils.torchutils import atleast_2d_float32_tensor


class MixedSimulator(Simulator):
    """Build the mixed simulator used in
    https://github.com/mackelab/sbi/blob/main/tutorials/14_SBI_with_trial-based_mixed_data.ipynb.

    Attributes
    ----------
    inputs: List[str]
        Specifies inputs passed to the simulator functions.
        Corresponds to parameters and input values specified in the config simulation file.
    simulator_results: List[str]
        Specifies the structure of the simulator output.
        Corresponds to the observations specified in the config simulation file.

    Methods
    -------
    __init__(self, inputs: List[str], simulator_results: List[str], **kwargs) -> None
        Use the config file to initialize attributes.
    __call__(self, input_data: torch.Tensor) -> torch.Tensor
        Use the given batch of input_data to simulate data.
    generate_data(self, input_dict: Dict(torch.Tensor)) -> Union[torch.Tensor, np.ndarray]
        Use the given input_dict sample to simulate one simulation result.
    """

    def __init__(
        self, inputs: List[str], simulator_results: List[str], **kwargs
    ) -> None:
        super().__init__(inputs, simulator_results)

    def generate_data(self, input_dict: Dict) -> Union[torch.Tensor, np.ndarray]:
        """Use the given input_dict sample to simulate one simulation result.

        Parameters
        ----------
        input_dict
            The current sample of parameters and experimental conditions to be used for simulating data.
            The input names correspond to self.inputs. Access the inputs e.g. via `input_dict['drift]`.

        Returns
        -------
        sample
            Contains one simulation result in the order specified by self.simulator_results.
            If no valid results has been computed, return a tensor or array containing `NaN` or `±∞`.
        """
        beta, ps = input_dict["shape"], input_dict["prob"]

        if isinstance(beta, torch.Tensor):
            temp = torch.ones_like(beta)
        else:
            temp = 1

        choices = Binomial(probs=ps).sample()
        rts = InverseGamma(concentration=2 * temp, rate=beta).sample()
        rts, choices = atleast_2d_float32_tensor(rts), atleast_2d_float32_tensor(
            choices
        )

        return torch.cat((rts, choices), dim=1)
