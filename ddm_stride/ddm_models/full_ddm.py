from typing import Dict, List, Union

import numpy as np
import sdeint
import torch

from ddm_stride.ddm_models.base_simulator import Simulator


class FullDDM(Simulator):
    """Build the model used for simulating data.

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

    def generate_data(
        self, input_dict: Dict
    ) -> Union[torch.Tensor, np.ndarray]:
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
        # Sample drift, starting point and non-decision time using mean values and across-trial variability
        # as specified in https://osf.io/xb75f/
        sign = np.random.choice([-1, 1])
        drift = np.random.normal(
            sign * input_dict["drift"], input_dict["atv_drift"]
        )
        relative_starting_point = np.random.uniform(
            low=input_dict["starting_point"] - input_dict["atv_starting_point"] / 2,
            high=input_dict["starting_point"] + input_dict["atv_starting_point"] / 2,
        )
        non_decision_time = np.random.uniform(
            low=input_dict["non_decision_time"]
            - input_dict["atv_non_decision_time"] / 2,
            high=input_dict["non_decision_time"]
            + input_dict["atv_non_decision_time"] / 2,
        )

        # the reaction time starts at 0
        rt = 0
        # define time interval and number of intermediate steps
        tmax = 7.0
        tspan = np.linspace(0.0, tmax, 1001)

        def f(x, t):
            return drift

        def g(x, t):
            return 1

        starting_point = relative_starting_point * input_dict["boundary_separation"]
        traj = sdeint.itoSRI2(f, g, starting_point, tspan).flatten()

        # check which boundary has been crossed (first)
        lower_bound = 0
        upper_bound = input_dict["boundary_separation"]
        pass_lower_bound = np.where(traj < lower_bound)[0]
        pass_upper_bound = np.where(traj > upper_bound)[0]

        if pass_lower_bound.size > 0 and pass_upper_bound.size > 0:
            if pass_lower_bound[0] < pass_upper_bound[0]:
                rt, choice = tspan[pass_lower_bound[0]] + non_decision_time, 0
            else:
                rt, choice = tspan[pass_upper_bound[0]] + non_decision_time, 1
        elif pass_lower_bound.size > 0:
            rt, choice = tspan[pass_lower_bound[0]] + non_decision_time, 0
        elif pass_upper_bound.size > 0:
            rt, choice = tspan[pass_upper_bound[0]] + non_decision_time, 1
        # if no boundary has been crossed, return nan
        else:
            rt, choice = torch.nan, torch.nan

        return rt, choice
