from typing import Dict, List, Union

import numpy as np
import torch

from sbi.utils.torchutils import atleast_2d_float32_tensor


class Simulator:
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
        """Use the config file to initialize attributes.

        Parameters
        -------
        inputs: List[str]
            Specifies the parameters and experimental conditions passed to the simulator functions.
            Corresponds to parameters and experimental conditions specified in the config simulation file.
        simulator_results: List[str]
            Specifies how the simulator output should be structured.
            Corresponds to the observations specified in the config simulation file.
        """
        self.inputs = inputs
        self.simulator_results = simulator_results
        print("parameter names and experimental conditions: ", self.inputs)
        print("simulation results: ", self.simulator_results)

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Use the given batch of input_data to simulate data.

        Parameters
        ----------
        input_data
            A batch of parameter samples and experimental conditions to be used for simulating data. The input names
            correspond to self.inputs.

        Returns
        -------
        samples
            Each row of the tensor contains one simulation result in the order specified by self.simulator_results.
        """
        # Reshape to 2-dim in case that only one parameter sample is passed
        input_data = atleast_2d_float32_tensor(input_data)

        sample = torch.zeros(size=(input_data.shape[0], len(self.simulator_results)))

        # Convert input to dictionary in order to facilitate data generating
        for i, parameter_sample in enumerate(input_data):
            input_dict = {}
            for name, input in zip(self.inputs, parameter_sample):
                input_dict[name] = input.item()
            sample[i] = torch.as_tensor(
                self.generate_data(input_dict), dtype=torch.float32
            )

        return sample

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
        raise NotImplementedError
