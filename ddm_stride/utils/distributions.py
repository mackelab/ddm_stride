from typing import Optional

import torch
from torch.distributions.distribution import Distribution

# See https://github.com/mackelab/sbi/blob/93f64dcf6c821beb4a37802536d7a67a0adde1e7/docs/docs/faq/question_07.md for information about
# the creation of custom priors
# The pipeline does not check the support of a custom prior


class Categorical(Distribution):
    """
    Discrete distribution based on torch.Categorical.
    Returns predefined values instead of class indices.
    """

    def __init__(
        self, discrete_values: torch.Tensor, probs: Optional[torch.Tensor] = None
    ):
        """
        Parameters
        ----------
        discrete_values:
            2-dimensional tensor containing the discrete values to sample from.
        probs:
            2-dimensional tensor containing the probabilites corresponding to the discrete values.
        """

        self.discrete_values = discrete_values
        self._batch_shape = (
            self.discrete_values.size()[:-1]
            if self.discrete_values.ndimension() > 1
            else torch.Size()
        )
        self.probs = probs if probs else torch.ones_like(self.discrete_values)
        self.categorical = torch.distributions.Categorical(probs=self.probs)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.

        Parameters
        ----------
        sample_shape:
            Shape of returned sample.

        Returns
        -------
        sample:
            Tensor containing samples from self.discrete_values.
        """
        idx_sample = self.categorical.sample(sample_shape)
        sample = self.discrete_values[0, idx_sample]

        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log prob of the given value.

        Parameters
        ----------
        value:
            Tensor containing value(s) specified in self.discrete_values.

        Returns
        -------
        log_prob:
            Log prob of value.
        """
        idx = torch.where(self.discrete_values == value)[0]
        return self.categorical.log_prob(idx)
