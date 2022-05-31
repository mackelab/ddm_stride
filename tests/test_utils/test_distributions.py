import torch

from ddm_stride.utils.distributions import Categorical


def test_discrete_uniform():

    discrete_values = torch.Tensor([[1, 5, 8, 3]])
    du = Categorical(discrete_values)
    sample = du.sample(torch.Size((20,)))

    # Check that only passed values are sampled
    assert all([s in discrete_values for s in sample])
    # All values have the same probability
    assert du.log_prob(3) == du.log_prob(5)
