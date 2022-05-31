import torch

from ddm_stride.pipeline.infer import *
from ddm_stride.utils.config import load_config_file
from ddm_stride.utils.data_names import *


def test_prior_and_proposal(toy_config):

    prior = build_prior(toy_config, "cpu")
    proposal = build_proposal(toy_config, "cpu")

    prior_sample = prior.sample((3,))
    proposal_sample = proposal.sample((5,))

    # Check for correct shape of samples
    assert prior_sample.shape[-1] == len(
        get_parameter_names(toy_config)
    ), "Every parameter needs a sample value"
    assert prior_sample.shape[0] == 3, "Sample shape incorrect"
    assert proposal_sample.shape[-1] == len(
        get_param_exp_cond_names(toy_config)
    ), "Every parameter and experimental condition needs a sample value"
    assert proposal_sample.shape[0] == 5, "Sample shape incorrect"
    # Test if tensors are returned
    assert isinstance(prior_sample, torch.Tensor), "Function should return a tensor"
    assert isinstance(proposal_sample, torch.Tensor), "Function should return a tensor"


def test_load_density_estimator():

    x = torch.Tensor([[0.3, 1], [0.12, 0]])
    cfg = load_config_file("basic_ddm")
    context = build_proposal(cfg, "cpu").sample((2,))

    density_est = load_density_estimator("basic_ddm", "cpu")
    log_prob = density_est.log_prob(x, context)

    assert isinstance(
        density_est, torch.nn.Module
    ), "Density estimator needs to be a torch module"
    # Check if log prob available and working
    assert isinstance(log_prob, torch.Tensor), "Log prob needs to return a tensor"
    assert (
        log_prob.shape[0] == x.shape[0]
    ), "Shape of log prob needs to correspond to first dimension of data"


def test_posterior(toy_config):

    x = torch.Tensor([[0.3, 1], [0.12, 0]])

    posterior = build_posterior(toy_config, x, None, device="cpu")

    sample = posterior.sample()

    # Check if the posterior can be sampled from
    assert isinstance(
        sample, torch.Tensor
    ), 'posterior needs sample function'
