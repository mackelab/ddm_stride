import torch

from ddm_stride.pipeline.infer import (
    build_prior,
    build_proposal,
    load_density_estimator,
)
from ddm_stride.pipeline.simulate import build_simulator
from ddm_stride.sbi_extensions.potential_fn_exp_cond import mnle_potential_exp_cond
from ddm_stride.sbi_extensions.simulate_iid import simulate_for_sbi_iid
from ddm_stride.utils.config import load_config_file
from sbi.utils.torchutils import atleast_2d_float32_tensor


def test_simulate_for_sbi_iid(toy_config):

    simulator = build_simulator(toy_config)
    proposal = build_proposal(toy_config, "cpu")

    theta, x = simulate_for_sbi_iid(
        simulator, proposal, num_params=3, num_iid_simulations_per_param=10
    )

    # Check that the function returns two tensors
    assert isinstance(theta, torch.Tensor) and isinstance(x, torch.Tensor)
    # Check that theta and x have the same shape and the correct number of rows
    assert theta.shape[0] == x.shape[0]
    assert theta.shape[0] == 30
    # Check if num_params theta have been selected
    assert torch.unique(theta, dim=0).shape[0] == 3


