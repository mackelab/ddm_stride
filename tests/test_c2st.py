from pyro.distributions import InverseGamma
from torch.distributions import Binomial

from ddm_stride.pipeline.infer import *
from ddm_stride.pipeline.simulate import build_simulator
from ddm_stride.utils.config import load_config_file
from sbi.inference import MCMCPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import mcmc_transform
from sbi.utils.torchutils import atleast_2d
from sbibm.metrics import c2st


def test_c2st(toy_config):
    class PotentialFunctionProvider(BasePotential):
        allow_iid_x = True

        def __init__(self, prior, x_o, device="cpu"):
            super().__init__(prior, x_o, device)

        def __call__(self, theta, track_gradients: bool = True):

            theta = atleast_2d(theta)

            with torch.set_grad_enabled(track_gradients):
                iid_ll = self.iid_likelihood(theta)

            return iid_ll + self.prior.log_prob(theta)

        def iid_likelihood(self, theta):

            lp_choices = torch.stack(
                [
                    Binomial(probs=th.reshape(1, -1)).log_prob(self.x_o[:, 1:])
                    for th in theta[:, 1:]
                ],
                dim=1,
            )

            lp_rts = torch.stack(
                [
                    InverseGamma(
                        concentration=2 * torch.ones_like(beta_i), rate=beta_i
                    ).log_prob(self.x_o[:, :1])
                    for beta_i in theta[:, :1]
                ],
                dim=1,
            )

            joint_likelihood = (lp_choices + lp_rts).squeeze()

            assert joint_likelihood.shape == torch.Size(
                [self.x_o.shape[0] * theta.shape[0]]
            )
            return joint_likelihood.sum(0)

    simulator = build_simulator(toy_config)
    prior = build_prior(toy_config, "cpu")

    num_trials = 10
    num_samples = 1000
    theta_o = prior.sample((2,))
    x_o = simulator(theta_o.repeat(num_trials, 1))

    true_posterior = MCMCPosterior(
        potential_fn=PotentialFunctionProvider(prior, x_o),
        proposal=prior,
        theta_transform=mcmc_transform(prior, enable_transform=True),
        **toy_config["task"]["posterior_params"],
    )
    true_samples = true_posterior.sample((num_samples,))
    mnle_posterior = build_posterior(toy_config, None, None, "cpu")
    mnle_samples = mnle_posterior.sample((num_samples,), x=x_o)

    scores = c2st(true_samples, mnle_samples)

    assert scores < 0.7
