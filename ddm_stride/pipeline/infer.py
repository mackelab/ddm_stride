from copy import deepcopy
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import sbi.utils
from ddm_stride.sbi_extensions.mcmc import MCMCPosterior
from ddm_stride.sbi_extensions.potential_fn_exp_cond import mnle_potential_exp_cond
from ddm_stride.utils.config import load_config_file, load_wandb_config, results_path
from ddm_stride.utils.data_names import *
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.neural_nets.flow import *
from sbi.neural_nets.flow import build_nsf
from sbi.utils.get_nn_models import likelihood_nn
from sbi.utils.user_input_checks import process_prior
from sbi.utils.user_input_checks_utils import MultipleIndependent


def build_prior(cfg: DictConfig, device: str or torch.device) -> MultipleIndependent:
    """Build the prior distribution for all parameters theta.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    device:
        Device the prior distribution is moved to.

    Returns
    -------
    prior:
        Torch distribution built from one or multiple independent torch distributions.
    """
    prior = []

    for datum in cfg["ddm_model"]["parameters"]:
        # Convert distribution arguments to tensors
        distribution_args = {}
        for key in datum["distribution"].keys():
            if key == "_target_":
                continue
            distribution_args[key] = torch.FloatTensor([datum["distribution"][key]]).to(
                device
            )
        d_prior = hydra.utils.instantiate(datum["distribution"], **distribution_args)
        prior.append(d_prior)

    # Return either a sequence of distributions or one distribution
    if len(prior) > 1:
        prior, _, _ = process_prior(prior)
        # TODO: no prior support for custom prior, https://github.com/mackelab/sbi/blob/93f64dcf6c821beb4a37802536d7a67a0adde1e7/docs/docs/faq/question_07.md
    else:
        prior, _, _ = process_prior(prior[0])

    return prior


def build_proposal(cfg: DictConfig, device: str or torch.device) -> MultipleIndependent:
    """Build the proposal distribution for all parameters theta and experimental conditions pi.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    device:
        Device the proposal distribution is moved to.

    Returns
    -------
    proposal:
        Torch distribution built from one or multiple independent torch distributions.
    """
    proposal = []
    data = cfg["ddm_model"]["parameters"].copy()
    if cfg["ddm_model"]["experimental_conditions"]:
        data += cfg["ddm_model"]["experimental_conditions"]

    for datum in data:
        d_prior = build_distribution(datum, device)
        proposal.append(d_prior)

    # Return either a sequence of distributions or one distribution
    if len(proposal) > 1:
        prior, _, _ = process_prior(proposal)
    else:
        prior, _, _ = process_prior(proposal[0])

    return prior


def build_distribution(datum: DictConfig, device):
    """Build a distribution from the configurations passed via hydra.

    Parameters
    ----------
    datum:
        Configuration of a parameter or experimental condition.
        Contains a name, a distribution specified by _target_ and distribution parameters.
    device:
        Device the distribution is moved to.

    Returns
    -------
    dist:
        Torch distribution.
    """
    # Convert distribution arguments to tensors
    distribution_args = {}
    for key in datum["distribution"].keys():
        if key == "_target_":
            continue
        distribution_args[key] = torch.FloatTensor([datum["distribution"][key]]).to(
            device
        )
    dist = hydra.utils.instantiate(datum["distribution"], **distribution_args)
    return dist


def load_density_estimator(rel_path: str, device: str or torch.device) -> nn.Module:
    """Load a trained neural likelihood estimator.

    Parameters
    ----------
    rel_path:
        Relative path to the results subfolder containing the model state dict.
    device:
        Device the density estimator is moved to.

    Returns
    -------
    density_estimator:
        A pretrained density estimator.
    """

    cfg = load_config_file(rel_path=rel_path)
    wandb_hyperparams = load_wandb_config(rel_path)

    default_params = torch.rand(
        size=(2, len(get_param_exp_cond_names(cfg))),
        dtype=torch.float32,
    ).to(device)
    default_obs = torch.rand(
        size=(2, len(get_observation_names(cfg))), dtype=torch.float32
    ).to(device)

    model_hyperparams = cfg["algorithm"]["model_hyperparams"]
    density_estimator_build = likelihood_nn(
        **model_hyperparams,
        hidden_features=wandb_hyperparams["hidden_features"],
        num_transforms=wandb_hyperparams["num_transforms"],
        hidden_layers=wandb_hyperparams["hidden_layers"]
    )

    density_estimator = density_estimator_build(default_params, default_obs).to(device)

    model_filepath = Path(results_path() + rel_path + "/model_state_dict.pt")
    density_estimator.load_state_dict(torch.load(model_filepath))

    return density_estimator


def build_posterior(
    cfg: DictConfig,
    x: Optional[Tensor] = None,
    exp_cond: Optional[Tensor] = None,
    device: str = "cpu",
) -> NeuralPosterior:
    """Build a posterior.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    x:
        Observations.
    exp_cond:
        Experimental conditions under which x was observed.
    device:
        Device the posterior and its inputs are moved to.

    Returns
    -------
    posterior:
        A MCMC posterior.
    """

    prior = build_prior(cfg, device)

    if cfg["task"]["model_path"]:
        rel_path = cfg["task"]["model_path"]
    else:
        rel_path = cfg["result_folder"]

    if cfg["algorithm"]["model_hyperparams"]["model"] == "mnle":
        density_estimator = load_density_estimator(rel_path, device)
        potential_fn, theta_transform = mnle_potential_exp_cond(
            density_estimator, prior, x, exp_cond
        )
    else:
        raise NotImplementedError

    if cfg["task"]["posterior"] == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            theta_transform=theta_transform,
            device=device,
            x_shape=torch.Size([1, len(get_observation_names(cfg))]),
            **cfg["task"]["posterior_params"]
        )
    else:
        raise NotImplementedError

    return deepcopy(posterior)
