import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from black import Dict
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from ddm_stride.ddm_models.base_simulator import Simulator
from ddm_stride.pipeline.infer import build_posterior, build_prior
from ddm_stride.pipeline.simulate import build_simulator
from ddm_stride.utils.config import data_path
from ddm_stride.utils.data_names import *
from ddm_stride.utils.plot import plot_pdf, plot_samples
from sbi.analysis.plot import pairplot
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.types import ScalarFloat
from sbi.utils import tensor2numpy
from sbi.utils.torchutils import atleast_2d_float32_tensor


def evaluate(cfg: DictConfig) -> None:

    # Check if experimental data has been specified
    if not cfg["task"]["experimental_data_path"]:
        return

    os.makedirs("evaluate", exist_ok=True)

    exp_data = load_experimental_data(cfg)
    exp_cond_names = get_experimental_condition_names(cfg)

    # Range of observations to compute the pdf on
    cont_min = np.min(exp_data[get_continuous_observation_names(cfg)].values)
    cont_max = np.max(exp_data[get_continuous_observation_names(cfg)].values)
    x_cont_range = torch.linspace(cont_min, cont_max, 1000)
    x_discr = np.sort(np.unique(exp_data[get_discrete_observation_names(cfg)]))
    n_post_pred_subplots = len(get_observation_names(cfg))

    simulator = build_simulator(cfg)
    n_posterior_samples = cfg["task"]["n_posterior_samples"]
    best_thetas = {}

    # Create figures
    plt.rc("font", size=15)

    if exp_cond_names and cfg["task"]["group_by"]:

        # The posterior is computed and plotted separately for each group
        groupby = (
            list(cfg["task"]["group_by"])
            if isinstance(cfg["task"]["group_by"], ListConfig)
            else cfg["task"]["group_by"]
        )
        grouped_data = exp_data.groupby(by=groupby)

        # Pdf figure
        fig1, ax1 = plt.subplots(
            grouped_data.ngroups,
            1,
            constrained_layout=True,
            facecolor="white",
            figsize=(12, grouped_data.ngroups * 7),
            num=1,
        )
        plt.suptitle("Probability density function", fontsize=20)

        # Posterior samples figure
        figwidth = len(get_parameter_names(cfg)) * 3
        fig2 = plt.figure(
            2,
            constrained_layout=True,
            facecolor="white",
            figsize=(figwidth, grouped_data.ngroups * figwidth),
        )
        outer_grid2 = fig2.add_gridspec(
            grouped_data.ngroups, 1, wspace=0.05, hspace=0.2
        )
        outer_grid2.tight_layout(fig2)

        # Posterior predictive figure, plan height 3 for each subplot of the 2x5 plots
        fig3 = plt.figure(
            3,
            figsize=(
                n_post_pred_subplots * 3 * 5,
                grouped_data.ngroups * n_post_pred_subplots * 3 * 2,
            ),
            facecolor="white",
        )
        outer_grid3 = fig3.add_gridspec(
            grouped_data.ngroups, 1, wspace=0.02, hspace=0.2
        )
        outer_grid3.tight_layout(fig3)

    else:
        # Pdf figure
        fig1, ax1 = plt.subplots(
            1, 1, constrained_layout=True, facecolor="white", figsize=(12, 7), num=1
        )
        plt.title("Probability density function", fontsize=20)

        # Posterior samples figure
        figwidth = len(get_parameter_names(cfg)) * 3
        fig2 = plt.figure(
            2, constrained_layout=True, facecolor="white", figsize=(figwidth, figwidth)
        )
        outer_grid2 = fig2.add_gridspec(1, 1, wspace=0.05, hspace=0.2)
        outer_grid2.tight_layout(fig2)

        # Posterior predictive figure
        fig3 = plt.figure(
            3,
            figsize=(n_post_pred_subplots * 3 * 5, n_post_pred_subplots * 3 * 2),
            facecolor="white",
        )

    # Sample posterior, perform posterior predictive check, plot results
    # Group by specified experimental conditions, marginalize the others out
    if exp_cond_names:
        if cfg["task"]["group_by"]:

            best_thetas["experimental_condition_groups"] = cfg["task"]["group_by"]

            for idx, (key, group) in enumerate(grouped_data):

                x_o = group.loc[:, get_observation_names(cfg)].values
                exp_cond = group.loc[:, get_experimental_condition_names(cfg)].values

                posterior = build_posterior(
                    cfg, torch.FloatTensor(x_o), torch.FloatTensor(exp_cond)
                )
                potential_fn = posterior.potential_fn

                post_sample = tensor2numpy(posterior.sample((n_posterior_samples,)))
                # Compute metrics and log probability
                best_theta = tensor2numpy(
                    posterior.map(
                        num_iter=cfg["task"]["num_iter"],
                        num_to_optimize=cfg["task"]["num_to_optimize"],
                        num_init_samples=cfg["task"]["num_init_samples"],
                        init_method=cfg["task"]["init_method"],
                    )[0]
                )
                potential_prob = posterior.potential(best_theta) / group.shape[0]
                best_thetas[key] = theta_metrics(
                    cfg, best_theta, post_sample, potential_prob
                )

                # Flow likelihood fixed to MAP
                pdfs = nflow_pdf(
                    cfg, best_theta, group, potential_fn, x_discr, x_cont_range
                )

                # Plot posterior samples
                title = f"conditions: {cfg['task']['group_by']} = {key}"
                plot_samples(
                    cfg,
                    post_sample,
                    prior_samples=None,
                    reference_points=[best_theta],
                    fig=fig2,
                    grid=outer_grid2[idx],
                    title=title,
                )

                # Plot posterior predictive
                posterior_predictive(
                    cfg, post_sample, group, simulator, key, fig3, outer_grid3[idx]
                )

                # If the discrete observations are binary, move data corresponding to one of the discr_obs to negative space for plotting
                if (
                    len(get_continuous_observation_names(cfg)) == 1
                    and x_discr.shape[0] == 2
                ):
                    group.loc[
                        group[get_discrete_observation_names(cfg).pop()] == x_discr[0],
                        get_continuous_observation_names(cfg),
                    ] = (
                        group.loc[
                            group[get_discrete_observation_names(cfg).pop()].values
                            == x_discr[0],
                            get_continuous_observation_names(cfg),
                        ].values
                        * -1
                    )

                # Plot experimental data and pdf
                plot_pdf(
                    cfg,
                    pdfs=pdfs,
                    x_discr=x_discr,
                    x_cont_range=x_cont_range,
                    data=group,
                    exp_cond=key,
                    axis=ax1[idx],
                )

    # Marginalize out all experimental conditions
    else:

        x_o = exp_data.loc[:, get_observation_names(cfg)].values

        posterior = build_posterior(cfg, torch.FloatTensor(x_o), None)
        potential_fn = posterior.potential_fn

        post_sample = tensor2numpy(posterior.sample((n_posterior_samples,)))
        # Compute metrics and log probability
        best_theta = tensor2numpy(
            posterior.map(
                num_iter=cfg["task"]["num_iter"],
                num_to_optimize=cfg["task"]["num_to_optimize"],
                num_init_samples=cfg["task"]["num_init_samples"],
                init_method=cfg["task"]["init_method"],
            )[0]
        )  
        potential_prob = tensor2numpy(
            posterior.potential(best_theta) / exp_data.shape[0]
        )
        best_thetas = theta_metrics(cfg, best_theta, post_sample, potential_prob)

        # flow likelihood fixed to MAP
        pdfs = nflow_pdf(cfg, best_theta, exp_data, potential_fn, x_discr, x_cont_range)

        # Plot posterior samples
        plot_samples(
            cfg,
            post_sample,
            prior_samples=None,
            reference_points=[best_theta],
            fig=fig2,
        )

        # Plot posterior predictive
        posterior_predictive(
            cfg, post_sample, exp_data, simulator, exp_cond=None, fig=fig3
        )

        # If the discrete observations are binary, move data corresponding to one of the discr_obs to negative space for plotting
        if len(get_continuous_observation_names(cfg)) == 1 and x_discr.shape[0] == 2:
            exp_data.loc[
                exp_data[get_discrete_observation_names(cfg).pop()] != 1,
                get_continuous_observation_names(cfg),
            ] = (
                exp_data.loc[
                    exp_data[get_discrete_observation_names(cfg).pop()].values != 1,
                    get_continuous_observation_names(cfg),
                ].values
                * -1
            )

        # Plot experimental data and pdf
        plot_pdf(
            cfg,
            pdfs=pdfs,
            x_discr=x_discr,
            x_cont_range=x_cont_range,
            data=exp_data,
            exp_cond=None,
            axis=ax1,
        )
        ax1.legend(loc="upper left", frameon=False)

    # Save figures
    plt.figure(1)
    plt.savefig("evaluate/pdf.png")
    plt.show()

    plt.figure(2)
    fig2.suptitle("Posterior samples", y=1, fontsize=20)
    plt.savefig("evaluate/posterior.png")
    plt.show()

    plt.figure(3)
    plt.savefig("evaluate/posterior_predictive.png")
    plt.show()
    plt.close()

    with open("evaluate/best_thetas.json", "w") as file:
        json.dump(best_thetas, file)

    return


def load_experimental_data(
    cfg: DictConfig, drop_invalid_data: bool = True
) -> pd.DataFrame:
    """Load the dataframe containing experimental data.
    The dataframe should contain the experimental conditions and observations specified in the ddm_model config file.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.
        drop_invalid_data:
            exclude rows containing inf and NaN

        Returns
        -------
        exp_data:
            Dataframe containing experimental data.
    """
    filepath = Path(data_path() + cfg["task"]["experimental_data_path"])
    exp_data = pd.read_csv(filepath, index_col=0)

    if get_observation_names(cfg) + get_experimental_condition_names(cfg):
        for key in get_observation_names(cfg) + get_experimental_condition_names(cfg):
            if not key in exp_data.columns:
                raise KeyError(
                    f"observation or experimental condition {key} specified in the ddm_model configuration not found in the experimental data"
                )

    if drop_invalid_data:
        exp_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        exp_data.dropna(axis=0, how="any", inplace=True)

    return exp_data


def theta_metrics(
    cfg: DictConfig,
    map: np.ndarray,
    post_sample: np.ndarray,
    potential_prob: np.ndarray,
) -> Dict:
    """Create a dict containing each parameter name as key and its best posterior value as value.
    Additionally, p(x | theta, pi) is saved.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.
        map:
            Maximum a posteriori of each parameter.
        post_sample:
            Samples from the posterior.
        potential

        Returns
        -------
        param_dict:
            Dictionary containing the parameter names and metrics.
    """
    post_mean = np.mean(post_sample, axis=0)
    post_std = np.std(post_sample, axis=0)
    post_median = np.median(post_sample, axis=0)

    # Compute confidence interval of median via bootstrapping
    n_bootstrap_samples = 300
    bootstrap_idx = np.random.choice(
        post_sample.shape[0], size=(n_bootstrap_samples * post_sample.shape[0])
    )
    bootstrap_samples = post_sample[bootstrap_idx].reshape(
        n_bootstrap_samples, post_sample.shape[0], post_sample.shape[1]
    )
    bootstrap_median = np.median(bootstrap_samples, axis=1)
    errors = bootstrap_median - post_median
    confidence_left = post_median - np.abs(np.quantile(errors, q=0.01, axis=0))
    confidence_right = post_median + np.abs(np.quantile(errors, q=0.99, axis=0))
    quantile_5 = np.quantile(post_sample, q=0.05, axis=0)
    quantile_95 = np.quantile(post_sample, q=0.95, axis=0)

    param_dict = {}
    param_dict["log_prob"] = float(potential_prob)
    for i, param in enumerate(get_parameter_names(cfg)):
        param_dict[param] = {}
        param_dict[param]["map"] = float(map[i])
        param_dict[param]["median"] = float(post_median[i])
        param_dict[param]["confidence"] = [
            float(confidence_left[i]),
            np.float(confidence_right[i]),
        ]
        param_dict[param]["mean"] = float(post_mean[i])
        param_dict[param]["std"] = float(post_std[i])
        param_dict[param]["quantile_5"] = float(quantile_5[i])
        param_dict[param]["quantile_95"] = float(quantile_95[i])

    return param_dict


def nflow_pdf(
    cfg: DictConfig,
    theta: np.ndarray or torch.Tensor,
    data: pd.DataFrame,
    potential_fn: LikelihoodBasedPotential,
    x_discr: np.ndarray,
    x_cont_range: torch.linspace,
) -> List:
    """Build the prior distribution for all parameters theta.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    theta:
        Parameters the pdf is computed for.
    data:
        Experimental data the pdf is computed for.
    exp_cond:
        Experimental conditions the pdf is computed for.
    x_discr:
        All unique discrete values available in the data.
    x_cont_range:
        Linspace the pdfs are computed on.

    Returns
    -------
    pdfs:
        List of pdfs over linspace for each dicrete value.
    """
    n_steps = len(x_cont_range)
    theta = atleast_2d_float32_tensor(theta)

    if len(get_continuous_observation_names(cfg)) > 1:
        print("pdf for multiple observations is not implemented yet")
        return []

    else:
        log_pdfs = []

        for xd in x_discr:
            # Object to save the pdf to
            log_pdf = torch.zeros_like(x_cont_range)
            # Stack continuous observations to choices
            obs = torch.cat(
                (x_cont_range.reshape(-1, 1), torch.Tensor([xd]).repeat((n_steps, 1))),
                dim=1,
            )

            if get_experimental_condition_names(cfg):

                exp_cond_groups = data.groupby(get_experimental_condition_names(cfg))

                for exp_cond, group in exp_cond_groups:
                    # Compute probability of experimental condition
                    exp_cond_prob = group.shape[0] / data.shape[0]

                    potential_fn.set_exp_cond(torch.Tensor([exp_cond]))

                    # Compute PDFs
                    for i in range(n_steps):
                        potential_fn.set_x(obs[i])
                        log_pdf[i] += potential_fn(theta).item() * exp_cond_prob

            else:
                # Compute PDFs
                for i in range(n_steps):
                    potential_fn.set_x(obs[i])
                    log_pdf[i] = potential_fn(theta).item()

            log_pdfs.append(log_pdf)

        # Remove log, normalize the pdf to 1 over x_cont
        pdfs = []
        pdf_sum = np.exp(np.vstack(log_pdfs)).sum()
        bin_weight = n_steps / (x_cont_range[-1] - x_cont_range[0])

        for log_pdf in log_pdfs:
            pdfs.append((np.exp(log_pdf) / pdf_sum) * bin_weight)

        return pdfs


def posterior_predictive(
    cfg: DictConfig,
    post_samples: np.ndarray,
    data: pd.DataFrame,
    simulator: Simulator,
    exp_cond: Tuple or ScalarFloat,
    fig,
    grid=None,
) -> None:
    """Compute and plot posterior predictive using experimental data: For each plot select one posterior sample that has been inferred
    from the experimental data and use it to simulate new data. If the posterior sample is suitable, experimental data and simulated data
    should look similar.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    post_samples:
        Posterior samples inferred from the data.
    data:
        Data used for inferring the posterior samples.
    simulator:
        Simulator used to train the density estimator of the posterior.
    exp_cond:
        Experimental condition of posterior predictive if available.
    fig:
        Matplotlib figure used for plotting.
    grid:
        Matplotlib grid to plot on if available.
    """

    n_samples = 200
    n_plots = 10

    if grid:
        mid_grid = grid.subgridspec(2, 5, wspace=0.09, hspace=0.2)
    else:
        mid_grid = fig.add_gridspec(2, 5, wspace=0.09, hspace=0.2)

    for s in range(n_plots):
        # Draw observations to plot
        data_samples = data.sample(n_samples)
        x_o = data_samples[get_observation_names(cfg)].values
        # Draw experimental conditions to simulate
        exp_conds = data_samples[get_experimental_condition_names(cfg)].values

        rand_idx = np.random.choice(post_samples.shape[0])
        rand_sample = post_samples[rand_idx]

        if get_experimental_condition_names(cfg):
            rand_idx = np.random.choice(data.shape[0])
            input = np.hstack(
                (
                    np.repeat(rand_sample.reshape(1, -1), repeats=n_samples, axis=0),
                    exp_conds,
                )
            )
        else:
            input = np.repeat(rand_sample.reshape(1, -1), repeats=n_samples, axis=0)

        # Simulate observations for the posterior sample
        sim_samples = simulator(input)
        # Remove nan samples
        sim_samples = tensor2numpy(sim_samples[np.isnan(sim_samples).any(axis=1) == 0])

        # Plot the test x and simulator observations
        inner_grid = mid_grid[s].subgridspec(
            len(get_observation_names(cfg)),
            len(get_observation_names(cfg)),
            wspace=0.09,
            hspace=0.3,
        )
        ax1 = inner_grid.subplots()
        if exp_cond != None and s == 2:
            ax1[0, 0].set_title("exp cond: " + str(exp_cond), fontsize=18, y=1.1)

        all_samples = np.vstack((sim_samples, x_o))

        _, _ = pairplot(
            samples=sim_samples,
            samples_colors=["palevioletred"],
            diag="hist",
            hist_diag=dict(
                bins=np.arange(
                    np.min(all_samples) - 1e-1, np.max(all_samples) + 1e-1, 0.1
                ),
                histtype="stepfilled",
                density=True,
                align="mid",
            ),
            upper="scatter",
            scatter_offdiag=dict(alpha=0.6),
            limits=[
                (np.min(all_samples[:, i]) - 0.2, np.max(all_samples[:, i]) + 0.2)
                for i in range(x_o.shape[1])
            ],
            labels=[obs for obs in get_observation_names(cfg)],
            legend=True,
            fig=fig,
            axes=ax1,
        )

        _, _ = pairplot(
            samples=x_o,
            samples_colors=["slategrey"],
            diag="hist",
            hist_diag=dict(
                bins=np.arange(
                    np.min(all_samples) - 1e-1, np.max(all_samples) + 1e-1, 0.1
                ),
                histtype="step",
                density=True,
                alpha=0.7,
                linewidth=1.5,
                align="mid",
            ),
            upper="scatter",
            scatter_offdiag=dict(ec="slategrey", fc="none", alpha=0.5),
            limits=[
                (np.min(all_samples[:, i]) - 0.2, np.max(all_samples[:, i]) + 0.2)
                for i in range(all_samples.shape[1])
            ],
            labels=[obs for obs in get_observation_names(cfg)],
            legend=True,
            title="Posterior predictive check",
            title_format={"fontsize": 20},
            fig=fig,
            axes=ax1,
        )

        if s == 0:
            ax1[0, 0].legend(["simulations", "experimental data"])
