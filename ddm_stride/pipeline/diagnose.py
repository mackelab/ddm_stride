import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from ddm_stride.ddm_models.base_simulator import Simulator
from ddm_stride.pipeline.infer import (
    build_posterior,
    build_prior,
    build_proposal,
    load_density_estimator,
)
from ddm_stride.pipeline.simulate import build_simulator, load_simulation_data
from ddm_stride.sbi_extensions.local_coverage_test import local_coverage_test
from ddm_stride.sbi_extensions.sbc_exp_cond import run_sbc
from ddm_stride.utils.data_names import (
    get_experimental_condition_names,
    get_observation_names,
    get_param_exp_cond_names,
    get_parameter_names,
)
from ddm_stride.utils.plot import (
    compare_observations,
    plot_samples,
    pp_plots_local_coverage,
)
from sbi.analysis import pairplot, sbc_rank_plot
from sbi.utils.torchutils import atleast_2d_float32_tensor
from sbi.utils.user_input_checks import prepare_for_sbi


def diagnose(cfg: DictConfig):

    os.makedirs("diagnose", exist_ok=True)

    device = "cpu"

    simulator = build_simulator(cfg)
    simulator, _ = prepare_for_sbi(simulator, build_proposal(cfg, device))
    density_estimator = load_density_estimator(cfg["result_folder"], device)

    (
        _,
        simulation_test_data,
        simulation_iid_test_data,
    ) = load_simulation_data(cfg, drop_invalid_data=True)
    test_data_sbc = simulation_test_data.iloc[:-100, :]
    test_data_local_cov = simulation_test_data.iloc[-100:, :]

    # Run fast diagnose methods for a first overview
    if cfg["run_diagnose_fast"]:

        # Compare observations sampled from flow to simulated test observations
        print("\nCompare observations")
        compare_observations(cfg, density_estimator, simulation_iid_test_data)

        # Perform posterior predictive check
        print("\nPosterior predictive")
        posterior_predictive_check(
            cfg,
            iid_test_data=simulation_iid_test_data,
            simulator=simulator,
            device=device,
        )

    # Run slow diagnose methods to check if the posterior is well defined
    if cfg["run_diagnose_slow"]:

        # Run sbc
        print("\nSBC check")
        ranks, xs = sbc_check(cfg, test_data_sbc, device=device)

        # Perform local coverage test
        print("\nLocal coverage test")
        local_coverage_check(cfg, xs, test_data_local_cov, ranks, device=device)

    return


def posterior_predictive_check(
    cfg: DictConfig,
    iid_test_data: pd.DataFrame,
    simulator: Simulator,
    device: str = "cpu",
):
    """Compute and plot a posterior predictive check, using the posterior samples to simulate observations.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    iid_test_data:
        Multiple simulated observations per configuration of parameters and experimental conditions.
    simulator:
        The simulator the iid_test_data has been simulated with.
    """
    # Separate the test data into the groups of parameters and experimental conditions.
    param_groups = iid_test_data.groupby(by=get_param_exp_cond_names(cfg))
    n_groups = param_groups.ngroups

    # Use prior params as a baseline for plots
    prior_samples = build_prior(cfg, "cpu").sample((5000,))

    fig = plt.figure(figsize=(20, n_groups * 7), facecolor="white")
    outer_grid = fig.add_gridspec(n_groups, 1, wspace=0.02, hspace=0.2)
    outer_grid.tight_layout(fig)

    for idx, (key, data) in enumerate(param_groups):

        # Draw x and experimental condition from the test data
        n_obs = min(250, data.shape[0])
        obs_idx = np.random.choice(data.index, n_obs, replace=False)
        x = data.loc[obs_idx, get_observation_names(cfg)].values
        x = atleast_2d_float32_tensor(x)
        exp_cond = data.loc[obs_idx, get_experimental_condition_names(cfg)].values
        exp_cond = atleast_2d_float32_tensor(exp_cond)

        # Build a posterior and sample from it
        posterior = build_posterior(cfg, x, exp_cond, device=device)
        posterior_samples = posterior.sample((n_obs,))

        # Simulate observations for the posterior samples
        simulator_input = torch.cat((posterior_samples, exp_cond), dim=1)
        sim_samples = simulator(simulator_input).cpu().numpy()
        # Remove nan samples
        sim_samples = sim_samples[~np.isnan(sim_samples).any(axis=1)]

        # Convert observations and posterior samples to numpy for plotting
        x = x.numpy()
        posterior_samples = posterior_samples.numpy()

        mid_grid = outer_grid[idx, 0].subgridspec(
            1, 2, wspace=0.1, hspace=0.2, width_ratios=[0.55, 0.45]
        )

        # Plot the posterior samples
        plot_samples(
            cfg,
            posterior_samples,
            prior_samples,
            reference_points=key,
            fig=fig,
            grid=mid_grid[0, 0],
            legend=["posterior", "ground truth", "prior"],
        )

        # Plot the test x and simulator observations
        inner_grid_1 = mid_grid[0, 1].subgridspec(2, 2, wspace=0.09, hspace=0.09)
        ax1 = inner_grid_1.subplots()

        all_samples = np.vstack((sim_samples, x))
        limits = [
            (np.min(all_samples[:, i]) - 0.2, np.max(all_samples[:, i]) + 0.2)
            for i in range(all_samples.shape[1])
        ]

        _, _ = pairplot(
            samples=sim_samples,
            samples_colors=["thistle"],
            diag="hist",
            hist_diag=dict(bins=20, histtype="stepfilled", density=True, align="mid"),
            upper="scatter",
            scatter_offdiag=dict(alpha=0.7),
            limits=limits,
            labels=[obs for obs in get_observation_names(cfg)],
            fig=fig,
            axes=ax1,
        )

        _, _ = pairplot(
            samples=x,
            samples_colors=["palevioletred"],
            diag="hist",
            hist_diag=dict(
                bins=20, histtype="step", density=True, linewidth=1.5, align="mid"
            ),
            upper="scatter",
            scatter_offdiag=dict(ec="palevioletred", fc="none", alpha=0.7),
            limits=limits,
            labels=[obs for obs in get_observation_names(cfg)],
            fig_subplots_adjust={
                "top": 0.965,
            },
            title="Posterior predictive check and posterior samples",
            title_format={"fontsize": 22},
            fig=fig,
            axes=ax1,
        )
        ax1[-1, -1].legend(["simulated samples", "observations"], bbox_to_anchor=(0,0), loc="lower right")

    plt.rc("font", size=15)
    plt.savefig("diagnose/posterior_predictive.png")
    plt.show()
    plt.close()

    return


def sbc_check(
    cfg: DictConfig, simulation_test_data: pd.DataFrame, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute and plot the simuation based calibration.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    simulation_test_data:
        Simulated observations, parameters and experimental conditions.
    """

    n_sbc_runs = cfg["task"]["n_sbc_runs"]
    num_workers = 1  # Parallel posterior and sbc run does not seem to work
    sbc_batch_size = 1

    idx = np.random.choice(
        simulation_test_data.shape[0], size=(n_sbc_runs,), replace=True
    )
    xs = atleast_2d_float32_tensor(
        simulation_test_data.loc[idx, get_observation_names(cfg)].values
    ).to(device)
    thetas = atleast_2d_float32_tensor(
        simulation_test_data.loc[idx, get_parameter_names(cfg)].values
    ).to(device)
    exp_cond = atleast_2d_float32_tensor(
        simulation_test_data.loc[idx, get_experimental_condition_names(cfg)].values
    ).to(device)

    posterior = build_posterior(cfg, None, None, device)

    ranks, dap_samples = run_sbc(
        thetas,
        xs,
        exp_cond,
        posterior,
        num_posterior_samples=300,
        num_workers=num_workers,
        sbc_batch_size=sbc_batch_size,
    )

    sbc_rank_plot(
        ranks,
        num_bins=100,
        num_posterior_samples=300,
        parameter_labels=get_parameter_names(cfg),
        colors=None,
    )

    plt.rc("font", size=15)
    plt.suptitle("SBC test")
    plt.savefig("diagnose/sbc.png")
    plt.show()
    plt.close()

    return ranks, xs


def local_coverage_check(
    cfg,
    xs_train: np.ndarray,
    test_data: pd.DataFrame,
    xs_ranks: np.ndarray,
    device="cpu",
):
    """Compute and plot the local coverage check using the SBC results.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    xs_train:
        Observations the SBC check has been run on.
    test_data:
        Simulated observations, parameters and experimental conditions not used for SBC.
    xs_ranks:
        SBC ranks.
    """
    xs_test = test_data.loc[:, get_observation_names(cfg)].values
    data_test = test_data.loc[
        :, get_observation_names(cfg) + get_experimental_condition_names(cfg)
    ].values

    (
        global_pvalues_per_dim,
        local_pvalues_per_dim,
        rank_predictions_per_dim,
        uniform_predictions_per_dim,
    ) = local_coverage_test(xs_test, xs_train, xs_ranks, device=device)

    # Save global p values
    print("global p-value per parameter dimension:", global_pvalues_per_dim)
    global_p_values = {}
    for param, pvalue in zip(get_parameter_names(cfg), global_pvalues_per_dim):
        global_p_values[param] = np.round(float(pvalue), 4)
    with open("diagnose/global_pvalues.json", "w") as f:
        json.dump(global_p_values, f)

    fig = pp_plots_local_coverage(
        cfg["task"]["n_local_coverage_plots"],
        rank_predictions_per_dim,
        uniform_predictions_per_dim,
        local_pvalues=local_pvalues_per_dim,
        thetas=get_parameter_names(cfg),
        data_test=data_test,
        observation_names=get_observation_names(cfg)
        + get_experimental_condition_names(cfg),
    )

    if fig:
        plt.savefig("diagnose/local_coverage_test.png")
        plt.close()
