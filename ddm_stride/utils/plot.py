from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from ddm_stride.utils.data_names import (
    get_continuous_observation_names,
    get_discrete_observation_names,
    get_observation_names,
    get_param_exp_cond_names,
    get_parameter_names,
)
from sbi.analysis.plot import pairplot


def compare_observations(
    cfg: DictConfig, model: nn.Module, test_data: pd.DataFrame
) -> None:
    """Plot pdf on experimental data as well as the posterior over the parameters.

    Parameters
    ----------
    cfg:
        Config file passed by hydra.
    model:
        Trained likelihood estimator.
    test_data:
        Simulated test data containing multiple i.i.d. observations per parameter and experimental condition.
    """

    model.eval()

    param_groups = test_data.groupby(by=get_param_exp_cond_names(cfg))
    n_groups = param_groups.ngroups

    n_observations = len(get_observation_names(cfg))

    # Plot observations in simulated test data as well as observations sampled from flow using the same inputs
    fig = plt.figure(
        constrained_layout=True,
        figsize=(n_observations * 8, n_groups * 5),
        facecolor="white",
    )
    plt.rc("font", size=15)
    fig.suptitle("Compare observations between simulated and flow data", fontsize=20)

    subfigs = fig.subfigures(nrows=n_groups, ncols=1)

    for idx, (key, data) in enumerate(param_groups):

        n_samples = data.shape[0]

        # Simulated observations
        test_samples = data.loc[:, get_observation_names(cfg)]

        with torch.no_grad():
            model_input = torch.tensor(key, dtype=torch.float32).reshape(1, -1)
            # draw observation samples
            samples = model.sample(model_input, num_samples=n_samples)
        # Create dataframe for the sampled observations
        model_samples = pd.DataFrame(
            samples.cpu().numpy(), columns=get_observation_names(cfg)
        )

        # Set the key as a title for each row
        title = ""
        for i, name in enumerate(get_param_exp_cond_names(cfg)):
            title += name + ": " + str(round(key[i], 3)) + "  "
        subfigs[idx].suptitle(title, fontsize=17)
        # Create subplots for each key/row
        ax = subfigs[idx].subplots(1, n_observations)

        for i, obs in enumerate(get_observation_names(cfg)):

            bin_range = (
                min(
                    0,
                    min(min(test_samples[obs].values), min(model_samples[obs].values)),
                )
                - 0.2,
                max(max(test_samples[obs].values), max(model_samples[obs].values))
                + 0.2,
            )

            # Plot the simulated observation
            ax[i].hist(
                test_samples[obs].values,
                range=bin_range,
                bins=25,
                histtype="step",
                linewidth=2,
                color="palevioletred",
                label="simulated " + str(obs),
                alpha=0.7,
            )
            ax[i].hist(
                test_samples[obs].values,
                range=bin_range,
                bins=25,
                histtype="stepfilled",
                color="palevioletred",
                alpha=0.6,
            )
            # Plot the flow observation
            ax[i].hist(
                model_samples[obs].values,
                range=bin_range,
                bins=25,
                histtype="step",
                linewidth=2,
                color="steelblue",
                label="MNLE " + str(obs),
                alpha=0.7,
            )
            ax[i].hist(
                model_samples[obs].values,
                range=bin_range,
                bins=25,
                histtype="stepfilled",
                color="steelblue",
                alpha=0.6,
            )
            ax[i].set_title(str(obs), fontsize=17)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].legend()

    plt.savefig("diagnose/compare_observations.png")
    plt.show()
    plt.close()

    return


def pp_plots_local_coverage(
    n_plots: int,
    rank_predictions: Tensor,
    uniform_predictions: Tensor,
    local_pvalues: Tensor,
    thetas: List,
    data_test: Tensor = None,
    observation_names="observations",
    alphas: Tensor = torch.linspace(0.05, 0.95, 21),
    conf_alpha: float = 0.05,
) -> Union[None, plt.figure]:
    """PP-plots for alpha and data_test rank predictions. Only coverage test with significant p-value are plotted.

    Parameters
    ----------
    n_plots:
        Maximum number of plots per parameter.
    rank_predictions:
        Alpha predictions using ranks for each observation in data_test.
        rank_predictions should be equivalent to the output rank_predictions of local_test().
    uniform_predictions:
        Alpha predictions using the uniform distribution for each observation in data_test.
        uniform_predictions should be equivalent to the output uniform_predictions of local_test().
    local_pvalues:
        List of p-values per dimension for each observation.
        local_pvalues should be equivalent to local_pvalues returned by local_coverage_test().
    thetas:
        Parameter names.
    data_test:
        Test observations and experimental conditions the posterior will be evaluated at.
        The test observations should be equivalent to the observations used for local_coverage_test().
        data_test is only used for plot titles.
    observation_names:
        Names of observations (and experimental conditions) used for the plot title.
    alphas:
        Posterior quantiles used for running local_coverage_test().
    conf_alpha:
        Confidence value used for creating confidence bounds.
    """

    theta_significant = []
    for theta_dim in range(len(local_pvalues)):
        xs_idx = torch.where(local_pvalues[theta_dim] < conf_alpha)[0]
        if len(xs_idx) > 0:
            theta_significant.append(theta_dim)
        else:
            print(f"no significant p-values for theta {thetas[theta_dim]} found")
    if len(theta_significant) == 0:
        return None

    fig = plt.figure(
        constrained_layout=True,
        facecolor="white",
        figsize=(20, len(theta_significant) * 7),
    )
    subfigs = fig.subfigures(nrows=len(theta_significant), ncols=1)

    # Create plots for significant local p-values
    for idx, theta_dim in enumerate(theta_significant):

        if len(theta_significant) == 1:
            subfig = subfigs
        else:
            subfig = subfigs[idx]

        subfig.supxlabel("posterior rank")
        subfig.supylabel("empirical CDF")

        xs_idx = torch.where(local_pvalues[theta_dim] < conf_alpha)[0]
        # Plot a maximum of n_plots plots
        n_plots = min(len(xs_idx), n_plots)

        # Compute confidence bands
        lower_band = torch.quantile(
            uniform_predictions[theta_dim], q=conf_alpha / 2, axis=0
        )
        upper_band = torch.quantile(
            uniform_predictions[theta_dim], q=1 - conf_alpha / 2, axis=0
        )

        # Plot 5 plots in a row
        n_yplots = 5
        n_xplots = int(np.ceil(n_plots / n_yplots))

        axs = subfig.subplots(n_xplots, n_yplots, sharex=True, sharey=True)
        subfig.suptitle(f"P({thetas[theta_dim]}|{observation_names})", fontsize=18)

        for idx, ax in zip(xs_idx, axs.ravel()[:n_plots]):

            ax.plot(alphas, rank_predictions[theta_dim][idx], color="darkblue")
            ax.plot(alphas, alphas, "--", color="royalblue", alpha=0.5)
            ax.fill_between(
                alphas, lower_band[idx], upper_band[idx], alpha=0.2, color="grey"
            )
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            if isinstance(data_test, torch.Tensor):
                ax.set_title(str(np.round(data_test[idx].tolist(), 3).tolist()))

    plt.suptitle("Local coverage test", fontsize=20)
    plt.rc("font", size=15)
    plt.show()

    return fig


def plot_pdf(
    cfg,
    pdfs: np.ndarray,
    x_discr: np.ndarray,
    x_cont_range: torch.linspace,
    data: pd.DataFrame,
    exp_cond: float = None,
    axis=None,
) -> None:
    """Plot pdf and experimental data. Currently only available for one continuous observation.

    Parameters
    ----------
    cfg:
        Config file passed by hydra.
    pdfs:
        Normalized pdfs per x_discr computed by the likelihood estimator.
    x_discr:
        All unique discrete values available in the data.
    x_cont_range:
        X-axis for pdfs.
    data:
        Experimental data containing the observations.
    exp_cond:
        Experimental condition shown in title of plot.
    axis:
        Matplotlib axis to plot on.
    """

    data_min = data.loc[:, get_continuous_observation_names(cfg)].min().min() - 0.2
    data_max = data.loc[:, get_continuous_observation_names(cfg)].max().max() + 0.2

    n_cont_obs = len(get_continuous_observation_names(cfg))
    n_discr_obs = x_discr.shape[0]

    if n_cont_obs == 1 and n_discr_obs == 2:

        axis.hist(
            data.loc[:, get_continuous_observation_names(cfg)].values,
            bins=np.arange(data_min, data_max, 0.1),
            density=True,
            alpha=0.7,
            color="slategrey",
            label="experimental data",
        )
        axis.plot(-1 * x_cont_range, pdfs[0], color="steelblue", linewidth=2.5)
        axis.plot(
            x_cont_range,
            pdfs[1],
            color="steelblue",
            linewidth=2.5,
            label=r"$q(x|\hat{\theta}, \pi) \cdot p(\hat{\theta})$ ddm stride",
        )
        axis.set_xlabel(
            f"experimental data (negative for {get_discrete_observation_names(cfg)[0]} {x_discr[0]})"
        )

    else:
        print("not implemented yet")
        # Compute and normalize histograms for each choice, plot afterwards
        # colors = sns.color_palette('pastel')[:len(pdfs)]
        # for i, xd in enumerate(x_discr):
        #     xd_data = data.loc[data.loc[:, get_discrete_observation_names(cfg)].values == xd]
        #     axis.hist(xd_data.loc[:, get_continuous_observation_names(cfg)].values, bins=np.arange(data_min, data_max, 0.1), \
        #            density=False, alpha=0.7, color=colors[i])
        #     axis.plot(x_cont_range, pdfs[i], color=colors[i], label='pdf and data for choice ' + str(xd))
        return

    axis.legend(loc="upper left", frameon=False)
    axis.set_ylabel("pdf")
    ylim = np.max(np.vstack(pdfs)) + 0.5
    axis.set_ylim(0, ylim)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    if exp_cond:
        axis.set_title(f"conditions: {cfg['task']['group_by']} = {exp_cond}")

    return


def plot_samples(
    cfg: DictConfig,
    posterior_samples: np.ndarray,
    prior_samples: np.ndarray,
    reference_points: List[np.ndarray],
    fig,
    grid=None,
    title=None,
    legend=["posterior", "map", "prior"],
) -> None:
    """Plot pdf on experimental data as well as the posterior over the parameters.
    The last column in data is supposed to

    Parameters
    ----------
    cfg:
        Config file passed by hydra.
    posterior_samples:
        Samples from the posterior over each parameter.
    prior_samples:
        Samples from the prior over each parameter.
    reference_points:
        Reference for posterior samples, e.g. ground truth or posterior samples from another source.
    fig:
        Matplotlib figure to plot on.
    grid:
        Matplotlib grid to plot on, if available.
    title:
        Title for each grid.
    """

    n_subplots = len(get_parameter_names(cfg))
    if grid == None:
        grid = fig.add_gridspec(n_subplots, n_subplots, wspace=0.09, hspace=0.1)
        axis = grid.subplots()
    else:
        inner_grid = grid.subgridspec(n_subplots, n_subplots, wspace=0.09, hspace=0.1)
        axis = inner_grid.subplots()

    # Plot posterior samples
    _, _ = pairplot(
        samples=posterior_samples,
        samples_colors=["steelblue"],
        points=reference_points,
        points_colors=["palevioletred"] * len(reference_points),
        diag="hist",
        upper="scatter",
        hist_diag=dict(bins=20, density=True, histtype="stepfilled", alpha=1),
        points_diag=dict(alpha=1, linewidth=2),
        points_offdiag=dict(marker=".", markersize=7),
        labels=[theta for theta in get_parameter_names(cfg)],
        fig_subplots_adjust={
            "top": 0.98,
        },
        fig=fig,
        axes=axis,
    )

    # Plotting the prior samples will set the plot limits to the prior range.
    if prior_samples is not None:

        all_samples = np.vstack((prior_samples, posterior_samples))

        # Plot prior samples as a reference
        _, _ = pairplot(
            samples=prior_samples,
            samples_colors=["gainsboro"],
            diag="hist",
            upper="contour",
            hist_diag=dict(bins=20, density=True),
            contour_offdiag=dict(levels=[1], alpha=0, color="white"),
            labels=[theta for theta in get_parameter_names(cfg)],
            limits=[
                (np.min(all_samples[:, i]), np.max(all_samples[:, i]))
                for i in range(all_samples.shape[1])
            ],
            fig=fig,
            axes=axis,
        )

    if title:
        mid_idx = int((n_subplots - 1) / 2)
        axis[0, mid_idx].set_title(title, fontsize=18)

    axis[0, -1].legend(legend)

    return
