import warnings
from typing import Tuple

import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.simulators.simutils import tqdm_joblib


# Iterate over thetas, xs and exp_conds, if available.
# This allows to set exp_cond in posterior.sample similar to set_x.
def run_sbc(
    thetas: Tensor,
    xs: Tensor,
    exp_cond: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    sbc_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Run simulation-based calibration (SBC) (parallelized across sbc runs).
    Returns sbc ranks, log probs of the true parameters under the posterior and samples
    from the data averaged posterior, one for each sbc run, respectively.
    SBC is implemented as proposed in Talts et al., "Validating Bayesian Inference
    Algorithms with Simulation-Based Calibration", https://arxiv.org/abs/1804.06788.
    Args:
        thetas: ground-truth parameters for sbc, simulated from the prior.
        xs: observed data for sbc, simulated from thetas.
        exp_cond: experimental conditions used for simulating.
        posterior: a posterior obtained from sbi.
        num_posterior_samples: number of approximate posterior samples used for ranking.
        num_workers: number of CPU cores to use in parallel for running num_sbc_samples
            inferences.
        sbc_batch_size: batch size for workers.
        show_progress_bar: whether to display a progress over sbc runs.
    Returns:
        ranks: ranks of the ground truth parameters under the inferred posterior.
        dap_samples: samples from the data averaged posterior.
    """
    num_sbc_samples = thetas.shape[0]

    if num_sbc_samples < 1000:
        warnings.warn(
            """Number of SBC samples should be on the order of 100s to give realiable
            results. We recommend using 300."""
        )
    if num_posterior_samples < 100:
        warnings.warn(
            """Number of posterior samples for ranking should be on the order
            of 100s to give reliable SBC results. We recommend using at least 300."""
        )

    assert (
        thetas.shape[0] == xs.shape[0]
    ), "Unequal number of parameters and observations."
    assert (
        exp_cond.shape[0] == thetas.shape[0]
    ), "Unequal number of parameters and experimental conditions."

    thetas_batches = torch.split(thetas, sbc_batch_size, dim=0)
    xs_batches = torch.split(xs, sbc_batch_size, dim=0)
    exp_cond_batches = torch.split(exp_cond, sbc_batch_size, dim=0)

    if num_workers != 1:
        # Parallelize the sequence of batches across workers.
        # We use the solution proposed here: https://stackoverflow.com/a/61689175
        # to update the pbar only after the workers finished a task.
        with tqdm_joblib(
            tqdm(
                thetas_batches,
                disable=not show_progress_bar,
                desc=f"""Running {num_sbc_samples} sbc runs in {len(thetas_batches)}
                    batches.""",
                total=len(thetas_batches),
            )
        ) as _:
            sbc_outputs = Parallel(n_jobs=num_workers)(
                delayed(sbc_on_batch)(
                    thetas_batch,
                    xs_batch,
                    exp_cond_batch,
                    posterior,
                    num_posterior_samples,
                )
                for thetas_batch, xs_batch, exp_cond_batch in zip(
                    thetas_batches, xs_batches, exp_cond_batches
                )
            )
    else:
        pbar = tqdm(
            total=num_sbc_samples,
            disable=not show_progress_bar,
            desc=f"Running {num_sbc_samples} sbc samples.",
        )

        with pbar:
            sbc_outputs = []
            for thetas_batch, xs_batch, exp_cond_batch in zip(
                thetas_batches, xs_batches, exp_cond_batches
            ):
                sbc_outputs.append(
                    sbc_on_batch(
                        thetas_batch,
                        xs_batch,
                        exp_cond_batch,
                        posterior,
                        num_posterior_samples,
                    )
                )
                pbar.update(sbc_batch_size)

    # Aggregate results.
    ranks = []
    dap_samples = []
    for out in sbc_outputs:
        ranks.append(out[0])
        dap_samples.append(out[1])

    ranks = torch.cat(ranks)
    dap_samples = torch.cat(dap_samples)

    return ranks, dap_samples


def sbc_on_batch(
    thetas: Tensor,
    xs: Tensor,
    exp_cond: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int,
) -> Tuple[Tensor, Tensor]:
    """Return SBC results for a batch of SBC parameters and data from prior.
    Args:
        thetas: ground truth parameters.
        xs: corresponding observations.
        exp_cond: corresponding experimental conditions.
        posterior: sbi posterior.
        num_posterior_samples: number of samples to draw from the posterior in each sbc
            run.
    Returns
        ranks: ranks of true parameters vs. posterior samples under the specified RV,
            for each posterior dimension.
        log_prob_thetas: log prob of true parameters under the approximate posterior.
            Note that this is interpretable only for normalized log probs, i.e., when
            using (S)NPE.
        dap_samples: samples from the data averaged posterior for the current batch,
            i.e., a single sample from each approximate posterior.
    """

    dap_samples = torch.zeros_like(thetas)
    ranks = torch.zeros_like(thetas)

    for idx, (tho, xo, exp) in enumerate(zip(thetas, xs, exp_cond)):
        # Draw posterior samples and save one for the data average posterior.
        ths = posterior.sample(
            (num_posterior_samples,), x=xo, exp_cond=exp, show_progress_bars=False
        )

        # Save one random sample for data average posterior (dap).
        dap_samples[idx] = ths[0]

        # rank for each posterior dimension as in Talts et al. section 4.1.
        for dim in range(thetas.shape[1]):
            ranks[idx, dim] = (ths[:, dim] < tho[dim]).sum().item()

    return ranks, dap_samples
