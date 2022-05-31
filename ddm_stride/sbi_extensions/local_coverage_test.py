import copy
from typing import Callable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch.distributions import Uniform

from sbi.utils.torchutils import atleast_2d_float32_tensor


def local_coverage_test(
    xs_test: Tensor,
    xs_train: Tensor,
    xs_ranks: Tensor,
    num_posterior_samples: int = 300,
    alphas: Tensor = torch.linspace(0.05, 0.95, 21),
    classifier: Callable = LogisticRegression(
        penalty="none", solver="saga", max_iter=10000
    ),
    null_distr_samples: int = 500,
    device: str = "cpu",
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Compute local coverage tests using the ranks computed by sbc.
    Returns for each dimension of theta the global and local p-values as well as quantile predictions based on sbc ranks at test points xs_test and quantile predictions based on uniform samples at test points xs_test.
    The local coverage test is implemented as proposed in Zhao et al., "Validating Conditional Density Models and Bayesian Inference Algorithms", https://proceedings.mlr.press/v161/zhao21b/zhao21b.pdf.
    Parameters
    ----------
    xs_test:
        Test observations the posterior will be evaluated at.
    xs_train:
        Training observations used as input to run_sbc.
    xs_ranks:
        Ranks returned by run_sbc.
    num_posterior_samples:
        Number of posterior samples used for run_sbc.
    alphas:
        Posterior quantiles that will be compared to the normalized sbc ranks.
        A linspace from 0 to 1 might lead to errors due to no ranks being smaller/larger than 0/1.
    classifier:
        Regression classifier that will be used for predicting the posterior
        quantiles at the test observations based on normalized sbc ranks or uniform samples.
    null_distr_samples:
        Determines how many uniform test statistics will be used for computing the p-values.
        Reasonable values for null_distr_samples might lie in (200, 1000).
    device:
        cuda or cpu
    Returns
    -------
    global_pvalues_per_dim:
        List of p-values per dimension of theta averaged over all test observations. These values are supposed to not be significant if the posterior is correct.
    local_pvalues_per_dim:
        List of p-values per dimension of theta and test observation. These values are supposed to not be significant if the posterior is correct.
    rank_predictions_per_dim:
        List of posterior quantile predictions per alpha per dimension of theta, xs_test point and alpha quantile. The predictions are based on the normalized sbc ranks.
    uniform_predictions_per_dim:
        List of alpha predictions per dimension of theta, null_distr_samples value, xs_test point and alpha quantile. The predictions are based on uniform distribution samples.
    """
    xs_test = atleast_2d_float32_tensor(xs_test)
    xs_train = atleast_2d_float32_tensor(xs_train)
    xs_ranks = atleast_2d_float32_tensor(xs_ranks)

    rank_predictions_per_dim = []
    uniform_predictions_per_dim = []
    local_pvalues_per_dim = []
    global_pvalues_per_dim = []

    for dim in range(xs_ranks.shape[1]):

        # Select dim, normalize ranks
        ranks = torch.ravel(xs_ranks[:, dim]) / num_posterior_samples

        ### Calculate local test at points of interest xs_test
        rank_predictions = torch.zeros(size=(xs_test.shape[0], len(alphas)))

        for i, alpha in enumerate(alphas):
            # Fit training samples and PIT indicators/ranks
            ind_train = [1 * (rank <= alpha) for rank in ranks]

            # If all ind_train are 0 or 1, no classifier needs to be trained
            if np.sum(ind_train) == len(ind_train):
                print(f"all ranks are smaller than {alpha}")
                rank_predictions[:, i] = 1
            elif np.sum(ind_train) == 0:
                print(f"all ranks are larger than {alpha}")
                rank_predictions[:, i] = 0
            else:
                rhat_rank = copy.deepcopy(classifier)
                rhat_rank.fit(X=xs_train, y=ind_train)

                # Predict on test samples
                pred = rhat_rank.predict_proba(xs_test)[:, 1]
                rank_predictions[:, i] = torch.FloatTensor(pred)

        # Compute test statistic T for the rank predictions
        T_rank = torch.mean((rank_predictions - alphas) ** 2, dim=1)
        # Compute test statistic S for the rank predictions
        S_rank = torch.mean(T_rank)

        rank_predictions_per_dim.append(rank_predictions)

        ### Refit the classifier using Uniform(0,1) values in place of true PIT values/rank
        T_uni = torch.zeros(size=(null_distr_samples, xs_test.shape[0]))
        uniform_predictions = torch.zeros(
            size=(null_distr_samples, xs_test.shape[0], len(alphas))
        )

        for b in range(null_distr_samples):

            uniform_predictions_b = torch.zeros(size=(xs_test.shape[0], len(alphas)))

            # Sample from uniform distribution instead of using PIT values/ranks
            uni_sample = Uniform(0, 1).sample((xs_ranks.shape[0],))

            for i, alpha in enumerate(alphas):
                # Fit training samples and uniform indicators
                ind_train = [1 * (sample <= alpha) for sample in uni_sample]

                # If all ind_train are 0 or 1, no classifier needs to be trained
                if np.sum(ind_train) == len(ind_train):
                    print(f"all ranks are smaller than {alpha}")
                    uniform_predictions_b[:, i] = 1
                elif np.sum(ind_train) == 0:
                    print(f"all ranks are larger than {alpha}")
                    uniform_predictions_b[:, i] = 0
                else:
                    rhat_uni = copy.deepcopy(classifier)
                    rhat_uni.fit(X=xs_train, y=ind_train)

                    # Predict on test samples
                    preds = rhat_uni.predict_proba(xs_test)[:, 1]
                    uniform_predictions_b[:, i] = torch.FloatTensor(preds)

            # Compute test statistic T for uniform samples
            T_uni[b] = torch.mean((uniform_predictions_b - alphas) ** 2, dim=1)
            # Save predictions in order to compute confidence bands
            uniform_predictions[b] = uniform_predictions_b

        # Compute test statistic S for uniform samples
        S_uni = torch.mean(T_uni, dim=1)

        uniform_predictions_per_dim.append(uniform_predictions)

        # Compute local p-value
        local_pvalues = torch.mean(1.0 * (T_rank < T_uni), dim=0)
        local_pvalues_per_dim.append(local_pvalues)
        # Compute global p-value
        global_pvalue = torch.mean(1.0 * (S_rank < S_uni))
        global_pvalues_per_dim.append(global_pvalue)

    return (
        global_pvalues_per_dim,
        local_pvalues_per_dim,
        rank_predictions_per_dim,
        uniform_predictions_per_dim,
    )
