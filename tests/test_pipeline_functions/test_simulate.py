import torch

from ddm_stride.pipeline.infer import build_prior, build_proposal
from ddm_stride.pipeline.simulate import *


def test_simulate(toy_config):

    ## Test build_simulator
    simulator = build_simulator(toy_config)
    prior = build_prior(toy_config, "cpu")
    proposal = build_proposal(toy_config, "cpu")

    theta = prior.sample((25,))
    sim_res = simulator(theta)

    # Check that simulator returns a tensor
    assert isinstance(sim_res, torch.Tensor), "Simulation result should be a tensor"
    # Check that number of simulation results is equal to number of inputs
    assert sim_res.shape[0] == theta.shape[0], "Incorrect number of simulation results"

    ## Test load_simulation_data
    train_data, test_data, test_data_iid = load_simulation_data(toy_config)

    # Check that dataframes are returned
    assert (
        isinstance(train_data, pd.DataFrame)
        and isinstance(test_data, pd.DataFrame)
        and isinstance(test_data_iid, pd.DataFrame)
    ), "Data needs to be a pandas dataframe"
    # Check that data contains no NaNs
    assert not (
        train_data.isna().any().any()
        or test_data.isna().any().any()
        or test_data_iid.isna().any().any()
    ), "Data contains NaNs"

    ## Test simulate_append
    simulation_train_data = simulate_append(
        toy_config, simulator, proposal, "sim_training_data_params", train_data
    )

    # Check that dataframe is returned
    assert isinstance(
        simulation_train_data, pd.DataFrame
    ), "Data needs to be a pandas dataframe"
    # Check if number of simulated data is correct
    assert (
        simulation_train_data.shape[0]
        == train_data.shape[0]
        + toy_config["task"]["sim_training_data_params"]["num_simulations"]
    ), "Incorrect number of simulations: do new simulations get appended to previous simulations?"
    # Check if column names have been kept the same
    assert all(
        simulation_train_data.columns == train_data.columns
    ), "Columns of new simulations need to correspond to columns of previous simulations"
