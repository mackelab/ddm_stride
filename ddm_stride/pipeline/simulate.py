import os
from pathlib import Path
from typing import Any, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ddm_stride.ddm_models import *
from ddm_stride.ddm_models.base_simulator import Simulator
from ddm_stride.pipeline.infer import build_proposal
from ddm_stride.sbi_extensions.simulate_iid import simulate_for_sbi_iid
from ddm_stride.utils.config import results_path
from ddm_stride.utils.data_names import *
from sbi.inference import prepare_for_sbi, simulate_for_sbi


def simulate(cfg: DictConfig):
    """Uses the currently active config file to sample parameters and simulate data given those parameters.

    Parameters
    ----------
    cfg:
        The config file passed via hydra.
    """

    os.makedirs("simulation_data", exist_ok=True)

    proposal = build_proposal(cfg, "cpu")
    simulator = build_simulator(cfg)
    simulator, proposal = prepare_for_sbi(simulator, proposal)

    # Load previously simulated data if available
    (
        simulation_train_data,
        simulation_test_data,
        simulation_iid_test_data,
    ) = load_simulation_data(cfg, simulation_stage=True, drop_invalid_data=True)

    if cfg["task"]["sim_training_data_params"]["num_simulations"] > 0:
        # Simulate new training data, save data
        simulation_train_data = simulate_append(
            cfg, simulator, proposal, "sim_training_data_params", simulation_train_data
        )
        simulation_train_data.to_csv(
            Path(
                results_path()
                + cfg["result_folder"]
                + "/simulation_data/training_data.csv"
            ),
            index=False,
        )

    if cfg["task"]["sim_test_data_params"]["num_simulations"] > 0:
        # Simulate new test data, save data
        simulation_test_data = simulate_append(
            cfg, simulator, proposal, "sim_test_data_params", simulation_test_data
        )
        simulation_test_data.to_csv(
            Path(
                results_path() + cfg["result_folder"] + "/simulation_data/test_data.csv"
            ),
            index=False,
        )

    if cfg["task"]["sim_iid_test_data_params"]["num_params"] > 0:
        # Simulate new iid test data, save data
        simulation_iid_test_data = simulate_append(
            cfg,
            simulator,
            proposal,
            "sim_iid_test_data_params",
            simulation_iid_test_data,
        )
        simulation_iid_test_data.to_csv(
            Path(
                results_path()
                + cfg["result_folder"]
                + "/simulation_data/iid_test_data.csv"
            ),
            index=False,
        )

    return


def build_simulator(cfg: DictConfig) -> Simulator:
    """Instantiate the simulator.

    Parameters
    ----------
    cfg:
        Config file passed by hydra

    Returns
    -------
    simulator:
        A simulator takes parameters $\theta$ and inputs $\pi$ and maps them to simulations, or observations,
        `x`, $\text{sim}(\theta, \pi)\to x$.
    """

    inputs = get_param_exp_cond_names(cfg)
    simulator_results = get_observation_names(cfg)
    simulator = hydra.utils.instantiate(
        cfg["ddm_model"]["simulator"],
        inputs=inputs,
        simulator_results=simulator_results,
    )

    return simulator


def load_simulation_data(
    cfg: DictConfig, simulation_stage: bool = False, drop_invalid_data: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the simulation training data and the simulation test data.

    Parameters
    ----------
    cfg:
        Config file passed by hydra
    simulation_stage:
        Specifies if data is loaded for the simulation stage.
    drop_invalid_data:
        Exclude rows containing inf and NaN

    Returns
    -------
    training_data:
        DataFrame containing the simulated training data or empty DataFrame if no simulated training data available.
    test_data:
        DataFrame containing the simulated test data or empty DataFrame if no simulated test data available.
    test_data:
        DataFrame containing the simulated test data containing multiple observations per parameter or empty DataFrame if no simulated test data available.
    """

    dfs = []
    # Column names of simulation data
    col_names = get_data_names(cfg)

    # Load one dataframe
    def load_df(simulation_folder, simulation_file, error_msg):

        filepath = Path(results_path() + simulation_folder + simulation_file)
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(error_msg)
            raise

        # Sort columns and check if all values used for tranining and testing are contained in the dataframe
        try:
            df = df[col_names]
        except KeyError:
            print(
                f"""Not all parameters, inputs and observations specified in cfg[ddm_model] can be found in the loaded training and test dataframes.
                If you are trying to reuse previously simulated data, make sure you use the same parameters, inputs and observations.
                """
            )
            raise

        # Drop invalid values similar to handle_invalid_x in the sbi package
        if drop_invalid_data:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(axis=0, how="any", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    # Load all data
    for dir, n_sim, file in zip(
        [
            cfg["task"]["presim_training_data_path"],
            cfg["task"]["presim_test_data_path"],
            cfg["task"]["presim_iid_test_data_path"],
        ],
        [
            cfg["task"]["sim_training_data_params"]["num_simulations"],
            cfg["task"]["sim_test_data_params"]["num_simulations"],
            cfg["task"]["sim_iid_test_data_params"]["num_params"],
        ],
        ["/training_data.csv", "/test_data.csv", "/iid_test_data.csv"],
    ):
        # After the simulation stage data is stored in the current results folder unless n_sim = 0
        if not simulation_stage and n_sim > 0:
            error_msg = f"""No simulation data can be found for the current run. 
            Run the simulation stage first or specify a path to load the data from and set num_simulaions to 0."""
            dfs.append(
                load_df(cfg["result_folder"] + "/simulation_data/", file, error_msg)
            )
        # Load data from the path specified in dir
        elif dir:
            error_msg = f"""Simulation data file specified in cfg[task][{dir}] with path {dir + file} not found."""
            dfs.append(load_df(dir, file, error_msg))
        # Create empty DataFrames in case there is no previously simulated data available
        else:
            dfs.append(pd.DataFrame(columns=col_names))

    return tuple(dfs)


def simulate_append(
    cfg: DictConfig,
    simulator: Simulator,
    proposal: Any,
    sim_params: str,
    simulation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Load the simulation training data and the simulation test data.

    Parameters
    ----------
    cfg:
        Config file passed by hydra
    simulator:
        A simulator takes parameters $\theta$ and input values $\pi$ and maps them to
        simulations, or observations, `x`, $\text{sim}(\theta, \pi)\to x$.
    proposal:
        Probability distribution that the parameters $\theta$ and $\pi$ are sampled from.
    sim_params:
        One of {"sim_training_data_params", "sim_test_data_params", "sim_iid_test_data_params"}.
    simulation_data:
        Dataframe containing previously simulated data or empty dataframe if no simulated data is available.

    Returns
    -------
    simulation_data:
        DataFrame containing the previously simulated data and/or new simulation data.
    """
    data = []

    if sim_params == "sim_iid_test_data_params":
        params, observations = simulate_for_sbi_iid(
            simulator=simulator,
            proposal=proposal,
            num_iid_simulations_per_param=500,
            **cfg["task"][sim_params],
        )
        data = np.hstack((params, observations))

    else:
        params, observations = simulate_for_sbi(
            simulator=simulator, proposal=proposal, **cfg["task"][sim_params]
        )
        data = np.hstack((params, observations))

    new_simulation_data = pd.DataFrame(data, columns=get_data_names(cfg))

    simulation_data = pd.concat(
        [simulation_data, new_simulation_data], ignore_index=True
    )

    return simulation_data
