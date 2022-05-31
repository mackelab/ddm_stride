import json
import os
from pathlib import Path
from typing import Dict

import yaml
from omegaconf import DictConfig, OmegaConf

import ddm_stride.__init__

def load_config_file(
    rel_path: str, stage: str = None, print_config: bool = None
) -> DictConfig:
    """Load a config file from the results folder.

    Parameters
    ----------
    rel_path
        Name of the experiment subfolder inside the results folder that contains the configuration file.
    stage
        Specifies if only part of the config file will be loaded.
    print_config
        Specifies if the config file will be printed to the console.

    Returns
    -------
    config_data
        Content of config file or parts of a config file.
    """

    path = Path(results_path() + rel_path + "/.hydra/config.yaml")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    if stage:
        if print_config:
            print(OmegaConf.to_yaml(config_data[stage]))
        return config_data[stage]
    else:
        if print_config:
            print(OmegaConf.to_yaml(config_data))
        return config_data


def results_path() -> str:
    """Find path to results directory.

    Returns
    -------
    results_path
        Absolute path to results directory.
    """
    results_path = os.environ["DDM_STRIDE_DIR"] + "/results/"
    return results_path


def data_path() -> str:
    """Find path to data directory.

    Returns
    -------
    data_path
        Absolute path to data directory.
    """
    data_path = os.environ["DDM_STRIDE_DIR"] + "/data/"
    return data_path


def load_wandb_config(model_path: str) -> Dict:
    """Load the WandB hyperparameters.

    Parameters
    ----------
    model_path:
        Path to wandb_config.json, relative to the results folder.

    Returns
    -------
    wandb_best_config:
        Hyperparameters found by WandB.
    """

    # Load wandb config of loaded model and save it in the current results folder
    f = open(Path(results_path() + model_path + "/wandb_config.json"))
    wandb_best_config = json.load(f)
    f.close()

    return wandb_best_config
