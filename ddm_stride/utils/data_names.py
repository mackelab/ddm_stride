from typing import List

from omegaconf import DictConfig


def get_parameter_names(cfg: DictConfig) -> List[str]:
    """Names of all parameters theta. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        parameter_names:
            List of parameter names.
    """
    if cfg["ddm_model"]["parameters"]:
        return [
            param["name"] for param in cfg["ddm_model"]["parameters"] if param["name"]
        ]
    else:
        return []


def get_experimental_condition_names(cfg: DictConfig) -> List[str]:
    """Names of all experimental conditions pi. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        experimental_condition_names:
            List of experimental condition names.
    """
    if cfg["ddm_model"]["experimental_conditions"]:
        return [
            param["name"]
            for param in cfg["ddm_model"]["experimental_conditions"]
            if param["name"]
        ]
    else:
        return []


def get_param_exp_cond_names(cfg: DictConfig) -> List[str]:
    """Names of all parameters theta and experimental conditions pi. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        param_exp_cond_names:
            List containing first the names of parameters and second the names of experimental conditions.
    """
    return get_parameter_names(cfg) + get_experimental_condition_names(cfg)


def get_continuous_observation_names(cfg: DictConfig) -> List[str]:
    """Names of all continuous observations x. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        continuous_observation_names:
            List containing the names of continuous observations.
    """
    if cfg["ddm_model"]["observations"]:
        return [
            param["name"]
            for param in cfg["ddm_model"]["observations"]
            if param["variable_type"] == "continuous"
        ]
    else:
        return []


def get_discrete_observation_names(cfg: DictConfig) -> List[str]:
    """Names of all discrete observations x. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        discrete_observation_names:
            List containing the names of discrete observations.
    """
    if cfg["ddm_model"]["observations"]:
        return [
            param["name"]
            for param in cfg["ddm_model"]["observations"]
            if param["variable_type"] == "discrete"
        ]
    else:
        return []


def get_observation_names(cfg: DictConfig) -> List[str]:
    """Names of all parameters theta and experimental conditions pi. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        param_exp_cond_names:
            List containing first the names of continuous observations and second the names of discrete observations.
    """
    return get_continuous_observation_names(cfg) + get_discrete_observation_names(cfg)


def get_data_names(cfg: DictConfig) -> List[str]:
    """Names of all parameters theta, experimental conditions pi and observations x. These names are used for loading, saving and sorting
    dataframes containing simulations or experimental data.

        Parameters
        ----------
        cfg:
            The config file passed via hydra.

        Returns
        -------
        data_names:
            List containing first the names of parameters, second the names of experimental conditions,
            third the names of continuous observations and last the names of discrete observations.
    """
    return get_param_exp_cond_names(cfg) + get_observation_names(cfg)
