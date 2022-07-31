import json

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import sbi.utils
from ddm_stride.pipeline.infer import build_proposal
from ddm_stride.pipeline.simulate import load_simulation_data
from ddm_stride.utils.data_names import *
from sbi.inference.snle import MNLE
from sbi.neural_nets.flow import *
from sbi.utils.get_nn_models import likelihood_nn
from sbi.utils.torchutils import atleast_2d_float32_tensor


def train(cfg: DictConfig):

    device = "cpu"

    # If a pretrained model is used, skip the training step
    if cfg["task"]["model_path"]:
        return

    # Train model using hyperparameter search by wandb
    else:
        training_data, _, _ = load_simulation_data(cfg, drop_invalid_data=True)
        proposal = build_proposal(cfg, device)

        params = training_data.loc[:, get_param_exp_cond_names(cfg)]
        params = atleast_2d_float32_tensor(params.values)
        observations = training_data.loc[:, get_observation_names(cfg)]
        observations = atleast_2d_float32_tensor(observations.values)

        wandb.login()
        dict = OmegaConf.to_container(cfg["algorithm"]["wandb"])
        # Specify the project name
        sweep_id = wandb.sweep(
            dict, project=str(cfg["result_folder"]).replace("/", "_")
        )

        if cfg["algorithm"]["wandb"]["metric"]["goal"] != "maximize":
            raise ValueError("validation log prob has to be maximized")

        # Save best sweep log prob, config and model state dict
        best_sweep_info = {"best_val_log_prob": -np.inf}

        # Training loop using the currently selected hyperparameters
        def train_wandb(config=None, best_sweep_info=best_sweep_info):

            with wandb.init(config=config):

                wandb_config = wandb.config

                model_hyperparams = cfg["algorithm"]["model_hyperparams"]
                density_estimator_build = likelihood_nn(
                    **model_hyperparams,
                    hidden_features=wandb_config.hidden_features,
                    num_transforms=wandb_config.num_transforms,
                    hidden_layers=wandb_config.hidden_layers
                )

                inference = MNLE(
                    prior=proposal,
                    density_estimator=density_estimator_build,
                    device=device,
                    logging_level="WARNING",
                    summary_writer=None,
                    show_progress_bars=True,
                )
                inference = inference.append_simulations(params, observations)

                density_estimator = inference.train(
                    **cfg["algorithm"]["train_params"],
                    training_batch_size=wandb_config.training_batch_size,
                    learning_rate=wandb_config.learning_rate,
                    validation_fraction=wandb_config.validation_fraction,
                    stop_after_epochs=wandb_config.stop_after_epochs
                )
                val_log_prob = inference._summary["best_validation_log_probs"][-1]
                wandb.log({"validation_log_prob": val_log_prob})

                # Save best hyperparameters and model
                if val_log_prob > best_sweep_info["best_val_log_prob"]:
                    best_sweep_info["best_val_log_prob"] = val_log_prob
                    torch.save(density_estimator.state_dict(), "model_state_dict.pt")
                    wandb_best_config = wandb_config.__dict__["_items"]
                    with open("wandb_config.json", "w") as f:
                        json.dump(wandb_best_config, f)
            return

        # Use the agent to run the training function with different hyperparameter configurations
        wandb.agent(
            sweep_id,
            train_wandb,
            count=cfg["algorithm"]["n_wandb_sweeps"],
            project=str(cfg["result_folder"]).replace("/", "_"),
        )



    return