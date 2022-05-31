import os

import hydra

from ddm_stride.pipeline.diagnose import diagnose
from ddm_stride.pipeline.evaluate import evaluate
from ddm_stride.pipeline.simulate import simulate
from ddm_stride.pipeline.train import train


@hydra.main(config_path="../config", config_name="config")
def run_pipeline(cfg):

    if cfg["run_simulate"]:
        simulate(cfg)
    if cfg["run_train"]:
        train(cfg)
    if cfg["run_diagnose_fast"] or cfg["run_diagnose_slow"]:
        diagnose(cfg)
    if cfg["run_evaluate"]:
        evaluate(cfg)


if __name__ == "__main__":

    run_pipeline()
