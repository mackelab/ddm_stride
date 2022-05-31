import os
from pathlib import Path

from ddm_stride.utils.config import *


def test_load_config():

    cfg1 = load_config_file("toy_example")
    cfg2 = load_config_file("toy_example", "ddm_model")

    # Check if config is dict
    assert isinstance(cfg1, dict)
    assert isinstance(cfg2, dict)


def test_path():

    ddm_stride_path_ = Path(os.environ["BASE_DIR"] + "/ddm_stride")
    results_path_ = results_path()
    data_path_ = data_path()

    # Check if relevant folders exist at the specified path
    assert os.path.isdir(
        ddm_stride_path_
    ), "ddm_stride directory not found in msc directory"
    assert os.path.isdir(results_path_), "results directory not found in msc directory"
    assert os.path.isdir(data_path_), "data directory not found in msc directory"
