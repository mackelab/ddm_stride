import pytest

from ddm_stride.utils.config import load_config_file


@pytest.fixture(scope="module")
def toy_config():

    cfg = load_config_file("toy_example")
    return cfg


@pytest.fixture(scope="module")
def toy_folder():

    return "toy_example"
