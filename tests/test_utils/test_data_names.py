from ddm_stride.utils.config import load_config_file
from ddm_stride.utils.data_names import *


def test_data_names(toy_config):

    data_names = get_data_names(toy_config)

    # Data names should be a list
    assert isinstance(data_names, list)
    # Check for unique entries
    assert len(set(data_names)) == len(data_names)
    # Check correspondence to toy_config entries
    assert all(
        [el["name"] in data_names for el in toy_config["ddm_model"]["parameters"]]
    )
