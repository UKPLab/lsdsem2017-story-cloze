from os import path

import yaml

from util import merge_dicts


def load_config(config_path):
    """Loads and reads a yaml configuration file which extends the default_config of this project

    :param config_path: path of the configuration file to load
    :type config_path: str
    :return: the configuration dictionary
    :rtype: dict
    """

    with open(config_path, 'r') as user_config_file:
        return read_config(user_config_file.read())


def read_config(config_str):
    """Reads a yaml configuration string which extends the default_config string, if there is one

    :param config_str: the configuration to read
    :type config_str: str
    :return: the configuration dictionary
    :rtype: dict
    """
    user_config = yaml.load(config_str)
    return user_config
