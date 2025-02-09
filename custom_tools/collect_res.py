# This file defines several tool calsses and functions for collecting results from run result folder.

import os
import glob
import logging
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def collect_config(folder: str):
    config_file = glob.glob(os.path.join(folder, "**", "config.yaml"), recursive=True)
    if len(config_file) > 1:
        LOGGER.warning("Multiple config files found in %s", folder)
        LOGGER.warning("Using the first one: %s", config_file[0])

    config_file = config_file[0]
    config = OmegaConf.load(config_file)
    json_config = OmegaConf.to_container(config, resolve=True)
    return json_config
