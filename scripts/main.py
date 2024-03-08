import os
import csv
import json
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import init_script, set_progress

@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(hparams):
    LOGGER = init_script(hparams)
    LOGGER.info("Configs", configs=hparams)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    device = f'cuda:{hparams.gpu.gpu_id}'

    # Main logic

if __name__=="__main__":
    main()