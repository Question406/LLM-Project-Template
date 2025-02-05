import os
import structlog
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, ListConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from .log import configure_structlog

def init_script(hparams):
    #* Initialize all configs and return a structlog LOGGER object
    OmegaConf.resolve(hparams)
    unfilled_paths= find_unfilled_paths(hparams)
    if len(unfilled_paths) > 0:
        err = "\n".join(
            [f"{'.'.join(map(str, path))}" for path in unfilled_paths]
        )
        raise ValueError(f"Unfilled paths in config:\n {err}")
    hydraconf = HydraConfig.get()
    configure_structlog(f"{hydraconf.runtime.output_dir}/{hydraconf.job.name}.log")
    LOGGER = structlog.getLogger()
    return LOGGER

def find_unfilled_paths(conf, path=None):
    if path is None:
        path = []
    paths_with_unfilled = []

    if isinstance(conf, DictConfig):
        for key in conf.keys():
            try:
                value = conf[key]
            except Exception as e:
                value = '???'
            new_path = path + [key]  
            if isinstance(value, (DictConfig, ListConfig)):
                paths_with_unfilled.extend(find_unfilled_paths(value, new_path))  
            elif value == '???':
                paths_with_unfilled.append(new_path)
    elif isinstance(conf, ListConfig):
        for index in range(len(conf)):
            try:
                item = conf[index]
            except Exception as e:
                item = '???'
            new_path = path + [index]
            if isinstance(item, (DictConfig, ListConfig)):
                paths_with_unfilled.extend(find_unfilled_paths(item, new_path))
            elif item == '???':
                paths_with_unfilled.append(new_path)
        
    return paths_with_unfilled

def set_progress(disable=False):
    return Progress(
        TextColumn("[bold blue]{task.fields[name]}", justify="left"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        disable=disable,
    ) 