import os
from omegaconf import OmegaConf


def read_configs(config_dir="./configs"):
    conf = OmegaConf.create()
    for f in os.listdir(config_dir):
        if f.endswith(".yaml"):
            tmpconf = OmegaConf.load(os.path.join(config_dir, f))
            try:
                OmegaConf.resolve(tmpconf)
            except Exception as _:
                pass
            if len(conf) == 0:
                conf = tmpconf
            else:
                conf = OmegaConf.merge(conf, tmpconf)
    OmegaConf.resolve(conf)
    return conf
