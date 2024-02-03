import argparse
from omegaconf import OmegaConf
import torch

def load_config(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/dog.yaml',
                        help="Config file path")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    base_config = OmegaConf.load(config.base_config)
    config = OmegaConf.merge(base_config, config)
    OmegaConf.resolve(config)
    if print_config:
        print("[INFO] loaded config:")
        print(OmegaConf.to_yaml(config))

    return config