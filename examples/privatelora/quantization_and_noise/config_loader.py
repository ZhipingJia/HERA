"""YAML quantization-config loader for the PrivateLoRA INT8/CGRA pipeline.

Vendored from the experiment code: merges the base quant YAML, the unified-CGRA
lora_mobile YAML, and the optional Head LB (lora_lm_head_B) YAML into one
attribute-accessible config consumed by ``prepare_quant_model_unified_cgra``.
"""

from __future__ import annotations

import os

import munch
import yaml


def get_config_unified_cgra(config_file, config_file_2, lm_head_file=None):
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Cannot find a configuration file at', config_file)
    with open(config_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    if not os.path.isfile(config_file_2):
        raise FileNotFoundError(f"Cannot find a configuration file at {config_file_2}")
    with open(config_file_2) as yaml_file:
        cfg_2 = yaml.safe_load(yaml_file)
    cfg['quan']['lora_mobile_config_2'] = cfg_2['quan']['lora_mobile']

    if lm_head_file:
        if not os.path.isfile(lm_head_file):
            raise FileNotFoundError(f"Cannot find a configuration file at {lm_head_file}")
        with open(lm_head_file) as yaml_file:
            lm_head_cfg = yaml.safe_load(yaml_file)
        cfg['quan']['lora_lm_head_B'] = lm_head_cfg['quan']['lora_lm_head_B']

    return munch.munchify(cfg)
