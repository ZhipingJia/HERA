"""YAML quantization-config loader (vendored from the experiment code).

Merges a default quantization YAML with one or more override files and returns an
attribute-accessible config (``cfg.quan.conv.weight.bit`` style) used by
``prepare_quant_model2``.
"""

from __future__ import annotations

import os

import munch
import yaml


def merge_nested_dict(d, other):
    new = dict(d)
    for k, v in other.items():
        if d.get(k, None) is not None and type(v) is dict:
            new[k] = merge_nested_dict(d[k], v)
        else:
            new[k] = v
    return new


def get_config(default_file, config_file):
    with open(default_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for f in config_file:
        if not os.path.isfile(f):
            raise FileNotFoundError('Cannot find a configuration file at', f)
        with open(f) as yaml_file:
            c = yaml.safe_load(yaml_file)
            cfg = merge_nested_dict(cfg, c)

    return munch.munchify(cfg)
