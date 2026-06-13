"""ACIM layer-construction utilities for the Faster R-CNN workload.

These helpers replace a quantized DCNM layer (``conv2d_quant_noise`` /
``linear_quant_noise``) with its ACIM sample-noise counterpart, cloning the LSQ
quantizers into the ACIM scale convention.  They are used to rebuild a hybrid
ACIM/DCNM detector for evaluation (``eval_voc.py``) and for hybrid QAT
(``train_dcnm_int8_qat.py``); they are model-construction utilities, independent
of any layer-selection / profiling logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn

from quantization_and_noise.quant_layer import (
    conv2d_quant_noise,
    conv2d_quant_sample_noise,
    linear_quant_noise,
    linear_quant_sample_noise,
)
from quantization_and_noise.quant_util import LSQ_act_quantizer, LSQ_weight_quantizer


def to_jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def get_submodule_by_name(model: nn.Module, full_name: str) -> nn.Module:
    module = model
    for part in full_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def parse_layer_indices(spec: str, target_layers: list[dict]) -> list[int]:
    if spec.strip().lower() == "all":
        return [int(layer["index"]) for layer in target_layers]
    selected: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            selected.extend(range(int(start), int(end) + 1))
        else:
            selected.append(int(item))
    valid = {int(layer["index"]) for layer in target_layers}
    bad = sorted(set(selected) - valid)
    if bad:
        raise ValueError(f"Unknown layer indices: {bad}")
    return selected


def set_submodule_by_name(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _infer_lsq_kwargs(old_quantizer: nn.Module, bit: int, int_flag: bool) -> dict:
    thd_neg = int(getattr(old_quantizer, "thd_neg"))
    thd_pos = int(getattr(old_quantizer, "thd_pos"))
    all_positive = thd_neg == 0
    symmetric = (not all_positive) and (abs(thd_neg) == abs(thd_pos))
    return {
        "bit": bit,
        "all_positive": all_positive,
        "symmetric": symmetric,
        "per_channel": bool(getattr(old_quantizer, "per_channel", False)),
        "noise_scale": 0,
        "noise_method": getattr(old_quantizer, "noise_method", "add"),
        "noise_range": getattr(old_quantizer, "noise_range", "max"),
        "s_init": getattr(old_quantizer, "s_init", 2),
        "init_mode": getattr(old_quantizer, "init_mode", "origin"),
        "init_percent": getattr(old_quantizer, "init_percent", 0.95),
        "int_flag": int_flag,
    }


def clone_lsq_act_for_acim(old_quantizer: LSQ_act_quantizer) -> LSQ_act_quantizer:
    quantizer = LSQ_act_quantizer(**_infer_lsq_kwargs(old_quantizer, bit=5, int_flag=True))
    quantizer.s = nn.Parameter(old_quantizer.s.detach().clone() * 8.0)
    return quantizer


def clone_lsq_weight_for_acim(old_quantizer: LSQ_weight_quantizer) -> LSQ_weight_quantizer:
    quantizer = LSQ_weight_quantizer(**_infer_lsq_kwargs(old_quantizer, bit=4, int_flag=True))
    quantizer.s = nn.Parameter(old_quantizer.s.detach().clone() * 16.0)
    return quantizer


def parse_layer_value_overrides(spec: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not spec:
        return overrides
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad override item '{item}', expected '<layer-index-or-name>:<value>'")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Bad override item '{item}', expected '<layer-index-or-name>:<value>'")
        overrides[key] = value
    return overrides


def layer_override_value(layer: dict, spec: str, default: str) -> str:
    overrides = parse_layer_value_overrides(spec)
    index_key = str(layer.get("index", ""))
    name_key = str(layer.get("full_name", ""))
    short_key = str(layer.get("short_name", ""))
    if index_key in overrides:
        return overrides[index_key]
    if name_key in overrides:
        return overrides[name_key]
    if short_key in overrides:
        return overrides[short_key]
    return default


def replace_one_layer_with_acim(model: nn.Module, layer: dict, args: argparse.Namespace) -> dict:
    full_name = layer["full_name"]
    module = get_submodule_by_name(model, full_name)
    if not isinstance(module, (conv2d_quant_noise, linear_quant_noise)):
        raise TypeError(f"{full_name} is {type(module).__name__}, expected quant_noise module")
    if not isinstance(module.a_quantizer, LSQ_act_quantizer):
        raise TypeError(f"{full_name}.a_quantizer is {type(module.a_quantizer).__name__}, expected LSQ_act_quantizer")
    if not isinstance(module.w_quantizer, LSQ_weight_quantizer):
        raise TypeError(f"{full_name}.w_quantizer is {type(module.w_quantizer).__name__}, expected LSQ_weight_quantizer")

    old_act_scale = module.a_quantizer.s.detach().clone()
    old_weight_scale = module.w_quantizer.s.detach().clone()
    a_quantizer = clone_lsq_act_for_acim(module.a_quantizer)
    w_quantizer = clone_lsq_weight_for_acim(module.w_quantizer)
    sample_out_scale = 2.0 * a_quantizer.s.detach() * w_quantizer.s.detach()
    sample_noise_kwargs = {
        "noise_std": args.sample_noise_std,
        "scale_factor": args.sample_noise_scale,
    }
    sample_output_min = layer_override_value(
        layer,
        getattr(args, "sample_output_min_overrides", ""),
        getattr(args, "sample_output_min", None),
    )
    sample_output_max = layer_override_value(
        layer,
        getattr(args, "sample_output_max_overrides", ""),
        getattr(args, "sample_output_max", None),
    )
    if sample_output_min is not None:
        sample_noise_kwargs["output_min"] = float(sample_output_min)
    if sample_output_max is not None:
        sample_noise_kwargs["output_max"] = float(sample_output_max)
    sample_noise_mode = layer_override_value(
        layer,
        getattr(args, "sample_noise_mode_overrides", ""),
        args.sample_noise_mode,
    )

    if isinstance(module, conv2d_quant_noise):
        new_module = conv2d_quant_sample_noise(
            module,
            w_quantizer=w_quantizer,
            a_quantizer=a_quantizer,
            a_out_quantizer=getattr(module, "a_out_quantizer", None),
            int_flag=True,
            sample_out_scale=sample_out_scale,
            sample_noise_mode=sample_noise_mode,
            sample_noise_kwargs=sample_noise_kwargs,
        )
    else:
        new_module = linear_quant_sample_noise(
            module,
            w_quantizer=w_quantizer,
            a_quantizer=a_quantizer,
            a_out_quantizer=getattr(module, "a_out_quantizer", None),
            int_flag=True,
            sample_out_scale=sample_out_scale,
            sample_noise_mode=sample_noise_mode,
            sample_noise_kwargs=sample_noise_kwargs,
        )

    set_submodule_by_name(model, full_name, new_module)
    return {
        "full_name": full_name,
        "old_act_scale": to_jsonable(old_act_scale),
        "old_weight_scale": to_jsonable(old_weight_scale),
        "acim_act_scale": to_jsonable(a_quantizer.s.detach()),
        "acim_weight_scale": to_jsonable(w_quantizer.s.detach()),
        "sample_out_scale": to_jsonable(sample_out_scale),
        "sample_noise_mode": sample_noise_mode,
        "sample_noise_kwargs": sample_noise_kwargs,
        "new_class": type(new_module).__name__,
    }
