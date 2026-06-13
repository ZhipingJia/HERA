"""VGG16 layer-wise ACIM / DCNM EDP profiling via the shared HERA hardware model.

Shape extraction (forward hooks over Conv2d / Linear) is VGG16-specific; the EDP
itself is computed by ``hera.hardware`` using the calibrated folded energy
coefficients, so no raw device-level power quantities appear here. Per-layer EDP
values match the manuscript's hardware model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hera.hardware import LayerShape, calculate_acim, calculate_dcnm  # noqa: E402
from hera.hardware.hera_config import DEFAULT_HERA_HARDWARE as HW  # noqa: E402
from vgg16_pipeline.layers import FULL_TO_SHORT  # noqa: E402

HARDWARE_MODEL_VERSION = "hera_envelope_vgg16_batch_shape_20260515"


def hardware_constants() -> dict[str, float | int | str]:
    return {"hardware_model_version": HARDWARE_MODEL_VERSION, **HW.as_dict()}


def _acim_edp(flops: float, r_dim: int, c_dim: int, active_rows: float, row_parallel: bool):
    est = calculate_acim(LayerShape("x", r_dim, c_dim, float(active_rows)), row_parallel=row_parallel)
    d = est.details
    detail = {
        "base_rows": HW.acim_base_rows,
        "base_cols": HW.acim_base_cols,
        "max_parallel_groups": HW.acim_max_parallel_groups,
        "arrays_per_vector": d["groups_per_vector"],
        "cycles_per_vector_batch": d["cycles_per_batch"],
        "vector_parallel": d["vector_parallel"],
        "vector_batches": d["vector_batches"],
        "row_parallel": row_parallel,
        "parallel_mode": "row_parallel" if row_parallel else "single_vector",
        "latency_cycles": d["total_cycles"],
        "active_rows": active_rows,
        "throughput_ops": flops / est.latency_s if est.latency_s > 0 else 0.0,
        "latency": est.latency_s,
        "efficiency_tops_per_w": (flops / est.energy_j) / 1e12 if est.energy_j > 0 else 0.0,
        "power": est.energy_j / est.latency_s if est.latency_s > 0 else 0.0,
        "energy": est.energy_j,
        "dynamic_energy": d["dynamic_energy_j"],
        "static_energy": d["static_energy_j"],
        "edp": est.edp,
        "R": r_dim,
        "C": c_dim,
    }
    return est.edp, detail


def _dcnm_edp(flops: float, r_dim: int, c_dim: int):
    active_rows = flops / (2.0 * r_dim * c_dim) if r_dim > 0 and c_dim > 0 else 1.0
    est = calculate_dcnm(LayerShape("x", r_dim, c_dim, max(active_rows, 1e-9)))
    d = est.details
    detail = {
        "throughput_ops": HW.dcnm_effective_throughput_int8_ops,
        "latency": est.latency_s,
        "efficiency_tops_per_w": (flops / est.energy_j) / 1e12 if est.energy_j > 0 else 0.0,
        "power": est.energy_j / est.latency_s if est.latency_s > 0 else 0.0,
        "energy": est.energy_j,
        "energy_mem": d["memory_energy_j"],
        "energy_compute": d["compute_energy_j"],
        "energy_overhead": d["overhead_energy_j"],
        "memory_bits_per_element": HW.dcnm_memory_bits_per_element,
        "memory_energy_per_element": HW.dcnm_memory_energy_per_element_j,
        "compute_energy_per_op": HW.dcnm_compute_energy_per_op_j,
        "effective_throughput_int8": HW.dcnm_effective_throughput_int8_ops,
        "edp": est.edp,
        "R": r_dim,
        "C": c_dim,
    }
    return est.edp, detail


def _unwrap_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        tensor = value[0]
        if len(value) > 1 and value[1] != 0.0:
            return tensor * value[1]
        return tensor
    return value


def _leading_active_rows(input_tensor: torch.Tensor) -> int:
    if input_tensor.dim() <= 1:
        return 1
    return int(math.prod(input_tensor.shape[:-1]))


def layer_forward_stats(full_name: str, module: nn.Module, module_input, module_output) -> dict[str, Any] | None:
    input_tensor = _unwrap_tensor(module_input[0])
    output_tensor = _unwrap_tensor(module_output)
    if not torch.is_tensor(input_tensor) or not torch.is_tensor(output_tensor):
        return None

    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size
        batch, c_out, h_out, w_out = output_tensor.shape
        c_in_eff = module.in_channels // module.groups
        macs = batch * h_out * w_out * module.out_channels * (k_h * k_w * c_in_eff)
        flops = 2 * macs
        r_dim = k_h * k_w * c_in_eff
        c_dim = module.out_channels
        active_rows = batch * h_out * w_out
        row_parallel = active_rows > 1
        shape = {
            "input": list(input_tensor.shape),
            "output": list(output_tensor.shape),
            "kernel_size": [k_h, k_w],
            "groups": module.groups,
            "matrix_convention": "R=in_channels_per_group*kernel_h*kernel_w, C=out_channels",
        }
        c_in, c_out_n, kernel_size_sq = c_in_eff, module.out_channels, k_h * k_w
    elif isinstance(module, nn.Linear):
        active_rows = _leading_active_rows(input_tensor)
        macs = active_rows * module.in_features * module.out_features
        flops = 2 * macs
        r_dim = module.in_features
        c_dim = module.out_features
        row_parallel = active_rows > 1
        shape = {
            "input": list(input_tensor.shape),
            "output": list(output_tensor.shape),
            "active_rows": active_rows,
            "matrix_convention": "R=in_features, C=out_features",
            "note": "For VGG16 batch=1 classifier layers, active_rows is usually 1.",
        }
        c_in, c_out_n, kernel_size_sq = module.in_features, module.out_features, 1
    else:
        return None

    edp_acim, acim_details = _acim_edp(float(flops), int(r_dim), int(c_dim), float(active_rows), row_parallel)
    edp_dcnm, dcnm_details = _dcnm_edp(float(flops), int(r_dim), int(c_dim))
    return {
        "full_name": full_name,
        "short_name": FULL_TO_SHORT.get(full_name, full_name),
        "module_type": type(module).__name__,
        "weight_shape": list(module.weight.shape) if hasattr(module, "weight") else None,
        "shape": shape,
        "flops": float(flops),
        "macs": float(macs),
        "kernel_size_sq": int(kernel_size_sq),
        "c_in": int(c_in),
        "c_out": int(c_out_n),
        "R": int(r_dim),
        "C": int(c_dim),
        "active_rows": float(active_rows),
        "row_parallel": bool(row_parallel),
        "edp_acim": float(edp_acim),
        "edp_dcnm": float(edp_dcnm),
        "edp_diff": float(edp_dcnm - edp_acim),
        "better_accelerator_by_edp": "ACIM" if edp_dcnm - edp_acim > 0 else "DCNM",
        "acim_details": acim_details,
        "dcnm_details": dcnm_details,
    }


def get_vgg16_layer_stats_detailed(
    model: nn.Module | None = None,
    model_name: str = "cifar100_vgg16_bn",
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    if model is None:
        from pytorch_cifar_models import __dict__ as model_zoo

        model = model_zoo[model_name](pretrained=False)
    model = model.to(device).eval()

    layer_stats: list[dict[str, Any]] = []

    def make_hook(full_name: str):
        def hook_fn(module, module_input, module_output):
            stats = layer_forward_stats(full_name, module, module_input, module_output)
            if stats is not None:
                layer_stats.append(stats)

        return hook_fn

    hooks = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))
    for hook in hooks:
        hook.remove()
    return layer_stats
