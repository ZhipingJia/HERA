from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
QUANT_DEFAULT_FILE = CONFIGS_DIR / "quant_default.yaml"

DEFAULT_QUANT_CONFIG = CONFIGS_DIR / "config_dcnm_int8.yaml"
DEFAULT_TARGET_LAYERS = CONFIGS_DIR / "target_layers_v0.json"

HARDWARE_MODEL_VERSION = "hera_envelope_new_bl78ma_dcnm0p28pj_conv_linear56pe_20260506"

ACIM_BASE_ROWS = 576
ACIM_BASE_COLS = 32
ACIM_MAX_PARALLEL_GROUPS = 56
ACIM_SUPPLY_VOLTAGE = 0.9
ACIM_ADC_CURRENT_PER_COL = 800e-6
ACIM_BL_CALC_TOP_CURRENT_PER_GROUP = 78e-3
ACIM_MODULE_STATIC_POWER = 0.05
ACIM_LATENCY_PER_CYCLE = 50e-9

DCNM_MEMORY_BITS_PER_ELEMENT = 8
DCNM_MEMORY_ENERGY_PER_BIT = 0.2e-12
DCNM_COMPUTE_ENERGY_PER_OP = 0.28e-12
DCNM_FIXED_OVERHEAD = 2.0e-9
DCNM_PEAK_THROUGHPUT_32BIT = 64e9
DCNM_PEAK_THROUGHPUT_8BIT = DCNM_PEAK_THROUGHPUT_32BIT * 4
DCNM_UTILIZATION = 0.80
DCNM_EFFECTIVE_THROUGHPUT_8BIT = DCNM_PEAK_THROUGHPUT_8BIT * DCNM_UTILIZATION
DCNM_CALIBRATION_FACTOR = 1.0


def setup_paths() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


setup_paths()

from build_dcnm_baseline import get_submodule_by_name, load_target_layers, to_jsonable, verify_dcnm_layers  # noqa: E402
from data.dataset import TestDataset  # noqa: E402
from eval_voc import build_model  # noqa: E402
from quantization_and_noise import prepare_quant_model2  # noqa: E402
from quantization_and_noise.config_loader import get_config  # noqa: E402
from utils.config import opt  # noqa: E402


class LayerStatsAccumulator:
    def __init__(self, layer: dict):
        self.layer = dict(layer)
        self.count = 0
        self.sums: dict[str, float] = {
            "flops": 0.0,
            "macs": 0.0,
            "edp_acim": 0.0,
            "edp_dcnm": 0.0,
            "edp_diff": 0.0,
            "active_rows": 0.0,
        }
        self.detail_sums: dict[str, dict[str, float]] = {"acim": {}, "dcnm": {}}
        self.last_shape: dict | None = None

    def update(self, stats: dict) -> None:
        self.count += 1
        for key in self.sums:
            self.sums[key] += float(stats.get(key, 0.0))
        for detail_name in ("acim", "dcnm"):
            details = stats[f"{detail_name}_details"]
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    self.detail_sums[detail_name][key] = self.detail_sums[detail_name].get(key, 0.0) + float(value)
        self.last_shape = stats["shape"]

    def mean(self) -> dict:
        if self.count == 0:
            raise RuntimeError(f"Layer {self.layer['full_name']} was not executed during EDP profiling")
        row = dict(self.layer)
        row["profiled_forwards"] = self.count
        for key, value in self.sums.items():
            row[key] = value / self.count
        row["edp_diff"] = row["edp_dcnm"] - row["edp_acim"]
        row["better_accelerator_by_edp"] = "ACIM" if row["edp_diff"] > 0 else "DCNM"
        row["shape"] = self.last_shape
        row["hardware_params"] = {
            "hardware_model_version": HARDWARE_MODEL_VERSION,
            "acim_base_rows": ACIM_BASE_ROWS,
            "acim_base_cols": ACIM_BASE_COLS,
            "acim_max_parallel_groups": ACIM_MAX_PARALLEL_GROUPS,
            "acim_supply_voltage": ACIM_SUPPLY_VOLTAGE,
            "acim_adc_current_per_col": ACIM_ADC_CURRENT_PER_COL,
            "acim_bl_calc_top_current_per_group": ACIM_BL_CALC_TOP_CURRENT_PER_GROUP,
            "acim_module_static_power": ACIM_MODULE_STATIC_POWER,
            "acim_latency_per_cycle": ACIM_LATENCY_PER_CYCLE,
            "dcnm_memory_bits_per_element": DCNM_MEMORY_BITS_PER_ELEMENT,
            "dcnm_memory_energy_per_bit": DCNM_MEMORY_ENERGY_PER_BIT,
            "dcnm_compute_energy_per_op": DCNM_COMPUTE_ENERGY_PER_OP,
            "dcnm_fixed_overhead": DCNM_FIXED_OVERHEAD,
            "dcnm_peak_throughput_32bit": DCNM_PEAK_THROUGHPUT_32BIT,
            "dcnm_peak_throughput_8bit": DCNM_PEAK_THROUGHPUT_8BIT,
            "dcnm_utilization": DCNM_UTILIZATION,
            "dcnm_effective_throughput_8bit": DCNM_EFFECTIVE_THROUGHPUT_8BIT,
            "dcnm_calibration_factor": DCNM_CALIBRATION_FACTOR,
        }
        row["acim_details"] = {
            key: value / self.count for key, value in self.detail_sums["acim"].items()
        }
        row["dcnm_details"] = {
            key: value / self.count for key, value in self.detail_sums["dcnm"].items()
        }
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FasterRCNN per-layer EDP and combine it with KLD into affinity scores."
    )
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--kld-metrics", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, default=DEFAULT_QUANT_CONFIG)
    parser.add_argument("--target-layers", type=Path, default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="stage2_best_noise20_200img_hera_new_hw_weight_rc_bl78ma_dcnm0p28pj_conv_linear56pe_eps0p1median",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--test-num-workers", type=int, default=4)
    parser.add_argument("--epsilon-alpha", type=float, default=0.1)
    parser.add_argument("--voc-data-dir", default=None)
    return parser.parse_args()


def apply_runtime_options(args: argparse.Namespace) -> None:
    opt.device = args.device
    opt.test_num_workers = args.test_num_workers
    opt.test_num = args.max_images if args.max_images > 0 else 10000
    if args.voc_data_dir is not None:
        opt.voc_data_dir = args.voc_data_dir


def build_test_dataloader() -> data_.DataLoader:
    testset = TestDataset(opt)
    return data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        pin_memory=True,
    )


def build_dcnm_model(quant_config: Path) -> nn.Module:
    model = build_model()
    args_ = get_config(
        default_file=str(QUANT_DEFAULT_FILE),
        config_file=[str(quant_config)],
    )
    prepare_quant_model2(model.extractor, None, args_.quan)
    prepare_quant_model2(model.rpn, None, args_.quan)
    prepare_quant_model2(model.head, None, args_.quan)
    return model


def load_checkpoint_strict(model: nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    result = model.load_state_dict(state_dict, strict=True)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def calculate_acim_edp(
    flops: float,
    r_dim: int,
    c_dim: int,
    active_rows: float,
    row_parallel: bool,
) -> tuple[float, dict]:
    arrays_per_vector = math.ceil(c_dim / ACIM_BASE_COLS)
    cycles_per_vector_batch = math.ceil(arrays_per_vector / ACIM_MAX_PARALLEL_GROUPS)
    if row_parallel:
        vector_parallel = max(1, ACIM_MAX_PARALLEL_GROUPS // arrays_per_vector)
    else:
        vector_parallel = 1
    vector_batches = math.ceil(active_rows / vector_parallel)
    latency_cycles = vector_batches * cycles_per_vector_batch
    single_input_latency = cycles_per_vector_batch * ACIM_LATENCY_PER_CYCLE
    t_acim = latency_cycles * ACIM_LATENCY_PER_CYCLE
    adc_current_per_group = ACIM_ADC_CURRENT_PER_COL * ACIM_BASE_COLS
    group_current = adc_current_per_group + ACIM_BL_CALC_TOP_CURRENT_PER_GROUP
    group_power = group_current * ACIM_SUPPLY_VOLTAGE
    active_pe_per_full_batch = min(ACIM_MAX_PARALLEL_GROUPS, vector_parallel * arrays_per_vector)
    dynamic_energy = active_rows * arrays_per_vector * group_power * ACIM_LATENCY_PER_CYCLE
    static_energy = ACIM_MODULE_STATIC_POWER * t_acim
    e_acim = dynamic_energy + static_energy
    p_acim = e_acim / t_acim if t_acim > 0 else 0.0
    edp_acim = e_acim * t_acim
    throughput_ops = flops / t_acim if t_acim > 0 else 0.0
    eff_acim = flops / e_acim if e_acim > 0 else 0.0
    return edp_acim, {
        "base_rows": ACIM_BASE_ROWS,
        "base_cols": ACIM_BASE_COLS,
        "max_parallel_groups": ACIM_MAX_PARALLEL_GROUPS,
        "num_base_arrays": arrays_per_vector,
        "arrays_per_vector": arrays_per_vector,
        "vector_parallel": vector_parallel,
        "vector_batches": vector_batches,
        "active_pe_per_full_batch": active_pe_per_full_batch,
        "row_parallel": row_parallel,
        "parallel_mode": "row_parallel" if row_parallel else "single_vector",
        "spatial_parallel": row_parallel,
        "cycles_per_vector_batch": cycles_per_vector_batch,
        "latency_cycles": latency_cycles,
        "active_rows": active_rows,
        "single_input_latency": single_input_latency,
        "adc_current_per_group": adc_current_per_group,
        "bl_calc_top_current_per_group": ACIM_BL_CALC_TOP_CURRENT_PER_GROUP,
        "group_power": group_power,
        "dynamic_power": active_pe_per_full_batch * group_power if row_parallel else arrays_per_vector * group_power,
        "module_static_power": ACIM_MODULE_STATIC_POWER,
        "throughput_ops": throughput_ops,
        "latency": t_acim,
        "efficiency": eff_acim,
        "efficiency_tops_per_w": eff_acim / 1e12,
        "power": p_acim,
        "energy": e_acim,
        "dynamic_energy": dynamic_energy,
        "static_energy": static_energy,
        "edp": edp_acim,
        "R": r_dim,
        "C": c_dim,
    }


def calculate_dcnm_edp(flops: float, r_dim: int, c_dim: int) -> tuple[float, dict]:
    energy_mem = r_dim * c_dim * DCNM_MEMORY_BITS_PER_ELEMENT * DCNM_MEMORY_ENERGY_PER_BIT
    energy_compute = flops * DCNM_COMPUTE_ENERGY_PER_OP
    raw_energy = energy_mem + energy_compute + DCNM_FIXED_OVERHEAD
    e_dcnm = raw_energy / DCNM_CALIBRATION_FACTOR
    dcnm_throughput_ops = DCNM_EFFECTIVE_THROUGHPUT_8BIT
    t_dcnm = flops / dcnm_throughput_ops
    p_dcnm = e_dcnm / t_dcnm if t_dcnm > 0 else 0.0
    eff_dcnm = flops / e_dcnm if e_dcnm > 0 else 0.0
    edp_dcnm = e_dcnm * t_dcnm
    return edp_dcnm, {
        "throughput_ops": dcnm_throughput_ops,
        "latency": t_dcnm,
        "efficiency": eff_dcnm,
        "efficiency_tops_per_w": eff_dcnm / 1e12,
        "power": p_dcnm,
        "energy": e_dcnm,
        "energy_mem": energy_mem,
        "energy_compute": energy_compute,
        "energy_overhead": DCNM_FIXED_OVERHEAD,
        "raw_energy": raw_energy,
        "calibration_factor": DCNM_CALIBRATION_FACTOR,
        "memory_bits_per_element": DCNM_MEMORY_BITS_PER_ELEMENT,
        "memory_energy_per_bit": DCNM_MEMORY_ENERGY_PER_BIT,
        "compute_energy_per_op": DCNM_COMPUTE_ENERGY_PER_OP,
        "peak_throughput_32bit": DCNM_PEAK_THROUGHPUT_32BIT,
        "peak_throughput_8bit": DCNM_PEAK_THROUGHPUT_8BIT,
        "utilization": DCNM_UTILIZATION,
        "edp": edp_dcnm,
        "R": r_dim,
        "C": c_dim,
    }


def unwrap_tensor(value):
    if isinstance(value, tuple):
        tensor = value[0]
        if len(value) > 1 and value[1] != 0.0:
            return tensor * value[1]
        return tensor
    return value


def layer_forward_stats(module: nn.Module, module_input, module_output) -> dict:
    input_tensor = unwrap_tensor(module_input[0])
    output_tensor = unwrap_tensor(module_output)
    if not torch.is_tensor(output_tensor):
        raise TypeError(f"Expected tensor output from {type(module).__name__}, got {type(output_tensor)}")

    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size
        batch, c_out, h_out, w_out = output_tensor.shape
        c_in_eff = module.in_channels // module.groups
        macs = batch * h_out * w_out * module.out_channels * (k_h * k_w * c_in_eff)
        flops = 2 * macs
        kernel_size_sq = k_h * k_w
        r_dim = kernel_size_sq * c_in_eff
        c_dim = module.out_channels
        active_rows = batch * h_out * w_out
        shape = {
            "input": list(input_tensor.shape),
            "output": list(output_tensor.shape),
            "kernel_size": [k_h, k_w],
            "groups": module.groups,
            "matrix_convention": "R=in_channels_per_group*kernel_h*kernel_w, C=out_channels",
        }
    elif isinstance(module, nn.Linear):
        if output_tensor.dim() == 1:
            active = 1
        else:
            active = int(output_tensor.shape[0])
        macs = active * module.in_features * module.out_features
        flops = 2 * macs
        kernel_size_sq = 1
        c_in_eff = module.in_features
        r_dim = module.in_features
        c_dim = module.out_features
        active_rows = active
        shape = {
            "input": list(input_tensor.shape),
            "output": list(output_tensor.shape),
            "active_rows": active,
            "matrix_convention": "R=in_features, C=out_features",
        }
    else:
        raise TypeError(f"Unsupported module type: {type(module).__name__}")

    row_parallel = isinstance(module, nn.Conv2d) or (
        isinstance(module, nn.Linear) and active_rows > 1
    )
    edp_acim, acim_details = calculate_acim_edp(
        flops,
        r_dim,
        c_dim,
        float(active_rows),
        row_parallel,
    )
    edp_dcnm, dcnm_details = calculate_dcnm_edp(flops, r_dim, c_dim)
    return {
        "flops": float(flops),
        "macs": float(macs),
        "kernel_size_sq": int(kernel_size_sq),
        "c_in": int(c_in_eff),
        "c_out": int(module.out_channels if isinstance(module, nn.Conv2d) else module.out_features),
        "R": int(r_dim),
        "C": int(c_dim),
        "active_rows": float(active_rows),
        "edp_acim": float(edp_acim),
        "edp_dcnm": float(edp_dcnm),
        "edp_diff": float(edp_dcnm - edp_acim),
        "acim_details": acim_details,
        "dcnm_details": dcnm_details,
        "shape": shape,
    }


def iter_eval_batches(dataloader, max_images: int):
    for batch_idx, batch in enumerate(dataloader):
        if max_images > 0 and batch_idx >= max_images:
            break
        imgs, _sizes, _gt_bboxes, _gt_labels, _gt_difficults, scale = batch
        scale_value = float(scale.item() if torch.is_tensor(scale) else scale)
        yield imgs, scale_value


def profile_edp(model: nn.Module, dataloader, target_layers: list[dict], device: torch.device, max_images: int) -> list[dict]:
    accumulators = {layer["full_name"]: LayerStatsAccumulator(layer) for layer in target_layers}
    hooks = []
    for layer in target_layers:
        module = get_submodule_by_name(model, layer["full_name"])

        def hook_fn(module, module_input, module_output, layer_name=layer["full_name"]):
            stats = layer_forward_stats(module, module_input, module_output)
            accumulators[layer_name].update(stats)

        hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        for imgs, scale in tqdm(
            iter_eval_batches(dataloader, max_images),
            total=max_images if max_images > 0 else len(dataloader),
            desc="EDP profiling",
        ):
            imgs = imgs.to(device).float()
            model(imgs, scale=scale)

    for hook in hooks:
        hook.remove()

    return [accumulators[layer["full_name"]].mean() for layer in target_layers]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of an empty list")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def combine_affinity(kld_metrics: dict, edp_rows: list[dict], epsilon_alpha: float) -> dict:
    kld_by_index = {int(layer["index"]): layer for layer in kld_metrics["layers"]}
    total_kld_values = [
        float(kld_by_index[int(edp["index"])]["total_kld_v1"])
        for edp in edp_rows
    ]
    median_total_kld = median(total_kld_values)
    epsilon = epsilon_alpha * median_total_kld
    combined_layers = []
    for edp in edp_rows:
        idx = int(edp["index"])
        if idx not in kld_by_index:
            raise KeyError(f"KLD metrics missing layer index {idx}")
        kld = kld_by_index[idx]
        total_kld = float(kld["total_kld_v1"])
        affinity_score = float(edp["edp_diff"]) / (total_kld + epsilon)
        combined = {
            **kld,
            "edp": edp,
            "flops_per_frame": edp["flops"],
            "macs_per_frame": edp["macs"],
            "edp_acim": edp["edp_acim"],
            "edp_dcnm": edp["edp_dcnm"],
            "edp_diff": edp["edp_diff"],
            "affinity_score": affinity_score,
            "affinity_rule": "edp_diff / (total_kld_v1 + epsilon_alpha * median_total_kld_v1)",
            "epsilon": epsilon,
            "epsilon_alpha": epsilon_alpha,
            "median_total_kld_v1": median_total_kld,
            "better_accelerator_by_edp": edp["better_accelerator_by_edp"],
            "better_accelerator_by_affinity_sign": "ACIM" if affinity_score > 0 else "DCNM",
        }
        combined_layers.append(combined)

    combined_layers.sort(key=lambda item: int(item["index"]))
    return {
        "kld_source": kld_metrics,
        "epsilon": epsilon,
        "epsilon_alpha": epsilon_alpha,
        "median_total_kld_v1": median_total_kld,
        "epsilon_rule": "epsilon = epsilon_alpha * median(total_kld_v1 over target layers)",
        "layers": combined_layers,
        "ranking_by_affinity_desc": sorted(combined_layers, key=lambda item: item["affinity_score"], reverse=True),
        "ranking_by_total_kld_asc": sorted(combined_layers, key=lambda item: item["total_kld_v1"]),
        "ranking_by_edp_diff_desc": sorted(combined_layers, key=lambda item: item["edp_diff"], reverse=True),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def main() -> None:
    args = parse_args()
    apply_runtime_options(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(opt.device)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / Path(__file__).name)
    shutil.copyfile(args.target_layers, output_dir / args.target_layers.name)

    with open(args.kld_metrics, "r") as f:
        kld_metrics = json.load(f)

    target_layers = load_target_layers(args.target_layers)
    dataloader = build_test_dataloader()
    model = build_dcnm_model(args.quant_config)
    load_result = load_checkpoint_strict(model, args.reference_checkpoint)
    verification = verify_dcnm_layers(model, target_layers)
    if not verification["ok"]:
        raise RuntimeError(f"DCNM verification failed: {verification['bad_layers']}")
    model.to(device).eval()

    edp_rows = profile_edp(model, dataloader, target_layers, device, args.max_images)
    affinity_payload = combine_affinity(kld_metrics, edp_rows, args.epsilon_alpha)
    affinity_payload.update({
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "kld_metrics": str(args.kld_metrics.resolve()),
        "quant_config": str(args.quant_config.resolve()),
        "target_layers": str(args.target_layers.resolve()),
        "max_images": args.max_images,
        "device": args.device,
        "load_result": load_result,
        "verification": verification,
        "hardware_constants": {
            "hardware_model_version": HARDWARE_MODEL_VERSION,
            "acim_base_rows": ACIM_BASE_ROWS,
            "acim_base_cols": ACIM_BASE_COLS,
            "acim_max_parallel_groups": ACIM_MAX_PARALLEL_GROUPS,
            "acim_supply_voltage": ACIM_SUPPLY_VOLTAGE,
            "acim_adc_current_per_col": ACIM_ADC_CURRENT_PER_COL,
            "acim_bl_calc_top_current_per_group": ACIM_BL_CALC_TOP_CURRENT_PER_GROUP,
            "acim_module_static_power": ACIM_MODULE_STATIC_POWER,
            "acim_latency_per_cycle": ACIM_LATENCY_PER_CYCLE,
            "dcnm_memory_bits_per_element": DCNM_MEMORY_BITS_PER_ELEMENT,
            "dcnm_memory_energy_per_bit": DCNM_MEMORY_ENERGY_PER_BIT,
            "dcnm_compute_energy_per_op": DCNM_COMPUTE_ENERGY_PER_OP,
            "dcnm_fixed_overhead": DCNM_FIXED_OVERHEAD,
            "dcnm_peak_throughput_32bit": DCNM_PEAK_THROUGHPUT_32BIT,
            "dcnm_peak_throughput_8bit": DCNM_PEAK_THROUGHPUT_8BIT,
            "dcnm_utilization": DCNM_UTILIZATION,
            "dcnm_effective_throughput_8bit": DCNM_EFFECTIVE_THROUGHPUT_8BIT,
            "dcnm_calibration_factor": DCNM_CALIBRATION_FACTOR,
        },
    })

    write_json(output_dir / "edp_metrics.json", {"layers": edp_rows})
    write_json(output_dir / "affinity_metrics.json", affinity_payload)
    write_json(
        output_dir / "affinity_ranking.json",
        {
            "ranking_key": "affinity_score descending",
            "layers": affinity_payload["ranking_by_affinity_desc"],
        },
    )

    print(f"Wrote EDP metrics to {output_dir / 'edp_metrics.json'}")
    print(f"Wrote affinity metrics to {output_dir / 'affinity_metrics.json'}")
    print(
        f"Adaptive epsilon = {affinity_payload['epsilon']:.6g} "
        f"(alpha={affinity_payload['epsilon_alpha']:.6g}, "
        f"median_kld={affinity_payload['median_total_kld_v1']:.6g})"
    )
    print("Top affinity layers:")
    for item in affinity_payload["ranking_by_affinity_desc"][:5]:
        print(
            f"  {item['index']:02d} {item['short_name']:<10} "
            f"affinity={item['affinity_score']:.6e} "
            f"kld={item['total_kld_v1']:.6g} edp_diff={item['edp_diff']:.6e}"
        )


if __name__ == "__main__":
    main()
