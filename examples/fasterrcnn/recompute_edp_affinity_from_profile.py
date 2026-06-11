from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute FasterRCNN EDP and affinity from a saved EDP profile without model inference."
    )
    parser.add_argument("--old-edp", type=Path, required=True)
    parser.add_argument("--kld-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="stage2_best_noise20_200img_hera_new_hw_weight_rc_bl78ma_dcnm0p28pj_conv_linear56pe_eps0p1median",
    )
    parser.add_argument("--epsilon-alpha", type=float, default=0.1)
    return parser.parse_args()


def hardware_constants() -> dict:
    return {
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


def calculate_acim_edp(
    flops: float,
    r_dim: float,
    c_dim: float,
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
    latency = latency_cycles * ACIM_LATENCY_PER_CYCLE
    adc_current_per_group = ACIM_ADC_CURRENT_PER_COL * ACIM_BASE_COLS
    group_current = adc_current_per_group + ACIM_BL_CALC_TOP_CURRENT_PER_GROUP
    group_power = group_current * ACIM_SUPPLY_VOLTAGE
    active_pe_per_full_batch = min(ACIM_MAX_PARALLEL_GROUPS, vector_parallel * arrays_per_vector)
    dynamic_energy = active_rows * arrays_per_vector * group_power * ACIM_LATENCY_PER_CYCLE
    static_energy = ACIM_MODULE_STATIC_POWER * latency
    energy = dynamic_energy + static_energy
    power = energy / latency if latency > 0 else 0.0
    edp = energy * latency
    throughput_ops = flops / latency if latency > 0 else 0.0
    efficiency = flops / energy if energy > 0 else 0.0
    return edp, {
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
        "latency": latency,
        "efficiency": efficiency,
        "efficiency_tops_per_w": efficiency / 1e12,
        "power": power,
        "energy": energy,
        "dynamic_energy": dynamic_energy,
        "static_energy": static_energy,
        "edp": edp,
        "R": r_dim,
        "C": c_dim,
    }


def calculate_dcnm_edp(flops: float, r_dim: float, c_dim: float) -> tuple[float, dict]:
    energy_mem = r_dim * c_dim * DCNM_MEMORY_BITS_PER_ELEMENT * DCNM_MEMORY_ENERGY_PER_BIT
    energy_compute = flops * DCNM_COMPUTE_ENERGY_PER_OP
    raw_energy = energy_mem + energy_compute + DCNM_FIXED_OVERHEAD
    energy = raw_energy / DCNM_CALIBRATION_FACTOR
    throughput_ops = DCNM_EFFECTIVE_THROUGHPUT_8BIT
    latency = flops / throughput_ops
    power = energy / latency if latency > 0 else 0.0
    efficiency = flops / energy if energy > 0 else 0.0
    edp = energy * latency
    return edp, {
        "throughput_ops": throughput_ops,
        "latency": latency,
        "efficiency": efficiency,
        "efficiency_tops_per_w": efficiency / 1e12,
        "power": power,
        "energy": energy,
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
        "edp": edp,
        "R": r_dim,
        "C": c_dim,
    }


def get_weight_matrix_rc(row: dict) -> tuple[float, float]:
    weight_shape = row.get("weight_shape")
    module_type = row.get("module_type")
    if module_type == "Conv2d" and weight_shape and len(weight_shape) == 4:
        out_channels, in_channels_per_group, kernel_h, kernel_w = weight_shape
        return float(in_channels_per_group * kernel_h * kernel_w), float(out_channels)
    if module_type == "Linear" and weight_shape and len(weight_shape) == 2:
        out_features, in_features = weight_shape
        return float(in_features), float(out_features)
    raise KeyError(
        f"Cannot derive weight-matrix R/C for layer {row.get('full_name', row.get('index'))}: "
        f"module_type={module_type}, weight_shape={weight_shape}"
    )


def recompute_edp_rows(old_rows: list[dict]) -> list[dict]:
    rows = []
    for old in old_rows:
        row = dict(old)
        flops = float(row["flops"])
        active_rows = float(row.get("active_rows", 1.0))
        r_dim, c_dim = get_weight_matrix_rc(row)
        row_parallel = row.get("module_type") == "Conv2d" or (
            row.get("module_type") == "Linear" and active_rows > 1.0
        )
        edp_acim, acim_details = calculate_acim_edp(flops, r_dim, c_dim, active_rows, row_parallel)
        edp_dcnm, dcnm_details = calculate_dcnm_edp(flops, r_dim, c_dim)
        row["R"] = r_dim
        row["C"] = c_dim
        row["matrix_convention"] = "R=input dimension, C=output dimension"
        row["edp_acim"] = edp_acim
        row["edp_dcnm"] = edp_dcnm
        row["edp_diff"] = edp_dcnm - edp_acim
        row["better_accelerator_by_edp"] = "ACIM" if row["edp_diff"] > 0 else "DCNM"
        row["hardware_params"] = hardware_constants()
        row["acim_details"] = acim_details
        row["dcnm_details"] = dcnm_details
        rows.append(row)
    return rows


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
        "hardware_constants": hardware_constants(),
        "layers": combined_layers,
        "ranking_by_affinity_desc": sorted(combined_layers, key=lambda item: item["affinity_score"], reverse=True),
        "ranking_by_total_kld_asc": sorted(combined_layers, key=lambda item: item["total_kld_v1"]),
        "ranking_by_edp_diff_desc": sorted(combined_layers, key=lambda item: item["edp_diff"], reverse=True),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / Path(__file__).name)

    with open(args.old_edp, "r") as f:
        old_edp = json.load(f)
    with open(args.kld_metrics, "r") as f:
        kld_metrics = json.load(f)

    edp_rows = recompute_edp_rows(old_edp["layers"])
    affinity_payload = combine_affinity(kld_metrics, edp_rows, args.epsilon_alpha)
    affinity_payload.update({
        "old_edp": str(args.old_edp.resolve()),
        "kld_metrics": str(args.kld_metrics.resolve()),
        "run_name": args.run_name,
        "note": "Recomputed from saved EDP profile; no model inference was run.",
    })

    write_json(output_dir / "edp_metrics.json", {"layers": edp_rows, "hardware_constants": hardware_constants()})
    write_json(output_dir / "affinity_metrics.json", affinity_payload)
    write_json(
        output_dir / "affinity_ranking.json",
        {
            "ranking_key": "affinity_score descending",
            "hardware_constants": hardware_constants(),
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
    for item in affinity_payload["ranking_by_affinity_desc"][:10]:
        print(
            f"  {item['index']:02d} {item['short_name']:<10} "
            f"affinity={item['affinity_score']:.6e} "
            f"kld={item['total_kld_v1']:.6g} edp_diff={item['edp_diff']:.6e} "
            f"better={item['better_accelerator_by_edp']}"
        )


if __name__ == "__main__":
    main()
