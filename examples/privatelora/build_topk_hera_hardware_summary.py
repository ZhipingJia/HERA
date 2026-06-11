#!/usr/bin/env python3
"""Build PrivateLLM HERA hardware summaries for affinity-ranked mappings."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_K = "8,12,16,20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize HERA energy/latency for PrivateLLM affinity-ranked mappings. "
            "By default, bottomK_dcnm maps the K lowest-affinity layers to DCNM and "
            "all other profiled layers to ACIM."
        )
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--k", "--top-k", dest="k", default=DEFAULT_K)
    parser.add_argument(
        "--scheme-mode",
        choices=("bottom-dcnm", "top-acim"),
        default="bottom-dcnm",
    )
    parser.add_argument("--include-lm-head-b-dcnm", action="store_true")
    parser.add_argument("--lm-head-rank", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--workload",
        choices=("profiled", "decode"),
        default="profiled",
        help="Use saved profiled active rows, or recompute every layer with decode active_rows=batch size.",
    )
    parser.add_argument("--decode-batch-size", type=float, default=1.0)
    parser.add_argument("--json-name", default="hera_bottom_dcnm_hardware_summary.json")
    parser.add_argument("--md-name", default="hera_bottom_dcnm_hardware_summary.md")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_k_values(spec: str) -> list[int]:
    values = []
    for raw in spec.split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    return values


def calculate_acim_edp(
    flops: float,
    r_dim: int,
    c_dim: int,
    active_rows: float,
    row_parallel: bool,
    constants: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    base_cols = int(constants["acim_base_cols"])
    max_parallel_groups = int(constants["acim_max_parallel_groups"])
    latency_per_cycle = float(constants["acim_latency_per_cycle"])
    arrays_per_vector = math.ceil(c_dim / base_cols)
    cycles_per_vector_batch = math.ceil(arrays_per_vector / max_parallel_groups)
    vector_parallel = max(1, max_parallel_groups // arrays_per_vector) if row_parallel else 1
    vector_batches = math.ceil(active_rows / vector_parallel)
    latency_cycles = vector_batches * cycles_per_vector_batch
    t_acim = latency_cycles * latency_per_cycle
    adc_current_per_group = float(constants["acim_adc_current_per_col"]) * base_cols
    group_current = adc_current_per_group + float(constants["acim_bl_calc_top_current_per_group"])
    group_power = group_current * float(constants["acim_supply_voltage"])
    active_pe_per_full_batch = min(max_parallel_groups, vector_parallel * arrays_per_vector)
    dynamic_energy = active_rows * arrays_per_vector * group_power * latency_per_cycle
    static_energy = float(constants["acim_module_static_power"]) * t_acim
    e_acim = dynamic_energy + static_energy
    p_acim = e_acim / t_acim if t_acim > 0 else 0.0
    edp_acim = e_acim * t_acim
    throughput_ops = flops / t_acim if t_acim > 0 else 0.0
    eff_acim = flops / e_acim if e_acim > 0 else 0.0
    return edp_acim, {
        "base_rows": constants["acim_base_rows"],
        "base_cols": base_cols,
        "max_parallel_groups": max_parallel_groups,
        "arrays_per_vector": arrays_per_vector,
        "cycles_per_vector_batch": cycles_per_vector_batch,
        "vector_parallel": vector_parallel,
        "vector_batches": vector_batches,
        "active_pe_per_full_batch": active_pe_per_full_batch,
        "row_parallel": row_parallel,
        "parallel_mode": "row_parallel" if row_parallel else "single_vector",
        "latency_cycles": latency_cycles,
        "active_rows": active_rows,
        "adc_current_per_group": adc_current_per_group,
        "bl_calc_top_current_per_group": constants["acim_bl_calc_top_current_per_group"],
        "group_power": group_power,
        "dynamic_power": active_pe_per_full_batch * group_power if row_parallel else arrays_per_vector * group_power,
        "module_static_power": constants["acim_module_static_power"],
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


def calculate_dcnm_edp(
    flops: float,
    r_dim: int,
    c_dim: int,
    constants: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    energy_mem = (
        r_dim
        * c_dim
        * float(constants["dcnm_memory_bits_per_element"])
        * float(constants["dcnm_memory_energy_per_bit"])
    )
    energy_compute = flops * float(constants["dcnm_compute_energy_per_op"])
    raw_energy = energy_mem + energy_compute + float(constants["dcnm_fixed_overhead"])
    e_dcnm = raw_energy / float(constants["dcnm_calibration_factor"])
    t_dcnm = flops / float(constants["dcnm_effective_throughput_8bit"])
    p_dcnm = e_dcnm / t_dcnm if t_dcnm > 0 else 0.0
    eff_dcnm = flops / e_dcnm if e_dcnm > 0 else 0.0
    edp_dcnm = e_dcnm * t_dcnm
    return edp_dcnm, {
        "throughput_ops": constants["dcnm_effective_throughput_8bit"],
        "latency": t_dcnm,
        "efficiency": eff_dcnm,
        "efficiency_tops_per_w": eff_dcnm / 1e12,
        "power": p_dcnm,
        "energy": e_dcnm,
        "energy_mem": energy_mem,
        "energy_compute": energy_compute,
        "energy_overhead": constants["dcnm_fixed_overhead"],
        "raw_energy": raw_energy,
        "calibration_factor": constants["dcnm_calibration_factor"],
        "memory_bits_per_element": constants["dcnm_memory_bits_per_element"],
        "memory_energy_per_bit": constants["dcnm_memory_energy_per_bit"],
        "compute_energy_per_op": constants["dcnm_compute_energy_per_op"],
        "peak_throughput_32bit": constants["dcnm_peak_throughput_32bit"],
        "peak_throughput_8bit": constants["dcnm_peak_throughput_8bit"],
        "utilization": constants["dcnm_utilization"],
        "edp": edp_dcnm,
        "R": r_dim,
        "C": c_dim,
    }


def layer_edp_for_workload(
    layer: dict[str, Any],
    workload: str,
    constants: dict[str, Any],
    decode_batch_size: float,
) -> dict[str, Any]:
    if workload == "profiled":
        return layer["edp"]
    if workload != "decode":
        raise ValueError(f"Unsupported workload: {workload}")

    r_dim = int(layer["edp"].get("R", layer.get("R")))
    c_dim = int(layer["edp"].get("C", layer.get("C")))
    active_rows = float(decode_batch_size)
    macs = active_rows * r_dim * c_dim
    flops = 2.0 * macs
    row_parallel = active_rows > 1
    edp_acim, acim_details = calculate_acim_edp(flops, r_dim, c_dim, active_rows, row_parallel, constants)
    edp_dcnm, dcnm_details = calculate_dcnm_edp(flops, r_dim, c_dim, constants)
    return {
        "flops": flops,
        "macs": macs,
        "R": r_dim,
        "C": c_dim,
        "active_rows": active_rows,
        "active_tokens": active_rows,
        "edp_acim": edp_acim,
        "edp_dcnm": edp_dcnm,
        "edp_diff": edp_dcnm - edp_acim,
        "acim_details": acim_details,
        "dcnm_details": dcnm_details,
        "shape": {
            "active_rows": active_rows,
            "matrix_convention": "R=in_features, C=out_features",
            "workload": workload,
            "decode_batch_size": decode_batch_size,
        },
    }


def layer_energy_latency(
    layer: dict[str, Any],
    accelerator: str,
    workload: str,
    constants: dict[str, Any],
    decode_batch_size: float,
) -> dict[str, Any]:
    edp = layer_edp_for_workload(layer, workload, constants, decode_batch_size)
    details = edp[f"{accelerator.lower()}_details"]
    energy = float(details["energy"])
    latency = float(details["latency"])
    return {
        "index": int(layer["index"]),
        "full_name": layer["full_name"],
        "short_name": layer["short_name"],
        "accelerator": accelerator,
        "energy_j": energy,
        "latency_s": latency,
        "power_w": energy / latency if latency > 0 else 0.0,
        "edp_j_s": energy * latency,
        "flops": float(edp["flops"]),
        "macs": float(edp["macs"]),
        "R": int(edp["R"]),
        "C": int(edp["C"]),
        "active_rows": float(edp["active_rows"]),
        "affinity_score": layer.get("affinity_score"),
        "total_kld_v1": layer.get("total_kld_v1"),
        "edp_diff_dcnm_minus_acim": layer.get("edp_diff"),
    }


def fixed_lm_head_b_dcnm_layer(
    rank: int,
    vocab_size: int,
    workload: str,
    constants: dict[str, Any],
    decode_batch_size: float,
) -> dict[str, Any]:
    active_rows = float(decode_batch_size) if workload == "decode" else 1.0
    r_dim = int(rank)
    c_dim = int(vocab_size)
    macs = active_rows * r_dim * c_dim
    flops = 2.0 * macs
    _edp, details = calculate_dcnm_edp(flops, r_dim, c_dim, constants)
    energy = float(details["energy"])
    latency = float(details["latency"])
    return {
        "index": 96,
        "full_name": "lora_lm_head_B",
        "short_name": "lm_head_B",
        "accelerator": "DCNM",
        "energy_j": energy,
        "latency_s": latency,
        "power_w": energy / latency if latency > 0 else 0.0,
        "edp_j_s": energy * latency,
        "flops": flops,
        "macs": macs,
        "R": r_dim,
        "C": c_dim,
        "active_rows": active_rows,
        "affinity_score": None,
        "total_kld_v1": None,
        "edp_diff_dcnm_minus_acim": None,
        "fixed_mapping_note": "lora_lm_head_B is fixed to DCNM and is not part of affinity rank.",
    }


def build_scheme(
    k: int,
    model_layers: list[dict[str, Any]],
    ranked_layers: list[dict[str, Any]],
    workload: str,
    constants: dict[str, Any],
    scheme_mode: str,
    include_lm_head_b_dcnm: bool,
    lm_head_rank: int,
    vocab_size: int,
    decode_batch_size: float,
) -> dict[str, Any]:
    if scheme_mode == "bottom-dcnm":
        dcnm_layers = list(reversed(ranked_layers))[:k]
        dcnm_indices = {int(layer["index"]) for layer in dcnm_layers}
        acim_indices = {int(layer["index"]) for layer in model_layers} - dcnm_indices
        scheme_name = f"bottom{k}_dcnm"
        selected_layers = dcnm_layers
        selected_label = "dcnm_layers_by_bottom_affinity_order"
    elif scheme_mode == "top-acim":
        selected_layers = ranked_layers[:k]
        acim_indices = {int(layer["index"]) for layer in selected_layers}
        dcnm_indices = {int(layer["index"]) for layer in model_layers} - acim_indices
        scheme_name = f"top{k}_acim"
        selected_label = "acim_layers_by_affinity_order"
    else:
        raise ValueError(f"Unsupported scheme_mode: {scheme_mode}")

    per_layer = []
    for layer in sorted(model_layers, key=lambda item: int(item["index"])):
        accelerator = "ACIM" if int(layer["index"]) in acim_indices else "DCNM"
        per_layer.append(layer_energy_latency(layer, accelerator, workload, constants, decode_batch_size))
    if include_lm_head_b_dcnm:
        per_layer.append(
            fixed_lm_head_b_dcnm_layer(
                lm_head_rank,
                vocab_size,
                workload,
                constants,
                decode_batch_size,
            )
        )

    total_energy = sum(item["energy_j"] for item in per_layer)
    total_latency = sum(item["latency_s"] for item in per_layer)
    acim_energy = sum(item["energy_j"] for item in per_layer if item["accelerator"] == "ACIM")
    acim_latency = sum(item["latency_s"] for item in per_layer if item["accelerator"] == "ACIM")
    dcnm_energy = total_energy - acim_energy
    dcnm_latency = total_latency - acim_latency

    return {
        "scheme": scheme_name,
        "scheme_mode": scheme_mode,
        "workload": workload,
        "decode_batch_size": decode_batch_size if workload == "decode" else None,
        "k": k,
        "acim_count": len(acim_indices),
        "dcnm_count": len(dcnm_indices) + (1 if include_lm_head_b_dcnm else 0),
        "lora_mobile_layer_count": len(model_layers),
        "include_lm_head_b_dcnm": include_lm_head_b_dcnm,
        selected_label: [
            {
                "rank": rank,
                "index": int(layer["index"]),
                "full_name": layer["full_name"],
                "short_name": layer["short_name"],
                "affinity_score": float(layer["affinity_score"]),
                "total_kld_v1": float(layer["total_kld_v1"]),
            }
            for rank, layer in enumerate(selected_layers, start=1)
        ],
        "energy_j": total_energy,
        "energy_mj": total_energy * 1e3,
        "latency_s": total_latency,
        "latency_ms": total_latency * 1e3,
        "average_power_w": total_energy / total_latency if total_latency > 0 else 0.0,
        "edp_j_s": total_energy * total_latency,
        "edp_nj_s": total_energy * total_latency * 1e9,
        "acim_energy_j": acim_energy,
        "dcnm_energy_j": dcnm_energy,
        "acim_latency_s": acim_latency,
        "dcnm_latency_s": dcnm_latency,
        "per_layer": per_layer,
    }


def add_baseline_ratios(schemes: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    for scheme in schemes:
        scheme["relative_to_baseline"] = {
            "energy_ratio": scheme["energy_j"] / baseline["energy_j"],
            "latency_ratio": scheme["latency_s"] / baseline["latency_s"],
            "edp_ratio": scheme["edp_j_s"] / baseline["edp_j_s"],
            "energy_reduction_percent": (1.0 - scheme["energy_j"] / baseline["energy_j"]) * 100.0,
            "latency_reduction_percent": (1.0 - scheme["latency_s"] / baseline["latency_s"]) * 100.0,
            "edp_reduction_percent": (1.0 - scheme["edp_j_s"] / baseline["edp_j_s"]) * 100.0,
        }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# PrivateLLM GSM8K HERA Bottom-DCNM Hardware Summary",
        "",
        f"Metrics: `{payload['source_metrics']}`",
        f"Layer count: `{payload['profiled_layer_count']}`",
        f"Include lm_head_B on DCNM: `{payload['include_lm_head_b_dcnm']}`",
        f"Workload: `{payload['workload']}`",
        f"Decode batch size: `{payload['decode_batch_size']}`",
        f"Scheme mode: `{payload['scheme_mode']}`",
        f"Baseline for ratios: `{payload['baseline_scheme']}`",
        f"Aggregation: `{payload['aggregation_rule']}`",
        "",
        "## Schemes",
        "",
        "| scheme | ACIM | DCNM | energy (mJ) | latency (ms) | avg power (W) | EDP (nJ*s) | EDP change vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["schemes"]:
        rel = row["relative_to_baseline"]
        lines.append(
            f"| {row['scheme']} | {row['acim_count']} | {row['dcnm_count']} | "
            f"{row['energy_mj']:.6f} | {row['latency_ms']:.6f} | "
            f"{row['average_power_w']:.6f} | {row['edp_nj_s']:.6f} | "
            f"{(rel['edp_ratio'] - 1.0) * 100.0:.2f}% |"
        )
    lines.extend(["", "## Top-20 Affinity Order", ""])
    lines.append("| rank | index | layer | affinity | total_kld_v1 |")
    lines.append("|---:|---:|---|---:|---:|")
    for rank, row in enumerate(payload["affinity_rank_order"][:20], start=1):
        lines.append(
            f"| {rank} | {row['index']} | `{row['short_name']}` | "
            f"{row['affinity_score']:.6e} | {row['total_kld_v1']:.6g} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    model_layers = metrics["model_order"]
    ranked_layers = metrics["positive_affinity_rank"]
    k_values = parse_k_values(args.k)
    constants = metrics["hardware_constants"]

    baseline = build_scheme(
        0,
        model_layers,
        ranked_layers,
        args.workload,
        constants,
        "top-acim" if args.scheme_mode == "top-acim" else "bottom-dcnm",
        args.include_lm_head_b_dcnm,
        args.lm_head_rank,
        args.vocab_size,
        args.decode_batch_size,
    )
    baseline["scheme"] = "all_dcnm" if args.scheme_mode == "top-acim" else "all_acim"
    schemes = [baseline] + [
        build_scheme(
            k,
            model_layers,
            ranked_layers,
            args.workload,
            constants,
            args.scheme_mode,
            args.include_lm_head_b_dcnm,
            args.lm_head_rank,
            args.vocab_size,
            args.decode_batch_size,
        )
        for k in k_values
    ]
    add_baseline_ratios(schemes, baseline)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metrics": str(args.metrics.resolve()),
        "hardware_model_version": metrics.get("hardware_model_version"),
        "hardware_constants": metrics.get("hardware_constants"),
        "workload": args.workload,
        "decode_batch_size": args.decode_batch_size if args.workload == "decode" else None,
        "scheme_mode": args.scheme_mode,
        "baseline_scheme": baseline["scheme"],
        "profiled_layer_count": len(model_layers),
        "include_lm_head_b_dcnm": args.include_lm_head_b_dcnm,
        "lm_head_b_shape": {
            "R": args.lm_head_rank,
            "C": args.vocab_size,
        },
        "ranked_positive_layer_count": len(ranked_layers),
        "aggregation_rule": (
            "bottomK_dcnm maps the K lowest-affinity layers to DCNM and all remaining profiled "
            "layers to ACIM. topK_acim maps the K highest-affinity layers to ACIM and all "
            "remaining profiled layers to DCNM. Per-layer energy and latency are read from or "
            "recomputed with the saved HERA ACIM/DCNM EDP model and summed over target layers."
        ),
        "note": (
            "This GSM8K affinity run profiles 96 q/k/v lora_mobile layers. lora_lm_head_B "
            "can be added as a fixed DCNM layer with --include-lm-head-b-dcnm; it is not part "
            "of the affinity rank."
        ),
        "run_config": metrics.get("run_config"),
        "affinity_rank_order": [
            {
                "rank": rank,
                "index": int(row["index"]),
                "full_name": row["full_name"],
                "short_name": row["short_name"],
                "affinity_score": float(row["affinity_score"]),
                "total_kld_v1": float(row["total_kld_v1"]),
                "edp_diff": float(row["edp_diff"]),
            }
            for rank, row in enumerate(ranked_layers, start=1)
        ],
        "schemes": schemes,
    }

    output_json = args.metrics.parent / args.json_name
    output_md = args.metrics.parent / args.md_name
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(output_md, payload)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    for row in schemes:
        print(
            f"{row['scheme']}: ACIM={row['acim_count']} DCNM={row['dcnm_count']} "
            f"energy={row['energy_mj']:.6f} mJ latency={row['latency_ms']:.6f} ms "
            f"power={row['average_power_w']:.6f} W EDP={row['edp_nj_s']:.6f} nJ*s"
        )


if __name__ == "__main__":
    main()
