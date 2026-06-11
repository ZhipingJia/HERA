#!/usr/bin/env python3
"""Compute PrivateLLM BoolQ layer affinity for LoRA ACIM mapping.

This script follows the newer FasterRCNN/VGG16 HERA convention:

    edp_diff = EDP_DCNM - EDP_ACIM
    epsilon = epsilon_ratio * median(total_kld_v1)
    affinity = edp_diff / (total_kld_v1 + epsilon)

For layers with edp_diff <= 0, final ACIM affinity is intentionally skipped and
the layer is reported as DCNM-preferred. A raw signed score is still recorded as
a diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = THIS_DIR / "configs"


def default_dataset_path(relative: str) -> Path | None:
    """Resolve a datasets.save_to_disk cache under $PRIVATE_LLM_DATASET_ROOT."""

    root = os.environ.get("PRIVATE_LLM_DATASET_ROOT", "")
    if root:
        candidate = Path(root) / relative
        if candidate.is_dir():
            return candidate
    return None

HARDWARE_MODEL_VERSION = "hera_envelope_new_bl78ma_dcnm0p28pj_private_llm_linear56pe_20260521"

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
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))


setup_paths()

from quantization_and_noise.config_loader import get_config_unified_cgra  # noqa: E402
from infras.model_utils import get_base_model_and_tokenizer  # noqa: E402
from mymodels.modeling_llama_pl import LlamaForCausalLM  # noqa: E402
from quantization_and_noise import prepare_quant_model_unified_cgra  # noqa: E402
from quantization_and_noise.quant_layer import linear_quant_noise, linear_quant_sample_noise  # noqa: E402
from quantization_and_noise.quant_util import LSQ_act_quantizer, LSQ_weight_quantizer  # noqa: E402


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update_tensor(self, value: torch.Tensor) -> None:
        value = value.detach().double().reshape(-1)
        self.total += float(value.sum().cpu().item())
        self.count += int(value.numel())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class LayerStatsAccumulator:
    def __init__(self, layer: dict[str, Any]):
        self.layer = dict(layer)
        self.count = 0
        self.sums = {
            "flops": 0.0,
            "macs": 0.0,
            "edp_acim": 0.0,
            "edp_dcnm": 0.0,
            "edp_diff": 0.0,
            "active_rows": 0.0,
            "active_tokens": 0.0,
        }
        self.detail_sums: dict[str, dict[str, float]] = {"acim": {}, "dcnm": {}}
        self.last_shape: dict[str, Any] | None = None

    def update(self, stats: dict[str, Any]) -> None:
        self.count += 1
        for key in self.sums:
            self.sums[key] += float(stats.get(key, 0.0))
        for detail_name in ("acim", "dcnm"):
            for key, value in stats[f"{detail_name}_details"].items():
                if isinstance(value, (int, float, bool)):
                    self.detail_sums[detail_name][key] = self.detail_sums[detail_name].get(key, 0.0) + float(value)
        self.last_shape = stats["shape"]

    def mean(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError(f"Layer {self.layer['full_name']} was not executed")
        row = dict(self.layer)
        row["profiled_forwards"] = self.count
        for key, value in self.sums.items():
            row[key] = value / self.count
        row["edp_diff"] = row["edp_dcnm"] - row["edp_acim"]
        row["better_accelerator_by_edp"] = "ACIM" if row["edp_diff"] > 0 else "DCNM"
        row["shape"] = self.last_shape
        row["acim_details"] = {
            key: value / self.count for key, value in self.detail_sums["acim"].items()
        }
        row["dcnm_details"] = {
            key: value / self.count for key, value in self.detail_sums["dcnm"].items()
        }
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PrivateLLM BoolQ layer affinity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path("super_glue/boolq/validation"),
        help="datasets.save_to_disk path of the BoolQ validation split; defaults to "
             "$PRIVATE_LLM_DATASET_ROOT/super_glue/boolq/validation when present.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="plllama-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--source-max-len", type=int, default=900)
    parser.add_argument("--layer-indices", default="all")
    parser.add_argument("--epsilon-ratio", type=float, default=0.1)
    parser.add_argument("--component-eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--sample-noise-mode", default="sample_noise_2")
    parser.add_argument("--sample-noise-std", type=float, default=20.0)
    parser.add_argument("--sample-noise-scale", type=float, default=0.5)
    parser.add_argument("--sample-output-min", type=float, default=None)
    parser.add_argument("--sample-output-max", type=float, default=None)
    parser.add_argument("--quant-file", default=str(CONFIGS_DIR / "config_pl_quant_int8.yaml"))
    parser.add_argument("--quant-file-2", default=str(CONFIGS_DIR / "config_pl_quant_int8_cgra.yaml"))
    parser.add_argument("--lm-head-file", default=str(CONFIGS_DIR / "config_pl_quant_lora_lm_head_int8.yaml"))
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_submodule_by_name(model: nn.Module, full_name: str) -> nn.Module:
    module: nn.Module = model
    for part in full_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_submodule_by_name(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def load_private_lora_weights(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    loaded = []
    skipped_shape = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append({"key": key, "model_shape": list(model_state[key].shape), "checkpoint_shape": list(value.shape)})
            continue
        model_state[key].copy_(value.to(dtype=model_state[key].dtype))
        loaded.append(key)
    missing_checkpoint_keys = [
        key for key in model_state if (
            "lora_mobile" in key
            or "lora_lm_head" in key
            or "embed_tokens" in key
            or "quantizer" in key
            or "sample_out_scale" in key
        ) and key not in state_dict
    ]
    return {
        "loaded_keys": len(loaded),
        "unexpected_keys": unexpected,
        "skipped_shape_mismatch": skipped_shape,
        "missing_private_lora_or_quant_keys": missing_checkpoint_keys,
    }


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[LlamaForCausalLM, Any, dict[str, Any]]:
    print(f"Loading base model/tokenizer: {args.model_name}")
    model, tokenizer = get_base_model_and_tokenizer(args.model_name)
    args_ = get_config_unified_cgra(
        config_file=args.quant_file,
        config_file_2=args.quant_file_2,
        lm_head_file=args.lm_head_file,
    )
    print("Preparing unified CGRA INT8 quant modules")
    model = prepare_quant_model_unified_cgra(model, args_.quan)
    print(f"Loading PrivateLoRA checkpoint: {args.checkpoint}")
    load_result = load_private_lora_weights(model, args.checkpoint)
    model.to(device).eval()
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return model, tokenizer, load_result


def boolq_prompt(sample: dict[str, Any]) -> str:
    question = sample["question"]
    question = question[0].upper() + question[1:] if question else question
    if not question.endswith("?"):
        question += "?"
    return f"{sample['passage']} {question}\n"


def load_boolq_samples(dataset_path: Path, max_samples: int) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(dataset_path))
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def make_batches(
    tokenizer,
    samples: list[dict[str, Any]],
    batch_size: int,
    source_max_len: int,
) -> list[dict[str, Any]]:
    prompts = [tokenizer.bos_token + boolq_prompt(sample) for sample in samples]
    batches = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        tokens = tokenizer(
            prompts[start:end],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=source_max_len,
        )
        last_indices = tokens["attention_mask"].sum(dim=1).long() - 1
        batches.append({
            "batch_index": len(batches),
            "sample_start": start,
            "sample_end": end,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "last_indices": last_indices,
        })
    return batches


def answer_token_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    tokenization: dict[str, list[int]] = {}
    for label in ("No", "Yes"):
        encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise RuntimeError(f"Tokenizer returned no token for BoolQ answer {label!r}")
        ids[label] = int(encoded[0])
        tokenization[label] = [int(x) for x in encoded]
    return {"No": ids["No"], "Yes": ids["Yes"], "tokenization": tokenization}  # type: ignore[return-value]


def hardware_constants() -> dict[str, float | int | str]:
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
    r_dim: int,
    c_dim: int,
    active_rows: float,
    row_parallel: bool,
) -> tuple[float, dict[str, Any]]:
    arrays_per_vector = math.ceil(c_dim / ACIM_BASE_COLS)
    cycles_per_vector_batch = math.ceil(arrays_per_vector / ACIM_MAX_PARALLEL_GROUPS)
    if row_parallel:
        vector_parallel = max(1, ACIM_MAX_PARALLEL_GROUPS // arrays_per_vector)
    else:
        vector_parallel = 1
    vector_batches = math.ceil(active_rows / vector_parallel)
    latency_cycles = vector_batches * cycles_per_vector_batch
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


def calculate_dcnm_edp(flops: float, r_dim: int, c_dim: int) -> tuple[float, dict[str, Any]]:
    energy_mem = r_dim * c_dim * DCNM_MEMORY_BITS_PER_ELEMENT * DCNM_MEMORY_ENERGY_PER_BIT
    energy_compute = flops * DCNM_COMPUTE_ENERGY_PER_OP
    raw_energy = energy_mem + energy_compute + DCNM_FIXED_OVERHEAD
    e_dcnm = raw_energy / DCNM_CALIBRATION_FACTOR
    t_dcnm = flops / DCNM_EFFECTIVE_THROUGHPUT_8BIT
    p_dcnm = e_dcnm / t_dcnm if t_dcnm > 0 else 0.0
    eff_dcnm = flops / e_dcnm if e_dcnm > 0 else 0.0
    edp_dcnm = e_dcnm * t_dcnm
    return edp_dcnm, {
        "throughput_ops": DCNM_EFFECTIVE_THROUGHPUT_8BIT,
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


def unwrap_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        tensor = value[0]
        if len(value) > 1 and value[1] != 0.0:
            return tensor * value[1]
        return tensor
    return value


def leading_active_rows(input_tensor: torch.Tensor) -> int:
    if input_tensor.dim() <= 1:
        return 1
    return int(math.prod(input_tensor.shape[:-1]))


def layer_forward_stats(module: nn.Linear, module_input, module_output) -> dict[str, Any] | None:
    input_tensor = unwrap_tensor(module_input[0])
    output_tensor = unwrap_tensor(module_output)
    if not torch.is_tensor(input_tensor) or not torch.is_tensor(output_tensor):
        return None
    active_rows = leading_active_rows(input_tensor)
    macs = active_rows * module.in_features * module.out_features
    flops = 2 * macs
    r_dim = module.in_features
    c_dim = module.out_features
    row_parallel = active_rows > 1
    edp_acim, acim_details = calculate_acim_edp(flops, r_dim, c_dim, float(active_rows), row_parallel)
    edp_dcnm, dcnm_details = calculate_dcnm_edp(flops, r_dim, c_dim)
    return {
        "flops": float(flops),
        "macs": float(macs),
        "R": int(r_dim),
        "C": int(c_dim),
        "active_rows": float(active_rows),
        "active_tokens": float(active_rows),
        "edp_acim": float(edp_acim),
        "edp_dcnm": float(edp_dcnm),
        "edp_diff": float(edp_dcnm - edp_acim),
        "acim_details": acim_details,
        "dcnm_details": dcnm_details,
        "shape": {
            "input": list(input_tensor.shape),
            "output": list(output_tensor.shape),
            "active_rows": active_rows,
            "matrix_convention": "R=in_features, C=out_features",
        },
    }


def target_layers_from_model(model: nn.Module) -> list[dict[str, Any]]:
    layers = []
    for name, module in model.named_modules():
        if name.endswith("lora_mobile") and isinstance(module, nn.Linear):
            parts = name.split(".")
            layer_id = int(parts[2])
            proj = parts[4].replace("_lora", "")
            layers.append({
                "index": layer_id * 3 + {"q": 0, "k": 1, "v": 2}[proj],
                "transformer_layer": layer_id,
                "projection": proj,
                "short_name": f"L{layer_id:02d}_{proj}_mobile",
                "full_name": name,
                "module_type": type(module).__name__,
                "R": int(module.in_features),
                "C": int(module.out_features),
            })
    lm_head = get_submodule_by_name(model, "lora_lm_head_B")
    layers.append({
        "index": 96,
        "transformer_layer": None,
        "projection": "lm_head_B",
        "short_name": "lm_head_B",
        "full_name": "lora_lm_head_B",
        "module_type": type(lm_head).__name__,
        "R": int(lm_head.in_features),
        "C": int(lm_head.out_features),
    })
    return sorted(layers, key=lambda item: int(item["index"]))


def parse_layer_indices(spec: str, layers: list[dict[str, Any]]) -> list[int]:
    valid = {int(layer["index"]) for layer in layers}
    if spec.strip().lower() == "all":
        return sorted(valid)
    selected: list[int] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            selected.extend(range(int(start), int(end) + 1))
        else:
            selected.append(int(item))
    bad = sorted(set(selected) - valid)
    if bad:
        raise ValueError(f"Unknown layer indices: {bad}")
    return sorted(dict.fromkeys(selected))


def register_edp_hooks(model: nn.Module, target_layers: list[dict[str, Any]]) -> tuple[list[Any], dict[str, LayerStatsAccumulator]]:
    accumulators = {layer["full_name"]: LayerStatsAccumulator(layer) for layer in target_layers}
    hooks = []
    for layer in target_layers:
        module = get_submodule_by_name(model, layer["full_name"])

        def hook_fn(module, module_input, module_output, layer_name=layer["full_name"]):
            stats = layer_forward_stats(module, module_input, module_output)
            if stats is not None:
                accumulators[layer_name].update(stats)

        hooks.append(module.register_forward_hook(hook_fn))
    return hooks, accumulators


def gather_last_logits(logits: torch.Tensor, last_indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices.to(logits.device), :]


def reference_forward_profile(
    model: nn.Module,
    batches: list[dict[str, Any]],
    target_layers: list[dict[str, Any]],
    choice_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hooks, accumulators = register_edp_hooks(model, target_layers)
    ref_batches = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="Reference DCNM"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            last_indices = batch["last_indices"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            last_logits = gather_last_logits(outputs.logits, last_indices)
            ref_log_prob = F.log_softmax(last_logits.float(), dim=-1).detach().cpu()
            ref_choice_log_prob = F.log_softmax(last_logits[:, choice_ids].float(), dim=-1).detach().cpu()
            ref_batches.append({
                **batch,
                "ref_log_prob": ref_log_prob,
                "ref_choice_log_prob": ref_choice_log_prob,
            })
            del outputs, last_logits, input_ids, attention_mask, last_indices
    for hook in hooks:
        hook.remove()
    edp_rows = [accumulators[layer["full_name"]].mean() for layer in target_layers]
    return ref_batches, edp_rows


def infer_lsq_kwargs(old_quantizer: nn.Module, bit: int, int_flag: bool) -> dict[str, Any]:
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
    quantizer = LSQ_act_quantizer(**infer_lsq_kwargs(old_quantizer, bit=5, int_flag=True))
    quantizer.s = nn.Parameter(old_quantizer.s.detach().clone() * 8.0)
    return quantizer


def clone_lsq_weight_for_acim(old_quantizer: LSQ_weight_quantizer) -> LSQ_weight_quantizer:
    quantizer = LSQ_weight_quantizer(**infer_lsq_kwargs(old_quantizer, bit=4, int_flag=True))
    quantizer.s = nn.Parameter(old_quantizer.s.detach().clone() * 16.0)
    return quantizer


def replace_one_layer_with_acim(
    model: nn.Module,
    layer: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    module = get_submodule_by_name(model, layer["full_name"])
    if not isinstance(module, linear_quant_noise):
        raise TypeError(f"{layer['full_name']} is {type(module).__name__}, expected linear_quant_noise")
    if not isinstance(module.a_quantizer, LSQ_act_quantizer):
        raise TypeError(f"{layer['full_name']}.a_quantizer is {type(module.a_quantizer).__name__}")
    if not isinstance(module.w_quantizer, LSQ_weight_quantizer):
        raise TypeError(f"{layer['full_name']}.w_quantizer is {type(module.w_quantizer).__name__}")

    old_module = module
    a_quantizer = clone_lsq_act_for_acim(module.a_quantizer)
    w_quantizer = clone_lsq_weight_for_acim(module.w_quantizer)
    dummy_out_quantizer = clone_lsq_act_for_acim(module.a_quantizer)
    sample_out_scale = 2.0 * a_quantizer.s.detach() * w_quantizer.s.detach()
    sample_noise_kwargs: dict[str, Any] = {
        "noise_std": args.sample_noise_std,
        "scale_factor": args.sample_noise_scale,
    }
    if args.sample_output_min is not None:
        sample_noise_kwargs["output_min"] = args.sample_output_min
    if args.sample_output_max is not None:
        sample_noise_kwargs["output_max"] = args.sample_output_max

    new_module = linear_quant_sample_noise(
        module,
        w_quantizer=w_quantizer,
        a_quantizer=a_quantizer,
        a_out_quantizer=dummy_out_quantizer,
        int_flag=True,
        sample_out_scale=sample_out_scale,
        sample_noise_mode=args.sample_noise_mode,
        sample_noise_kwargs=sample_noise_kwargs,
    )
    new_module.to(module.weight.device)
    set_submodule_by_name(model, layer["full_name"], new_module)
    summary = {
        "full_name": layer["full_name"],
        "old_class": type(old_module).__name__,
        "new_class": type(new_module).__name__,
        "old_act_scale": module.a_quantizer.s.detach().cpu(),
        "old_weight_scale": module.w_quantizer.s.detach().cpu(),
        "acim_act_scale": a_quantizer.s.detach().cpu(),
        "acim_weight_scale": w_quantizer.s.detach().cpu(),
        "sample_out_scale": sample_out_scale.detach().cpu(),
        "sample_noise_mode": args.sample_noise_mode,
        "sample_noise_kwargs": sample_noise_kwargs,
        "acim_quant_rule": "clone DCNM LSQ scales: act bit5 scale*8, weight bit4 scale*16; sample_out_scale=2*act_scale*weight_scale",
    }
    return old_module, summary


def restore_layer(model: nn.Module, layer: dict[str, Any], old_module: nn.Module) -> None:
    set_submodule_by_name(model, layer["full_name"], old_module)


def profile_single_layer_kld(
    model: nn.Module,
    layer: dict[str, Any],
    ref_batches: list[dict[str, Any]],
    choice_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    old_module, replacement = replace_one_layer_with_acim(model, layer, args)
    choice_accum = RunningMean()
    full_vocab_accum = RunningMean()
    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(ref_batches, desc=f"KLD {layer['short_name']}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                last_indices = batch["last_indices"].to(device)
                for repeat_idx in range(args.noise_repeats):
                    seed = args.seed + int(layer["index"]) * 1_000_000 + int(batch["batch_index"]) * 100 + repeat_idx
                    set_seed(seed)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    last_logits = gather_last_logits(outputs.logits, last_indices)
                    test_log_prob = F.log_softmax(last_logits.float(), dim=-1).detach().cpu()
                    test_choice_log_prob = F.log_softmax(last_logits[:, choice_ids].float(), dim=-1).detach().cpu()
                    ref_log_prob = batch["ref_log_prob"]
                    ref_choice_log_prob = batch["ref_choice_log_prob"]
                    full_vocab_kl = (ref_log_prob.exp() * (ref_log_prob - test_log_prob)).sum(dim=-1)
                    choice_kl = (ref_choice_log_prob.exp() * (ref_choice_log_prob - test_choice_log_prob)).sum(dim=-1)
                    full_vocab_accum.update_tensor(full_vocab_kl)
                    choice_accum.update_tensor(choice_kl)
                    del outputs, last_logits, test_log_prob, test_choice_log_prob
                del input_ids, attention_mask, last_indices
    finally:
        restore_layer(model, layer, old_module)
    return {
        **layer,
        "boolq_choice_kl": choice_accum.mean,
        "full_vocab_last_token_kl": full_vocab_accum.mean,
        "total_kld_v1": choice_accum.mean,
        "total_kld_rule": "private_llm_boolq_choice_kl_v1",
        "kld_direction": "KL(DCNM_INT8 || single_layer_ACIM)",
        "profiled_samples": sum(batch["input_ids"].shape[0] for batch in ref_batches),
        "noise_repeats": args.noise_repeats,
        "replacement": replacement,
    }


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty values")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def combine_affinity(
    kld_rows: list[dict[str, Any]],
    edp_rows: list[dict[str, Any]],
    epsilon_ratio: float,
    component_eps: float,
) -> dict[str, Any]:
    kld_by_index = {int(row["index"]): row for row in kld_rows}
    edp_by_index = {int(row["index"]): row for row in edp_rows}
    total_kld_values = [
        max(0.0, float(kld_by_index[idx]["total_kld_v1"]))
        for idx in sorted(kld_by_index)
    ]
    median_total_kld = median(total_kld_values)
    epsilon = epsilon_ratio * median_total_kld
    if epsilon <= 0.0:
        epsilon = component_eps

    rows = []
    for idx in sorted(kld_by_index):
        kld = kld_by_index[idx]
        edp = edp_by_index[idx]
        total_kld = float(kld["total_kld_v1"])
        denom = total_kld + epsilon
        raw_signed_affinity = float(edp["edp_diff"]) / denom if denom != 0.0 else float("inf")
        edp_positive = float(edp["edp_diff"]) > 0.0
        final_affinity = raw_signed_affinity if edp_positive else None
        dcnm_priority = abs(float(edp["edp_diff"])) / denom if not edp_positive and denom != 0.0 else None
        rows.append({
            **kld,
            "edp": edp,
            "flops_per_profile_batch": edp["flops"],
            "macs_per_profile_batch": edp["macs"],
            "active_rows": edp["active_rows"],
            "R": edp["R"],
            "C": edp["C"],
            "edp_acim": edp["edp_acim"],
            "edp_dcnm": edp["edp_dcnm"],
            "edp_diff": edp["edp_diff"],
            "edp_preferred_accelerator": edp["better_accelerator_by_edp"],
            "raw_signed_affinity_score": raw_signed_affinity,
            "affinity_score": final_affinity,
            "dcnm_priority": dcnm_priority,
            "affinity_rule": "edp_diff / (total_kld_v1 + epsilon)",
            "final_affinity_status": (
                "computed_for_positive_edp_diff"
                if edp_positive
                else "skipped_because_edp_diff_non_positive"
            ),
            "epsilon": epsilon,
            "epsilon_ratio": epsilon_ratio,
            "median_total_kld_v1": median_total_kld,
        })

    positive_rows = [row for row in rows if row["affinity_score"] is not None]
    negative_rows = [row for row in rows if row["affinity_score"] is None]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_model_version": HARDWARE_MODEL_VERSION,
        "hardware_constants": hardware_constants(),
        "epsilon_policy": {
            "formula": "epsilon = epsilon_ratio * median(total_kld_v1)",
            "epsilon_ratio": epsilon_ratio,
            "median_total_kld_v1": median_total_kld,
            "epsilon": epsilon,
            "median_scope": "all profiled target layers with nonnegative total_kld_v1; EDP-negative layers are still allowed to contribute to the robust median",
        },
        "notes": [
            "total_kld_v1 is BoolQ Yes/No choice-token KL, analogous to VGG final class-logit KL.",
            "full_vocab_last_token_kl is saved only as a diagnostic and is not used for final affinity.",
            "edp_diff = EDP_DCNM - EDP_ACIM; positive means ACIM reduces EDP.",
            "Layers with edp_diff <= 0 are DCNM-preferred and do not receive final affinity_score.",
        ],
        "model_order": rows,
        "positive_affinity_rank": sorted(positive_rows, key=lambda item: item["affinity_score"], reverse=True),
        "negative_edp_layers": sorted(negative_rows, key=lambda item: int(item["index"])),
        "negative_edp_dcnm_priority_rank": sorted(
            negative_rows,
            key=lambda item: item["dcnm_priority"] if item["dcnm_priority"] is not None else -float("inf"),
            reverse=True,
        ),
        "ranking_by_total_kld_asc": sorted(rows, key=lambda item: item["total_kld_v1"]),
        "ranking_by_edp_diff_desc": sorted(rows, key=lambda item: item["edp_diff"], reverse=True),
        "diagnostic_full_signed_affinity_rank": sorted(rows, key=lambda item: item["raw_signed_affinity_score"], reverse=True),
    }


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"samples{args.max_samples}_bs{args.batch_size}_{datetime.now().strftime('%m%d%H%M%S')}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / Path(__file__).name)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    model, tokenizer, load_result = build_model(args, device)
    all_layers = target_layers_from_model(model)
    selected_indices = set(parse_layer_indices(args.layer_indices, all_layers))
    selected_layers = [layer for layer in all_layers if int(layer["index"]) in selected_indices]
    write_json(output_dir / "target_layers_private_llm_boolq.json", {"layers": all_layers})

    answer_ids = answer_token_ids(tokenizer)
    choice_ids = torch.tensor([answer_ids["No"], answer_ids["Yes"]], dtype=torch.long, device=device)
    samples = load_boolq_samples(args.dataset, args.max_samples)
    batches = make_batches(tokenizer, samples, args.batch_size, args.source_max_len)

    run_config = {
        "checkpoint": args.checkpoint.resolve(),
        "dataset": args.dataset.resolve(),
        "output_dir": output_dir.resolve(),
        "model_name": args.model_name,
        "device": str(device),
        "max_samples": args.max_samples,
        "actual_samples": len(samples),
        "batch_size": args.batch_size,
        "source_max_len": args.source_max_len,
        "selected_layer_indices": sorted(selected_indices),
        "epsilon_ratio": args.epsilon_ratio,
        "seed": args.seed,
        "noise_repeats": args.noise_repeats,
        "sample_noise_mode": args.sample_noise_mode,
        "sample_noise_std": args.sample_noise_std,
        "sample_noise_scale": args.sample_noise_scale,
        "sample_output_min": args.sample_output_min,
        "sample_output_max": args.sample_output_max,
        "quant_file": args.quant_file,
        "quant_file_2": args.quant_file_2,
        "lm_head_file": args.lm_head_file,
        "answer_token_ids": answer_ids,
        "load_result": load_result,
    }
    write_json(output_dir / "run_config.json", run_config)

    print(f"Profiling {len(selected_layers)} layers on {len(samples)} BoolQ samples; output={output_dir}")
    ref_batches, edp_rows = reference_forward_profile(model, batches, selected_layers, choice_ids, device)
    write_json(output_dir / "edp_profile.json", {
        "hardware_constants": hardware_constants(),
        "layers": edp_rows,
    })

    kld_rows = []
    for layer in selected_layers:
        print(f"Profiling KLD layer {layer['index']} {layer['full_name']}")
        metric = profile_single_layer_kld(model, layer, ref_batches, choice_ids, args, device)
        kld_rows.append(metric)
        write_json(output_dir / "raw_kld_metrics_partial.json", {
            "total_kld_rule": "private_llm_boolq_choice_kl_v1",
            "layers": kld_rows,
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output_dir / "kld_metrics.json", {
        "total_kld_rule": "private_llm_boolq_choice_kl_v1",
        "kld_direction": "KL(DCNM_INT8 || single_layer_ACIM)",
        "sample_noise": {
            "sample_noise_mode": args.sample_noise_mode,
            "noise_std": args.sample_noise_std,
            "scale_factor": args.sample_noise_scale,
            "output_min": args.sample_output_min,
            "output_max": args.sample_output_max,
            "noise_repeats": args.noise_repeats,
        },
        "layers": kld_rows,
    })

    affinity = combine_affinity(kld_rows, edp_rows, args.epsilon_ratio, args.component_eps)
    affinity.update({
        "run_config": run_config,
        "sample_noise": {
            "sample_noise_mode": args.sample_noise_mode,
            "noise_std": args.sample_noise_std,
            "scale_factor": args.sample_noise_scale,
            "output_min": args.sample_output_min,
            "output_max": args.sample_output_max,
            "noise_repeats": args.noise_repeats,
        },
    })
    write_json(output_dir / "affinity_metrics.json", affinity)
    write_json(output_dir / "affinity_rank.json", {
        "generated_at_utc": affinity["generated_at_utc"],
        "epsilon_policy": affinity["epsilon_policy"],
        "positive_affinity_rank": affinity["positive_affinity_rank"],
        "negative_edp_layers": affinity["negative_edp_layers"],
        "negative_edp_dcnm_priority_rank": affinity["negative_edp_dcnm_priority_rank"],
    })

    print(f"Wrote affinity metrics to {output_dir / 'affinity_metrics.json'}")
    print("Top positive affinity layers:")
    for row in affinity["positive_affinity_rank"][:10]:
        print(
            f"  {row['index']:02d} {row['short_name']:<14} "
            f"affinity={row['affinity_score']:.6e} "
            f"edp_diff={row['edp_diff']:.6e} kld={row['total_kld_v1']:.6g}"
        )
    print("DCNM-preferred layers:")
    for row in affinity["negative_edp_layers"]:
        print(
            f"  {row['index']:02d} {row['short_name']:<14} "
            f"edp_diff={row['edp_diff']:.6e} kld={row['total_kld_v1']:.6g} "
            f"final_affinity={row['affinity_score']}"
        )


if __name__ == "__main__":
    main()
