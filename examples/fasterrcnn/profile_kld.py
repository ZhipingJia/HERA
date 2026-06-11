from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data_
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
QUANT_DEFAULT_FILE = CONFIGS_DIR / "quant_default.yaml"

DEFAULT_QUANT_CONFIG = CONFIGS_DIR / "config_dcnm_int8.yaml"
DEFAULT_TARGET_LAYERS = CONFIGS_DIR / "target_layers_v0.json"
DEFAULT_KLD_RULES = CONFIGS_DIR / "kld_rules_v1.json"

COMPONENT_NAMES = (
    "rpn_cls_kl",
    "rpn_loc_kl_gaussian",
    "roi_cls_kl",
    "roi_loc_kl_gaussian",
)


def setup_paths() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


setup_paths()

from build_dcnm_baseline import get_submodule_by_name, load_target_layers, to_jsonable, verify_dcnm_layers  # noqa: E402
from data.dataset import TestDataset  # noqa: E402
from eval_voc import build_model  # noqa: E402
from quantization_and_noise import prepare_quant_model2  # noqa: E402
from quantization_and_noise.quant_layer import (  # noqa: E402
    conv2d_quant_noise,
    conv2d_quant_sample_noise,
    linear_quant_noise,
    linear_quant_sample_noise,
)
from quantization_and_noise.quant_util import LSQ_act_quantizer, LSQ_weight_quantizer  # noqa: E402
from quantization_and_noise.config_loader import get_config  # noqa: E402
from utils.config import opt  # noqa: E402


@dataclass
class RunningScalar:
    total: float = 0.0
    count: int = 0

    def update_tensor(self, value: torch.Tensor) -> None:
        value = value.detach().double()
        self.total += float(value.sum().cpu().item())
        self.count += int(value.numel())

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class RunningMoments:
    def __init__(self, dim: int):
        self.dim = dim
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.total_sq = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, value: torch.Tensor) -> None:
        value = value.detach().double().reshape(-1, self.dim).cpu()
        if value.numel() == 0:
            return
        self.total += value.sum(dim=0)
        self.total_sq += (value * value).sum(dim=0)
        self.count += int(value.shape[0])

    def as_dict(self, eps: float) -> dict:
        if self.count == 0:
            mean = torch.zeros(self.dim, dtype=torch.float64)
            std = torch.ones(self.dim, dtype=torch.float64)
        else:
            mean = self.total / self.count
            var = self.total_sq / self.count - mean * mean
            std = torch.sqrt(torch.clamp(var, min=eps * eps))
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "eps": eps,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile FasterRCNN single-layer ACIM KLD for affinity-aware mapping."
    )
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, default=DEFAULT_QUANT_CONFIG)
    parser.add_argument("--target-layers", type=Path, default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--kld-rules", type=Path, default=DEFAULT_KLD_RULES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="stage2_best_noise20")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-image", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--test-num-workers", type=int, default=4)
    parser.add_argument("--layer-indices", default="all")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--loc-std-eps", type=float, default=1e-6)
    parser.add_argument("--component-eps", type=float, default=1e-12)
    parser.add_argument("--sample-noise-mode", default="sample_noise_2")
    parser.add_argument(
        "--sample-noise-mode-overrides",
        default="",
        help="Comma-separated per-layer overrides, e.g. '7:sample_noise_5,rpn.conv1:sample_noise_5'.",
    )
    parser.add_argument("--sample-noise-std", type=float, default=20.0)
    parser.add_argument("--sample-noise-scale", type=float, default=0.5)
    parser.add_argument("--sample-output-min", type=float, default=None)
    parser.add_argument("--sample-output-max", type=float, default=None)
    parser.add_argument(
        "--sample-output-min-overrides",
        default="",
        help="Comma-separated per-layer output_min overrides, e.g. '7:-255,rpn.conv1:-255'.",
    )
    parser.add_argument(
        "--sample-output-max-overrides",
        default="",
        help="Comma-separated per-layer output_max overrides, e.g. '7:255,rpn.conv1:255'.",
    )
    parser.add_argument("--voc-data-dir", default=None)
    return parser.parse_args()


def apply_runtime_options(args: argparse.Namespace) -> None:
    opt.device = args.device
    opt.test_num_workers = args.test_num_workers
    opt.test_num = args.start_image + args.max_images if args.max_images > 0 else 10000
    if args.voc_data_dir is not None:
        opt.voc_data_dir = args.voc_data_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_loaded_dcnm_model(args: argparse.Namespace) -> tuple[nn.Module, dict]:
    model = build_dcnm_model(args.quant_config)
    load_result = load_checkpoint_strict(model, args.reference_checkpoint)
    return model, load_result


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


def categorical_kl(ref_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    ref_log_prob = F.log_softmax(ref_logits.float(), dim=-1)
    test_log_prob = F.log_softmax(test_logits.float(), dim=-1)
    return (ref_log_prob.exp() * (ref_log_prob - test_log_prob)).sum(dim=-1)


def foreground_roi_locs(roi_cls_locs: torch.Tensor, n_class: int) -> torch.Tensor:
    if n_class < 2:
        raise ValueError(f"Expected n_class >= 2, got {n_class}")
    locs = roi_cls_locs.reshape(-1, n_class, 4)
    return locs[:, 1, :]


def extract_outputs(
    model: nn.Module,
    imgs: torch.Tensor,
    scale: float,
    fixed_rois=None,
    fixed_roi_indices=None,
) -> dict:
    img_size = imgs.shape[2:]
    h = model.extractor(imgs)
    rpn_locs, rpn_scores, rois, roi_indices, _anchor = model.rpn(h, img_size, scale)
    head_rois = fixed_rois if fixed_rois is not None else rois
    head_roi_indices = fixed_roi_indices if fixed_roi_indices is not None else roi_indices
    roi_cls_locs, roi_scores = model.head(h, head_rois, head_roi_indices)
    return {
        "rpn_locs": rpn_locs.detach(),
        "rpn_scores": rpn_scores.detach(),
        "rois": rois,
        "roi_indices": roi_indices,
        "roi_cls_locs": roi_cls_locs.detach(),
        "roi_scores": roi_scores.detach(),
    }


def iter_eval_batches(dataloader, start_image: int, max_images: int):
    seen = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_image:
            continue
        if max_images > 0 and seen >= max_images:
            break
        imgs, _sizes, _gt_bboxes, _gt_labels, _gt_difficults, scale = batch
        scale_value = float(scale.item() if torch.is_tensor(scale) else scale)
        seen += 1
        yield batch_idx, imgs, scale_value


def compute_reference_loc_stats(
    model: nn.Module,
    dataloader,
    device: torch.device,
    start_image: int,
    max_images: int,
    loc_std_eps: float,
) -> dict:
    rpn_stats = RunningMoments(dim=4)
    roi_stats = RunningMoments(dim=4)
    model.eval()
    with torch.no_grad():
        for _batch_idx, imgs, scale in tqdm(
            iter_eval_batches(dataloader, start_image, max_images),
            total=max_images if max_images > 0 else len(dataloader),
            desc="Reference loc stats",
        ):
            imgs = imgs.to(device).float()
            outputs = extract_outputs(model, imgs, scale)
            rpn_stats.update(outputs["rpn_locs"])
            roi_stats.update(foreground_roi_locs(outputs["roi_cls_locs"], model.n_class))
    return {
        "rpn_locs": rpn_stats.as_dict(loc_std_eps),
        "roi_fg_locs": roi_stats.as_dict(loc_std_eps),
    }


def profile_single_layer(
    ref_model: nn.Module,
    acim_model: nn.Module,
    dataloader,
    device: torch.device,
    layer: dict,
    loc_stats: dict,
    args: argparse.Namespace,
) -> dict:
    accum = {name: RunningScalar() for name in COMPONENT_NAMES}
    rpn_std = torch.tensor(loc_stats["rpn_locs"]["std"], dtype=torch.float32, device=device).reshape(1, 1, 4)
    roi_std = torch.tensor(loc_stats["roi_fg_locs"]["std"], dtype=torch.float32, device=device).reshape(1, 4)

    ref_model.eval()
    acim_model.eval()
    with torch.no_grad():
        for batch_idx, imgs, scale in tqdm(
            iter_eval_batches(dataloader, args.start_image, args.max_images),
            total=args.max_images if args.max_images > 0 else len(dataloader),
            desc=f"KLD {layer['short_name']}",
        ):
            imgs = imgs.to(device).float()
            ref_outputs = extract_outputs(ref_model, imgs, scale)
            for repeat_idx in range(args.noise_repeats):
                seed = args.seed + int(layer["index"]) * 1_000_000 + batch_idx * 100 + repeat_idx
                set_seed(seed)
                acim_outputs = extract_outputs(
                    acim_model,
                    imgs,
                    scale,
                    fixed_rois=ref_outputs["rois"],
                    fixed_roi_indices=ref_outputs["roi_indices"],
                )

                accum["rpn_cls_kl"].update_tensor(
                    categorical_kl(ref_outputs["rpn_scores"], acim_outputs["rpn_scores"])
                )
                accum["roi_cls_kl"].update_tensor(
                    categorical_kl(ref_outputs["roi_scores"], acim_outputs["roi_scores"])
                )

                rpn_loc_kl = 0.5 * ((acim_outputs["rpn_locs"] - ref_outputs["rpn_locs"]) / rpn_std).pow(2)
                accum["rpn_loc_kl_gaussian"].update_tensor(rpn_loc_kl)

                ref_roi_fg = foreground_roi_locs(ref_outputs["roi_cls_locs"], ref_model.n_class)
                acim_roi_fg = foreground_roi_locs(acim_outputs["roi_cls_locs"], acim_model.n_class)
                roi_loc_kl = 0.5 * ((acim_roi_fg - ref_roi_fg) / roi_std).pow(2)
                accum["roi_loc_kl_gaussian"].update_tensor(roi_loc_kl)

    raw = {name: accum[name].mean for name in COMPONENT_NAMES}
    counts = {f"{name}_count": accum[name].count for name in COMPONENT_NAMES}
    return {
        **layer,
        **raw,
        **counts,
        "start_image": args.start_image,
        "profiled_images": args.max_images,
        "noise_repeats": args.noise_repeats,
    }


def add_normalized_metrics(layer_metrics: list[dict], component_eps: float) -> dict:
    component_means = {
        name: float(np.mean([float(item[name]) for item in layer_metrics])) if layer_metrics else 0.0
        for name in COMPONENT_NAMES
    }

    for item in layer_metrics:
        norm_values = []
        for name in COMPONENT_NAMES:
            norm_name = f"norm_{name}"
            item[norm_name] = float(item[name]) / (component_means[name] + component_eps)
            norm_values.append(item[norm_name])
        item["total_kld_v1"] = float(np.mean(norm_values))
        item["total_kld_cls_only"] = float(np.mean([item["norm_rpn_cls_kl"], item["norm_roi_cls_kl"]]))
        item["total_kld_roi_only"] = float(np.mean([
            item["norm_roi_cls_kl"],
            item["norm_roi_loc_kl_gaussian"],
        ]))
        item["total_kld_rule"] = "fasterrcnn_kld_aggregation_v1"

    return component_means


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def main() -> None:
    args = parse_args()
    apply_runtime_options(args)
    set_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.set_device(opt.device)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / Path(__file__).name)
    shutil.copyfile(args.kld_rules, output_dir / args.kld_rules.name)
    shutil.copyfile(args.target_layers, output_dir / args.target_layers.name)

    with open(args.kld_rules, "r") as f:
        kld_rules = json.load(f)
    target_layers = load_target_layers(args.target_layers)
    selected_indices = set(parse_layer_indices(args.layer_indices, target_layers))
    selected_layers = [layer for layer in target_layers if int(layer["index"]) in selected_indices]
    dataloader = build_test_dataloader()

    print(f"Building reference DCNM model from {args.reference_checkpoint}")
    ref_model, load_result = build_loaded_dcnm_model(args)
    verification = verify_dcnm_layers(ref_model, target_layers)
    if not verification["ok"]:
        raise RuntimeError(f"Reference DCNM verification failed: {verification['bad_layers']}")
    ref_model.to(device).eval()

    loc_stats = compute_reference_loc_stats(
        ref_model,
        dataloader,
        device=device,
        start_image=args.start_image,
        max_images=args.max_images,
        loc_std_eps=args.loc_std_eps,
    )
    write_json(output_dir / "reference_loc_stats.json", loc_stats)

    layer_metrics = []
    replacements = []
    for layer in selected_layers:
        print(f"Profiling layer {layer['index']} {layer['full_name']} ({layer['short_name']})")
        acim_model, acim_load_result = build_loaded_dcnm_model(args)
        replacement_summary = replace_one_layer_with_acim(acim_model, layer, args)
        acim_model.to(device).eval()
        metric = profile_single_layer(ref_model, acim_model, dataloader, device, layer, loc_stats, args)
        metric["replacement"] = replacement_summary
        metric["acim_load_result"] = acim_load_result
        layer_metrics.append(metric)
        replacements.append(replacement_summary)
        write_json(output_dir / "raw_kld_metrics_partial.json", {"layers": layer_metrics})
        del acim_model
        torch.cuda.empty_cache()

    component_means = add_normalized_metrics(layer_metrics, args.component_eps)
    payload = {
        "run_name": args.run_name,
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "quant_config": str(args.quant_config.resolve()),
        "target_layers": str(args.target_layers.resolve()),
        "kld_rules": kld_rules,
        "device": args.device,
        "start_image": args.start_image,
        "max_images": args.max_images,
        "noise_repeats": args.noise_repeats,
        "seed": args.seed,
        "sample_noise_mode": args.sample_noise_mode,
        "sample_noise_mode_overrides": args.sample_noise_mode_overrides,
        "sample_noise_std": args.sample_noise_std,
        "sample_noise_scale": args.sample_noise_scale,
        "sample_output_min": args.sample_output_min,
        "sample_output_max": args.sample_output_max,
        "sample_output_min_overrides": args.sample_output_min_overrides,
        "sample_output_max_overrides": args.sample_output_max_overrides,
        "component_eps": args.component_eps,
        "loc_std_eps": args.loc_std_eps,
        "selected_layer_indices": sorted(selected_indices),
        "reference_load_result": load_result,
        "reference_verification": verification,
        "reference_loc_stats": loc_stats,
        "component_means_for_normalization": component_means,
        "layers": layer_metrics,
    }
    write_json(output_dir / "affinity_kld_metrics.json", payload)

    ranked = sorted(layer_metrics, key=lambda item: item["total_kld_v1"])
    write_json(
        output_dir / "affinity_kld_ranking_by_low_kld.json",
        {
            "ranking_key": "total_kld_v1 ascending",
            "layers": ranked,
        },
    )
    print(f"Wrote KLD metrics to {output_dir / 'affinity_kld_metrics.json'}")
    print("Lowest total_kld_v1 layers:")
    for item in ranked[: min(5, len(ranked))]:
        print(
            f"  {item['index']:02d} {item['short_name']:<10} "
            f"total_kld_v1={item['total_kld_v1']:.6g}"
        )


if __name__ == "__main__":
    main()
