from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils import data as data_
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
QUANT_DEFAULT_FILE = CONFIGS_DIR / "quant_default.yaml"

DEFAULT_CONFIG = CONFIGS_DIR / "config_dcnm_int8.yaml"
DEFAULT_TARGET_LAYERS = CONFIGS_DIR / "target_layers_v0.json"


def setup_paths() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


setup_paths()

from data.dataset import TestDataset  # noqa: E402
from eval_voc import build_model, eval_detector  # noqa: E402
from quantization_and_noise import prepare_quant_model2  # noqa: E402
from quantization_and_noise.config_loader import get_config  # noqa: E402
from quantization_and_noise.quant_layer import (  # noqa: E402
    conv2d_quant_noise,
    conv2d_quant_sample_noise,
    linear_quant_noise,
    linear_quant_sample_noise,
)
from quantization_and_noise.quant_util import LSQ_act_quantizer  # noqa: E402
from utils.config import opt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the FasterRCNN DCNM INT8/noise=0 baseline checkpoint."
    )
    parser.add_argument("--fp-checkpoint", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target-layers", type=Path, default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="fasterrcnn_dcnm_int8_from_fp_best.pth")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=10000)
    parser.add_argument("--test-num-workers", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--eval-map", action="store_true")
    parser.add_argument("--voc-data-dir", default=None)
    return parser.parse_args()


def apply_runtime_options(args: argparse.Namespace) -> None:
    opt.device = args.device
    opt.test_num = args.test_num
    opt.test_num_workers = args.test_num_workers
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


def load_fp_weights(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def quantize_as_dcnm(model: torch.nn.Module, dataloader: data_.DataLoader, quant_config: Path) -> dict:
    args_ = get_config(
        default_file=str(QUANT_DEFAULT_FILE),
        config_file=[str(quant_config)],
    )
    prepare_quant_model2(model.extractor, dataloader, args_.quan)
    prepare_quant_model2(model.rpn, dataloader, args_.quan)
    prepare_quant_model2(model.head, dataloader, args_.quan)
    return args_


def calibrate_activation_scales(
    model: torch.nn.Module,
    dataloader: data_.DataLoader,
    num_batches: int,
    device: torch.device,
) -> dict:
    quantizers = [(name, module) for name, module in model.named_modules() if isinstance(module, LSQ_act_quantizer)]
    if num_batches <= 0 or not quantizers:
        return {
            "enabled": False,
            "requested_batches": num_batches,
            "used_batches": 0,
            "lsq_act_quantizers": len(quantizers),
        }

    for _name, module in quantizers:
        module.init_batch_mode = True
        module.init_batch_num = 0

    model.to(device)
    model.eval()
    used_batches = 0
    with torch.no_grad():
        for batch_idx, (imgs, _sizes, _gt_bboxes, _gt_labels, _gt_difficults, scale) in tqdm(
            enumerate(dataloader),
            total=min(num_batches, len(dataloader)),
            desc="Calibrating DCNM activation scales",
        ):
            if batch_idx >= num_batches:
                break
            imgs = imgs.to(device).float()
            scale_value = float(scale.item() if torch.is_tensor(scale) else scale)
            model(imgs, scale=scale_value)
            used_batches += 1

    for _name, module in quantizers:
        module.init_batch_mode = False

    return {
        "enabled": True,
        "requested_batches": num_batches,
        "used_batches": used_batches,
        "lsq_act_quantizers": len(quantizers),
        "first_scales": {
            name: float(module.s.detach().cpu().reshape(-1)[0])
            for name, module in quantizers[:10]
        },
    }


def load_target_layers(path: Path) -> list[dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["layers"]


def get_submodule_by_name(model: torch.nn.Module, full_name: str) -> torch.nn.Module:
    module = model
    for part in full_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def scalar_attr(module: torch.nn.Module, attr_path: str):
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return list(obj.detach().cpu().shape)
    return obj


def verify_dcnm_layers(model: torch.nn.Module, target_layers: list[dict]) -> dict:
    quant_noise_types = (conv2d_quant_noise, linear_quant_noise)
    sample_noise_types = (conv2d_quant_sample_noise, linear_quant_sample_noise)
    layer_summaries = []
    bad_layers = []

    for layer in target_layers:
        module = get_submodule_by_name(model, layer["full_name"])
        is_quant_noise = isinstance(module, quant_noise_types)
        is_sample_noise = isinstance(module, sample_noise_types)
        summary = {
            "index": layer["index"],
            "full_name": layer["full_name"],
            "short_name": layer["short_name"],
            "class_name": type(module).__name__,
            "is_quant_noise": is_quant_noise,
            "is_sample_noise": is_sample_noise,
        }
        if hasattr(module, "w_quantizer"):
            summary["weight_noise_scale"] = scalar_attr(module.w_quantizer, "noise_scale")
            summary["weight_scale"] = scalar_attr(module.w_quantizer, "s")
        if hasattr(module, "a_quantizer"):
            summary["act_noise_scale"] = scalar_attr(module.a_quantizer, "noise_scale")
            if hasattr(module.a_quantizer, "s"):
                summary["act_scale"] = scalar_attr(module.a_quantizer, "s")
        layer_summaries.append(summary)

        if not is_quant_noise or is_sample_noise:
            bad_layers.append(summary)
        if summary.get("weight_noise_scale", 0) != 0 or summary.get("act_noise_scale", 0) != 0:
            bad_layers.append(summary)

    remaining_sample_noise = [
        name
        for name, module in model.named_modules()
        if isinstance(module, sample_noise_types)
    ]

    return {
        "ok": len(bad_layers) == 0 and len(remaining_sample_noise) == 0,
        "target_layer_count": len(target_layers),
        "bad_layers": bad_layers,
        "remaining_sample_noise_layers": remaining_sample_noise,
        "layers": layer_summaries,
    }


def save_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    summary: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": opt._state_dict(),
            "affinity_summary": summary,
        },
        str(output_path),
    )


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


def main() -> None:
    args = parse_args()
    apply_runtime_options(args)
    torch.cuda.set_device(opt.device)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    dataloader = build_test_dataloader()
    faster_rcnn = build_model()

    load_result = load_fp_weights(faster_rcnn, args.fp_checkpoint)
    quant_args = quantize_as_dcnm(faster_rcnn, dataloader, args.quant_config)
    calibration = calibrate_activation_scales(
        faster_rcnn,
        dataloader,
        num_batches=args.calib_batches,
        device=device,
    )

    target_layers = load_target_layers(args.target_layers)
    verification = verify_dcnm_layers(faster_rcnn, target_layers)
    if not verification["ok"]:
        raise RuntimeError(f"DCNM verification failed: {verification['bad_layers']}")

    output_path = args.output_dir / args.output_name
    summary = {
        "source_fp_checkpoint": str(args.fp_checkpoint.resolve()),
        "output_checkpoint": str(output_path.resolve()),
        "quant_config": str(args.quant_config.resolve()),
        "target_layers": str(args.target_layers.resolve()),
        "load_result": load_result,
        "calibration": calibration,
        "verification": verification,
        "dcnm_definition": {
            "conv_linear": "conv2d_quant_noise / linear_quant_noise",
            "activation": "LSQ 8-bit",
            "weight": "LSQ 8-bit",
            "noise_scale": 0,
            "sample_noise": False,
            "act_out_quantizer": "NoQuan",
            "int_flag": False,
        },
    }

    if args.eval_map:
        faster_rcnn.to(device)
        result = eval_detector(dataloader, faster_rcnn, test_num=opt.test_num)
        summary["eval"] = {"map": float(result["map"]), "raw": result}
        print(f"DCNM INT8 baseline mAP: {result['map']}")

    save_checkpoint(faster_rcnn, output_path, summary)
    shutil.copyfile(args.quant_config, output_path.parent / args.quant_config.name)
    with open(output_path.parent / "dcnm_int8_baseline_summary.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print(f"Saved DCNM INT8 baseline checkpoint: {output_path}")
    print(f"Verified DCNM target layers: {verification['target_layer_count']}")


if __name__ == "__main__":
    main()
