#!/usr/bin/env python3
"""VGG16 layer-wise KLD profiling (single-layer ACIM substitution) on CIFAR-100.

Builds the DCNM INT8 baseline (reference), then replaces one Conv2d/Linear layer
at a time with its ACIM sample-noise counterpart (5-bit activation / 4-bit weight,
output clamp [-127, 127]) and measures the Kullback-Leibler divergence of the
logits against the baseline, plus the top-1 accuracy drop. This is the accuracy-
sensitivity half of the affinity-aware mapping framework (Methods, Step 2).

The full-precision weights default to the public chenyaofo CIFAR-100 VGG16-BN
checkpoint (auto-downloaded); the INT8 baseline is the released quantized
checkpoint. CIFAR-100 is fetched by torchvision.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pytorch_cifar_models import __dict__ as model_zoo  # noqa: E402
from quantization_and_noise.config_loader import get_config  # noqa: E402
from quantization_and_noise.quant_layer import conv2d_quant_sample_noise, linear_quant_sample_noise  # noqa: E402
from quantization_and_noise.util import get_act_quantizer, get_weight_quantizer, prepare_quant_model2  # noqa: E402
from vgg16_pipeline.dataset_profiles import build_cifar_transform, get_profile  # noqa: E402
from vgg16_pipeline.kl import calculate_accuracy, calculate_kl_divergence, extract_logits  # noqa: E402
from vgg16_pipeline.layers import FULL_TO_SHORT, LAYER_ORDER  # noqa: E402

CONFIGS_DIR = SCRIPT_DIR / "configs"
QUANT_DEFAULT_FILE = CONFIGS_DIR / "quant_default.yaml"
BASELINE_CONFIG_FILE = CONFIGS_DIR / "vgg16_quant_8bit.yaml"
SAMPLE_NOISE_CONFIG_FILE = CONFIGS_DIR / "vgg16_quant_5bit_4bit.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
ACT_SCALE_MULTIPLIER = 2 ** (8 - 5)   # 5-bit ACIM activation
WEIGHT_SCALE_MULTIPLIER = 2 ** (8 - 4)  # 4-bit ACIM weight


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch)

    def __len__(self):
        return len(self.dataloader)


class LimitedDataLoader:
    def __init__(self, dataloader, max_batches: int | None):
        self.dataloader = dataloader
        self.max_batches = max_batches

    def __iter__(self):
        for idx, batch in enumerate(self.dataloader):
            if self.max_batches is not None and idx >= self.max_batches:
                break
            yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self.dataloader)
        return min(len(self.dataloader), self.max_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG16 layer-wise KLD via single-layer ACIM substitution.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default="./cifar_data")
    parser.add_argument("--weights-path", default=None, help="FP weights; default uses chenyaofo pretrained.")
    parser.add_argument("--int8-checkpoint", default=str(SCRIPT_DIR / "weights" / "vgg16_int8_baseline_cifar100.pth"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kl-output-name", default=None)
    parser.add_argument("--acc-output-name", default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--max-train-batches-for-init", type=int, default=None)
    parser.add_argument("--sample-noise-mode", choices=["sample_noise_2", "sample_noise_4"], default="sample_noise_4")
    parser.add_argument("--sample-noise-std", type=float, default=20.0)
    parser.add_argument("--sample-noise-scale", type=float, default=0.5)
    parser.add_argument("--sample-output-min", type=float, default=-127.0)
    parser.add_argument("--sample-output-max", type=float, default=127.0)
    parser.add_argument("--seed", type=int, default=20260515)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace, profile):
    dataset_cls = profile.dataset_cls
    transform = build_cifar_transform(train=False)
    testset = dataset_cls(root=args.data_root, train=False, download=True, transform=transform)
    trainset = dataset_cls(root=args.data_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        LimitedDataLoader(trainloader, args.max_train_batches_for_init),
        LimitedDataLoader(testloader, args.max_test_batches),
    )


def load_fp_model(device: torch.device, profile, weights_path: str | None) -> nn.Module:
    if weights_path:
        model = model_zoo[profile.model_name](pretrained=False)
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    else:
        model = model_zoo[profile.model_name](pretrained=True)  # chenyaofo public weights
    return model.to(device).eval()


def build_int8_baseline(device, trainloader, profile, weights_path, int8_checkpoint) -> nn.Module:
    args_ = get_config(default_file=str(QUANT_DEFAULT_FILE), config_file=[str(BASELINE_CONFIG_FILE)])
    model = load_fp_model(device, profile, weights_path)
    model = prepare_quant_model2(model, DeviceDataLoader(trainloader, device), args_.quan)
    checkpoint_path = Path(int8_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"INT8 baseline checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    return model.to(device).eval()


def find_named_module(model: nn.Module, layer_name: str):
    modules = dict(model.named_modules())
    target = modules.get(layer_name)
    if target is None:
        return None, None, None
    if "." in layer_name:
        parent = modules[".".join(layer_name.split(".")[:-1])]
        attr = layer_name.split(".")[-1]
    else:
        parent, attr = model, layer_name
    return target, parent, attr


def sample_noise_no_input_clamp(x, scale_factor=0.5, output_min=-127.0, output_max=127.0, noise_std=20.0):
    y = x * torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    if noise_std:
        y = y + torch.randn_like(y) * torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    return y.clamp(
        torch.as_tensor(output_min, dtype=x.dtype, device=x.device),
        torch.as_tensor(output_max, dtype=x.dtype, device=x.device),
    )


def resolve_sample_noise_mode(mode: str) -> str | Callable:
    return sample_noise_no_input_clamp if mode == "sample_noise_4" else mode


def replace_layer_with_sample_noise(model, layer_name, sample_noise_config, args):
    target, parent, attr = find_named_module(model, layer_name)
    if target is None:
        raise ValueError(f"Layer {layer_name} not found")

    w_scale_8bit = target.w_quantizer.s.item() if hasattr(target, "w_quantizer") else 1.0
    act_scale_8bit = target.a_quantizer.s.item() if hasattr(target, "a_quantizer") else 1.0
    act_scale_5bit = act_scale_8bit * ACT_SCALE_MULTIPLIER
    w_scale_4bit = w_scale_8bit * WEIGHT_SCALE_MULTIPLIER
    sample_out_scale = 2.0 * w_scale_4bit * act_scale_5bit
    sample_noise_kwargs = {
        "scale_factor": args.sample_noise_scale,
        "output_min": args.sample_output_min,
        "output_max": args.sample_output_max,
        "noise_std": args.sample_noise_std,
    }
    sample_noise_mode = resolve_sample_noise_mode(args.sample_noise_mode)

    is_conv = isinstance(target, nn.Conv2d) or hasattr(target, "in_channels")
    target_cfg = sample_noise_config["quan"]["conv" if is_conv else "fc"]
    factory = conv2d_quant_sample_noise if is_conv else linear_quant_sample_noise
    sample_module = factory(
        target,
        w_quantizer=get_weight_quantizer(target_cfg["weight"]),
        a_quantizer=get_act_quantizer(target_cfg["act"]),
        a_out_quantizer=get_act_quantizer(target_cfg.get("act_out", {"quant_name": None})),
        int_flag=True,
        sample_out_scale=sample_out_scale,
        sample_noise_mode=sample_noise_mode,
        sample_noise_kwargs=sample_noise_kwargs,
    )
    if hasattr(sample_module.w_quantizer, "s"):
        sample_module.w_quantizer.s.data.fill_(w_scale_4bit)
    if hasattr(sample_module.a_quantizer, "s"):
        sample_module.a_quantizer.s.data.fill_(act_scale_5bit)
    setattr(parent, attr, sample_module)
    return model, {
        "layer_name": layer_name,
        "short_name": FULL_TO_SHORT.get(layer_name, layer_name),
        "w_scale_4bit": w_scale_4bit,
        "act_scale_5bit": act_scale_5bit,
        "sample_out_scale": sample_out_scale,
        "sample_noise_mode": args.sample_noise_mode,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    profile = get_profile(args.dataset)

    with open(SAMPLE_NOISE_CONFIG_FILE) as f:
        sample_noise_config = yaml.safe_load(f)

    trainloader, testloader = build_dataloaders(args, profile)
    baseline_model = build_int8_baseline(device, trainloader, profile, args.weights_path, args.int8_checkpoint)

    print("Extracting baseline logits...")
    baseline_logits = extract_logits(baseline_model, testloader, device)
    baseline_accuracy = calculate_accuracy(baseline_model, testloader, device)
    print(f"baseline_acc={baseline_accuracy:.4f}%")

    layers, kl_by_layer, acc_by_layer = [], {}, {}
    for layer_name in tqdm(LAYER_ORDER, desc="Layer-wise KLD"):
        set_seed(args.seed)
        model_copy = copy.deepcopy(baseline_model)
        model_copy, _ = replace_layer_with_sample_noise(model_copy, layer_name, sample_noise_config, args)
        model_copy.to(device).eval()
        sample_logits = extract_logits(model_copy, testloader, device)
        accuracy = calculate_accuracy(model_copy, testloader, device)
        kl_div = calculate_kl_divergence(baseline_logits, sample_logits)
        short = FULL_TO_SHORT.get(layer_name, layer_name)
        layers.append({
            "layer_name": layer_name, "short_name": short, "kl_divergence": kl_div,
            "accuracy": accuracy, "accuracy_drop_from_baseline": baseline_accuracy - accuracy,
        })
        kl_by_layer[layer_name] = kl_div
        acc_by_layer[layer_name] = accuracy
        print(f"{short:>6s} {layer_name:>14s}: KL={kl_div:.8f}, acc={accuracy:.4f}%, drop={baseline_accuracy - accuracy:.4f}%")
        del model_copy, sample_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model_name": profile.model_name,
        "baseline_accuracy": baseline_accuracy,
        "sample_noise": {"mode": args.sample_noise_mode, "std": args.sample_noise_std, "scale": args.sample_noise_scale},
        "layers": layers,
        "kl_by_layer": kl_by_layer,
        "accuracy_by_layer": acc_by_layer,
    }
    kl_path = args.output_dir / (args.kl_output_name or f"vgg16_kld_{args.dataset}.json")
    acc_path = args.output_dir / (args.acc_output_name or f"vgg16_kld_accuracy_{args.dataset}.json")
    with open(kl_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(acc_path, "w") as f:
        json.dump({"dataset": args.dataset, "baseline_accuracy": baseline_accuracy,
                   "accuracy_by_layer": acc_by_layer, "layers": layers}, f, indent=2)
    print(f"Wrote {kl_path}\nWrote {acc_path}")


if __name__ == "__main__":
    main()
