#!/usr/bin/env python3
"""VGG16 (CIFAR-100) accuracy evaluation — the affinity framework's audit network.

Two modes:
  * ``fp``   — full-precision baseline; uses the public chenyaofo CIFAR-100 VGG16-BN
               weights (auto-downloaded by ``pytorch_cifar_models``).
  * ``int8`` — DCNM INT8 baseline; wraps the model with the LSQ INT8 quantization
               layers and loads the released quantized checkpoint (the KLD-profiling
               reference model).

CIFAR-100 is fetched automatically by torchvision into ``--data-root``.

Examples:
    python eval.py --mode fp
    python eval.py --mode int8 --int8-checkpoint weights/vgg16_int8_baseline_cifar100.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torchvision
from torch.utils.data import DataLoader

from pytorch_cifar_models import cifar100_vgg16_bn
from vgg16_pipeline.dataset_profiles import build_cifar_transform
from vgg16_pipeline.kl import calculate_accuracy

CONFIGS_DIR = SCRIPT_DIR / "configs"
QUANT_DEFAULT_FILE = CONFIGS_DIR / "quant_default.yaml"


def build_testloader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=build_cifar_transform(train=False)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_calibration_loader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=build_cifar_transform(train=False)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def wrap_int8_and_load(model, quant_config: Path, calib_loader, checkpoint: Path, device):
    """Wrap the model with DCNM INT8 quant layers and load the quantized checkpoint."""

    from quantization_and_noise.config_loader import get_config
    from quantization_and_noise.util import prepare_quant_model2

    args_ = get_config(default_file=str(QUANT_DEFAULT_FILE), config_file=[str(quant_config)])

    class _DeviceLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                yield tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)

        def __len__(self):
            return len(self.loader)

    model = model.to(device)
    model = prepare_quant_model2(model, _DeviceLoader(calib_loader), args_.quan)
    state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="VGG16 CIFAR-100 accuracy evaluation.")
    parser.add_argument("--mode", choices=["fp", "int8"], default="fp")
    parser.add_argument("--data-root", default="./cifar_data")
    parser.add_argument("--int8-checkpoint", type=Path, default=SCRIPT_DIR / "weights" / "vgg16_int8_baseline_cifar100.pth")
    parser.add_argument("--quant-config", type=Path, default=CONFIGS_DIR / "vgg16_quant_8bit.yaml")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    testloader = build_testloader(args.data_root, args.batch_size, args.num_workers)

    model = cifar100_vgg16_bn(pretrained=True)
    if args.mode == "int8":
        calib = build_calibration_loader(args.data_root, args.batch_size, args.num_workers)
        model = wrap_int8_and_load(model, args.quant_config, calib, args.int8_checkpoint, device)
    model = model.to(device).eval()

    acc = calculate_accuracy(model, testloader, device)
    print(f"VGG16 CIFAR-100 {args.mode} top-1 accuracy: {acc:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
