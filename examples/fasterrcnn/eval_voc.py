#!/usr/bin/env python3
"""Faster R-CNN inference and VOC mAP evaluation on HERA's lightweight detector.

Builds the paper's channel-32 lightweight Faster R-CNN (``FasterRCNNVGG16LIGHTV3``),
loads a checkpoint (full-precision or DCNM INT8), runs detection over a VOC-style
dataset, and reports mAP50 (VOC07 11-point metric).

Example (full-precision checkpoint shipped with this repository):

    python eval_voc.py \
        --checkpoint weights/fasterrcnn_fp_map0889_epoch10.pth \
        --voc-data-dir /path/to/infrared_voc_root \
        --device 0

For a DCNM INT8 checkpoint produced by ``build_dcnm_baseline.py``, add
``--quant-config configs/config_dcnm_int8.yaml`` so the model is wrapped with the
INT8 quantization layers before the state dict is loaded.

The infrared human-detection dataset is not distributed with this repository;
``--voc-data-dir`` must point to a user-provided VOC-style directory (see
README for the expected layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch.utils import data as data_
from tqdm import tqdm

from data.dataset import TestDataset
from model import FasterRCNNVGG16LIGHTV3
from quantization_and_noise import prepare_quant_model2
from quantization_and_noise.config_loader import get_config
from utils.config import opt
from utils.eval_tool import eval_detection_voc

DEFAULT_QUANT_DEFAULT_FILE = SCRIPT_DIR / "configs" / "quant_default.yaml"


def build_model() -> torch.nn.Module:
    """Build the paper's lightweight channel-32 Faster R-CNN variant."""

    return FasterRCNNVGG16LIGHTV3(
        use_resnet=opt.use_resnet,
        use_maxpool=opt.use_maxpool,
        use_conv=opt.use_conv,
        use_rois_s=opt.use_rois_s,
        anchor_scales=opt.anchor_scales,
        channel=opt.channel,
        block_num=opt.block_num,
    )


def build_test_dataloader() -> data_.DataLoader:
    if not opt.voc_data_dir:
        raise ValueError("--voc-data-dir is required (VOC-style dataset root)")
    testset = TestDataset(opt)
    return data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        pin_memory=True,
    )


def quantize_as_dcnm(model: torch.nn.Module, dataloader, quant_config: Path) -> None:
    """Wrap all target layers with the DCNM INT8 quantization layers."""

    args_ = get_config(
        default_file=str(DEFAULT_QUANT_DEFAULT_FILE),
        config_file=[str(quant_config)],
    )
    prepare_quant_model2(model.extractor, dataloader, args_.quan)
    prepare_quant_model2(model.rpn, dataloader, args_.quan)
    prepare_quant_model2(model.head, dataloader, args_.quan)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, strict: bool = True):
    """Load a trainer-format checkpoint ``{'model': state_dict, ...}`` or a plain state dict."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return model.load_state_dict(state_dict, strict=strict)


def eval_detector(dataloader, faster_rcnn, test_num: int = 10000, show_progress: bool = True) -> dict:
    """Run detection over the dataloader and return the VOC07 mAP report."""

    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, scale) in tqdm(
        enumerate(dataloader),
        total=min(test_num, len(dataloader)),
        disable=not show_progress,
    ):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    return eval_detection_voc(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        use_07_metric=True,
    )


def apply_effective_config(model: torch.nn.Module, effective_config: Path, target_layers_path: Path) -> list[dict]:
    """Rebuild the hybrid ACIM/DCNM layer assignment recorded by a QAT run.

    ``effective_config.json`` is written by ``train_dcnm_int8_qat.py`` and records
    the ACIM layer indices plus the sample-noise layer settings, so a hybrid
    checkpoint (e.g. the paper's HERA-A / HERA-P / all-ACIM detectors) can be
    reconstructed for evaluation without re-running training.
    """

    import json
    from argparse import Namespace

    from profile_kld import parse_layer_indices, replace_one_layer_with_acim

    with open(effective_config) as f:
        cfg = json.load(f)
    with open(target_layers_path) as f:
        target_layers = json.load(f)["layers"]

    noise_args = Namespace(
        sample_noise_mode=cfg.get("sample_noise_mode", "sample_noise_4"),
        sample_noise_mode_overrides=cfg.get("sample_noise_mode_overrides", ""),
        sample_noise_std=float(cfg.get("sample_noise_std", 0.0)),
        sample_noise_scale=float(cfg.get("sample_noise_scale", 0.5)),
        sample_output_min=cfg.get("sample_output_min"),
        sample_output_max=cfg.get("sample_output_max"),
        sample_output_min_overrides=cfg.get("sample_output_min_overrides", ""),
        sample_output_max_overrides=cfg.get("sample_output_max_overrides", ""),
    )
    indices_spec = cfg.get("acim_layer_indices", "")
    if not indices_spec:
        return []
    selected = parse_layer_indices(str(indices_spec), target_layers)
    replacements = []
    for idx in selected:
        layer = target_layers[idx] if isinstance(idx, int) else idx
        replacements.append(replace_one_layer_with_acim(model, layer, noise_args))
    print(f"hybrid rebuild: {len(replacements)} ACIM layer(s) "
          f"[mode={noise_args.sample_noise_mode}, std={noise_args.sample_noise_std}]")
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser(description="Faster R-CNN VOC mAP evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-data-dir", type=Path, required=True)
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=None,
        help="DCNM INT8 quant YAML; omit for a full-precision checkpoint.",
    )
    parser.add_argument(
        "--effective-config",
        type=Path,
        default=None,
        help="effective_config.json of a hybrid ACIM/DCNM QAT run; rebuilds the "
             "recorded ACIM layer assignment before loading the checkpoint "
             "(implies --quant-config).",
    )
    parser.add_argument(
        "--target-layers",
        type=Path,
        default=SCRIPT_DIR / "configs" / "target_layers_v0.json",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260502,
                        help="Random seed; ACIM layers inject sampled noise at eval time.")
    args = parser.parse_args()

    opt.voc_data_dir = str(args.voc_data_dir)
    opt.device = args.device
    opt.test_num = args.test_num
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    quant_config = args.quant_config
    if args.effective_config is not None and quant_config is None:
        quant_config = SCRIPT_DIR / "configs" / "config_dcnm_int8.yaml"

    dataloader = build_test_dataloader()
    model = build_model()
    if quant_config is not None:
        quantize_as_dcnm(model, dataloader, quant_config)
    if args.effective_config is not None:
        apply_effective_config(model, args.effective_config, args.target_layers)
    load_result = load_checkpoint(model, args.checkpoint)
    print(f"checkpoint loaded: missing={list(load_result.missing_keys)} "
          f"unexpected={list(load_result.unexpected_keys)}")
    model = model.cuda().eval()

    result = eval_detector(dataloader, model, test_num=args.test_num)
    print(f"mAP50 (VOC07 metric): {result['map']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
