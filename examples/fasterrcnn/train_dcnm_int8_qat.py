from __future__ import annotations

import argparse
import json
import logging
import math
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

DEFAULT_QUANT_CONFIG = CONFIGS_DIR / "config_dcnm_int8.yaml"
DEFAULT_TARGET_LAYERS = CONFIGS_DIR / "target_layers_v0.json"


def setup_paths() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


setup_paths()

from build_dcnm_baseline import (  # noqa: E402
    get_submodule_by_name,
    load_target_layers,
    to_jsonable,
    verify_dcnm_layers,
)
from data.dataset import Dataset, TestDataset  # noqa: E402
from eval_voc import build_model, eval_detector  # noqa: E402
from quantization_and_noise import prepare_quant_model2  # noqa: E402
from quantization_and_noise.quant_layer import (  # noqa: E402
    conv2d_quant_noise,
    conv2d_quant_sample_noise,
    linear_quant_noise,
    linear_quant_sample_noise,
)
from profile_kld import parse_layer_indices, replace_one_layer_with_acim  # noqa: E402
from trainer import FasterRCNNTrainer  # noqa: E402
from quantization_and_noise.config_loader import get_config  # noqa: E402
from utils import array_tool as at  # noqa: E402
from utils.config import opt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT train the FasterRCNN DCNM INT8/noise=0 model.")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, default=None)
    parser.add_argument("--quant-config", type=Path, default=DEFAULT_QUANT_CONFIG)
    parser.add_argument("--target-layers", type=Path, default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="stage_lr")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--decay-epoch", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--test-num-workers", type=int, default=8)
    parser.add_argument("--eval-test-num", type=int, default=10000)
    parser.add_argument("--initial-test-num", type=int, default=2000)
    parser.add_argument("--plot-every", type=int, default=500)
    parser.add_argument("--initial-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-count-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trainable-scope", choices=["all", "scales"], default="all")
    parser.add_argument("--acim-layer-indices", default="")
    parser.add_argument(
        "--reset-new-acim-layer-indices",
        default="",
        help=(
            "Comma-separated ACIM layer indices newly added in this stage. "
            "Their ACIM a/w/sample_out scales are restored after warm-start loading "
            "so DCNM checkpoint scale keys cannot overwrite the new ACIM scale setup."
        ),
    )
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


def make_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "vis").mkdir(exist_ok=True)

    logger = logging.getLogger("dcnm_int8_qat")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_formatter = logging.Formatter("%(levelname)s:%(message)s")
    stream_formatter = logging.Formatter("\n%(levelname)s:%(message)s")
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def apply_runtime_options(args: argparse.Namespace) -> dict:
    original_weight_count_loss = dict(getattr(opt, "weight_count_loss", {}))
    opt.device = args.device
    opt.epoch = args.epochs
    opt.lr = args.lr
    opt.lr_decay = args.lr_decay
    opt.decay_epoch = args.decay_epoch
    opt.weight_decay = args.weight_decay
    opt.use_adam = args.use_adam
    opt.num_workers = args.num_workers
    opt.test_num_workers = args.test_num_workers
    opt.test_num = args.eval_test_num
    opt.plot_every = args.plot_every
    opt.load_path = str(args.source_checkpoint)
    if args.voc_data_dir is not None:
        opt.voc_data_dir = args.voc_data_dir
    if args.disable_count_loss:
        opt.weight_count_loss = {}
    return original_weight_count_loss


def build_dataloaders(args: argparse.Namespace):
    trainset = Dataset(opt)
    train_dataloader = data_.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def build_dcnm_model(quant_config: Path):
    faster_rcnn = build_model()
    args_ = get_config(
        default_file=str(QUANT_DEFAULT_FILE),
        config_file=[str(quant_config)],
    )
    prepare_quant_model2(faster_rcnn.extractor, None, args_.quan)
    prepare_quant_model2(faster_rcnn.rpn, None, args_.quan)
    prepare_quant_model2(faster_rcnn.head, None, args_.quan)
    return faster_rcnn


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    logger.info("strict load result: %s", load_result)
    logger.info("loaded checkpoint: %s", checkpoint_path)
    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def load_checkpoint_matching(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()

    compatible = {}
    skipped_shape = {}
    unexpected = []
    for key, tensor in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if hasattr(tensor, "shape") and hasattr(model_state[key], "shape") and tensor.shape != model_state[key].shape:
            skipped_shape[key] = {
                "checkpoint_shape": list(tensor.shape),
                "model_shape": list(model_state[key].shape),
            }
            continue
        compatible[key] = tensor

    load_result = model.load_state_dict(compatible, strict=False)
    logger.info("warm-start checkpoint: %s", checkpoint_path)
    logger.info(
        "warm-start loaded compatible keys: %d, missing after load: %d, unexpected skipped: %d, shape skipped: %d",
        len(compatible),
        len(load_result.missing_keys),
        len(unexpected),
        len(skipped_shape),
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "loaded_key_count": len(compatible),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_skipped_keys": unexpected,
        "shape_skipped_keys": skipped_shape,
    }


def apply_trainable_scope(model: torch.nn.Module, scope: str, logger: logging.Logger) -> list[str]:
    if scope == "all":
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        logger.info("trainable_scope=all, trainable parameter tensors: %d", len(trainable))
        return trainable

    selected = []
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(".s")
        if param.requires_grad:
            selected.append(name)
    if not selected:
        raise RuntimeError("trainable_scope=scales selected no parameters")
    logger.info("trainable_scope=scales, trainable parameter tensors: %d", len(selected))
    for name in selected:
        logger.info("  trainable: %s", name)
    return selected


def selected_acim_layers(args: argparse.Namespace, target_layers: list[dict]) -> list[dict]:
    if not args.acim_layer_indices.strip():
        return []
    selected_indices = set(parse_layer_indices(args.acim_layer_indices, target_layers))
    return [layer for layer in target_layers if int(layer["index"]) in selected_indices]


def selected_reset_new_acim_layers(args: argparse.Namespace, target_layers: list[dict]) -> list[dict]:
    if not args.reset_new_acim_layer_indices.strip():
        return []
    selected_indices = set(parse_layer_indices(args.reset_new_acim_layer_indices, target_layers))
    acim_indices = set(parse_layer_indices(args.acim_layer_indices, target_layers)) if args.acim_layer_indices.strip() else set()
    missing_from_acim = sorted(selected_indices - acim_indices)
    if missing_from_acim:
        raise ValueError(
            "--reset-new-acim-layer-indices must be a subset of --acim-layer-indices; "
            f"missing from ACIM set: {missing_from_acim}"
        )
    return [layer for layer in target_layers if int(layer["index"]) in selected_indices]


def replace_acim_layers(
    model: torch.nn.Module,
    target_layers: list[dict],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> list[dict]:
    replacements = []
    for layer in selected_acim_layers(args, target_layers):
        summary = replace_one_layer_with_acim(model, layer, args)
        replacements.append(summary)
        logger.info(
            "ACIM layer %s (%s) -> %s, sample_out_scale=%s",
            layer["full_name"],
            layer["short_name"],
            summary["new_class"],
            summary["sample_out_scale"],
        )
    if replacements:
        logger.info("ACIM replacements: %d layer(s)", len(replacements))
    else:
        logger.info("ACIM replacements: none; training pure DCNM INT8 model")
    return replacements


def capture_acim_scale_state(model: torch.nn.Module, layers: list[dict]) -> dict[str, dict]:
    scale_state = {}
    for layer in layers:
        module = get_submodule_by_name(model, layer["full_name"])
        if not isinstance(module, (conv2d_quant_sample_noise, linear_quant_sample_noise)):
            raise TypeError(
                f"{layer['full_name']} is {type(module).__name__}, expected ACIM sample-noise module"
            )
        scale_state[layer["full_name"]] = {
            "index": layer["index"],
            "short_name": layer["short_name"],
            "a_scale": module.a_quantizer.s.detach().clone(),
            "w_scale": module.w_quantizer.s.detach().clone(),
            "sample_out_scale": module.sample_out_scale.detach().clone(),
        }
    return scale_state


def restore_acim_scale_state_after_warm_start(
    model: torch.nn.Module,
    scale_state: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    reset_summaries = []
    for full_name, desired in scale_state.items():
        module = get_submodule_by_name(model, full_name)
        if not isinstance(module, (conv2d_quant_sample_noise, linear_quant_sample_noise)):
            raise TypeError(f"{full_name} is {type(module).__name__}, expected ACIM sample-noise module")

        before = {
            "a_scale": module.a_quantizer.s.detach().clone(),
            "w_scale": module.w_quantizer.s.detach().clone(),
            "sample_out_scale": module.sample_out_scale.detach().clone(),
        }

        module.a_quantizer.s.data.copy_(
            desired["a_scale"].to(device=module.a_quantizer.s.device, dtype=module.a_quantizer.s.dtype)
        )
        module.w_quantizer.s.data.copy_(
            desired["w_scale"].to(device=module.w_quantizer.s.device, dtype=module.w_quantizer.s.dtype)
        )
        module.sample_out_scale.data.copy_(
            desired["sample_out_scale"].to(
                device=module.sample_out_scale.device,
                dtype=module.sample_out_scale.dtype,
            )
        )

        after = {
            "a_scale": module.a_quantizer.s.detach().clone(),
            "w_scale": module.w_quantizer.s.detach().clone(),
            "sample_out_scale": module.sample_out_scale.detach().clone(),
        }
        summary = {
            "index": desired["index"],
            "full_name": full_name,
            "short_name": desired["short_name"],
            "before_warm_reset": {key: to_jsonable(value) for key, value in before.items()},
            "restored_to": {
                "a_scale": to_jsonable(desired["a_scale"]),
                "w_scale": to_jsonable(desired["w_scale"]),
                "sample_out_scale": to_jsonable(desired["sample_out_scale"]),
            },
            "after_warm_reset": {key: to_jsonable(value) for key, value in after.items()},
        }
        reset_summaries.append(summary)
        logger.info(
            "reset new ACIM scale %s (%s): a %s -> %s, w %s -> %s, sample_out %s -> %s",
            full_name,
            desired["short_name"],
            to_jsonable(before["a_scale"]),
            to_jsonable(after["a_scale"]),
            to_jsonable(before["w_scale"]),
            to_jsonable(after["w_scale"]),
            to_jsonable(before["sample_out_scale"]),
            to_jsonable(after["sample_out_scale"]),
        )
    return reset_summaries


def infer_quantizer_bit_width(quantizer: torch.nn.Module) -> int | None:
    if hasattr(quantizer, "bit"):
        bit = getattr(quantizer, "bit")
        return int(bit) if bit is not None else None

    thd_neg = getattr(quantizer, "thd_neg", None)
    thd_pos = getattr(quantizer, "thd_pos", None)
    if thd_neg is None or thd_pos is None:
        return None

    levels = int(thd_pos) - int(thd_neg) + 1
    if levels <= 0:
        return None

    bit = int(round(math.log2(levels)))
    if 2**bit == levels:
        return bit

    # Symmetric LSQ often uses [-2^(b-1)+1, 2^(b-1)-1], e.g. [-7, 7].
    # The representable integer count is then 2^b-1 rather than 2^b.
    if int(thd_neg) == -int(thd_pos):
        symmetric_bit = int(round(math.log2(int(thd_pos) + 1))) + 1
        if int(thd_pos) == 2 ** (symmetric_bit - 1) - 1:
            return symmetric_bit

    return None


def verify_hybrid_layers(model: torch.nn.Module, target_layers: list[dict], acim_layers: list[dict]) -> dict:
    acim_names = {layer["full_name"] for layer in acim_layers}
    quant_noise_types = (conv2d_quant_noise, linear_quant_noise)
    sample_noise_types = (conv2d_quant_sample_noise, linear_quant_sample_noise)
    layers = []
    bad_layers = []

    for layer in target_layers:
        module = get_submodule_by_name(model, layer["full_name"])
        should_be_acim = layer["full_name"] in acim_names
        is_quant_noise = isinstance(module, quant_noise_types)
        is_sample_noise = isinstance(module, sample_noise_types)
        summary = {
            "index": layer["index"],
            "full_name": layer["full_name"],
            "short_name": layer["short_name"],
            "expected": "ACIM" if should_be_acim else "DCNM",
            "class_name": type(module).__name__,
            "is_quant_noise": is_quant_noise,
            "is_sample_noise": is_sample_noise,
        }
        if hasattr(module, "w_quantizer"):
            summary["weight_bit"] = infer_quantizer_bit_width(module.w_quantizer)
            summary["weight_thd_neg"] = getattr(module.w_quantizer, "thd_neg", None)
            summary["weight_thd_pos"] = getattr(module.w_quantizer, "thd_pos", None)
            summary["weight_noise_scale"] = getattr(module.w_quantizer, "noise_scale", None)
        if hasattr(module, "a_quantizer"):
            summary["act_bit"] = infer_quantizer_bit_width(module.a_quantizer)
            summary["act_thd_neg"] = getattr(module.a_quantizer, "thd_neg", None)
            summary["act_thd_pos"] = getattr(module.a_quantizer, "thd_pos", None)
            summary["act_noise_scale"] = getattr(module.a_quantizer, "noise_scale", None)
        if hasattr(module, "sample_noise_mode"):
            summary["sample_noise_mode"] = module.sample_noise_mode
            summary["sample_noise_kwargs"] = dict(module.sample_noise_kwargs)
        layers.append(summary)

        expected_ok = is_sample_noise if should_be_acim else is_quant_noise
        if not expected_ok:
            bad_layers.append(summary)

    return {
        "ok": len(bad_layers) == 0,
        "acim_layer_count": len(acim_layers),
        "dcnm_layer_count": len(target_layers) - len(acim_layers),
        "bad_layers": bad_layers,
        "layers": layers,
    }


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def main() -> None:
    args = parse_args()
    original_weight_count_loss = apply_runtime_options(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(opt.device)

    log_dir = args.output_root / args.run_name
    logger = make_logger(log_dir)
    shutil.copyfile(__file__, log_dir / Path(__file__).name)
    shutil.copyfile(args.quant_config, log_dir / args.quant_config.name)

    logger.info("building dataloaders")
    train_dataloader, test_dataloader = build_dataloaders(args)

    logger.info("building DCNM INT8 model")
    faster_rcnn = build_dcnm_model(args.quant_config)
    load_result = load_checkpoint_strict(faster_rcnn, args.source_checkpoint, logger)

    target_layers = load_target_layers(args.target_layers)
    acim_layers = selected_acim_layers(args, target_layers)
    acim_replacements = replace_acim_layers(faster_rcnn, target_layers, args, logger)
    reset_new_acim_layers = selected_reset_new_acim_layers(args, target_layers)
    desired_new_acim_scales = capture_acim_scale_state(faster_rcnn, reset_new_acim_layers)
    warm_start_result = None
    reset_new_acim_scales = []
    if args.warm_start_checkpoint is not None:
        warm_start_result = load_checkpoint_matching(faster_rcnn, args.warm_start_checkpoint, logger)
        reset_new_acim_scales = restore_acim_scale_state_after_warm_start(
            faster_rcnn,
            desired_new_acim_scales,
            logger,
        )
    if acim_layers:
        verification = verify_hybrid_layers(faster_rcnn, target_layers, acim_layers)
        if not verification["ok"]:
            raise RuntimeError(f"Hybrid ACIM/DCNM verification failed: {verification['bad_layers']}")
    else:
        verification = verify_dcnm_layers(faster_rcnn, target_layers)
        if not verification["ok"]:
            raise RuntimeError(f"DCNM verification failed: {verification['bad_layers']}")

    trainable_names = apply_trainable_scope(faster_rcnn, args.trainable_scope, logger)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    run_config = {
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "quant_config": str(args.quant_config.resolve()),
        "target_layers": str(args.target_layers.resolve()),
        "log_dir": str(log_dir.resolve()),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "decay_epoch": args.decay_epoch,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam" if args.use_adam else "SGD(momentum=0.9)",
        "disable_count_loss": args.disable_count_loss,
        "weight_count_loss": dict(getattr(opt, "weight_count_loss", {})),
        "original_weight_count_loss": original_weight_count_loss,
        "trainable_scope": args.trainable_scope,
        "trainable_parameter_tensors": len(trainable_names),
        "eval_test_num": args.eval_test_num,
        "initial_eval": args.initial_eval,
        "quiet_progress": args.quiet_progress,
        "initial_test_num": args.initial_test_num,
        "load_result": load_result,
        "warm_start_result": warm_start_result,
        "reset_new_acim_layer_indices": args.reset_new_acim_layer_indices,
        "reset_new_acim_scales": reset_new_acim_scales,
        "verification": verification,
        "acim_layer_indices": args.acim_layer_indices,
        "acim_layers": acim_layers,
        "acim_replacements": acim_replacements,
        "sample_noise_mode": args.sample_noise_mode,
        "sample_noise_mode_overrides": args.sample_noise_mode_overrides,
        "sample_noise_std": args.sample_noise_std,
        "sample_noise_scale": args.sample_noise_scale,
        "sample_output_min": args.sample_output_min,
        "sample_output_max": args.sample_output_max,
        "sample_output_min_overrides": args.sample_output_min_overrides,
        "sample_output_max_overrides": args.sample_output_max_overrides,
    }
    write_json(log_dir / "effective_config.json", run_config)
    logger.info("effective config saved to: %s", log_dir / "effective_config.json")
    logger.info("effective run config: %s", run_config)

    initial_map = None
    best_map = -1.0
    best_path = None
    if args.initial_eval:
        logger.info("initial validation before QAT")
        eval_result = eval_detector(
            test_dataloader,
            faster_rcnn,
            test_num=args.initial_test_num,
            show_progress=not args.quiet_progress,
        )
        initial_map = float(eval_result["map"])
        logger.info("initial mAP: %s", initial_map)
        best_path = trainer.save(best_map=initial_map, save_path=str(log_dir), save_str="initial")
        best_map = initial_map
        logger.info("initial checkpoint saved: %s", best_path)

    history = []

    if args.epochs <= 0:
        if best_path is None:
            best_path = str(args.source_checkpoint.resolve())

    for epoch in range(args.epochs):
        trainer.reset_meters()
        for ii, (img, _sizes, bbox_, label_, scale) in tqdm(
            enumerate(train_dataloader),
            disable=args.quiet_progress,
        ):
            scale = at.scalar(scale)
            img = img.cuda().float()
            bbox = bbox_.cuda()
            label = label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                logger.info("epoch [%d/%d] %s", epoch, ii, trainer.get_meter_data())

        eval_result = eval_detector(
            test_dataloader,
            faster_rcnn,
            test_num=args.eval_test_num,
            show_progress=not args.quiet_progress,
        )
        current_map = float(eval_result["map"])
        current_lr = trainer.faster_rcnn.optimizer.param_groups[0]["lr"]
        meter_data = trainer.get_meter_data()
        logger.info("epoch:%d, lr:%s, map:%s, loss:%s", epoch, current_lr, current_map, meter_data)

        saved_path = trainer.save(best_map=current_map, save_path=str(log_dir), save_str=f"epoch{epoch}")
        if current_map > best_map:
            best_map = current_map
            best_path = saved_path

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "map": current_map,
            "loss": meter_data,
            "checkpoint": saved_path,
            "best_map_so_far": best_map,
            "best_path_so_far": best_path,
        }
        history.append(epoch_record)

        summary = {
            **run_config,
            "initial_map": initial_map,
            "best_map": best_map,
            "best_path": best_path,
            "history": history,
        }
        write_json(log_dir / "train_summary.json", summary)

        if epoch == opt.decay_epoch:
            if best_path is not None:
                trainer.load(best_path, load_optimizer=False)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)

    final_summary = {
        **run_config,
        "initial_map": initial_map,
        "best_map": best_map,
        "best_path": best_path,
        "history": history,
    }
    write_json(log_dir / "train_summary.json", final_summary)
    print(json.dumps({"best_map": best_map, "best_path": best_path, "log_dir": str(log_dir)}, indent=2))


if __name__ == "__main__":
    main()
