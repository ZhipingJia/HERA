#!/usr/bin/env python3
"""Compute PrivateLLM GSM8K layer affinity for LoRA ACIM mapping.

GSM8K exact-match evaluation is generation-heavy and very slow for the INT
modules, so this script uses a teacher-forced answer-token KL as the model
quality term:

    total_kld_v1 = mean KL(DCNM_INT8 logits || single-layer ACIM logits)
                   over answer-token prediction positions

The EDP/epsilon/final-affinity convention is shared with the BoolQ/VGG/FasterRCNN
pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm

# GSM8K paper affinity now uses the r256 PrivateLoRA models. The model module
# reads PRIVATE_LLM_LORA_RANK at import time, so set it before importing the
# shared BoolQ helpers that import modeling_llama_pl. Use a GSM8K-specific
# override to avoid accidentally inheriting PRIVATE_LLM_LORA_RANK=128.
os.environ["PRIVATE_LLM_LORA_RANK"] = os.environ.get("PRIVATE_LLM_GSM8K_AFFINITY_LORA_RANK", "256")

from compute_boolq_affinity import (
    HARDWARE_MODEL_VERSION,
    default_dataset_path,
    build_model,
    combine_affinity,
    hardware_constants,
    parse_layer_indices,
    register_edp_hooks,
    replace_one_layer_with_acim,
    restore_layer,
    set_seed,
    target_layers_from_model,
    to_jsonable,
    write_json,
)
from mymodels import modeling_llama_pl





class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update_tensor(self, value: torch.Tensor) -> None:
        value = value.detach().double().reshape(-1)
        self.total += float(value.sum().cpu().item())
        self.count += int(value.numel())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PrivateLLM GSM8K layer affinity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path("gsm8k/test"),
        help="datasets.save_to_disk path of the GSM8K test split; defaults to "
             "$PRIVATE_LLM_DATASET_ROOT/gsm8k/test when present.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="plllama-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--quant-file", default="config/config_pl_quant_int8.yaml")
    parser.add_argument("--quant-file-2", default="config/config_pl_quant_int8_cgra.yaml")
    parser.add_argument("--lm-head-file", default="config/config_pl_quant_lora_lm_head_int8.yaml")
    parser.add_argument(
        "--disable-lm-head-lora",
        action="store_true",
        help="Zero and skip lm_head LoRA B for legacy GSM8K checkpoints that do not contain it.",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def disable_lm_head_lora(model: nn.Module) -> None:
    modeling_llama_pl.LM_HEAD_LORA = False
    for name in ("lora_lm_head_A", "lora_lm_head_B"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "weight"):
            with torch.no_grad():
                module.weight.zero_()
            module.weight.requires_grad = False


def gsm8k_prompt(sample: dict[str, Any]) -> str:
    return f"Question: {sample['question']}\nAnswer:"


def load_gsm8k_samples(dataset_path: Path, max_samples: int) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(dataset_path))
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def encode_sample(tokenizer, sample: dict[str, Any], source_max_len: int) -> dict[str, Any] | None:
    prompt_ids = tokenizer(
        tokenizer.bos_token + gsm8k_prompt(sample),
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        return None

    full_ids = prompt_ids + answer_ids
    if len(full_ids) > source_max_len:
        full_ids = full_ids[:source_max_len]

    prompt_len = len(prompt_ids)
    available_answer_tokens = max(0, len(full_ids) - prompt_len)
    if available_answer_tokens <= 0:
        return None

    # Logit at position i predicts token i+1.  We score only answer-token
    # prediction positions, i.e. prompt_len-1 predicts the first answer token.
    answer_positions = list(range(prompt_len - 1, prompt_len + available_answer_tokens - 1))
    answer_positions = [pos for pos in answer_positions if 0 <= pos < len(full_ids) - 1]
    if not answer_positions:
        return None

    return {
        "input_ids": full_ids,
        "answer_positions": answer_positions,
        "question": sample["question"],
        "answer": sample["answer"],
    }


def make_batches(
    tokenizer,
    samples: list[dict[str, Any]],
    batch_size: int,
    source_max_len: int,
) -> list[dict[str, Any]]:
    encoded = [
        item for item in (encode_sample(tokenizer, sample, source_max_len) for sample in samples)
        if item is not None
    ]
    batches = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        max_len = max(len(item["input_ids"]) for item in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        answer_mask = torch.zeros((len(chunk), max_len - 1), dtype=torch.bool)
        answer_token_count = 0
        for row, item in enumerate(chunk):
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            input_ids[row, : ids.numel()] = ids
            attention_mask[row, : ids.numel()] = 1
            for pos in item["answer_positions"]:
                answer_mask[row, pos] = True
            answer_token_count += len(item["answer_positions"])

        batches.append({
            "batch_index": len(batches),
            "sample_start": start,
            "sample_end": start + len(chunk),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "answer_token_count": answer_token_count,
        })
    return batches


def answer_logits_from_outputs(logits: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    return shifted_logits[answer_mask.to(logits.device)]


def reference_forward_profile(
    model: nn.Module,
    batches: list[dict[str, Any]],
    target_layers: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hooks, accumulators = register_edp_hooks(model, target_layers)
    ref_batches = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="Reference DCNM"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            answer_logits = answer_logits_from_outputs(outputs.logits, batch["answer_mask"])
            ref_answer_log_prob = F.log_softmax(answer_logits.float(), dim=-1).detach().cpu().to(torch.float16)
            ref_batches.append({
                **batch,
                "ref_answer_log_prob": ref_answer_log_prob,
            })
            del outputs, answer_logits, input_ids, attention_mask
    for hook in hooks:
        hook.remove()
    edp_rows = [accumulators[layer["full_name"]].mean() for layer in target_layers]
    return ref_batches, edp_rows


def profile_single_layer_kld(
    model: nn.Module,
    layer: dict[str, Any],
    ref_batches: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    old_module, replacement = replace_one_layer_with_acim(model, layer, args)
    answer_accum = RunningMean()
    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(ref_batches, desc=f"KLD {layer['short_name']}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                for repeat_idx in range(args.noise_repeats):
                    seed = args.seed + int(layer["index"]) * 1_000_000 + int(batch["batch_index"]) * 100 + repeat_idx
                    set_seed(seed)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    answer_logits = answer_logits_from_outputs(outputs.logits, batch["answer_mask"])
                    test_log_prob = F.log_softmax(answer_logits.float(), dim=-1).detach().cpu()
                    ref_log_prob = batch["ref_answer_log_prob"].float()
                    answer_kl = (ref_log_prob.exp() * (ref_log_prob - test_log_prob)).sum(dim=-1)
                    answer_accum.update_tensor(answer_kl)
                    del outputs, answer_logits, test_log_prob
                del input_ids, attention_mask
    finally:
        restore_layer(model, layer, old_module)
    return {
        **layer,
        "gsm8k_answer_token_kl": answer_accum.mean,
        "total_kld_v1": answer_accum.mean,
        "total_kld_rule": "private_llm_gsm8k_teacher_forced_answer_token_kl_v1",
        "kld_direction": "KL(DCNM_INT8 || single_layer_ACIM)",
        "profiled_samples": sum(batch["input_ids"].shape[0] for batch in ref_batches),
        "profiled_answer_tokens": sum(int(batch["answer_token_count"]) for batch in ref_batches),
        "noise_repeats": args.noise_repeats,
        "replacement": replacement,
    }


def write_summary(path: Path, affinity: dict[str, Any], run_config: dict[str, Any]) -> None:
    lines = [
        "# PrivateLLM GSM8K INT8 Affinity Summary",
        "",
        f"Run: `{Path(run_config['output_dir']).name}`",
        f"Checkpoint: `{run_config['checkpoint']}`",
        f"Samples: `{run_config['actual_samples']}` GSM8K test samples, batch size `{run_config['batch_size']}`",
        "KLD rule: `total_kld_v1 = teacher-forced answer-token KL`",
        f"Epsilon: `{affinity['epsilon_policy']['epsilon']:.12g}` = `{run_config['epsilon_ratio']}` * median KLD `{affinity['epsilon_policy']['median_total_kld_v1']:.12g}`",
        f"Positive EDP layers: `{len(affinity['positive_affinity_rank'])}`; DCNM-preferred EDP-negative layers: `{len(affinity['negative_edp_layers'])}`",
        "",
        "## Top 20 Positive Affinity Layers",
        "",
        "| rank | index | layer | affinity | edp_diff | total_kld_v1 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(affinity["positive_affinity_rank"][:20], start=1):
        lines.append(
            f"| {rank} | {row['index']} | `{row['short_name']}` | "
            f"{row['affinity_score']:.6e} | {row['edp_diff']:.6e} | {row['total_kld_v1']:.6g} |"
        )
    lines.extend([
        "",
        "## Notes",
        "- total_kld_v1 is GSM8K teacher-forced answer-token KL, used to avoid slow generation during affinity profiling.",
        "- edp_diff = EDP_DCNM - EDP_ACIM; positive means ACIM reduces EDP.",
        "- This run profiles q/k/v lora_mobile layers and, when enabled, lm_head LoRA B.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"samples{args.max_samples}_bs{args.batch_size}_all_layers_gpu0_{datetime.now().strftime('%m%d%H%M%S')}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / Path(__file__).name)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    model, tokenizer, load_result = build_model(args, device)
    if args.disable_lm_head_lora:
        disable_lm_head_lora(model)

    all_layers = target_layers_from_model(model)
    if args.disable_lm_head_lora:
        all_layers = [layer for layer in all_layers if int(layer["index"]) < 96]
    selected_indices = set(parse_layer_indices(args.layer_indices, all_layers))
    selected_layers = [layer for layer in all_layers if int(layer["index"]) in selected_indices]
    write_json(output_dir / "target_layers_private_llm_gsm8k.json", {"layers": all_layers})

    samples = load_gsm8k_samples(args.dataset, args.max_samples)
    batches = make_batches(tokenizer, samples, args.batch_size, args.source_max_len)
    actual_samples = sum(batch["input_ids"].shape[0] for batch in batches)
    answer_tokens = sum(int(batch["answer_token_count"]) for batch in batches)
    if actual_samples == 0 or answer_tokens == 0:
        raise RuntimeError("No valid GSM8K samples/answer tokens to profile")

    run_config = {
        "checkpoint": args.checkpoint.resolve(),
        "dataset": args.dataset.resolve(),
        "output_dir": output_dir.resolve(),
        "model_name": args.model_name,
        "device": str(device),
        "max_samples": args.max_samples,
        "actual_samples": actual_samples,
        "profiled_answer_tokens": answer_tokens,
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
        "lm_head_lora_disabled": bool(args.disable_lm_head_lora),
        "load_result": load_result,
    }
    write_json(output_dir / "run_config.json", run_config)

    print(f"Profiling {len(selected_layers)} layers on {actual_samples} GSM8K samples ({answer_tokens} answer tokens); output={output_dir}")
    ref_batches, edp_rows = reference_forward_profile(model, batches, selected_layers, device)
    write_json(output_dir / "edp_profile.json", {
        "hardware_constants": hardware_constants(),
        "layers": edp_rows,
    })

    kld_rows = []
    for layer in selected_layers:
        print(f"Profiling KLD layer {layer['index']} {layer['full_name']}")
        metric = profile_single_layer_kld(model, layer, ref_batches, args, device)
        kld_rows.append(metric)
        write_json(output_dir / "raw_kld_metrics_partial.json", {
            "total_kld_rule": "private_llm_gsm8k_teacher_forced_answer_token_kl_v1",
            "layers": kld_rows,
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output_dir / "kld_metrics.json", {
        "total_kld_rule": "private_llm_gsm8k_teacher_forced_answer_token_kl_v1",
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_model_version": HARDWARE_MODEL_VERSION,
        "run_config": run_config,
        "notes": [
            "total_kld_v1 is GSM8K teacher-forced answer-token KL.",
            "Generation exact-match is not used during affinity profiling because INT GSM8K generation is too slow.",
            "lm_head LoRA B is included unless --disable-lm-head-lora is set.",
            "edp_diff = EDP_DCNM - EDP_ACIM; positive means ACIM reduces EDP.",
        ],
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
    write_summary(output_dir / "affinity_summary.md", affinity, to_jsonable(run_config))

    print(f"Wrote affinity metrics to {output_dir / 'affinity_metrics.json'}")
    print("Top positive affinity layers:")
    for row in affinity["positive_affinity_rank"][:10]:
        print(
            f"  {row['index']:02d} {row['short_name']:<14} "
            f"affinity={row['affinity_score']:.6e} "
            f"edp_diff={row['edp_diff']:.6e} kld={row['total_kld_v1']:.6g}"
        )


if __name__ == "__main__":
    main()
