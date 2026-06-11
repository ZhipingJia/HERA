#!/usr/bin/env python3
"""Evaluate a full-precision PrivateLoRA checkpoint on BoolQ / GSM8K with lm-eval.

Loads the frozen Llama-2-7b base model, overlays a PrivateLoRA checkpoint (the
96 PLM ``lora_mobile`` matrices plus the Head LB ``lora_lm_head_B``), and runs
the bundled lm-evaluation-harness tasks.

Environment / model resolution:
  * Base model: ``--base-model`` or env ``PRIVATE_LLM_BASE_MODEL`` (defaults to
    the ``meta-llama/Llama-2-7b-hf`` Hub id; not redistributed here).
  * Datasets: resolved from a local ``datasets.save_to_disk`` cache under env
    ``PRIVATE_LLM_DATASET_ROOT`` when present, otherwise from the HF Hub.
  * PrivateLoRA rank: ``--lora-rank`` (paper checkpoints use rank 256).

Example:

    python eval_pl.py --checkpoint-path /path/to/pl.bin --lora-rank 256 \
        --tasks boolq --batch-size 16 --output-json results/boolq_fp.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "lm-evaluation-harness"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="plllama-7b")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--base-model", default=None,
                        help="Path or HF Hub id of the Llama-2-7b base model.")
    parser.add_argument("--lora-rank", type=int, default=256,
                        help="PrivateLoRA rank used by the checkpoint (paper: 256).")
    parser.add_argument("--tasks", required=True, help="Comma/space-separated lm-eval tasks (boolq, gsm8k_yaml).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--log-samples", dest="log_samples", action="store_true")
    parser.add_argument("--no-log-samples", dest="log_samples", action="store_false")
    parser.set_defaults(log_samples=False)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[eval_pl] Ignoring launcher args: {unknown}")
    return args


def split_tasks(tasks):
    if isinstance(tasks, str):
        parts = []
        for item in tasks.split(","):
            parts.extend(item.split())
        return [x for x in parts if x]
    return list(tasks)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def main():
    args = parse_args()

    # Both must be set before mymodels.modeling_llama_pl is imported.
    os.environ["PRIVATE_LLM_LORA_RANK"] = str(args.lora_rank)
    if args.base_model:
        os.environ["PRIVATE_LLM_BASE_MODEL"] = args.base_model

    import torch
    from transformers.generation import GenerationConfig, GenerationMixin

    from infras.eval import simple_evaluate
    from infras.model_utils import get_base_model_and_tokenizer
    from lm_eval.models.huggingface import LoadedHFLM

    tasks = split_tasks(args.tasks)

    if not torch.cuda.is_available() or args.device == "cpu":
        device = torch.device("cpu")
    else:
        local_rank = args.local_rank
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    print(f"[eval_pl] device={device} checkpoint={args.checkpoint_path} tasks={tasks} "
          f"lora_rank={args.lora_rank}")

    print("[eval_pl] loading base model/tokenizer")
    model, tokenizer = get_base_model_and_tokenizer(args.model_name)

    print("[eval_pl] loading PrivateLoRA checkpoint")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    zeroed_lm_head = False
    if not any("lora_lm_head" in k for k in checkpoint):
        for name in ("lora_lm_head_A", "lora_lm_head_B"):
            module = getattr(model, name, None)
            if module is not None and hasattr(module, "weight"):
                with torch.no_grad():
                    module.weight.zero_()
                zeroed_lm_head = True

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(
        "[eval_pl] checkpoint keys: "
        f"{len(checkpoint)} total, "
        f"{sum('lora_mobile' in k for k in checkpoint)} lora_mobile, "
        f"{sum('quantizer' in k for k in checkpoint)} quantizer"
    )
    print(f"[eval_pl] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected, "
          f"zeroed_missing_lm_head_lora={zeroed_lm_head}")

    if not hasattr(model, "generate") and not isinstance(model, GenerationMixin):
        cls = model.__class__
        model.__class__ = type(f"{cls.__name__}WithGenerationMixin", (cls, GenerationMixin), {})
        print("[eval_pl] patched model with GenerationMixin for greedy generation tasks")
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    model.eval()
    model.to(device)

    lm = LoadedHFLM(
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
    )

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    if results is not None:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "model_name": args.model_name,
            "lora_rank": args.lora_rank,
            "tasks": tasks,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "results": results,
        }
        with output_path.open("w") as f:
            json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        print(f"[eval_pl] wrote {output_path}")
        print(json.dumps(jsonable(results.get("results", results)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
