#!/usr/bin/env python3
"""Profile VGG16 layer-wise ACIM / DCNM EDP with the shared HERA hardware model.

Runs a single forward pass over the VGG16 layers, records each Conv2d / Linear
shape, and computes per-layer ACIM and DCNM energy-delay product via
``hera.hardware``. Writes a JSON profile consumed by ``build_affinity.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hardware import get_vgg16_layer_stats_detailed, hardware_constants  # noqa: E402
from vgg16_pipeline.layers import LAYER_ORDER, LAYER_NAMES_SHORT  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def attach_layer_indices(rows: list[dict]) -> list[dict]:
    order_to_index = {name: idx for idx, name in enumerate(LAYER_ORDER)}
    short_by_full = {full: short for full, short in zip(LAYER_ORDER, LAYER_NAMES_SHORT)}
    enriched = []
    for row in rows:
        item = dict(row)
        item["layer_index"] = order_to_index.get(item["full_name"])
        item["short_name"] = short_by_full.get(item["full_name"], item["short_name"])
        enriched.append(item)
    return enriched


def summarize(rows: list[dict]) -> dict:
    return {
        "layer_count": len(rows),
        "total_flops": sum(float(r["flops"]) for r in rows),
        "total_macs": sum(float(r["macs"]) for r in rows),
        "all_dcnm": {
            "energy_j": sum(float(r["dcnm_details"]["energy"]) for r in rows),
            "latency_s": sum(float(r["dcnm_details"]["latency"]) for r in rows),
        },
        "all_acim": {
            "energy_j": sum(float(r["acim_details"]["energy"]) for r in rows),
            "latency_s": sum(float(r["acim_details"]["latency"]) for r in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile VGG16 layer-wise ACIM/DCNM EDP.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=32)
    args = parser.parse_args()

    model_name = f"{args.dataset}_vgg16_bn"
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    rows = attach_layer_indices(
        get_vgg16_layer_stats_detailed(model_name=model_name, input_shape=input_shape, device=args.device)
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model_name": model_name,
        "batch_size": args.batch_size,
        "input_shape": list(input_shape),
        "hardware_constants": hardware_constants(),
        "summary": summarize(rows),
        "layers": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (args.output_name or f"vgg16_edp_profile_{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {output_path}")
    print(f"layers={len(rows)} total_flops={payload['summary']['total_flops']:.3e}")


if __name__ == "__main__":
    main()
