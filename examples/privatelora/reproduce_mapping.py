#!/usr/bin/env python3
"""CLI skeleton for PrivateLoRA HERA-A/HERA-P mapping reports.

The input profile must be a sanitized BoolQ or GSM8K JSON list.  Head LB is
always assigned to DCNM by the reproduction convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hera.workloads.privatelora.reproduce_mapping import (
    build_privatelora_mapping_report,
    load_layer_profiles,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report PrivateLoRA HERA-A/HERA-P assignments.")
    parser.add_argument("--task", choices=("boolq", "gsm8k"), required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--hera-a-threshold", type=float, default=None)
    parser.add_argument("--hera-p-threshold", type=float, default=None)
    args = parser.parse_args()

    profiles = load_layer_profiles(args.profiles_json)
    report = build_privatelora_mapping_report(
        profiles,
        hera_a_threshold=args.hera_a_threshold,
        hera_p_threshold=args.hera_p_threshold,
    )
    print(f"Task: {args.task}")
    for objective, scheme in report.items():
        print(f"{objective}: {scheme.name}")
        print("  ACIM:", ", ".join(scheme.acim_layers) or "(none)")
        print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

