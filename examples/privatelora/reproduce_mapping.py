#!/usr/bin/env python3
"""CLI skeleton for PrivateLoRA HERA-A/HERA-P mapping reports.

The input profile must be a sanitized BoolQ or GSM8K JSON list.  Head LB is
always assigned to DCNM by the reproduction convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hera.workloads.privatelora.reproduce_mapping import (
    build_privatelora_mapping_report,
    load_layer_profiles,
)


def main() -> int:
    # Per-task DCNM retain counts that reproduce the manuscript PLM splits
    # (Fig. 5c): BoolQ -> HERA-A 80/16, HERA-P 88/8; GSM8K -> HERA-A 76/20,
    # HERA-P 84/12.  Head LB is always on DCNM and is reported separately.
    task_defaults = {"boolq": (16, 8), "gsm8k": (20, 12)}

    parser = argparse.ArgumentParser(description="Report PrivateLoRA HERA-A/HERA-P assignments.")
    parser.add_argument("--task", choices=("boolq", "gsm8k"), required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument(
        "--hera-a-dcnm-retain",
        type=int,
        default=None,
        help="Lowest-affinity PLM layers kept on DCNM for HERA-A (default: per-task).",
    )
    parser.add_argument(
        "--hera-p-dcnm-retain",
        type=int,
        default=None,
        help="Lowest-affinity PLM layers kept on DCNM for HERA-P (default: per-task).",
    )
    parser.add_argument(
        "--list-dcnm",
        action="store_true",
        help="List the DCNM-assigned layer names (otherwise only counts are shown).",
    )
    args = parser.parse_args()

    default_a, default_p = task_defaults[args.task]
    hera_a_retain = args.hera_a_dcnm_retain if args.hera_a_dcnm_retain is not None else default_a
    hera_p_retain = args.hera_p_dcnm_retain if args.hera_p_dcnm_retain is not None else default_p

    profiles = load_layer_profiles(args.profiles_json)
    report = build_privatelora_mapping_report(
        profiles,
        hera_a_dcnm_retain=hera_a_retain,
        hera_p_dcnm_retain=hera_p_retain,
    )
    print(f"Task: {args.task}")
    for objective, scheme in report.items():
        print(f"{objective} ({len(scheme.acim_layers)} ACIM / {len(scheme.dcnm_layers)} DCNM):")
        if args.list_dcnm:
            print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
