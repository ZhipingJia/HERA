#!/usr/bin/env python3
"""CLI skeleton for Faster R-CNN HERA-A/HERA-P mapping reports.

The script consumes sanitized layer profiles and prints tier assignments without
invoking any HERA-silicon driver or loading detector checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hera.workloads.fasterrcnn.model_layout import describe_lightweight_faster_rcnn
from hera.workloads.fasterrcnn.reproduce_mapping import (
    build_faster_rcnn_mapping_report,
    load_layer_profiles,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Faster R-CNN HERA-A/HERA-P assignments.")
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--hera-a-threshold", type=float, default=None)
    parser.add_argument("--hera-p-threshold", type=float, default=None)
    parser.add_argument("--show-layout", action="store_true")
    args = parser.parse_args()

    if args.show_layout:
        print("Lightweight Faster R-CNN layout:")
        for stage, layers in describe_lightweight_faster_rcnn().items():
            print(f"  {stage}: {', '.join(layers)}")

    profiles = load_layer_profiles(args.profiles_json)
    report = build_faster_rcnn_mapping_report(
        profiles,
        hera_a_threshold=args.hera_a_threshold,
        hera_p_threshold=args.hera_p_threshold,
    )
    for objective, scheme in report.items():
        print(f"{objective}: {scheme.name}")
        print("  ACIM:", ", ".join(scheme.acim_layers) or "(none)")
        print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

