#!/usr/bin/env python3
"""CLI skeleton for VGG16 HERA-A/HERA-P mapping reports.

The input profile must be a sanitized JSON list with fields: name, short_name,
kld, edp_acim, and edp_dcnm.  No paper result JSON is included in this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hera.workloads.vgg16.reproduce_mapping import (
    build_vgg16_mapping_report,
    load_layer_profiles,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report VGG16 HERA-A/HERA-P assignments.")
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument(
        "--hera-a-dcnm-retain",
        type=int,
        default=1,
        help="Lowest-affinity qualified layers kept on DCNM for HERA-A (default: 1).",
    )
    parser.add_argument(
        "--hera-p-dcnm-retain",
        type=int,
        default=0,
        help="Lowest-affinity qualified layers kept on DCNM for HERA-P (default: 0).",
    )
    args = parser.parse_args()

    profiles = load_layer_profiles(args.profiles_json)
    report = build_vgg16_mapping_report(
        profiles,
        hera_a_dcnm_retain=args.hera_a_dcnm_retain,
        hera_p_dcnm_retain=args.hera_p_dcnm_retain,
    )
    for objective, scheme in report.items():
        print(f"{objective} ({len(scheme.acim_layers)} ACIM / {len(scheme.dcnm_layers)} DCNM):")
        print("  ACIM:", ", ".join(scheme.acim_layers) or "(none)")
        print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
