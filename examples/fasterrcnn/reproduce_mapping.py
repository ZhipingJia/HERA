#!/usr/bin/env python3
"""Report Faster R-CNN HERA-A / HERA-P assignments from a sanitized layer profile.

Self-contained CPU demo: reads a per-layer profile (KLD + ACIM/DCNM EDP) and the
target-layer list (``configs/target_layers_v0.json``), then uses the shared
``hera.affinity`` framework to synthesize the affinity ranking and print the
HERA-A / HERA-P tier assignments. No detector checkpoint or dataset is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hera.affinity.core import (  # noqa: E402
    LayerProfile,
    build_scheme_family,
    compute_affinity_records,
    select_objective_scheme,
)

DEFAULT_TARGET_LAYERS = SCRIPT_DIR / "configs" / "target_layers_v0.json"


def load_layer_profiles(path: Path) -> list[LayerProfile]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        LayerProfile(
            name=str(row["name"]),
            short_name=str(row.get("short_name", row["name"])),
            kld=float(row["kld"]),
            edp_acim=float(row["edp_acim"]),
            edp_dcnm=float(row["edp_dcnm"]),
        )
        for row in rows
    ]


def target_layer_names(path: Path) -> list[str]:
    return [layer["full_name"] for layer in json.loads(path.read_text(encoding="utf-8"))["layers"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Faster R-CNN HERA-A/HERA-P assignments.")
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--target-layers", type=Path, default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--hera-a-dcnm-retain", type=int, default=2,
                        help="Lowest-affinity qualified layers kept on DCNM for HERA-A (default: 2).")
    parser.add_argument("--hera-p-dcnm-retain", type=int, default=0,
                        help="Lowest-affinity qualified layers kept on DCNM for HERA-P (default: 0).")
    args = parser.parse_args()

    profiles = load_layer_profiles(args.profiles_json)
    layer_names = target_layer_names(args.target_layers)
    rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
    schemes = build_scheme_family(rank, all_layer_names=layer_names)
    report = {
        "HERA-A": select_objective_scheme(schemes, "HERA-A", dcnm_retain=args.hera_a_dcnm_retain),
        "HERA-P": select_objective_scheme(schemes, "HERA-P", dcnm_retain=args.hera_p_dcnm_retain),
    }
    for objective, scheme in report.items():
        print(f"{objective} ({len(scheme.acim_layers)} ACIM / {len(scheme.dcnm_layers)} DCNM):")
        print("  ACIM:", ", ".join(scheme.acim_layers) or "(none)")
        print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
