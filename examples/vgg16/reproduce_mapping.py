#!/usr/bin/env python3
"""Reproduce the VGG16 affinity-aware mapping (Fig. 3) from real per-layer profiles.

This is the CPU, seconds-long demo of the affinity framework on its audit network:
it reads the real VGG16-CIFAR100 KLD and EDP profiles (produced by ``profile_kld.py``
and ``profile_edp.py``, shipped under ``data/``), synthesizes the per-layer affinity
ranking, builds the progressive scheme family, and prints the HERA-A / HERA-P tier
assignments. No GPU, model, or dataset is required.

To regenerate the underlying profiles on GPU:
    python profile_edp.py --dataset cifar100
    python profile_kld.py --dataset cifar100
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

from hera.affinity.core import LayerProfile, build_scheme_family, compute_affinity_records, select_objective_scheme  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"


def load_profiles(edp_json: Path, kld_json: Path) -> list[LayerProfile]:
    edp_rows = {r["full_name"]: r for r in json.load(open(edp_json))["layers"]}
    kld_payload = json.load(open(kld_json))
    kld_by_layer = kld_payload.get("kl_by_layer") or {
        r["layer_name"]: r["kl_divergence"] for r in kld_payload["layers"]
    }
    profiles = []
    for name, edp in edp_rows.items():
        if name in kld_by_layer:
            profiles.append(
                LayerProfile(
                    name=name,
                    short_name=edp.get("short_name", name),
                    kld=float(kld_by_layer[name]),
                    edp_acim=float(edp["edp_acim"]),
                    edp_dcnm=float(edp["edp_dcnm"]),
                )
            )
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce VGG16 affinity mapping (Fig. 3) on CPU.")
    parser.add_argument("--edp-json", type=Path, default=DATA_DIR / "vgg16_edp_profile_cifar100.json")
    parser.add_argument("--kld-json", type=Path, default=DATA_DIR / "vgg16_kld_cifar100.json")
    parser.add_argument("--hera-a-dcnm-retain", type=int, default=1,
                        help="Lowest-affinity qualified layers kept on DCNM for HERA-A (default: 1).")
    parser.add_argument("--hera-p-dcnm-retain", type=int, default=0,
                        help="Lowest-affinity qualified layers kept on DCNM for HERA-P (default: 0).")
    args = parser.parse_args()

    profiles = load_profiles(args.edp_json, args.kld_json)
    all_layers = [p.name for p in profiles]
    rank = compute_affinity_records(profiles, epsilon_ratio=0.1, require_positive_edp_diff=True)

    print(f"VGG16 / CIFAR-100 - {len(profiles)} layers, {len(rank)} ACIM-qualified (positive delta-EDP)")
    print("\nPer-layer affinity (Fig. 3c, descending):")
    print(f"  {'layer':>14s}  {'KLD':>10s}  {'delta-EDP':>12s}  {'affinity':>12s}")
    for r in rank:
        print(f"  {r.name:>14s}  {r.kld:10.5f}  {r.edp_diff:12.4e}  {r.affinity:12.4e}")
    excluded = [p.name for p in profiles if p.edp_diff <= 0.0]
    if excluded:
        print(f"  excluded (delta-EDP <= 0, kept on DCNM): {', '.join(excluded)}")

    schemes = build_scheme_family(rank, all_layer_names=all_layers)
    hera_a = select_objective_scheme(schemes, "HERA-A", dcnm_retain=args.hera_a_dcnm_retain)
    hera_p = select_objective_scheme(schemes, "HERA-P", dcnm_retain=args.hera_p_dcnm_retain)

    print("\nMapping configurations (Fig. 3):")
    for objective, scheme in (("HERA-A", hera_a), ("HERA-P", hera_p)):
        print(f"{objective} ({len(scheme.acim_layers)} ACIM / {len(scheme.dcnm_layers)} DCNM):")
        print("  ACIM:", ", ".join(scheme.acim_layers) or "(none)")
        print("  DCNM:", ", ".join(scheme.dcnm_layers) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
