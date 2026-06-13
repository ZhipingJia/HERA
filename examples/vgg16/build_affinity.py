#!/usr/bin/env python3
"""Synthesize VGG16 per-layer affinity from the KLD and EDP profiles.

Merges ``profile_kld.py`` (accuracy sensitivity) and ``profile_edp.py`` (physical
benefit) into the affinity score ``Affinity(l) = ΔEDP^l / (KLD^l + ε)`` with
``ε = 0.1 · median(KLD)`` (Methods, Step 3), and writes a descending affinity
ranking. The synthesis itself is the shared ``hera.affinity`` framework.
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

from hera.affinity.core import LayerProfile, compute_affinity_records  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def load_profiles(edp_json: Path, kld_json: Path) -> list[LayerProfile]:
    edp_rows = {r["full_name"]: r for r in json.load(open(edp_json))["layers"]}
    kld_payload = json.load(open(kld_json))
    kld_by_layer = kld_payload.get("kl_by_layer") or {
        r["layer_name"]: r["kl_divergence"] for r in kld_payload["layers"]
    }
    profiles = []
    for name, edp in edp_rows.items():
        if name not in kld_by_layer:
            continue
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VGG16 affinity ranking from KLD + EDP profiles.")
    parser.add_argument("--edp-json", type=Path, default=DEFAULT_OUTPUT_DIR / "vgg16_edp_profile_cifar100.json")
    parser.add_argument("--kld-json", type=Path, default=DEFAULT_OUTPUT_DIR / "vgg16_kld_cifar100.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="vgg16_affinity_rank_cifar100.json")
    parser.add_argument("--epsilon-ratio", type=float, default=0.1)
    parser.add_argument(
        "--include-negative-edp",
        action="store_true",
        help="Keep layers whose ACIM EDP exceeds DCNM (default: exclude, as in the paper).",
    )
    args = parser.parse_args()

    profiles = load_profiles(args.edp_json, args.kld_json)
    records = compute_affinity_records(
        profiles,
        epsilon_ratio=args.epsilon_ratio,
        require_positive_edp_diff=not args.include_negative_edp,
    )

    payload = {
        "epsilon_ratio": args.epsilon_ratio,
        "num_layers_total": len(profiles),
        "num_layers_ranked": len(records),
        "affinity_rank": [
            {
                "name": r.name,
                "short_name": r.short_name,
                "kld": r.kld,
                "edp_acim": r.edp_acim,
                "edp_dcnm": r.edp_dcnm,
                "edp_diff": r.edp_diff,
                "affinity": r.affinity,
                "preferred_tier": r.preferred_tier,
            }
            for r in records
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {output_path}")
    print("Affinity rank (descending):")
    for r in records:
        print(f"  {r.short_name:>6s} {r.name:>14s}: affinity={r.affinity:.6e} kld={r.kld:.5f} edp_diff={r.edp_diff:.6e}")


if __name__ == "__main__":
    main()
