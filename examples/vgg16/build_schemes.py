#!/usr/bin/env python3
"""Build the three mapping-scheme families for VGG16 and their hardware EDP.

Produces, for the affinity / AccDrop / GA-Search methods, the progressive
scheme family (all-DCNM → all-ACIM, scheme k = k layers on ACIM) and, per scheme,
the normalized sum of selected per-layer EDP — the hardware axis of the
method-comparison plot (Fig. 3d). All three methods reuse ``hera.affinity``.

Per-scheme *accuracy* is not recomputed here (training is out of scope for this
release); reference accuracies are read from the published results JSON.
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

from hera.affinity.baselines import AccuracyDropRecord, GALayerRecord, GASearchConfig, rank_by_accdrop, run_ga_search  # noqa: E402
from hera.affinity.core import AffinityRecord, build_scheme_family  # noqa: E402
from vgg16_pipeline.layers import FULL_TO_SHORT, LAYER_ORDER  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def load_inputs(affinity_json: Path, edp_json: Path, kld_json: Path):
    aff = json.load(open(affinity_json))["affinity_rank"]
    affinity_rank = [
        AffinityRecord(
            name=r["name"], short_name=r["short_name"], kld=r["kld"],
            edp_acim=r["edp_acim"], edp_dcnm=r["edp_dcnm"], edp_diff=r["edp_diff"],
            affinity=r["affinity"], preferred_tier=r["preferred_tier"],
        )
        for r in aff
    ]
    edp = {r["full_name"]: r for r in json.load(open(edp_json))["layers"]}
    kld_payload = json.load(open(kld_json))
    acc_drop = {r["layer_name"]: r["accuracy_drop_from_baseline"] for r in kld_payload["layers"]}
    return affinity_rank, edp, acc_drop


def scheme_hardware(scheme, edp: dict) -> dict:
    """Sum of selected per-layer EDP (ACIM for acim_layers, else DCNM)."""
    total = sum(
        float(edp[name]["edp_acim"]) if name in scheme.acim_layers else float(edp[name]["edp_dcnm"])
        for name in edp
    )
    all_dcnm = sum(float(r["edp_dcnm"]) for r in edp.values())
    return {
        "name": scheme.name,
        "num_acim": len(scheme.acim_layers),
        "acim_layers": list(scheme.acim_layers),
        "selected_edp_sum": total,
        "selected_edp_normalized": total / all_dcnm if all_dcnm > 0 else 0.0,
    }


def family_summary(schemes, edp: dict) -> list[dict]:
    return [scheme_hardware(s, edp) for s in schemes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build affinity/AccDrop/GA scheme families + hardware EDP for VGG16.")
    parser.add_argument("--affinity-json", type=Path, default=DEFAULT_OUTPUT_DIR / "vgg16_affinity_rank_cifar100.json")
    parser.add_argument("--edp-json", type=Path, default=DEFAULT_OUTPUT_DIR / "vgg16_edp_profile_cifar100.json")
    parser.add_argument("--kld-json", type=Path, default=DEFAULT_OUTPUT_DIR / "vgg16_kld_cifar100.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ga-alpha", type=float, default=0.8)
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=12)
    args = parser.parse_args()

    affinity_rank, edp, acc_drop = load_inputs(args.affinity_json, args.edp_json, args.kld_json)
    all_layers = list(LAYER_ORDER)

    # 1) Affinity (HERA) scheme family
    affinity_schemes = build_scheme_family(affinity_rank, all_layer_names=all_layers, name_prefix="scheme")

    # 2) AccDrop scheme family — rank by smallest accuracy drop first
    accdrop_records = [
        AccuracyDropRecord(name=n, short_name=FULL_TO_SHORT.get(n, n), accuracy_drop=acc_drop.get(n, 0.0))
        for n in all_layers
    ]
    accdrop_order = [r.name for r in rank_by_accdrop(accdrop_records)]
    accdrop_pseudo = [
        AffinityRecord(name=n, short_name=FULL_TO_SHORT.get(n, n), kld=0.0,
                       edp_acim=0.0, edp_dcnm=0.0, edp_diff=0.0, affinity=float(-i), preferred_tier="ACIM")
        for i, n in enumerate(accdrop_order)
    ]
    accdrop_schemes = build_scheme_family(accdrop_pseudo, all_layer_names=all_layers, name_prefix="accdrop")

    # 3) GA Search scheme family — one search per cardinality k
    ga_layers = [GALayerRecord(name=n, edp_acim=float(edp[n]["edp_acim"]),
                               edp_dcnm=float(edp[n]["edp_dcnm"]), kld=float(edp[n].get("kld", 0.0)))
                 for n in all_layers]
    ga_config = GASearchConfig(alpha=args.ga_alpha, population_size=args.ga_population, generations=args.ga_generations)
    ga_schemes_raw = [run_ga_search(ga_layers, cardinality=k, config=ga_config) for k in range(len(all_layers) + 1)]

    def ga_hw(entry, k):
        acim = set(entry["acim_layers"])
        total = sum(float(edp[n]["edp_acim"]) if n in acim else float(edp[n]["edp_dcnm"]) for n in all_layers)
        all_dcnm = sum(float(edp[n]["edp_dcnm"]) for n in all_layers)
        return {"name": f"ga{k}", "num_acim": k, "acim_layers": list(entry["acim_layers"]),
                "selected_edp_sum": total, "selected_edp_normalized": total / all_dcnm if all_dcnm > 0 else 0.0}

    payload = {
        "ga_config": {"alpha": args.ga_alpha, "population_size": args.ga_population, "generations": args.ga_generations},
        "affinity": family_summary(affinity_schemes, edp),
        "accdrop": family_summary(accdrop_schemes, edp),
        "ga": [ga_hw(e, k) for k, e in enumerate(ga_schemes_raw)],
        "note": "Per-scheme accuracy is read from the published results JSON, not recomputed here.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vgg16_scheme_families_cifar100.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {output_path}")
    for method in ("affinity", "accdrop", "ga"):
        print(f"{method}: {len(payload[method])} schemes "
              f"(scheme0 all-DCNM → scheme{len(all_layers)} all-ACIM)")


if __name__ == "__main__":
    main()
