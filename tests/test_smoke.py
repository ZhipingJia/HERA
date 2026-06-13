#!/usr/bin/env python3
"""Smoke tests for HERA.

Exercise the analytical hardware model, the KLD divergence, and the affinity
framework, and assert that the demos reproduce the manuscript's qualitative
mapping splits — VGG16 (the audit network, real CIFAR-100 profiles) and the
Faster R-CNN application. Run with ``python tests/test_smoke.py`` or ``pytest``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hera.affinity.core import (  # noqa: E402
    LayerProfile,
    build_scheme_family,
    compute_affinity_records,
    select_objective_scheme,
)
from hera.affinity.profiling import compute_kld_from_logits  # noqa: E402
from hera.hardware import ACIMParameters, DCNMParameters, LayerShape, compare_acim_dcnm  # noqa: E402

FRCNN_DATA = REPO_ROOT / "examples" / "data" / "fasterrcnn_synthetic_profile.json"
FRCNN_TARGETS = REPO_ROOT / "examples" / "fasterrcnn" / "configs" / "target_layers_v0.json"
VGG_DATA = REPO_ROOT / "examples" / "vgg16" / "data"


def _load_list_profiles(path: Path) -> list[LayerProfile]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        LayerProfile(name=r["name"], short_name=r.get("short_name", r["name"]),
                     kld=float(r["kld"]), edp_acim=float(r["edp_acim"]), edp_dcnm=float(r["edp_dcnm"]))
        for r in rows
    ]


def test_hardware_model_runs() -> None:
    """The analytical ACIM/DCNM model returns finite, non-negative EDP."""

    layer = LayerShape(name="demo", r_dim=288, c_dim=32, active_rows=64)
    result = compare_acim_dcnm(layer, ACIMParameters(), DCNMParameters())
    assert result["acim"].edp >= 0.0
    assert result["dcnm"].edp >= 0.0
    assert result["preferred_tier"] in {"ACIM", "DCNM"}


def test_kld_is_zero_for_identical_logits() -> None:
    """KLD of a distribution with itself is ~0; with a perturbation it is > 0."""

    reference = np.array([[2.0, 1.0, 0.1, -0.5]])
    assert compute_kld_from_logits(reference, reference) < 1e-9
    perturbed = reference + np.array([[0.0, 0.0, 1.5, 0.0]])
    assert compute_kld_from_logits(reference, perturbed) > 0.0


def test_vgg16_audit_network_split() -> None:
    """VGG16 (real CIFAR-100 profiles): the first conv has negative ACIM EDP
    benefit and is excluded / kept on DCNM, reproducing Fig. 3c."""

    edp = {r["full_name"]: r for r in json.loads((VGG_DATA / "vgg16_edp_profile_cifar100.json").read_text())["layers"]}
    kld = json.loads((VGG_DATA / "vgg16_kld_cifar100.json").read_text())["kl_by_layer"]
    profiles = [
        LayerProfile(name=n, short_name=edp[n]["short_name"], kld=float(kld[n]),
                     edp_acim=float(edp[n]["edp_acim"]), edp_dcnm=float(edp[n]["edp_dcnm"]))
        for n in edp if n in kld
    ]
    rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
    assert "features.0" not in [r.name for r in rank]  # negative ΔEDP, excluded
    schemes = build_scheme_family(rank, all_layer_names=[p.name for p in profiles])
    hera_p = select_objective_scheme(schemes, "HERA-P", dcnm_retain=0)
    assert "features.0" in hera_p.dcnm_layers


def test_faster_rcnn_split_matches_manuscript() -> None:
    """HERA-A keeps the two ROI-head FC layers on DCNM; HERA-P moves them to ACIM."""

    profiles = _load_list_profiles(FRCNN_DATA)
    layer_names = [layer["full_name"] for layer in json.loads(FRCNN_TARGETS.read_text())["layers"]]
    rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
    schemes = build_scheme_family(rank, all_layer_names=layer_names)
    hera_a = select_objective_scheme(schemes, "HERA-A", dcnm_retain=2)
    hera_p = select_objective_scheme(schemes, "HERA-P", dcnm_retain=0)
    assert len(hera_a.acim_layers) == 7
    assert len(hera_p.acim_layers) == 9
    assert "head.classifier.0" in hera_a.dcnm_layers
    assert "head.classifier.0" in hera_p.acim_layers
    for layer in ("extractor.0.0", "rpn.score", "rpn.loc", "head.score", "head.cls_loc"):
        assert layer in hera_a.dcnm_layers
        assert layer in hera_p.dcnm_layers


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
