#!/usr/bin/env python3
"""Smoke tests for the HERA reviewer-facing skeleton.

These checks exercise the analytical hardware model, the KLD divergence, and the
end-to-end affinity-mapping pipeline on the shipped synthetic profiles.  They
assert that the demo reproduces the qualitative HERA-A / HERA-P splits described
in the manuscript.  Run with ``python tests/test_smoke.py`` or ``pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "examples" / "data"

from hera.affinity.profiling import compute_kld_from_logits
from hera.hardware import ACIMParameters, DCNMParameters, LayerShape, compare_acim_dcnm
from hera.workloads.fasterrcnn.reproduce_mapping import (
    build_faster_rcnn_mapping_report,
    load_layer_profiles as load_frcnn,
)
from hera.workloads.privatelora.reproduce_mapping import (
    build_privatelora_mapping_report,
    load_layer_profiles as load_plora,
)
from hera.workloads.vgg16.reproduce_mapping import (
    build_vgg16_mapping_report,
    load_layer_profiles as load_vgg,
)


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


def test_faster_rcnn_split_matches_manuscript() -> None:
    """HERA-A keeps the two ROI-head FC layers on DCNM; HERA-P moves them to ACIM."""

    report = build_faster_rcnn_mapping_report(load_frcnn(DATA / "fasterrcnn_synthetic_profile.json"))
    assert len(report["HERA-A"].acim_layers) == 7
    assert len(report["HERA-P"].acim_layers) == 9
    assert "head.classifier.0" in report["HERA-A"].dcnm_layers
    assert "head.classifier.0" in report["HERA-P"].acim_layers
    # The first conv and RPN/ROI output branches stay on DCNM in both objectives.
    for layer in ("extractor.0.0", "rpn.score", "rpn.loc", "head.score", "head.cls_loc"):
        assert layer in report["HERA-A"].dcnm_layers
        assert layer in report["HERA-P"].dcnm_layers


def test_privatelora_splits_match_manuscript() -> None:
    """BoolQ -> 80/88 PLM on ACIM; GSM8K -> 76/84; Head LB always on DCNM."""

    boolq = build_privatelora_mapping_report(load_plora(DATA / "privatelora_boolq_synthetic_profile.json"))
    assert len(boolq["HERA-A"].acim_layers) == 80
    assert len(boolq["HERA-P"].acim_layers) == 88

    gsm8k = build_privatelora_mapping_report(
        load_plora(DATA / "privatelora_gsm8k_synthetic_profile.json"),
        hera_a_dcnm_retain=20,
        hera_p_dcnm_retain=12,
    )
    assert len(gsm8k["HERA-A"].acim_layers) == 76
    assert len(gsm8k["HERA-P"].acim_layers) == 84

    for report in (boolq, gsm8k):
        assert "lora_lm_head_B" in report["HERA-A"].dcnm_layers
        assert "lora_lm_head_B" in report["HERA-P"].dcnm_layers


def test_vgg16_keeps_first_conv_on_dcnm() -> None:
    """The first convolution has negative ACIM EDP benefit and stays on DCNM."""

    report = build_vgg16_mapping_report(load_vgg(DATA / "vgg16_synthetic_profile.json"))
    assert "features.0" in report["HERA-A"].dcnm_layers
    assert "features.0" in report["HERA-P"].dcnm_layers


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
