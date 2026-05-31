"""Report Faster R-CNN HERA-A and HERA-P assignments from sanitized profiles."""

from __future__ import annotations

import json
from pathlib import Path

from hera.affinity import build_scheme_family, compute_affinity_records, select_objective_scheme
from hera.affinity.core import LayerProfile, MappingScheme
from hera.workloads.fasterrcnn.model_layout import faster_rcnn_target_layers


def load_layer_profiles(path: Path) -> list[LayerProfile]:
    """Load external Faster R-CNN affinity profile rows."""

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


def build_faster_rcnn_mapping_report(
    profiles: list[LayerProfile],
    hera_a_threshold: float | None = None,
    hera_p_threshold: float | None = None,
) -> dict[str, MappingScheme]:
    """Return HERA-A and HERA-P assignments for Faster R-CNN."""

    layer_names = [layer.full_name for layer in faster_rcnn_target_layers()]
    rank = compute_affinity_records(profiles, require_positive_edp_diff=True)
    schemes = build_scheme_family(rank, all_layer_names=layer_names)
    return {
        "HERA-A": select_objective_scheme(schemes, "HERA-A", hera_a_threshold),
        "HERA-P": select_objective_scheme(schemes, "HERA-P", hera_p_threshold),
    }

