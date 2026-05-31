"""Faster R-CNN profiling helpers for HERA affinity-aware mapping."""

from __future__ import annotations

from hera.affinity.profiling import merge_kld_edp_profiles, profile_edp
from hera.workloads.fasterrcnn.model_layout import faster_rcnn_target_layers


def profile_faster_rcnn_edp_with_external_active_rows(
    active_rows_by_layer: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Profile Faster R-CNN layer EDP from external shape statistics."""

    shapes = [
        spec.matrix_shape(active_rows=active_rows_by_layer.get(spec.full_name, 1.0))
        for spec in faster_rcnn_target_layers()
    ]
    return profile_edp(shapes)


def build_faster_rcnn_layer_profiles(
    kld_by_layer: dict[str, float],
    active_rows_by_layer: dict[str, float],
):
    """Merge Faster R-CNN KLD and EDP profiles into affinity records."""

    specs = faster_rcnn_target_layers()
    edp = profile_faster_rcnn_edp_with_external_active_rows(active_rows_by_layer)
    short_names = {spec.full_name: spec.short_name for spec in specs}
    return merge_kld_edp_profiles(kld_by_layer, edp, short_names=short_names)


def profile_faster_rcnn_kld_single_layer_substitution(*args, **kwargs):
    """Profile detector logits-deviation KLD under single-layer ACIM substitution."""

    raise NotImplementedError("To be released in the revised manuscript version")

