"""VGG16 profiling entry points for HERA affinity-aware mapping."""

from __future__ import annotations

from hera.affinity.profiling import merge_kld_edp_profiles, profile_edp
from hera.workloads.vgg16.layers import vgg16_cifar_target_layers


def profile_vgg16_edp_with_external_active_rows(
    active_rows_by_layer: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Profile VGG16 EDP from externally supplied layer active-row counts."""

    shapes = [
        spec.matrix_shape(active_rows=active_rows_by_layer.get(spec.full_name, 1.0))
        for spec in vgg16_cifar_target_layers()
    ]
    return profile_edp(shapes)


def build_vgg16_layer_profiles(
    kld_by_layer: dict[str, float],
    active_rows_by_layer: dict[str, float],
):
    """Merge VGG16 KLD and EDP profiles into affinity LayerProfile records."""

    specs = vgg16_cifar_target_layers()
    edp = profile_vgg16_edp_with_external_active_rows(active_rows_by_layer)
    short_names = {spec.full_name: spec.short_name for spec in specs}
    return merge_kld_edp_profiles(kld_by_layer, edp, short_names=short_names)


def profile_vgg16_kld_single_layer_substitution(*args, **kwargs):
    """Run VGG16 pure-DCNM vs single-layer ACIM logits-deviation profiling."""

    raise NotImplementedError("To be released in the revised manuscript version")

