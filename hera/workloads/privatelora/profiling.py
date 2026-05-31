"""PrivateLoRA profiling helpers for HERA affinity-aware mapping."""

from __future__ import annotations

from hera.affinity.profiling import merge_kld_edp_profiles, profile_edp
from hera.workloads.privatelora.layers import privatelora_target_layers


def profile_privatelora_edp_with_external_shapes(
    rank: int,
    hidden_dim: int,
    active_tokens_by_layer: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Profile PrivateLoRA EDP from external rank, hidden size, and token counts."""

    shapes = [
        spec.matrix_shape(
            rank=rank,
            hidden_dim=hidden_dim,
            active_tokens=active_tokens_by_layer.get(spec.full_name, 1.0),
        )
        for spec in privatelora_target_layers()
    ]
    return profile_edp(shapes)


def build_privatelora_layer_profiles(
    kld_by_layer: dict[str, float],
    rank: int,
    hidden_dim: int,
    active_tokens_by_layer: dict[str, float],
):
    """Merge PrivateLoRA KLD and EDP profiles into LayerProfile records."""

    specs = privatelora_target_layers()
    edp = profile_privatelora_edp_with_external_shapes(rank, hidden_dim, active_tokens_by_layer)
    short_names = {spec.full_name: spec.short_name for spec in specs}
    return merge_kld_edp_profiles(kld_by_layer, edp, short_names=short_names)

