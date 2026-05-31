"""Profiling hooks for HERA affinity-aware mapping.

The functions here define the reviewer-facing interfaces for logits-deviation
KLD profiling and EDP profiling.  Full workload-specific model execution will be
released with the revised manuscript version; this initial skeleton keeps the
contracts explicit without shipping checkpoints, datasets, or HERA-silicon code.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hera.affinity.core import LayerProfile
from hera.hardware import ACIMParameters, DCNMParameters, LayerShape, compare_acim_dcnm


def compute_kld_from_logits(reference_logits: np.ndarray, substituted_logits: np.ndarray) -> float:
    """Compute Kullback-Leibler divergence between two logits tensors."""

    reference = np.asarray(reference_logits, dtype=np.float64)
    substituted = np.asarray(substituted_logits, dtype=np.float64)
    if reference.shape != substituted.shape:
        raise ValueError("reference and substituted logits must have the same shape")

    reference = reference - reference.max(axis=-1, keepdims=True)
    substituted = substituted - substituted.max(axis=-1, keepdims=True)
    p = np.exp(reference)
    q = np.exp(substituted)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))


def profile_kld_single_layer_substitution(*args, **kwargs) -> dict[str, float]:
    """Run single-layer ACIM substitution and return per-layer KLD.

    The expected experiment compares pure-DCNM logits against logits obtained by
    replacing exactly one Conv2d, Linear, or PLM layer with its ACIM sample-noise
    counterpart.
    """

    raise NotImplementedError("To be released in the revised manuscript version")


def profile_edp(
    layers: Iterable[LayerShape],
    acim_params: ACIMParameters | None = None,
    dcnm_params: DCNMParameters | None = None,
) -> dict[str, dict[str, float | str]]:
    """Run analytical ACIM/DCNM EDP profiling for layer shapes."""

    rows: dict[str, dict[str, float | str]] = {}
    for layer in layers:
        comparison = compare_acim_dcnm(layer, acim_params=acim_params, dcnm_params=dcnm_params)
        acim = comparison["acim"]
        dcnm = comparison["dcnm"]
        rows[layer.name] = {
            "edp_acim": acim.edp,
            "edp_dcnm": dcnm.edp,
            "edp_diff": float(comparison["edp_diff"]),
            "preferred_tier": str(comparison["preferred_tier"]),
        }
    return rows


def merge_kld_edp_profiles(
    kld_by_layer: dict[str, float],
    edp_by_layer: dict[str, dict[str, float | str]],
    short_names: dict[str, str] | None = None,
) -> list[LayerProfile]:
    """Merge KLD and EDP dictionaries into LayerProfile objects."""

    short_names = short_names or {}
    profiles: list[LayerProfile] = []
    for name, kld in kld_by_layer.items():
        if name not in edp_by_layer:
            raise KeyError(f"missing EDP profile for layer {name}")
        edp = edp_by_layer[name]
        profiles.append(
            LayerProfile(
                name=name,
                short_name=short_names.get(name, name),
                kld=float(kld),
                edp_acim=float(edp["edp_acim"]),
                edp_dcnm=float(edp["edp_dcnm"]),
            )
        )
    return profiles

