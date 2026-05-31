"""Affinity-aware mapping framework used by HERA.

This module implements the Methods pipeline: KLD profiling, EDP profiling,
affinity score synthesis, progressive scheme family construction, and
objective-based selection of HERA-A and HERA-P configurations.  It contains no
paper result constants and reads no real power/latency traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


TIER_ACIM = "ACIM"
TIER_DCNM = "DCNM"


@dataclass(frozen=True)
class LayerProfile:
    """Per-layer KLD and EDP profile."""

    name: str
    short_name: str
    kld: float
    edp_acim: float
    edp_dcnm: float
    metadata: dict[str, str | int | float | bool] | None = None

    @property
    def edp_diff(self) -> float:
        """Return ``EDP_DCNM - EDP_ACIM``."""

        return self.edp_dcnm - self.edp_acim


@dataclass(frozen=True)
class AffinityRecord:
    """Layer affinity record after combining KLD and EDP profiles."""

    name: str
    short_name: str
    kld: float
    edp_acim: float
    edp_dcnm: float
    edp_diff: float
    affinity: float
    preferred_tier: str


@dataclass(frozen=True)
class MappingScheme:
    """Assignment of workload layers to ACIM or DCNM."""

    name: str
    acim_layers: tuple[str, ...]
    dcnm_layers: tuple[str, ...]

    def tier_for(self, layer_name: str) -> str:
        """Return the assigned HERA tier for a layer."""

        if layer_name in self.acim_layers:
            return TIER_ACIM
        if layer_name in self.dcnm_layers:
            return TIER_DCNM
        raise KeyError(layer_name)


def median_epsilon(kld_values: list[float], epsilon_ratio: float = 0.1) -> float:
    """Return ``epsilon_ratio * median(KLD)`` for affinity denominator smoothing."""

    if not kld_values:
        return 0.0
    return float(epsilon_ratio) * float(median(kld_values))


def compute_affinity_records(
    layer_profiles: list[LayerProfile],
    epsilon_ratio: float = 0.1,
    require_positive_edp_diff: bool = False,
) -> list[AffinityRecord]:
    """Combine KLD and EDP profiles into ranked affinity records.

    Args:
        layer_profiles: Per-layer KLD and EDP values.
        epsilon_ratio: Coefficient in ``epsilon = epsilon_ratio * median(KLD)``.
        require_positive_edp_diff: When true, omit layers where ACIM does not
            reduce EDP under the analytical model.

    Returns:
        Records sorted by descending affinity.
    """

    epsilon = median_epsilon([profile.kld for profile in layer_profiles], epsilon_ratio)
    records: list[AffinityRecord] = []
    for profile in layer_profiles:
        if require_positive_edp_diff and profile.edp_diff <= 0.0:
            continue
        denominator = profile.kld + epsilon
        affinity = float("inf") if denominator == 0.0 else profile.edp_diff / denominator
        records.append(
            AffinityRecord(
                name=profile.name,
                short_name=profile.short_name,
                kld=profile.kld,
                edp_acim=profile.edp_acim,
                edp_dcnm=profile.edp_dcnm,
                edp_diff=profile.edp_diff,
                affinity=affinity,
                preferred_tier=TIER_ACIM if profile.edp_diff > 0.0 else TIER_DCNM,
            )
        )
    return sorted(records, key=lambda row: row.affinity, reverse=True)


def build_scheme_family(
    affinity_rank: list[AffinityRecord],
    all_layer_names: list[str] | None = None,
    force_dcnm_layers: set[str] | None = None,
    name_prefix: str = "scheme",
) -> list[MappingScheme]:
    """Build N+1 progressive top-k mapping schemes.

    Scheme zero maps every layer to DCNM.  Scheme k maps the top-k ranked layers
    to ACIM, except layers listed in ``force_dcnm_layers``.
    """

    force_dcnm_layers = force_dcnm_layers or set()
    ordered_names = [record.name for record in affinity_rank]
    layer_names = list(all_layer_names or ordered_names)
    schemes: list[MappingScheme] = []

    for k in range(len(ordered_names) + 1):
        acim = tuple(name for name in ordered_names[:k] if name not in force_dcnm_layers)
        dcnm = tuple(name for name in layer_names if name not in acim)
        schemes.append(MappingScheme(name=f"{name_prefix}{k}", acim_layers=acim, dcnm_layers=dcnm))
    return schemes


def select_objective_scheme(
    schemes: list[MappingScheme],
    objective: str,
    threshold: float | None = None,
) -> MappingScheme:
    """Select HERA-A or HERA-P from a ranked scheme family.

    The threshold is intentionally abstract in this skeleton.  In the manuscript
    experiments it is supplied by workload-specific validation criteria rather
    than hard-coded here.
    """

    if not schemes:
        raise ValueError("at least one mapping scheme is required")
    objective_normalized = objective.upper()
    if objective_normalized not in {"HERA-A", "HERA-P"}:
        raise ValueError("objective must be 'HERA-A' or 'HERA-P'")
    if threshold is None:
        return schemes[0] if objective_normalized == "HERA-A" else schemes[-1]

    index = max(0, min(len(schemes) - 1, int(round(threshold * (len(schemes) - 1)))))
    return schemes[index]

