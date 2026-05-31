"""Affinity-aware mapping framework and baseline mapping strategies."""

from .baselines import AccuracyDropRecord, GALayerRecord, GASearchConfig, rank_by_accdrop, run_ga_search
from .core import (
    AffinityRecord,
    LayerProfile,
    MappingScheme,
    build_scheme_family,
    compute_affinity_records,
    median_epsilon,
    select_objective_scheme,
)
from .profiling import compute_kld_from_logits, profile_edp, profile_kld_single_layer_substitution

__all__ = [
    "AccuracyDropRecord",
    "AffinityRecord",
    "GALayerRecord",
    "GASearchConfig",
    "LayerProfile",
    "MappingScheme",
    "build_scheme_family",
    "compute_affinity_records",
    "compute_kld_from_logits",
    "median_epsilon",
    "profile_edp",
    "profile_kld_single_layer_substitution",
    "rank_by_accdrop",
    "run_ga_search",
    "select_objective_scheme",
]

