"""Baseline mapping strategies for HERA affinity-aware mapping.

This module covers the AccDrop and GA Search baselines described in the Methods.
It avoids workload-specific numbers and can operate on sanitized profiles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyDropRecord:
    """Accuracy drop from single-layer ACIM substitution."""

    name: str
    short_name: str
    accuracy_drop: float


@dataclass(frozen=True)
class GALayerRecord:
    """Per-layer values used by GA Search."""

    name: str
    edp_acim: float
    edp_dcnm: float
    kld: float


@dataclass(frozen=True)
class GASearchConfig:
    """Configuration for GA Search baseline."""

    alpha: float = 0.5
    population_size: int = 8
    generations: int = 4
    mutation_rate: float = 0.2
    tournament_size: int = 2
    elite_count: int = 1
    seed: int = 1


def rank_by_accdrop(records: list[AccuracyDropRecord]) -> list[AccuracyDropRecord]:
    """Rank layers from smallest to largest accuracy drop."""

    return sorted(records, key=lambda row: (row.accuracy_drop, row.name))


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _repair(indices: set[int], num_layers: int, k: int, rng: random.Random) -> tuple[int, ...]:
    while len(indices) > k:
        indices.remove(rng.choice(sorted(indices)))
    while len(indices) < k:
        choices = [idx for idx in range(num_layers) if idx not in indices]
        indices.add(rng.choice(choices))
    return tuple(sorted(indices))


def _fitness(
    indices: tuple[int, ...],
    layers: list[GALayerRecord],
    config: GASearchConfig,
) -> tuple[float, float, float]:
    selected = set(indices)
    total_edp = sum(
        layer.edp_acim if idx in selected else layer.edp_dcnm
        for idx, layer in enumerate(layers)
    )
    total_kld = sum(layer.kld for idx, layer in enumerate(layers) if idx in selected)

    all_dcnm = sum(layer.edp_dcnm for layer in layers)
    all_acim = sum(layer.edp_acim for layer in layers)
    max_kld = sum(layer.kld for layer in layers)
    edp_norm = _normalize(total_edp, min(all_acim, all_dcnm), max(all_acim, all_dcnm))
    kld_norm = _normalize(total_kld, 0.0, max_kld)
    fitness = config.alpha * edp_norm + (1.0 - config.alpha) * kld_norm
    return fitness, total_edp, total_kld


def run_ga_search(
    layers: list[GALayerRecord],
    cardinality: int,
    config: GASearchConfig | None = None,
) -> dict[str, object]:
    """Search a binary ACIM/DCNM assignment under a cardinality constraint."""

    config = config or GASearchConfig()
    if cardinality < 0 or cardinality > len(layers):
        raise ValueError("cardinality must be within the number of layers")

    rng = random.Random(config.seed + cardinality)
    num_layers = len(layers)
    population = [
        tuple(sorted(rng.sample(range(num_layers), cardinality))) if cardinality else tuple()
        for _ in range(config.population_size)
    ]

    best: tuple[int, ...] | None = None
    best_score = float("inf")
    for _ in range(config.generations):
        scored = sorted((_fitness(indices, layers, config)[0], indices) for indices in set(population))
        if scored[0][0] < best_score:
            best_score, best = scored[0]

        elites = [indices for _, indices in scored[: config.elite_count]]
        next_population = list(elites)
        while len(next_population) < config.population_size:
            parent_pool = scored[: max(config.tournament_size, 1)]
            parent_a = rng.choice(parent_pool)[1]
            parent_b = rng.choice(parent_pool)[1]
            split = rng.randrange(num_layers + 1) if num_layers else 0
            child = set(parent_a[:split]) | set(parent_b[split:])
            repaired = set(_repair(child, num_layers, cardinality, rng))
            if cardinality and rng.random() < config.mutation_rate:
                on_idx = rng.choice(sorted(repaired))
                off_choices = [idx for idx in range(num_layers) if idx not in repaired]
                if off_choices:
                    repaired.remove(on_idx)
                    repaired.add(rng.choice(off_choices))
            next_population.append(tuple(sorted(repaired)))
        population = next_population

    assert best is not None
    _, total_edp, total_kld = _fitness(best, layers, config)
    return {
        "acim_layers": tuple(layers[idx].name for idx in best),
        "dcnm_layers": tuple(layer.name for idx, layer in enumerate(layers) if idx not in set(best)),
        "fitness": best_score,
        "total_edp": total_edp,
        "total_kld": total_kld,
    }
