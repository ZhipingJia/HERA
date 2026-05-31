"""Analytical models for the HERA ACIM, DCNM, and GPC tiers.

This module corresponds to the Methods section on analytical modeling of the
heterogeneous efficiency envelope.  It provides compact, parameterized formulas
for estimating latency, energy, and EDP for a matrix-vector style layer mapped to
ACIM or DCNM.  The defaults are deliberately illustrative placeholders; paper
calibration constants and measured power/latency data are not distributed in
this initial repository skeleton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerShape:
    """Matrix view of a neural-network layer.

    Attributes:
        name: Stable layer identifier used by the workload adapter.
        r_dim: Matrix input dimension R.  Conv2d uses input channels per group
            times kernel height times kernel width; Linear uses input features.
        c_dim: Matrix output dimension C.  Conv2d uses output channels; Linear
            uses output features.
        active_rows: Number of independent row positions or tokens evaluated by
            this layer for the profiled sample.
    """

    name: str
    r_dim: int
    c_dim: int
    active_rows: float = 1.0

    @property
    def flops(self) -> float:
        """Return multiply-add FLOPs using the Methods convention."""

        return 2.0 * float(self.r_dim) * float(self.c_dim) * float(self.active_rows)


@dataclass(frozen=True)
class HardwareEstimate:
    """Latency, energy, and EDP estimate for a HERA compute tier."""

    tier: str
    latency_s: float
    energy_j: float
    details: dict[str, float | int | str | bool]

    @property
    def edp(self) -> float:
        """Energy-delay product."""

        return self.latency_s * self.energy_j


@dataclass(frozen=True)
class ACIMParameters:
    """Parameters for the ACIM analytical model.

    Use manuscript-calibrated values through an external configuration file when
    reproducing the paper.  The small defaults only make the API importable and
    testable without releasing real HERA power/latency data.
    """

    n_group: int = 2
    n_col: int = 2
    max_parallel_groups: int = 2
    t_cycle_s: float = 1.0
    p_static_w: float = 0.0
    p_dynamic_per_group_w: float = 1.0


@dataclass(frozen=True)
class DCNMParameters:
    """Parameters for the DCNM analytical model."""

    bit_width: int = 1
    e_mem_per_bit_j: float = 1.0
    e_op_j: float = 1.0
    e_overhead_j: float = 0.0
    throughput_eff_ops_s: float = 1.0


@dataclass(frozen=True)
class GPCAnchor:
    """Scalar GPC baseline anchor used for normalization and reporting."""

    relative_latency: float = 1.0
    relative_energy: float = 1.0


def calculate_acim(
    layer: LayerShape,
    params: ACIMParameters | None = None,
    row_parallel: bool = True,
) -> HardwareEstimate:
    """Estimate ACIM latency and energy for a layer.

    Args:
        layer: Layer shape in the paper's R/C matrix convention.
        params: ACIM model parameters.  Defaults are illustrative placeholders.
        row_parallel: Whether independent active rows may be scheduled in
            parallel across available ACIM groups.

    Returns:
        HardwareEstimate with tier set to ``"ACIM"``.
    """

    params = params or ACIMParameters()
    if layer.r_dim <= 0 or layer.c_dim <= 0 or layer.active_rows <= 0:
        raise ValueError("layer dimensions and active_rows must be positive")
    if params.n_col <= 0 or params.max_parallel_groups <= 0:
        raise ValueError("ACIM column and parallel-group counts must be positive")

    groups_per_vector = math.ceil(layer.c_dim / params.n_col)
    vector_parallel = max(1, params.max_parallel_groups // groups_per_vector) if row_parallel else 1
    vector_batches = math.ceil(layer.active_rows / vector_parallel)
    cycles_per_batch = math.ceil(groups_per_vector / params.max_parallel_groups)
    total_cycles = vector_batches * cycles_per_batch

    latency_s = float(total_cycles) * params.t_cycle_s
    active_group_evaluations = float(layer.active_rows) * float(groups_per_vector)
    dynamic_energy_j = active_group_evaluations * params.p_dynamic_per_group_w * params.t_cycle_s
    static_energy_j = params.p_static_w * latency_s
    energy_j = dynamic_energy_j + static_energy_j

    return HardwareEstimate(
        tier="ACIM",
        latency_s=latency_s,
        energy_j=energy_j,
        details={
            "groups_per_vector": groups_per_vector,
            "vector_parallel": vector_parallel,
            "vector_batches": vector_batches,
            "cycles_per_batch": cycles_per_batch,
            "total_cycles": total_cycles,
            "row_parallel": row_parallel,
            "dynamic_energy_j": dynamic_energy_j,
            "static_energy_j": static_energy_j,
        },
    )


def calculate_dcnm(layer: LayerShape, params: DCNMParameters | None = None) -> HardwareEstimate:
    """Estimate DCNM latency and energy for a layer.

    Args:
        layer: Layer shape in the paper's R/C matrix convention.
        params: DCNM model parameters.  Defaults are illustrative placeholders.

    Returns:
        HardwareEstimate with tier set to ``"DCNM"``.
    """

    params = params or DCNMParameters()
    if layer.r_dim <= 0 or layer.c_dim <= 0 or layer.active_rows <= 0:
        raise ValueError("layer dimensions and active_rows must be positive")
    if params.throughput_eff_ops_s <= 0:
        raise ValueError("DCNM effective throughput must be positive")

    weight_elements = float(layer.r_dim) * float(layer.c_dim)
    memory_energy_j = weight_elements * float(params.bit_width) * params.e_mem_per_bit_j
    compute_energy_j = layer.flops * params.e_op_j
    energy_j = memory_energy_j + compute_energy_j + params.e_overhead_j
    latency_s = layer.flops / params.throughput_eff_ops_s

    return HardwareEstimate(
        tier="DCNM",
        latency_s=latency_s,
        energy_j=energy_j,
        details={
            "weight_elements": weight_elements,
            "flops": layer.flops,
            "memory_energy_j": memory_energy_j,
            "compute_energy_j": compute_energy_j,
            "overhead_energy_j": params.e_overhead_j,
        },
    )


def calculate_gpc_anchor(anchor: GPCAnchor | None = None) -> HardwareEstimate:
    """Return the GPC scalar anchor used by the Methods efficiency envelope."""

    anchor = anchor or GPCAnchor()
    return HardwareEstimate(
        tier="GPC",
        latency_s=anchor.relative_latency,
        energy_j=anchor.relative_energy,
        details={"role": "scalar baseline anchor"},
    )


def compare_acim_dcnm(
    layer: LayerShape,
    acim_params: ACIMParameters | None = None,
    dcnm_params: DCNMParameters | None = None,
) -> dict[str, HardwareEstimate | float | str]:
    """Compare ACIM and DCNM EDP for a layer.

    Returns:
        A dictionary containing both tier estimates and ``edp_diff`` defined as
        ``EDP_DCNM - EDP_ACIM``.  Positive values indicate that ACIM reduces EDP.
    """

    acim = calculate_acim(layer, acim_params)
    dcnm = calculate_dcnm(layer, dcnm_params)
    edp_diff = dcnm.edp - acim.edp
    return {
        "acim": acim,
        "dcnm": dcnm,
        "edp_diff": edp_diff,
        "preferred_tier": "ACIM" if edp_diff > 0.0 else "DCNM",
    }

