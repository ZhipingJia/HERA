"""Calibrated HERA hardware model used by the affinity-aware mapping framework.

This is the single source of truth for the analytical ACIM / DCNM hardware model
behind the per-layer EDP profiling (Methods, *Analytical Modeling of the
Heterogeneous Efficiency Envelope*).  Both the lightweight ``hera.hardware``
analytical models and the GPU workloads under ``examples/`` read their hardware
coefficients from here, so the mapping decisions are produced by one consistent,
manuscript-aligned model rather than ad-hoc per-script constants.

Coefficient conventions
------------------------
Structural and timing parameters (array geometry, cycle time, throughput,
utilisation) are given directly.  The energy terms are expressed as **calibrated
energy coefficients** in joules — the per-layer energy each one contributes when
multiplied by the corresponding activity count:

* ``acim_dynamic_energy_per_group_cycle_j`` — dynamic energy of one active ACIM
  PE group per cycle;
* ``acim_static_energy_per_cycle_j`` — static energy accrued per cycle;
* ``dcnm_memory_energy_per_element_j`` — weight-access energy per matrix element;
* ``dcnm_compute_energy_per_op_j`` — compute energy per MAC operation;
* ``dcnm_fixed_overhead_j`` — fixed structural energy overhead per layer.

These coefficients are the modelling inputs needed to reproduce the mapping;
they are not low-level device measurements and the device-level electrical
quantities they aggregate are not recoverable from them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeraHardwareModel:
    """Calibrated coefficients for the ACIM and DCNM analytical tiers."""

    # --- ACIM array geometry and timing ---
    acim_base_rows: int = 576
    acim_base_cols: int = 32
    acim_max_parallel_groups: int = 56
    acim_latency_per_cycle_s: float = 50e-9

    # --- ACIM calibrated energy coefficients (J) ---
    acim_dynamic_energy_per_group_cycle_j: float = 4.662e-9
    acim_static_energy_per_cycle_j: float = 2.5e-9

    # --- DCNM structure and throughput ---
    dcnm_memory_bits_per_element: int = 8
    dcnm_peak_throughput_32bit_ops: float = 64e9
    dcnm_throughput_int8_factor: float = 4.0
    dcnm_utilization: float = 0.80

    # --- DCNM calibrated energy coefficients (J) ---
    dcnm_memory_energy_per_element_j: float = 1.6e-12
    dcnm_compute_energy_per_op_j: float = 0.28e-12
    dcnm_fixed_overhead_j: float = 2.0e-9
    dcnm_calibration_factor: float = 1.0

    @property
    def dcnm_effective_throughput_int8_ops(self) -> float:
        """Effective INT8 throughput (ops/s) after the utilisation factor."""

        return (
            self.dcnm_peak_throughput_32bit_ops
            * self.dcnm_throughput_int8_factor
            * self.dcnm_utilization
        )

    def as_dict(self) -> dict[str, float | int]:
        """Flat dict of the coefficients for embedding in profile JSON outputs."""

        return {
            "acim_base_rows": self.acim_base_rows,
            "acim_base_cols": self.acim_base_cols,
            "acim_max_parallel_groups": self.acim_max_parallel_groups,
            "acim_latency_per_cycle_s": self.acim_latency_per_cycle_s,
            "acim_dynamic_energy_per_group_cycle_j": self.acim_dynamic_energy_per_group_cycle_j,
            "acim_static_energy_per_cycle_j": self.acim_static_energy_per_cycle_j,
            "dcnm_memory_bits_per_element": self.dcnm_memory_bits_per_element,
            "dcnm_peak_throughput_32bit_ops": self.dcnm_peak_throughput_32bit_ops,
            "dcnm_throughput_int8_factor": self.dcnm_throughput_int8_factor,
            "dcnm_utilization": self.dcnm_utilization,
            "dcnm_effective_throughput_int8_ops": self.dcnm_effective_throughput_int8_ops,
            "dcnm_memory_energy_per_element_j": self.dcnm_memory_energy_per_element_j,
            "dcnm_compute_energy_per_op_j": self.dcnm_compute_energy_per_op_j,
            "dcnm_fixed_overhead_j": self.dcnm_fixed_overhead_j,
            "dcnm_calibration_factor": self.dcnm_calibration_factor,
        }


DEFAULT_HERA_HARDWARE = HeraHardwareModel()
"""Module-level default model; mutate via ``dataclasses.replace`` for sweeps."""
