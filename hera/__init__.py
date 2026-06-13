"""HERA reviewer-facing code skeleton.

The package exposes the analytical hardware models (``hera.hardware``) and the
affinity-aware mapping utilities (``hera.affinity``) — the core methodology of the
manuscript. Workload-specific code lives under ``examples/``. It intentionally
excludes checkpoints, datasets, paper result tables, and HERA silicon driver code.
"""

__all__ = ["affinity", "hardware"]

