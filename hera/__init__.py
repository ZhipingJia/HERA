"""HERA reviewer-facing code skeleton.

The package exposes analytical hardware models, affinity-aware mapping utilities,
and workload adapters for the manuscript's Faster R-CNN and PrivateLoRA studies.
It intentionally excludes checkpoints, datasets, paper result tables, and HERA
silicon driver code.
"""

__all__ = ["affinity", "hardware", "workloads"]

