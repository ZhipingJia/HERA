"""Faster R-CNN workload skeleton for HERA affinity-aware mapping."""

from .model_layout import FasterRCNNLayerSpec, faster_rcnn_target_layers

__all__ = ["FasterRCNNLayerSpec", "faster_rcnn_target_layers"]

