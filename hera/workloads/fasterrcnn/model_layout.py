"""Lightweight Faster R-CNN layout used by the HERA affinity workflow.

The Methods mapping pipeline profiles Conv2d and Linear layers in the extractor,
RPN, and ROI head.  This module records the search space and matrix convention;
it does not include model weights, infrared images, or any HERA-silicon driver.
"""

from __future__ import annotations

from dataclasses import dataclass

from hera.hardware import LayerShape


@dataclass(frozen=True)
class FasterRCNNLayerSpec:
    """Target layer metadata for Faster R-CNN affinity profiling."""

    index: int
    full_name: str
    short_name: str
    stage: str
    module_type: str
    weight_shape: tuple[int, ...]

    def matrix_shape(self, active_rows: float = 1.0) -> LayerShape:
        """Return the R/C matrix convention used by ACIM/DCNM EDP profiling."""

        if self.module_type == "Conv2d":
            out_channels, in_channels, kernel_h, kernel_w = self.weight_shape
            return LayerShape(
                name=self.full_name,
                r_dim=in_channels * kernel_h * kernel_w,
                c_dim=out_channels,
                active_rows=active_rows,
            )
        if self.module_type == "Linear":
            out_features, in_features = self.weight_shape
            return LayerShape(
                name=self.full_name,
                r_dim=in_features,
                c_dim=out_features,
                active_rows=active_rows,
            )
        raise ValueError(f"unsupported module_type: {self.module_type}")


def faster_rcnn_target_layers() -> list[FasterRCNNLayerSpec]:
    """Return the Faster R-CNN Conv2d/Linear affinity target layers."""

    rows = [
        ("extractor.0.0", "E0_0", "Extractor", "Conv2d", (32, 3, 3, 3)),
        ("extractor.0.2", "E0_2", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("extractor.1.1", "E1_1", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("extractor.1.3", "E1_3", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("extractor.2.0", "E2_0", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("extractor.2.2", "E2_2", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("extractor.2.4", "E2_4", "Extractor", "Conv2d", (32, 32, 3, 3)),
        ("rpn.conv1", "RPN_C1", "RPN", "Conv2d", (32, 32, 3, 3)),
        ("rpn.score", "RPN_SCORE", "RPN", "Conv2d", (18, 32, 1, 1)),
        ("rpn.loc", "RPN_LOC", "RPN", "Conv2d", (36, 32, 1, 1)),
        ("head.classifier.0", "ROI_FC1", "ROIHead", "Linear", (32, 288)),
        ("head.classifier.2", "ROI_FC2", "ROIHead", "Linear", (32, 32)),
        ("head.cls_loc", "ROI_LOC", "ROIHead", "Linear", (8, 32)),
        ("head.score", "ROI_SCORE", "ROIHead", "Linear", (2, 32)),
    ]
    return [
        FasterRCNNLayerSpec(index, full_name, short_name, stage, module_type, weight_shape)
        for index, (full_name, short_name, stage, module_type, weight_shape) in enumerate(rows)
    ]


def describe_lightweight_faster_rcnn() -> dict[str, tuple[str, ...]]:
    """Describe the reviewer-facing Faster R-CNN block layout."""

    layers = faster_rcnn_target_layers()
    return {
        "Extractor": tuple(layer.full_name for layer in layers if layer.stage == "Extractor"),
        "RPN": tuple(layer.full_name for layer in layers if layer.stage == "RPN"),
        "ROIHead": tuple(layer.full_name for layer in layers if layer.stage == "ROIHead"),
    }

