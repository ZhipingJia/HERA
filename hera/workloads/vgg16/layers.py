"""VGG16 layer conventions for HERA affinity-aware mapping.

The VGG16 study uses the same Methods pipeline as the manuscript workloads:
single-layer ACIM substitution for KLD, ACIM/DCNM EDP profiling, affinity
ranking, and progressive top-k scheme construction.  This module only records
architecture-level layer metadata and does not include checkpoints or measured
hardware results.
"""

from __future__ import annotations

from dataclasses import dataclass

from hera.hardware import LayerShape


@dataclass(frozen=True)
class VGG16LayerSpec:
    """Layer metadata for the VGG16 affinity search space."""

    index: int
    full_name: str
    short_name: str
    module_type: str
    weight_shape: tuple[int, ...]

    def matrix_shape(self, active_rows: float = 1.0) -> LayerShape:
        """Return the paper's R/C matrix convention for this layer."""

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


def vgg16_cifar_target_layers() -> list[VGG16LayerSpec]:
    """Return the Conv2d/Linear layers used by the VGG16 mapping pipeline."""

    conv_specs = [
        ("features.0", "C1_1", (64, 3, 3, 3)),
        ("features.3", "C1_2", (64, 64, 3, 3)),
        ("features.7", "C2_1", (128, 64, 3, 3)),
        ("features.10", "C2_2", (128, 128, 3, 3)),
        ("features.14", "C3_1", (256, 128, 3, 3)),
        ("features.17", "C3_2", (256, 256, 3, 3)),
        ("features.20", "C3_3", (256, 256, 3, 3)),
        ("features.24", "C4_1", (512, 256, 3, 3)),
        ("features.27", "C4_2", (512, 512, 3, 3)),
        ("features.30", "C4_3", (512, 512, 3, 3)),
        ("features.34", "C5_1", (512, 512, 3, 3)),
        ("features.37", "C5_2", (512, 512, 3, 3)),
        ("features.40", "C5_3", (512, 512, 3, 3)),
    ]
    linear_specs = [
        ("classifier.0", "FC1", (512, 512)),
        ("classifier.3", "FC2", (512, 512)),
        ("classifier.6", "FC3", (100, 512)),
    ]
    rows = [
        VGG16LayerSpec(idx, full_name, short_name, "Conv2d", shape)
        for idx, (full_name, short_name, shape) in enumerate(conv_specs)
    ]
    offset = len(rows)
    rows.extend(
        VGG16LayerSpec(offset + idx, full_name, short_name, "Linear", shape)
        for idx, (full_name, short_name, shape) in enumerate(linear_specs)
    )
    return rows

