"""PrivateLoRA PLM and Head LB layer conventions.

The PrivateLoRA workload contains edge-side PLM matrices across transformer
blocks and the LM Head LoRA-B layer, called Head LB in the manuscript.  The Head
LB is included in profiling metadata, but the reproduction convention keeps
Head LB assigned to DCNM regardless of HERA-A or HERA-P objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from hera.hardware import LayerShape


@dataclass(frozen=True)
class PLMLayerSpec:
    """PrivateLoRA Mobile or Head LB layer metadata."""

    index: int
    full_name: str
    short_name: str
    block_index: int | None
    projection: str
    module_type: str = "Linear"
    is_head_lb: bool = False

    def matrix_shape(
        self,
        rank: int,
        hidden_dim: int,
        active_tokens: float = 1.0,
    ) -> LayerShape:
        """Return a symbolic R/C matrix shape for PrivateLoRA profiling."""

        if self.is_head_lb:
            return LayerShape(
                name=self.full_name,
                r_dim=rank,
                c_dim=hidden_dim,
                active_rows=active_tokens,
            )
        return LayerShape(
            name=self.full_name,
            r_dim=hidden_dim,
            c_dim=rank,
            active_rows=active_tokens,
        )


def privatelora_target_layers(num_blocks: int = 32) -> list[PLMLayerSpec]:
    """Return the 96 PLM matrices plus Head LB target layer list."""

    layers: list[PLMLayerSpec] = []
    for block in range(num_blocks):
        for projection in ("q", "k", "v"):
            full_name = f"model.layers.{block}.self_attn.{projection}_lora.lora_mobile"
            short_name = f"L{block}_{projection.upper()}_PLM"
            layers.append(
                PLMLayerSpec(
                    index=len(layers),
                    full_name=full_name,
                    short_name=short_name,
                    block_index=block,
                    projection=projection,
                )
            )
    layers.append(head_lb_layer(index=len(layers)))
    return layers


def head_lb_layer(index: int = 96) -> PLMLayerSpec:
    """Return the LM Head LoRA-B layer, named Head LB in the manuscript."""

    return PLMLayerSpec(
        index=index,
        full_name="lora_lm_head_B",
        short_name="Head LB",
        block_index=None,
        projection="head_lb",
        is_head_lb=True,
    )


def head_lb_name() -> str:
    """Return the canonical Head LB layer name."""

    return "lora_lm_head_B"

