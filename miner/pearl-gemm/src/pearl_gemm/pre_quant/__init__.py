"""Block-scaled INT8 pre-quantization with fused commit statistics."""

from ._host import PreQuantConfig, pre_quant, pre_quant_output_shapes

__all__ = [
    "PreQuantConfig",
    "pre_quant",
    "pre_quant_output_shapes",
]
