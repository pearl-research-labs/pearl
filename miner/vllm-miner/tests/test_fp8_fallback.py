from unittest.mock import Mock

import torch
from vllm_miner import fp8_fallback


def test_framework_quantizer_and_native_bias_are_used(monkeypatch):
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    q = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 1, dtype=torch.float32)
    weight = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(1, 4, dtype=torch.float32)
    bias = torch.ones(4, dtype=torch.bfloat16)
    quantize = Mock(return_value=(q, scale))
    scaled_mm = Mock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    monkeypatch.setattr(fp8_fallback, "_activation_quantizer", quantize)
    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)

    fp8_fallback.fp8_fallback_gemm(x, weight, weight_scale, bias, torch.bfloat16)

    assert quantize.call_count == 1
    assert quantize.call_args.args[0] is x
    assert scaled_mm.call_count == 1
    args, kwargs = scaled_mm.call_args
    assert args[0] is q
    assert args[1].data_ptr() == weight.data_ptr()
    assert args[1].shape == (weight.shape[1], weight.shape[0])
    assert args[1].stride() == weight.t().stride()
    assert kwargs["scale_a"] is scale
    assert kwargs["scale_b"] is weight_scale
    assert kwargs["bias"] is bias
    assert kwargs["out_dtype"] is torch.bfloat16
