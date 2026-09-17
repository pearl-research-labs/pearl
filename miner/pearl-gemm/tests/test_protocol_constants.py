"""The kernel package's constants must stay in sync with the reference."""

import miner_base.noise as ref_noise
import miner_base.prequant as ref_prequant
import miner_base.quantization as ref_quant

from pearl_gemm import protocol_constants


def test_noise_constants():
    assert protocol_constants.NOISE_TARGET_NORM == ref_noise.NOISE_TARGET_NORM
    assert protocol_constants._INT_SQRT_PREC == ref_noise._INT_SQRT_PREC


def test_quantization_constants():
    assert protocol_constants.QUANT_MAX == ref_quant.QUANT_MAX
    assert protocol_constants.DELTA == ref_quant.DELTA
    assert protocol_constants.L2_ROUNDED_BITS == ref_prequant.L2_ROUNDED_BITS


def test_pre_quant_constants():
    assert protocol_constants.BLOCK_SCALE_GROUP == ref_prequant.DEFAULT_BLOCK_SIZE
