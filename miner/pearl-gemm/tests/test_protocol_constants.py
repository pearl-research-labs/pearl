"""The kernel package's constants must stay in sync with the reference."""

import miner_base.noise as ref_noise
import miner_base.prequant as ref_prequant
import miner_base.quantization as ref_quant
from miner_base.hardware import Blackwell

from pearl_gemm import protocol_constants


def test_noise_constants():
    assert protocol_constants.NOISE_TARGET_NORM == ref_noise.NOISE_TARGET_NORM
    assert protocol_constants._INT_SQRT_PREC == ref_noise._INT_SQRT_PREC


def test_quantization_constants():
    assert protocol_constants.QUANT_MAX == ref_quant.QUANT_MAX
    # The SM100 kernel's noise fraction is Blackwell's device-dependent delta.
    assert Blackwell().noise_fraction == protocol_constants.DELTA
    assert protocol_constants.L2_ROUNDED_BITS == ref_prequant.L2_ROUNDED_BITS


def test_pre_quant_constants():
    assert protocol_constants.BLOCK_SCALE_GROUP == ref_prequant.DEFAULT_BLOCK_SIZE
