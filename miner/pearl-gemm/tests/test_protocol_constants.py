"""The kernel package's constants must stay in sync with miner-base's prequant.

The noise and quantization constants feed every bit-exact operand, so the
pinned digests in ``test_noise_lines`` / ``test_noisy_quant*`` gate them.
"""

import miner_base.prequant as ref_prequant

from pearl_gemm import protocol_constants


def test_quantization_constants():
    assert protocol_constants.L2_ROUNDED_BITS == ref_prequant.L2_ROUNDED_BITS


def test_pre_quant_constants():
    assert protocol_constants.BLOCK_SCALE_GROUP == ref_prequant.DEFAULT_BLOCK_SIZE
