"""Protocol constants shared by the kernels.

Mirrored from ``miner_base`` (``noise.py`` / ``quantization.py`` /
``prequant.py``) so the kernel package has no runtime dependency on miner-base;
``tests/test_protocol_constants.py`` asserts the two stay in sync.
"""

# noise.py
NOISE_TARGET_NORM = 256  # constant approximate L2 norm every noise line is renormalized to
_INT_SQRT_PREC = 32  # fixed-point factor carrying log2(32)=5 fractional norm bits through isqrt

# quantization.py
QUANT_MAX = 448.0  # largest finite e4m3 magnitude (the quant grid ceiling)
DELTA = 0.5  # noise-to-signal ratio (in L2) of the injected E@F noise
L2_ROUNDED_BITS = 2  # low explicit BF16 mantissa bits cleared from l2

# api.py
BLOCK_SCALE_GROUP = 8  # int8 codes sharing one BF16 block scale along k

# the noise rank / raw peel width the kernels are currently specialized for
R = 32
# f8f6f4 atom K. ``pack_noise_factor`` zero-pads F1/E1 to this width (R=16
# needed 16 pad bytes; R=32 fills the atom). Peel matrices stay ``(rows, 2R)``.
FP8_MMA_K = 32
PACKED_NOISE_K = FP8_MMA_K * ((R + FP8_MMA_K - 1) // FP8_MMA_K)
PEEL_COLS = 2 * R

# the only architecture the kernels are implemented for (SM100 / Blackwell)
SM100_CC_MAJOR = 10
