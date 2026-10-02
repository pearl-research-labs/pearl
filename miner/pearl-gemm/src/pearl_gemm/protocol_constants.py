"""Protocol constants shared by the kernels."""

# noise lines
NOISE_TARGET_NORM = 256  # constant approximate L2 norm every noise line is renormalized to
_INT_SQRT_PREC = 32  # fixed-point factor carrying log2(32)=5 fractional norm bits through isqrt

# quantization
QUANT_MAX = 448.0  # largest finite e4m3 magnitude (the quant grid ceiling)
# The noise-to-signal ratio (in L2) of the injected E@F noise is committed per
# device, as in the verifier: Hopper's 13-bit QGMMA window needs the
# full-strength noise, Blackwell's 25-bit tcgen05 window half of it. It
# enters the kernels only through the per-row alpha/beta scale constants.
DELTA_SM90 = 1.0
DELTA_SM100 = 0.5
L2_ROUNDED_BITS = 2  # low explicit BF16 mantissa bits cleared from l2

# pre_quant
BLOCK_SCALE_GROUP = 8  # int8 codes sharing one BF16 block scale along k

# the noise rank / raw peel width the kernels are currently specialized for
R = 32
# f8f6f4 atom K. ``pack_noise_factor`` zero-pads F1/E1 to this width (R=16
# needed 16 pad bytes; R=32 fills the atom). Peel matrices stay ``(rows, 2R)``.
FP8_MMA_K = 32
PACKED_NOISE_K = FP8_MMA_K * ((R + FP8_MMA_K - 1) // FP8_MMA_K)
PEEL_COLS = 2 * R

# compute-capability majors of the architecture families the kernels target;
# ``_utils/_arch.py`` is the gate and each host declares the families it runs on
SM90_CC_MAJOR = 9  # Hopper (H100/H200): WGMMA, register accumulators, promoted FP8
SM100_CC_MAJOR = 10  # datacenter Blackwell (B200/B300): tcgen05 UMMA + TMEM
SM120_CC_MAJOR = 12  # workstation/consumer Blackwell (RTX PRO 6000, RTX 50): mma.sync tensor cores

# Hopper FP8 accumulation promotes the chained-WGMMA window into an FP32 total
# every PROMOTE_GROUPS instructions of FP8_MMA_K terms (the verifier's H100 replay):
# ``c`` restarts from +0 per window, ``C <- RNE_FP32(C + c)``. Blackwell has
# no promotion stage (the atom accumulator is the FP32 total).
SM90_PROMOTE_GROUPS = 4
SM90_PROMOTE_K = SM90_PROMOTE_GROUPS * FP8_MMA_K  # 128 terms along K per window


def delta_for_capability(device_capability: tuple[int, int]) -> float:
    """The committed noise fraction of the device the kernels run on.

    SM120 is Blackwell-family arithmetic and commits ``Device.BLACKWELL``, so
    it shares Blackwell's delta.
    """
    major = device_capability[0]
    if major == SM90_CC_MAJOR:
        return DELTA_SM90
    if major in (SM100_CC_MAJOR, SM120_CC_MAJOR):
        return DELTA_SM100
    raise ValueError(f"no committed noise fraction for sm{major}{device_capability[1]}")
