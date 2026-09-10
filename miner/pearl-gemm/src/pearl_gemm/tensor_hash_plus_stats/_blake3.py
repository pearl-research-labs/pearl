"""Register-resident BLAKE3 compression primitives."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

# BLAKE3 constants
CHAINING_VALUE_SIZE = 32
CHAINING_VALUE_SIZE_U32 = CHAINING_VALUE_SIZE // 4  # 8
KEY_SIZE = 32
CHUNK_SIZE = 1024
MSG_BLOCK_SIZE = 64
MSG_BLOCK_SIZE_U32 = MSG_BLOCK_SIZE // 4  # 16
# Admissible values for the flags field:
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4

IV0 = 0x6A09E667
IV1 = 0xBB67AE85
IV2 = 0x3C6EF372
IV3 = 0xA54FF53A
IV4 = 0x510E527F
IV5 = 0x9B05688C
IV6 = 0x1F83D9AB
IV7 = 0x5BE0CD19

# Precomputed flags variants of CompressParams (blake3.cuh)
FLAGS_INNER_NODE = KEYED_HASH | PARENT
FLAGS_ROOT = KEYED_HASH | ROOT | PARENT

# The 8 G operations of one round: (a, b, c, d) state indices, fed message
# words (2*g, 2*g + 1), per the BLAKE3 round function.
_G_SCHEDULE = (
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
)

# Fixed message-word permutation applied between rounds.
_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

# Leaf-compression rotate amounts ``mad_rot`` rewrites as one
# ``clmad.hi.u64`` each, off the saturated ALU pipe. The exactness argument
# is at ``_clmad_rotr32``; the size-gated dispatch lives in ``_merkle_host``.
MAD_ROT_AMOUNTS = frozenset((12, 7))
# Inputs below this many bytes dispatch the plain-SHF kernel even when the
# config enables mad_rot: at low occupancy the ~21-cycle CLMAD latency is a
# net loss.
MAD_ROT_MIN_BYTES = 24 * 1024 * 1024


def _rotr32(x, n):
    """rightrotate32: (x << (32 - n)) | (x >> n) on uint32."""
    return (x >> n) | (x << (32 - n))


def _clmad_rotr32(x, n):
    """rotr32(x, n) as one CLMAD.HI on the FP64/CLMUL pipe (idle here).

    Bit-exact: ``clmul(x, 2^(32-n) | 2^(64-n))`` places ``x << (32-n)`` and
    ``x >> n`` in disjoint bit ranges, so the selected 32-bit slice equals
    rotr32 (XOR == OR, no carries). ``x`` rides the operand pair's HIGH
    register and the result is consumed from the product's low word -- same
    instruction count as the CLMAD.LO orientation but a shorter critical
    path. The widening mov and word extraction lower to FMA-pipe IMAD.MOVs.
    """
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm

    mask = (1 << (32 - n)) | (1 << (64 - n))
    body = (
        "mov.u32 z, 0;\n\t"
        "mov.b64 a, {z, $1};\n\t"
        f"mov.b64 m, {mask:#x};\n\t"
        "clmad.hi.u64 y, a, m, 0;\n\t"
        "mov.b64 {$0, z}, y;\n\t"
    )
    res = _llvm.inline_asm(
        _ir.IntegerType.get_signless(32),
        [Uint32(x).ir_value()],
        "{\n\t.reg .b64 a, m, y;\n\t.reg .b32 z;\n\t" + body + "}",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    return Uint32(res)


def _blake3_round(s, m, mad_rot=False):
    """One BLAKE3 round (8 G operations) over state list *s*, message list *m*.

    Trace-time helper mutating the SSA-value list in place. ``mad_rot``
    routes the ``MAD_ROT_AMOUNTS`` rotations through the CLMAD offload (leaf
    compressions only).
    """

    def rot(x, n):
        if mad_rot and n in MAD_ROT_AMOUNTS:
            return _clmad_rotr32(x, n)
        return _rotr32(x, n)

    for g, (a, b, c, d) in enumerate(_G_SCHEDULE):
        s[a] = s[a] + (s[b] + m[2 * g])
        s[d] = rot(s[d] ^ s[a], 16)
        s[c] = s[c] + s[d]
        s[b] = rot(s[b] ^ s[c], 12)
        s[a] = s[a] + (s[b] + m[2 * g + 1])
        s[d] = rot(s[d] ^ s[a], 8)
        s[c] = s[c] + s[d]
        s[b] = rot(s[b] ^ s[c], 7)


def _blake3_permute(m):
    """Message-word permutation between rounds."""
    return [m[i] for i in _PERMUTATION]


@cute.jit
def compress_msg_block(
    rBlock,
    rChainingValue,
    counter,
    flags,
    block_len: cutlass.Constexpr[int] = MSG_BLOCK_SIZE,
    mad_rot: cutlass.Constexpr[bool] = False,
):
    """Compress a 64-byte message block (compress_msg_block_u32).

    ``rBlock`` is a 16-word rmem tensor, ``rChainingValue`` an 8-word rmem
    tensor updated in place. ``counter`` is the 32-bit chunk counter (all
    callers build the 64-bit counter from a u32 chunk index, so the high
    word is 0). ``flags`` may be a dynamic value.

    ``mad_rot``: this is a LEAF compression eligible for the CLMAD rotate
    offload (see ``MAD_ROT_AMOUNTS`` above). Parent/tree compressions must
    leave this False.
    """
    # List comprehensions execute at trace time (plain range, not range_constexpr)
    m = [rBlock[i] for i in range(MSG_BLOCK_SIZE_U32)]
    s = [rChainingValue[i] for i in range(CHAINING_VALUE_SIZE_U32)]
    s += [Uint32(IV0), Uint32(IV1), Uint32(IV2), Uint32(IV3)]
    s += [Uint32(counter), Uint32(0), Uint32(block_len), Uint32(flags)]

    # 6 rounds and permutations
    for _ in cutlass.range_constexpr(6):
        _blake3_round(s, m, mad_rot)
        m = _blake3_permute(m)
    # Final round w/o permutation
    _blake3_round(s, m, mad_rot)
    # Real BLAKE3 has some operations here on state8-15, but we don't care about
    # these so we can only change state0-7.
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rChainingValue[i] = s[i] ^ s[i + 8]
