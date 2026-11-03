"""On-GPU end-to-end validation of the FP16 (A100 / ``sm_80``) miner driver.

Drives :func:`pearl_gemm.fp16_miner.search_block` on the local GA100: from a job
header + full plaintext FP16 operands it derives the bit-exact seed chain, commits
both operands, rebuilds ``A'``/``B'``, runs the full-matrix lottery search, and
decodes the latched hit into an opened tile. The decoded witness (opened rows +
sliced noise + ``seed_a``) is then fed to the Rust-anchored numpy oracle
(``tests/helpers/fp16_pipeline_reference.verify_tile``), which must:

  * ACCEPT the tile (policy-admissible),
  * reproduce the search's latched 32-byte jackpot ticket bit-for-bit, and
  * clear the EASY difficulty target.

A HARD (impossible) ``nbits`` must produce no winner. A separate test asserts the
host seed-chain reproduction (:mod:`pearl_gemm.fp16_miner._seed_chain`) is
internally consistent and matches the committed Rust cargo-dump vector
bit-for-bit (keys / p-encodings / seeds / ``pow_key`` / pattern encodings).

Gated to ``sm_80`` hardware; the full suite is CI.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the FP16 miner driver targets sm_80 (GA100) hardware",
)

_BLAKE, _FOLD = 2, 1


class _Gen:
    """Deterministic xorshift FP16 stream with spread magnitudes (the honest,
    policy-admissible regime), reused from the pipeline/search tests."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def _n(self):
        self.s ^= (self.s << 13) & 0xFFFFFFFFFFFFFFFF
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & 0xFFFFFFFFFFFFFFFF
        return self.s

    def operand(self, n):
        out = np.empty(n, np.float16)
        for i in range(n):
            r = self._n()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = int((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(sign * mant * (2.0 ** exp))
        return out


# (h, w, k, rp_dims, cp_dims, m, n): h/w == pattern tile_size, 16 blake lanes,
# whole lottery tiles in m and n.
_SHAPES = [
    (4, 64, 256, [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)], 16, 256),
    (16, 64, 128, [(4, _BLAKE), (4, _FOLD)], [(4, _BLAKE), (16, _FOLD)], 32, 128),
]


@pytest.mark.parametrize(("h", "w", "k", "rp", "cp", "m", "n"), _SHAPES)
def test_search_block_winner_verifies_against_rust_oracle(h, w, k, rp, cp, m, n):
    from pearl_gemm.fp16_miner import (
        Fp16JobParams,
        Fp16OperandParams,
        jackpot_pow_key,
        noise_seeds,
        search_block,
    )
    from pearl_gemm.fp16_pipeline import AxisPattern

    from tests.helpers import fp16_pipeline_reference as ref

    r = 32
    header = bytes((i * 7 + 11) & 0xFF for i in range(76))

    g = _Gen(0xBEEF_1234_5678_9ABC ^ (h * 131 + w * 17 + k + m * 7 + n * 3))
    a = g.operand(m * k).reshape(m, k)
    b = g.operand(n * k).reshape(n, k)
    a_dev = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    b_dev = torch.from_numpy(np.ascontiguousarray(b)).cuda()

    rp_a = AxisPattern.new(rp)
    cp_a = AxisPattern.new(cp)
    op_a = Fp16OperandParams(num_rows=m, pattern=rp_a, chunk_len=1024)
    op_b = Fp16OperandParams(num_rows=n, pattern=cp_a, chunk_len=1024)

    ref_rp = ref.AxisPattern.new(rp)
    ref_cp = ref.AxisPattern.new(cp)

    # EASY nbits: every tile clears difficulty, so the first flat tile latches.
    easy = 0x207F_FFFF
    wt = search_block(header, a_dev, b_dev, Fp16JobParams(k=k, a=op_a, b=op_b, nbits=easy))
    assert wt.found, "EASY nbits must latch a winner"
    assert (wt.tile_row, wt.tile_col) == (0, 0), "EASY first winner must be tile (0, 0)"

    # Feed the decoded witness to the Rust-anchored numpy oracle.
    rv = ref.verify_tile(
        wt.opened_a_rows.cpu().numpy(),
        wt.opened_b_rows.cpu().numpy(),
        wt.e_a.cpu().numpy(),
        wt.f_a.cpu().numpy(),
        wt.e_b.cpu().numpy(),
        wt.f_b.cpu().numpy(),
        ref_rp,
        ref_cp,
        wt.seed_a,
        r,
    )
    assert rv.ticket == wt.ticket, (
        f"oracle ticket {rv.ticket.hex()} != search latched ticket {wt.ticket.hex()}"
    )
    assert rv.report.accept, "oracle must ACCEPT the honest winning tile"
    assert bool(wt.report.accept) == bool(rv.report.accept), "GA100 and oracle policy disagree"
    assert ref.check_jackpot_difficulty(rv.ticket, easy, h, w, k), "winner must clear EASY difficulty"

    # HARD (impossible) nbits: no tile can win.
    wt_hard = search_block(header, a_dev, b_dev, Fp16JobParams(k=k, a=op_a, b=op_b, nbits=0))
    assert not wt_hard.found
    assert (wt_hard.tile_row, wt_hard.tile_col) == (-1, -1)
    assert wt_hard.report is None

    # The host seed chain is internally consistent.
    sc = wt.seeds
    assert sc.pow_key == jackpot_pow_key(sc.seed_a)
    sa, sb = noise_seeds(sc.key_a, sc.key_b, sc.root_a, sc.root_b, sc.p_a, sc.p_b)
    assert (sa, sb) == (sc.seed_a, sc.seed_b)


def test_seed_chain_matches_rust_cargo_dump():
    """The host seed-chain reproduction is bit-for-bit with the ``zk-pow`` cargo
    dump (``api::fp16::plain_proof::seed_chain_dump::dump_seed_chain``).

    The fixed vector below is the committed output of that ``#[ignore]`` test for
    ``(header=new_for_test(0x207fffff), fill(0x1111)/fill(0x2222) operands, M=8
    N=128 K=256 R=32, P_A=[(4,Blake)], P_B=[(4,Blake),(16,Fold)], chunk 1024)``.
    Regenerate with::

        PEARL_FP16_SEEDCHAIN_OUT=/tmp/sc.txt cargo test -p zk-pow --lib -- \\
          api::fp16::plain_proof::seed_chain_dump::dump_seed_chain --ignored --exact
    """
    from pearl_gemm.fp16_miner import (
        commitment_keys,
        encode_p_a,
        encode_p_b,
        encode_pattern,
        jackpot_pow_key,
        noise_seeds,
    )
    from pearl_gemm.fp16_pipeline import AxisPattern

    expected = {
        "key_a": "69085d9fe0824504f23261420abc30fe9265ec0c3ce4381f5064bdad9311fd01",
        "key_b": "bb7c90187928bb63994c0b1a0f1aeb9d75d014cdaac2276255451033c0989c8b",
        "root_a": "3d9c9b3fb3eaeff3ca6bc0169c9ceeedcb35a9910da2fac33bc147bb17f6cdba",
        "root_b": "aae6744237799891831b778b5a673db028878cec732f171b9e3229919b6c3e84",
        "p_a": "00010000200000000008000000030e0303030303",
        "p_b": (
            "000000000101010101010101010101010101010101010101010101010101010101010101"
            "020202020202020202020202020202020202020202020202020202020202020266666666"
            "ffff7f2000010000200000000080000000030e3d03030303"
        ),
        "seed_a": "f32e5853b457b58cbe90bfc387186a05553757e59512bf92b0ce284ce14edbdf",
        "seed_b": "762bd866355529f85aa7ba16d5b973821601ac9099e0c672b377fd29faa44ebd",
        "pow_key": "31a758e15808744125c4747eed7fc20e9245abc77109d31812ab0278f0efc8a3",
        "pattern_a": "0e0303030303",
        "pattern_b": "0e3d03030303",
    }
    header = bytes.fromhex(
        "000000000101010101010101010101010101010101010101010101010101010101010101"
        "020202020202020202020202020202020202020202020202020202020202020266666666ffff7f20"
    )
    ancestor = header  # depth-0 coincidence
    k, r, m, n, hash_id = 256, 32, 8, 128, 3
    rp = AxisPattern.new([(4, 2)])
    cp = AxisPattern.new([(4, 2), (16, 1)])

    ka, kb = commitment_keys(header, ancestor)
    assert ka.hex() == expected["key_a"]
    assert kb.hex() == expected["key_b"]
    assert encode_pattern(rp).hex() == expected["pattern_a"]
    assert encode_pattern(cp).hex() == expected["pattern_b"]
    p_a = encode_p_a(k, r, 0, m, hash_id, rp)
    p_b = encode_p_b(ancestor, k, r, 0, n, hash_id, cp)
    assert p_a.hex() == expected["p_a"]
    assert p_b.hex() == expected["p_b"]
    seed_a, seed_b = noise_seeds(
        ka, kb, bytes.fromhex(expected["root_a"]), bytes.fromhex(expected["root_b"]), p_a, p_b
    )
    assert seed_a.hex() == expected["seed_a"]
    assert seed_b.hex() == expected["seed_b"]
    assert jackpot_pow_key(seed_a).hex() == expected["pow_key"]
