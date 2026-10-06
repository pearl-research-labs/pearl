"""Winner -> ``Fp16PlainProof`` -> gateway submission glue for the FP16 (v5) scheme.

The non-GPU half of the A100 runtime integration: given a winning tile's full
committed operands (as :class:`miner_base.fp16_block_submission.Fp16OpenedBlock`),
:func:`~miner_base.fp16_block_submission.submit_fp16_block` must assemble the
cert, verify it against the proposed header, and hand it to the client. The GPU
search (``pearl_gemm.fp16_miner.search_block``) that *produces* the tile cannot
run here (no A100 / torch+pearl_mining together), so the winning operands are the
Rust ``cert`` honest fixture -- the same bit-exact source the opener/assembler
test uses -- and the client is a recorder.

Also covers the torch-free scheme classifier (:mod:`miner_base.schemes`), which
drives arch/layer admission, the per-forward launch branch, and this submission
dispatch alike. Skipped unless ``pearl_mining`` is importable (py3.12/3.13 here).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

pearl_mining = pytest.importorskip("pearl_mining")

from miner_base.fp16_block_submission import (  # noqa: E402
    Fp16OpenedBlock,
    submit_fp16_block,
)
from miner_base.layout import AxisPattern, DimType  # noqa: E402
from miner_base.schemes import (  # noqa: E402
    Scheme,
    is_plain_fp16_job,
    is_plain_fp8_job,
    is_submittable_plain_job,
    scheme_of,
)

_M, _N, _K, _R = 8, 128, 256, 32
_NBITS = 0x207FFFFF  # EASY_NBITS
_GEN_SEED = 0xDEADBEEF0BADF00D
_U64 = (1 << 64) - 1

# P_A = [(4, Blake)] -> h=4 ; P_B = [(4, Blake), (16, Fold)] -> w=64 (the
# committed A100 patterns; contiguous tile offsets).
_P_A = AxisPattern(((4, DimType.BLAKE),))
_P_B = AxisPattern(((4, DimType.BLAKE), (16, DimType.FOLD)))


class _Gen:
    """Byte-identical to the Rust ``Gen`` xorshift64 operand generator."""

    def __init__(self, seed: int) -> None:
        self.s = seed & _U64

    def _next(self) -> int:
        self.s ^= (self.s << 13) & _U64
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & _U64
        return self.s

    def operand(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float16)
        for i in range(n):
            r = self._next()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = ((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(np.float32(sign * mant * (2.0**exp)))
        return out.view(np.uint16)


def _honest_operands() -> tuple[np.ndarray, np.ndarray]:
    g = _Gen(_GEN_SEED)
    return g.operand(_M * _K), g.operand(_N * _K)


def _test_header() -> bytes:
    return (
        struct.pack("<I", 0)
        + bytes([1] * 32)
        + bytes([2] * 32)
        + struct.pack("<II", 0x66666666, _NBITS)
    )


class _Job:
    """Minimal ``MiningJob`` stand-in (torch-free): only the two attributes the
    submission path reads. The real ``pearl_gateway.comm.dataclasses.MiningJob``
    imports torch, which cannot coexist with ``pearl_mining`` in one interpreter
    here."""

    def __init__(self, cert_version: int, header: bytes) -> None:
        self.cert_version = cert_version
        self.incomplete_header_bytes = header


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def submit_plain_proof(self, plain_proof, mining_job) -> None:
        self.calls.append((plain_proof, mining_job))


# ---- scheme classifier -----------------------------------------------------


def test_scheme_classifier_maps_cert_versions() -> None:
    hdr = _test_header()
    assert scheme_of(_Job(4, hdr)) is Scheme.FP8
    assert scheme_of(_Job(5, hdr)) is Scheme.FP16
    assert scheme_of(_Job(3, hdr)) is Scheme.OTHER
    assert is_plain_fp8_job(_Job(4, hdr)) and not is_plain_fp8_job(_Job(5, hdr))
    assert is_plain_fp16_job(_Job(5, hdr)) and not is_plain_fp16_job(_Job(4, hdr))
    # Both plaintext schemes are submittable; a ZK version is not.
    assert is_submittable_plain_job(_Job(4, hdr))
    assert is_submittable_plain_job(_Job(5, hdr))
    assert not is_submittable_plain_job(_Job(3, hdr))


def test_classifier_matches_binding_constant() -> None:
    assert scheme_of(_Job(pearl_mining.CERT_VERSION_PLAIN_FP16, _test_header())) is Scheme.FP16
    assert scheme_of(_Job(pearl_mining.CERT_VERSION_PLAIN_FP8, _test_header())) is Scheme.FP8


# ---- winner -> submit glue --------------------------------------------------


def test_submit_fp16_block_builds_verifies_and_submits() -> None:
    a, b = _honest_operands()
    header = _test_header()
    opened = Fp16OpenedBlock(
        a=a, b=b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
        tile_row=0, tile_col=0,
    )
    client = _RecordingClient()
    proof = submit_fp16_block(opened, _Job(5, header), client)

    assert proof is not None
    assert isinstance(proof, pearl_mining.Fp16PlainProof)
    # The winner was handed to the client exactly once, with the same proof/job.
    assert len(client.calls) == 1
    submitted_proof, submitted_job = client.calls[0]
    assert submitted_proof is proof
    assert submitted_job.cert_version == 5
    # The submitted proof opens the origin tile and verifies against the header.
    assert list(proof.values_a.row_indices) == [0, 1, 2, 3]
    assert list(proof.values_b.row_indices) == list(range(64))
    ok, msg = pearl_mining.verify_fp16_plain_proof(
        pearl_mining.IncompleteBlockHeader.from_bytes(header), proof, None
    )
    assert ok, msg


def test_submit_fp16_block_non_origin_tile() -> None:
    a, b = _honest_operands()
    header = _test_header()
    opened = Fp16OpenedBlock(
        a=a, b=b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
        tile_row=1, tile_col=1,
    )
    client = _RecordingClient()
    proof = submit_fp16_block(opened, _Job(5, header), client)
    assert proof is not None
    assert list(proof.values_a.row_indices) == [4, 5, 6, 7]
    assert list(proof.values_b.row_indices) == list(range(64, 128))
    assert len(client.calls) == 1


def test_submit_fp16_block_rejects_non_v5_job() -> None:
    a, b = _honest_operands()
    opened = Fp16OpenedBlock(
        a=a, b=b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
    )
    client = _RecordingClient()
    with pytest.raises(ValueError, match="certificate version"):
        submit_fp16_block(opened, _Job(4, _test_header()), client)
    assert client.calls == []


def test_owned_copy_snapshots_operands_against_buffer_reuse() -> None:
    # The serving thread may overwrite the activation buffer on its next
    # forward; owned_copy must snapshot the committed bytes before handoff.
    a, b = _honest_operands()
    a = a.copy()
    opened = Fp16OpenedBlock(
        a=a, b=b, k=_K, m=_M, n=_N, rows_pattern=_P_A, cols_pattern=_P_B,
    )
    owned = opened.owned_copy()
    frozen = bytes(owned.a)
    a[:] = 0  # simulate in-place buffer reuse after the handoff
    assert bytes(owned.a) == frozen  # unchanged
    # And the snapshot still builds a verifying proof.
    header = _test_header()
    client = _RecordingClient()
    proof = submit_fp16_block(owned, _Job(5, header), client)
    ok, _ = pearl_mining.verify_fp16_plain_proof(
        pearl_mining.IncompleteBlockHeader.from_bytes(header), proof, None
    )
    assert ok
