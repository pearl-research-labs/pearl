"""AsyncLoopManager submission dispatch across the FP8 (v4) and FP16 (v5) schemes.

The shared submission plumbing (capacity, backpressure, one-shot clients) is
scheme-agnostic; ``_build_and_submit_proof`` is the one point that knows FP8 vs.
FP16, routing an ``OpenedBlockInfo`` to ``submit_opened_block`` and an
``Fp16OpenedBlock`` to ``submit_fp16_block`` by the job's cert version, and
refusing a mismatched opening/scheme.

Needs ``torch`` (``async_loop_manager`` -> ``block_submission`` import it), so it
is skipped in the proof-only interpreter and runs in the unified production env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("pearl_mining")

from miner_base import async_loop_manager as alm  # noqa: E402
from miner_base.block_submission import Scheme  # noqa: E402


class _Job:
    def __init__(self, cert_version: int) -> None:
        self.cert_version = cert_version
        self.incomplete_header_bytes = b"\x00" * 76


class _FakeFp8Opening:
    """Stand-in whose only requirement is being an ``OpenedBlockInfo`` instance;
    we monkeypatch the actual submit function, so no real planes are needed."""


class _FakeFp16Opening:
    pass


@pytest.fixture()
def manager():
    from miner_base.gateway_client import MinerRpcConfig
    from miner_base.settings import MinerSettings

    # Construct only (no start()): _build_and_submit_proof is a pure dispatch.
    conf = MinerSettings(no_gateway=True)
    return alm.AsyncLoopManager(MinerRpcConfig(), conf)


def test_v4_routes_to_fp8_submit(manager, monkeypatch):
    calls = {}
    monkeypatch.setattr(alm, "OpenedBlockInfo", _FakeFp8Opening)
    monkeypatch.setattr(alm, "submit_opened_block", lambda o, j, c: calls.setdefault("fp8", (o, j)))
    monkeypatch.setattr(
        alm, "submit_fp16_block", lambda o, j, c: calls.setdefault("fp16", (o, j))
    )
    opening = _FakeFp8Opening()
    manager._build_and_submit_proof(opening, _Job(4), client=object())
    assert "fp8" in calls and "fp16" not in calls


def test_v5_routes_to_fp16_submit(manager, monkeypatch):
    calls = {}
    monkeypatch.setattr(alm, "Fp16OpenedBlock", _FakeFp16Opening)
    monkeypatch.setattr(alm, "submit_opened_block", lambda o, j, c: calls.setdefault("fp8", (o, j)))
    monkeypatch.setattr(
        alm, "submit_fp16_block", lambda o, j, c: calls.setdefault("fp16", (o, j))
    )
    opening = _FakeFp16Opening()
    manager._build_and_submit_proof(opening, _Job(5), client=object())
    assert "fp16" in calls and "fp8" not in calls


def test_mismatched_opening_for_scheme_raises(manager, monkeypatch):
    monkeypatch.setattr(alm, "Fp16OpenedBlock", _FakeFp16Opening)
    # A V5 job with a non-Fp16 opening is a programming error, not a silent drop.
    with pytest.raises(TypeError):
        manager._build_and_submit_proof(object(), _Job(5), client=object())


def test_non_submittable_scheme_raises(manager):
    with pytest.raises(ValueError):
        manager._build_and_submit_proof(object(), _Job(3), client=object())


def test_submittable_gate_accepts_both_schemes():
    from miner_base.block_submission import is_submittable_plain_job

    assert is_submittable_plain_job(_Job(4))
    assert is_submittable_plain_job(_Job(5))
    assert not is_submittable_plain_job(_Job(2))
    assert Scheme.FP16.value == "fp16"
