"""FP16 (A100/v5) per-forward driver: search -> winner -> submission routing.

Exercises ``vllm_miner.fp16_mining.run_fp16_mining_forward`` with the A100
search (``pearl_gemm.fp16_miner.search_block``), the async manager, and the
capture/health gates all stubbed -- the pure runtime glue that turns a decoded
``WinningTile`` into an ``Fp16OpenedBlock`` handed to the manager's bounded
submission executor at the winning contiguous tile.

Needs ``torch`` (the driver imports it); skipped in the proof-only interpreter.
The GPU search and ``pearl_mining`` proof construction are not invoked here --
that chain is covered where it can run (``test_fp16_submission_glue`` in
miner-base and the GA100 kernel tests).
"""

from __future__ import annotations

import struct
import sys
import types
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from vllm_miner import fp16_mining  # noqa: E402
from vllm_miner.fp16_layer import FP16_COLS_PATTERN, FP16_ROWS_PATTERN  # noqa: E402


def _header(nbits: int = 0x207FFFFF) -> bytes:
    return struct.pack("<I", 0) + bytes([1] * 32) + bytes([2] * 32) + struct.pack("<II", 0, nbits)


@dataclass
class _Job:
    incomplete_header_bytes: bytes
    cert_version: int = 5
    target: int = 0


@dataclass
class _FakeState:
    weight: object
    weight_cpu: object
    n: int
    k: int
    mineable: bool = True
    rows_pattern: object = FP16_ROWS_PATTERN
    cols_pattern: object = FP16_COLS_PATTERN
    r: int = 32
    layer_name: str = "fp16.layer"
    disabled: list = None

    def disable_mining(self, reason):  # noqa: D401
        (self.disabled if self.disabled is not None else []).append(reason)
        self.mineable = False


@dataclass
class _WinningTile:
    found: bool
    tile_row: int = 0
    tile_col: int = 0
    report: object = None


class _Report:
    def __init__(self, accept: bool) -> None:
        self.accept = accept


class _FakeManager:
    def __init__(self, job, launch=True, retain=True):
        self._job = job
        self._decision = types.SimpleNamespace(launch=launch, retain_winner=retain)
        self.submitted = []

    def mining_launch_admission(self, job):
        from contextlib import contextmanager

        decision = self._decision

        @contextmanager
        def _cm():
            yield decision

        return _cm()

    def get_mining_job(self):
        return self._job

    def handle_submit_block(self, opening, job, *, timeout=None):
        self.submitted.append((opening, job, timeout))
        return True


@pytest.fixture()
def patched(monkeypatch):
    """Stub the admission/producer/attempt gates and inject a fake
    ``pearl_gemm.fp16_miner`` so the driver runs without a GPU."""
    from contextlib import contextmanager

    @contextmanager
    def _producer():
        yield True

    class _Attempt:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __bool__(self):
            return True

        def mark_success(self, *a):
            pass

    monkeypatch.setattr(fp16_mining, "mining_launches_suspended", lambda: False)
    monkeypatch.setattr(fp16_mining, "gpu_mining_producer", _producer)
    monkeypatch.setattr(fp16_mining, "mining_attempt", lambda device: _Attempt())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    # Fake pearl_gemm.fp16_miner.{search_block,Fp16JobParams,Fp16OperandParams}.
    recorded = {}

    def fake_search_block(header, a, b, params, device):
        recorded["call"] = (bytes(header), a.shape[0], b.shape[0], params)
        return recorded["winner"]

    fake_mod = types.ModuleType("pearl_gemm.fp16_miner")
    fake_mod.search_block = fake_search_block
    fake_mod.Fp16JobParams = lambda **kw: types.SimpleNamespace(**kw)
    fake_mod.Fp16OperandParams = lambda **kw: types.SimpleNamespace(**kw)
    # _search_patterns imports from pearl_gemm.fp16_pipeline; stub its AxisPattern.
    fake_pipe = types.ModuleType("pearl_gemm.fp16_pipeline")
    fake_pipe.AxisPattern = types.SimpleNamespace(new=lambda dims: ("pattern", tuple(dims)))
    monkeypatch.setitem(sys.modules, "pearl_gemm.fp16_miner", fake_mod)
    monkeypatch.setitem(sys.modules, "pearl_gemm.fp16_pipeline", fake_pipe)
    return recorded


def _state(n=128, k=256):
    w = torch.zeros(n, k, dtype=torch.float16)
    return _FakeState(weight=w, weight_cpu=w.clone(), n=n, k=k)


def test_admissible_winner_is_submitted_at_its_tile(patched, monkeypatch):
    from miner_base.fp16_block_submission import Fp16OpenedBlock

    job = _Job(_header())
    mgr = _FakeManager(job)
    monkeypatch.setattr(fp16_mining, "get_async_manager", lambda: mgr)
    patched["winner"] = _WinningTile(found=True, tile_row=1, tile_col=2, report=_Report(True))

    st = _state()
    x2d = torch.zeros(8, st.k, dtype=torch.bfloat16)  # 8 rows -> 2 whole h=4 tiles
    fp16_mining.run_fp16_mining_forward(st, job, x2d)

    assert len(mgr.submitted) == 1
    opening, submitted_job, _timeout = mgr.submitted[0]
    assert isinstance(opening, Fp16OpenedBlock)
    assert (opening.tile_row, opening.tile_col) == (1, 2)
    assert opening.m == 8 and opening.n == st.n and opening.k == st.k
    assert submitted_job is job
    # The search committed exactly the bucketed activation + the full weight.
    _hdr, m_searched, n_searched, _params = patched["call"]
    assert m_searched == 8 and n_searched == st.n


def test_no_winner_is_not_submitted(patched, monkeypatch):
    job = _Job(_header())
    mgr = _FakeManager(job)
    monkeypatch.setattr(fp16_mining, "get_async_manager", lambda: mgr)
    patched["winner"] = _WinningTile(found=False)
    fp16_mining.run_fp16_mining_forward(_state(), job, torch.zeros(8, 256, dtype=torch.bfloat16))
    assert mgr.submitted == []


def test_policy_inadmissible_winner_is_filtered(patched, monkeypatch):
    job = _Job(_header())
    mgr = _FakeManager(job)
    monkeypatch.setattr(fp16_mining, "get_async_manager", lambda: mgr)
    patched["winner"] = _WinningTile(found=True, report=_Report(False))
    fp16_mining.run_fp16_mining_forward(_state(), job, torch.zeros(8, 256, dtype=torch.bfloat16))
    assert mgr.submitted == []


def test_declined_admission_skips_search(patched, monkeypatch):
    job = _Job(_header())
    mgr = _FakeManager(job, retain=False)  # no proof could be submitted
    monkeypatch.setattr(fp16_mining, "get_async_manager", lambda: mgr)
    patched["winner"] = _WinningTile(found=True, report=_Report(True))
    fp16_mining.run_fp16_mining_forward(_state(), job, torch.zeros(8, 256, dtype=torch.bfloat16))
    assert "call" not in patched  # search_block never ran
    assert mgr.submitted == []


def test_sub_tile_activation_is_not_mined(patched, monkeypatch):
    # Fewer than h=4 rows -> no whole tile -> nothing searched.
    job = _Job(_header())
    mgr = _FakeManager(job)
    monkeypatch.setattr(fp16_mining, "get_async_manager", lambda: mgr)
    patched["winner"] = _WinningTile(found=True, report=_Report(True))
    fp16_mining.run_fp16_mining_forward(_state(), job, torch.zeros(3, 256, dtype=torch.bfloat16))
    assert "call" not in patched
    assert mgr.submitted == []


def test_header_nbits_parses_trailing_u32():
    assert fp16_mining._header_nbits(_header(0x1D00FFFF)) == 0x1D00FFFF
