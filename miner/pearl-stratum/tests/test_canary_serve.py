"""Offline tests for run_canary's --serve mode (the definitive non-stale path).

These cover the serve loop's core contract WITHOUT a GPU, a pool socket, or a
real ssh process:

  1. A HIT line from the (mock) serve binary is mapped back to the right job_id
     by its echoed `header`, the proof is built, verify is run, and on --submit a
     verified HIT IS submitted (to the live job_id). dry-run does NOT submit.
  2. A HIT whose echoed header has no job mapping is dropped (no submit).
  3. An unverifiable proof is never submitted (fail-safe).
  4. on_new_job pushes a well-formed `JOB <header> <target>` line to ssh stdin.

`pearl_mining` is faked in sys.modules and `_build_proof_from_hit` is
monkeypatched so no native stack (torch / pearl_mining / numpy) is needed. The
tests are plain sync functions that drive the coroutines via asyncio.run(), so
no pytest-asyncio plugin is required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run_canary  # noqa: E402


# A real 76-byte header (the captured fixture) and a second, distinct one.
H1 = run_canary.CAPTURED_HEADER_HEX
H2 = "11" + H1[2:]


class _Job:
    def __init__(self, job_id, header_hex, target=0x1234, height=1):
        self.job_id = job_id
        self.header_bytes = bytes.fromhex(header_hex)
        self.target = target
        self.height = height


class _MiningConfig:
    common_dim = 4096
    rank = 256


class _FakeClient:
    """Records submit_share calls; returns a configurable result."""

    def __init__(self, accepted=True, error=None, error_code=None):
        self.submitted = []
        self._accepted = accepted
        self._error = error
        self._error_code = error_code
        self.stopped = False

    async def submit_share(self, job_id, b64, hashrate=0.0):
        self.submitted.append((job_id, b64))
        return types.SimpleNamespace(accepted=self._accepted, latency_ms=1.0,
                                     error=self._error,
                                     error_code=self._error_code)

    async def stop(self):
        self.stopped = True


class _FakeStdin:
    def __init__(self):
        self.buf = b""
        self._closing = False

    def write(self, b):
        self.buf += b

    def is_closing(self):
        return self._closing

    def close(self):
        self._closing = True


def _install_fake_pearl_mining(verify_ok=True, verify_msg="ok"):
    fake = types.ModuleType("pearl_mining")

    class _BH:
        @staticmethod
        def from_bytes(b):
            return _BH()

    class _Proof:
        @staticmethod
        def from_base64(s):
            return _Proof()

    fake.IncompleteBlockHeader = _BH
    fake.PlainProof = _Proof
    fake.verify_plain_proof = lambda bh, proof: (verify_ok, verify_msg)
    sys.modules["pearl_mining"] = fake


# Patch _build_proof_from_hit once (module-global): mapping/submit logic under
# test never depends on the actual proof bytes.
run_canary._build_proof_from_hit = lambda hit, hb, mc: b"PROOFBYTES"  # noqa: E731


def _make_serve(client, submit, verify_ok=True):
    _install_fake_pearl_mining(verify_ok=verify_ok)
    loop = asyncio.get_event_loop()
    return run_canary.ServeLoop(
        mining_config=_MiningConfig(), submit=submit, client=client, loop=loop,
        rig="192.168.70.6", ssh_user="root",
        bin_path=run_canary.RIG_BIN_PATH, dev=0)


def _hit_line(header_hex, nonce=7):
    hit = {"nonce": nonce, "seed": 123, "tile": [1, 2],
           "a_rows": [0] * 8, "b_cols": [0] * 16,
           "transcript": ["00000000"] * 16, "gpu_hash": "00" * 32,
           "header": header_hex}
    return "HIT " + json.dumps(hit)


# ---------------------------------------------------------------------------
# 1) HIT maps to the right job_id; submit on --submit, none on dry-run
# ---------------------------------------------------------------------------


def test_serve_hit_maps_to_job_and_submits():
    async def body():
        client = _FakeClient(accepted=True)
        serve = _make_serve(client, submit=True)
        serve.on_new_job(_Job("job1", H1))
        serve.on_new_job(_Job("job2", H2))
        await serve._handle_hit(_hit_line(H2))
        return serve, client

    serve, client = asyncio.run(body())
    assert serve.accepted is True
    assert len(client.submitted) == 1
    submitted_job_id, _b64 = client.submitted[0]
    assert submitted_job_id == "job2", "HIT must map to the job whose header it echoed"


def test_serve_dry_run_does_not_submit():
    async def body():
        client = _FakeClient(accepted=True)
        serve = _make_serve(client, submit=False)
        serve.on_new_job(_Job("job1", H1))
        await serve._handle_hit(_hit_line(H1))
        return serve, client

    serve, client = asyncio.run(body())
    assert client.submitted == [], "dry-run must not submit"
    assert serve.accepted is False


# ---------------------------------------------------------------------------
# 2) Unknown header is dropped
# ---------------------------------------------------------------------------


def test_serve_unknown_header_dropped():
    async def body():
        client = _FakeClient(accepted=True)
        serve = _make_serve(client, submit=True)
        serve.on_new_job(_Job("job1", H1))
        await serve._handle_hit(_hit_line(H2))  # H2 never registered
        return serve, client

    serve, client = asyncio.run(body())
    assert client.submitted == []
    assert serve.accepted is False


# ---------------------------------------------------------------------------
# 3) Unverifiable proof is never submitted
# ---------------------------------------------------------------------------


def test_serve_verify_fail_never_submits():
    async def body():
        client = _FakeClient(accepted=True)
        serve = _make_serve(client, submit=True, verify_ok=False)
        serve.on_new_job(_Job("job1", H1))
        await serve._handle_hit(_hit_line(H1))
        return serve, client

    serve, client = asyncio.run(body())
    assert client.submitted == []
    assert serve.accepted is False


# ---------------------------------------------------------------------------
# 4) on_new_job pushes a well-formed JOB line to ssh stdin
# ---------------------------------------------------------------------------


def test_serve_on_new_job_writes_job_line():
    async def body():
        client = _FakeClient()
        serve = _make_serve(client, submit=False)
        stdin = _FakeStdin()
        serve._proc = types.SimpleNamespace(stdin=stdin, stdout=None, stderr=None)
        serve.on_new_job(_Job("job1", H1, target=0x1234))
        await asyncio.sleep(0)  # let the fire-and-forget drain task run
        return serve, stdin

    serve, stdin = asyncio.run(body())
    line = stdin.buf.decode()
    assert line.startswith("JOB "), line
    parts = line.strip().split(" ")
    assert len(parts) == 3, parts
    assert parts[1] == H1, "header must be the 76B job header hex"
    assert parts[2] == (0x1234).to_bytes(32, "big").hex()
    assert H1 in serve._jobs and serve._jobs[H1][0] == "job1"


# ---------------------------------------------------------------------------
# 5) The serve JOB-line + remote-argv helpers are well-formed
# ---------------------------------------------------------------------------


def test_serve_job_line_helper():
    hb = bytes.fromhex(H1)
    line = run_canary._serve_job_line(hb, 0xABCD).decode()
    assert line == f"JOB {H1} {(0xABCD).to_bytes(32, 'big').hex()}\n"


def test_serve_remote_argv_has_mode_serve_and_env():
    remote = run_canary._serve_argv_remote(run_canary.RIG_BIN_PATH, k=4096, dev=0)
    assert "mode=serve" in remote
    assert "m=131072" in remote and "n=131072" in remote and "k=4096" in remote
    assert "PEARL_SM89_SWIZZLE=2" in remote
    assert run_canary.RIG_BIN_PATH in remote
