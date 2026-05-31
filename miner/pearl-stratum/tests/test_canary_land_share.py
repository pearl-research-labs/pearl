"""Offline tests for run_canary's --land-share mode.

These cover the two properties that make land-share distinct from the
preemptive CanaryMineLoop, WITHOUT importing the native mining stack
(pearl_mining / torch / numpy) and WITHOUT opening a socket or sshing anywhere:

  1. The staleness guard is BYPASSED: a verified HIT for a job that is no longer
     current is STILL submitted (exploit pool grace), unlike handle_job which
     drops it. We exercise the real `land_share_mine` with a fake `pearl_mining`
     injected into sys.modules so no native lib is needed.

  2. Orphans-kill is invoked AT STARTUP of the land-share runner (and on exit),
     so a stale binary holding the GPU is reaped before the first mine. We mock
     the ssh subprocess so nothing actually runs.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

# run_canary.py lives at the pearl-stratum repo root (it is a script, not part
# of the installed `pearl_stratum` package), so put that dir on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run_canary  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Job:
    """Minimal stand-in for LuckyPoolJob (only fields land_share_mine touches)."""

    def __init__(self, job_id, target=1, height=1):
        self.job_id = job_id
        self.target = target
        self.height = height
        self.header_bytes = bytes(76)


class _FakeClient:
    """Records submit_share calls; returns a configurable SubmitResult."""

    def __init__(self, accepted=True, error=None, error_code=None):
        self.submitted = []  # list of (job_id, b64)
        self._accepted = accepted
        self._error = error
        self._error_code = error_code

    async def submit_share(self, job_id, b64, hashrate=0.0):
        self.submitted.append((job_id, b64))
        return run_canary_submit_result(self._accepted, self._error,
                                        self._error_code)


def run_canary_submit_result(accepted, error, error_code):
    """Build a SubmitResult-shaped object without importing the pool client."""
    return types.SimpleNamespace(accepted=accepted, latency_ms=1.0,
                                 error=error, error_code=error_code)


def _install_fake_pearl_mining(monkeypatch, verify_ok=True, verify_msg="ok"):
    """Inject a fake `pearl_mining` so land_share_mine runs with no native lib.

    Only the symbols land_share_mine uses are provided:
      - IncompleteBlockHeader.from_bytes
      - PlainProof.from_base64
      - verify_plain_proof
    """
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
    monkeypatch.setitem(sys.modules, "pearl_mining", fake)


# ---------------------------------------------------------------------------
# 1) Staleness guard is BYPASSED in land-share mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_land_share_submits_even_when_job_rotated(monkeypatch):
    """A verified HIT for a job that is no longer current is STILL submitted."""
    _install_fake_pearl_mining(monkeypatch, verify_ok=True)

    mined = _Job("job1")
    # Backend always "hits" with a proof for the mined job.
    backend = lambda hb, mc, tgt, nr, job_id=None: b"PROOF"  # noqa: E731

    client = _FakeClient(accepted=True)
    loop = asyncio.get_running_loop()

    # current_job_id differs from the mined job -> the job has rotated.
    res = await asyncio.to_thread(
        run_canary.land_share_mine, mined, mining_config=None, backend=backend,
        submit=True, client=client, loop=loop, nonce_range=range(0, 48),
        current_job_id="job2")

    assert res["verify"] is True
    assert res["stale"] is True, "should be flagged stale (job rotated)"
    assert res["submitted"] is True, "land-share must submit even when stale"
    assert res["accepted"] is True
    # The submit used the MINED job_id (job1), not the rotated current one.
    assert client.submitted == [("job1", "UFJPT0Y=")], client.submitted


@pytest.mark.asyncio
async def test_land_share_dry_run_does_not_submit(monkeypatch):
    """--dry-run: verify is logged/returned, but no submit happens."""
    _install_fake_pearl_mining(monkeypatch, verify_ok=True)

    mined = _Job("job1")
    backend = lambda hb, mc, tgt, nr, job_id=None: b"PROOF"  # noqa: E731
    client = _FakeClient(accepted=True)
    loop = asyncio.get_running_loop()

    res = await asyncio.to_thread(
        run_canary.land_share_mine, mined, mining_config=None, backend=backend,
        submit=False, client=client, loop=loop, nonce_range=range(0, 48),
        current_job_id="job2")

    assert res["verify"] is True
    assert res["stale"] is True
    assert res["submitted"] is False, "dry-run must NOT submit"
    assert client.submitted == [], "dry-run must not call submit_share"


@pytest.mark.asyncio
async def test_land_share_verify_fail_never_submits(monkeypatch):
    """Fail-safe: an unverifiable proof is never submitted, even in land-share."""
    _install_fake_pearl_mining(monkeypatch, verify_ok=False, verify_msg="bad")

    mined = _Job("job1")
    backend = lambda hb, mc, tgt, nr, job_id=None: b"PROOF"  # noqa: E731
    client = _FakeClient(accepted=True)
    loop = asyncio.get_running_loop()

    res = await asyncio.to_thread(
        run_canary.land_share_mine, mined, mining_config=None, backend=backend,
        submit=True, client=client, loop=loop, nonce_range=range(0, 48),
        current_job_id="job1")

    assert res["verify"] is False
    assert res["submitted"] is False
    assert client.submitted == []


# ---------------------------------------------------------------------------
# 2) Orphans-kill is invoked at startup (mock the ssh call)
# ---------------------------------------------------------------------------


def test_ssh_backend_kill_orphans_invokes_pkill(monkeypatch):
    """SshRigBackend.kill_orphans runs `ssh ... pkill -9 -f pearl_miner_sm89`."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_canary.subprocess, "run", fake_run)

    be = run_canary.SshRigBackend("192.168.70.6")
    be.kill_orphans()

    assert len(calls) == 1, calls
    argv = calls[0]
    assert argv[0] == "ssh"
    assert "root@192.168.70.6" in argv
    assert any("pkill -9 -f pearl_miner_sm89" in a for a in argv), argv


@pytest.mark.asyncio
async def test_land_share_runner_kills_orphans_at_startup(monkeypatch):
    """_run_land_share reaps GPU orphans BEFORE the first mine (startup)."""
    events = []

    # A backend exposing kill_orphans; record when it is called.
    class _Backend:
        def kill_orphans(self):
            events.append("kill_orphans")

        def reset_cancel(self):
            pass

    backend = _Backend()

    # A fake pool client whose run() blocks until stop(), so the mine loop has a
    # chance to start; we assert kill_orphans fired at startup, then tear down.
    class _Client:
        def __init__(self, **kwargs):
            self.on_new_job = kwargs.get("on_new_job")
            self._stop = asyncio.Event()

        async def run(self):
            # Let the startup kill_orphans + mine_task scheduling happen.
            await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            return 0

        async def stop(self):
            self._stop.set()

    # No job is ever delivered, so land_share_mine is never reached; we only
    # assert the STARTUP reap. Build minimal args.
    args = run_canary.build_parser().parse_args(
        ["--backend", "ssh-rig", "--rig", "192.168.70.6", "--wallet", "prl1x",
         "--worker", "cnry01", "--land-share"])
    args.mine_timeout = run_canary.LAND_SHARE_MINE_TIMEOUT_S

    loop = asyncio.get_running_loop()

    task = loop.create_task(
        run_canary._run_land_share(args, None, backend, loop, _Client))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert "kill_orphans" in events, "startup kill_orphans was not invoked"
