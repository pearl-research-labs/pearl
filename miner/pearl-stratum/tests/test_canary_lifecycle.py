"""Offline lifecycle tests for run_canary.CanaryMineLoop.

These exercise the PREEMPTION mechanism (generation counter + backend cancel +
nonce-cursor reset + stale-result drop) WITHOUT importing the native mining
stack (pearl_mining / torch / numpy) and WITHOUT opening a socket or sshing
anywhere. `handle_job` is replaced with a recording fake so we can drive the
loop deterministically and assert which job each window was asked to mine and
whether the in-flight window was cancelled when a new job arrived.

The real `handle_job` (which DOES import pearl_mining) is still covered by the
`--selftest` path; here we test only the orchestration around it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

# run_canary.py lives at the pearl-stratum repo root (it is a script, not part
# of the installed `pearl_stratum` package), so put that dir on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run_canary  # noqa: E402


async def _await_until(pred, timeout=2.0, step=0.005):
    """Poll `pred()` on the event loop until truthy or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(step)
    raise AssertionError("condition not met within timeout")


class _Job:
    """Minimal stand-in for LuckyPoolJob (only the fields the loop touches)."""

    def __init__(self, job_id, target=1, height=1):
        self.job_id = job_id
        self.target = target
        self.height = height
        self.header_bytes = bytes(76)


class _RecordingBackend:
    """A fake GPU backend that blocks on a (cancellable) event per window.

    Records every job_id it was asked to mine, and whether each window was
    cancelled (preempted) vs. ran to completion. `cancel()` releases the
    in-flight window immediately (mirrors killing the ssh/binary subprocess).
    """

    def __init__(self, hit_on=None):
        self.calls = []          # list of (job_id, was_cancelled)
        self._cond = threading.Condition()
        self._cancelled = False
        self._release_seq = 0    # bumped by release_window()
        self._consumed_seq = 0   # how many releases the windows have consumed
        self._in_window = False
        self.hit_on = hit_on     # job_id that should "find" a proof (HIT)
        self.proof_for = b"PROOF"

    def in_window(self):
        with self._cond:
            return self._in_window

    def reset_cancel(self):
        # Mirrors the real backend clearing its cancel flag before a new window.
        # It does NOT touch the release counter, so a pending release for the
        # next window is never lost (avoids a test-side lost-wakeup).
        with self._cond:
            self._cancelled = False

    def cancel(self):
        with self._cond:
            self._cancelled = True
            self._cond.notify_all()

    def __call__(self, header_bytes, mining_config, target, nonce_range,
                 job_id=None):
        # Block until either this window is released (NOHIT) or cancelled by a
        # new job. Mirrors the real backend blocking in subprocess.communicate.
        with self._cond:
            self._in_window = True
            my_seq = self._consumed_seq + 1
            self._cond.wait_for(
                lambda: self._cancelled or self._release_seq >= my_seq,
                timeout=2.0)
            cancelled = self._cancelled
            if not cancelled:
                self._consumed_seq = my_seq
            self._in_window = False
        self.calls.append((job_id, cancelled))
        if cancelled:
            return None
        if self.hit_on is not None and job_id == self.hit_on:
            return self.proof_for
        return None

    def release_window(self):
        """Let the currently-blocked window complete as a NOHIT."""
        with self._cond:
            self._release_seq += 1
            self._cond.notify_all()


def _patch_handle_job(monkeypatch, recorder):
    """Replace run_canary.handle_job with a thin wrapper that records the
    (job, gen-staleness) and calls the backend so the loop's cancel path runs.

    Returns nothing; appends result dicts to `recorder`.
    """

    def fake_handle_job(job, *, mining_config, backend, submit, client, loop,
                        nonce_range, is_current):
        proof = backend(job.header_bytes, mining_config, job.target,
                        nonce_range, job_id=job.job_id)
        res = {"job_id": job.job_id, "verify": None, "submitted": False,
               "accepted": None, "error": None, "stale": False,
               "nonce_start": nonce_range.start}
        if proof is not None:
            # Mimic the real verify=True path; honor the staleness guard.
            if not is_current(job.job_id):
                res["stale"] = True
            else:
                res["verify"] = True
                res["submitted"] = bool(submit)
        recorder.append(res)
        return res

    monkeypatch.setattr(run_canary, "handle_job", fake_handle_job)


@pytest.mark.asyncio
async def test_new_job_preempts_in_flight_window(monkeypatch):
    """Feed two notifies in sequence; assert the first job's window is cancelled
    and the loop switches to the second job."""
    results = []
    backend = _RecordingBackend()
    _patch_handle_job(monkeypatch, results)

    loop = asyncio.get_running_loop()
    ml = run_canary.CanaryMineLoop(
        mining_config=None, backend=backend, submit=False, client=object(),
        loop=loop, window=32)

    task = loop.create_task(ml.run())

    # Job #1 arrives -> loop starts a window that BLOCKS in the backend until
    # cancelled. Wait until the backend is actually inside that window.
    ml.on_new_job(_Job("job1"))
    assert ml.current_gen() == 1
    await _await_until(lambda: backend.in_window())

    # Job #2 arrives BEFORE job1's window completes -> on_new_job cancels the
    # in-flight window, switching the loop to job2.
    ml.on_new_job(_Job("job2"))
    assert ml.current_gen() == 2

    # The cancelled job1 window unwinds; the loop resets cancel and starts a
    # fresh window for job2. Wait until that job2 window is running, then
    # release it as a NOHIT so it records and the loop continues.
    await _await_until(lambda: any(c[0] == "job1" for c in backend.calls))
    await _await_until(lambda: backend.in_window())
    backend.release_window()
    await _await_until(lambda: any(c[0] == "job2" for c in backend.calls))

    ml.stop()
    backend.cancel()
    await asyncio.sleep(0.02)
    task.cancel()

    job_ids = [c[0] for c in backend.calls]
    assert "job1" in job_ids, f"job1 was never mined: {backend.calls}"
    assert "job2" in job_ids, f"job2 was never mined: {backend.calls}"
    # The job1 window must have been CANCELLED (preempted), not run to a result.
    job1_calls = [c for c in backend.calls if c[0] == "job1"]
    assert all(c[1] is True for c in job1_calls), \
        f"job1 window should have been cancelled: {backend.calls}"


@pytest.mark.asyncio
async def test_stale_hit_is_not_submitted(monkeypatch):
    """If a HIT comes back for a job that has since rotated, the is_current
    guard drops it (stale) rather than submitting."""
    results = []
    backend = _RecordingBackend(hit_on="job1")
    _patch_handle_job(monkeypatch, results)

    loop = asyncio.get_running_loop()
    ml = run_canary.CanaryMineLoop(
        mining_config=None, backend=backend, submit=True, client=object(),
        loop=loop, window=32)

    # Build the is_current predicate for gen=1, then rotate to gen=2 and prove
    # the predicate now reports the job1-gen as stale.
    ml.on_new_job(_Job("job1"))
    guard = ml.is_current(ml.current_gen())
    assert guard("job1") is True
    ml.on_new_job(_Job("job2"))
    assert guard("job1") is False, "guard must report gen=1 as stale after rotation"


@pytest.mark.asyncio
async def test_cursor_advances_across_windows_same_job(monkeypatch):
    """Successive NOHIT windows for the SAME job advance the nonce cursor."""
    results = []
    backend = _RecordingBackend()  # never hits
    _patch_handle_job(monkeypatch, results)

    loop = asyncio.get_running_loop()
    ml = run_canary.CanaryMineLoop(
        mining_config=None, backend=backend, submit=False, client=object(),
        loop=loop, window=32)
    task = loop.create_task(ml.run())

    ml.on_new_job(_Job("job1"))
    # Drive three NOHIT windows for the same job. The loop calls reset_cancel()
    # before each window; we wait for the backend to be blocked in a window,
    # release it, then wait for the recorded-call count to tick up before
    # releasing the next (so each window is distinct).
    for n in range(3):
        await _await_until(lambda: backend.in_window())
        backend.release_window()
        await _await_until(lambda n=n: len(backend.calls) >= n + 1)

    ml.stop()
    backend.cancel()
    await asyncio.sleep(0.02)
    task.cancel()

    starts = [r["nonce_start"] for r in results if r["job_id"] == "job1"]
    # At least the first two windows should show strictly increasing starts.
    assert len(starts) >= 2, f"expected multiple windows, got {results}"
    assert starts[1] > starts[0], f"nonce cursor did not advance: {starts}"
    assert starts[0] == 0, f"first window must start at nonce 0: {starts}"
