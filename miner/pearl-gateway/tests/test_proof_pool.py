"""ProofPool lifecycle against a real spawned worker process."""

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest
from pearl_gateway import proof_worker
from pearl_gateway.proof_pool import ProofPool


@pytest.mark.asyncio
async def test_stop_kills_a_worker_busy_with_a_proof():
    pool = ProofPool()
    await pool.start()
    executor = pool._executor
    busy = asyncio.get_running_loop().run_in_executor(executor, time.sleep, 60)
    while not executor._processes:  # wait for the worker to spawn
        await asyncio.sleep(0.05)
    workers = list(executor._processes.values())

    await pool.stop()

    for worker in workers:
        assert await _exited(worker)
    busy.cancel()


async def _exited(worker, timeout: float = 10.0) -> bool:
    """True once the worker process is gone. is_alive() alone can report a dead child
    as alive if the executor's own thread reaped it first, so also ask the OS."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.exitcode is not None:
            return True
        try:
            os.kill(worker.pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_prove_retries_after_the_worker_died_while_idle(monkeypatch):
    pool = ProofPool()
    await pool.start()
    try:
        executor = pool._executor
        await asyncio.get_running_loop().run_in_executor(executor, time.sleep, 0)
        for worker in list(executor._processes.values()):
            worker.kill()
            worker.join(timeout=10)

        # A builtin, so the spawned worker can unpickle it (it cannot import tests).
        monkeypatch.setattr(proof_worker, "prove", print)
        result = await pool.prove(4, b"header", "proof")

        assert result is None  # print's return value: the job ran on a fresh worker
        assert pool._executor is not executor
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_restart_after_stop_does_not_revive_the_pool():
    pool = ProofPool()
    await pool.start()
    executor = pool._executor
    await pool.stop()

    await pool._restart(executor)
    await pool._restart(None)

    assert pool._executor is None


def test_worker_exits_on_a_rust_panic(monkeypatch):
    class PanicException(BaseException):  # stand-in for pyo3's PanicException
        pass

    def panic(*_args):
        raise PanicException("circuit cache poisoned")

    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(proof_worker, "IncompleteBlockHeader", MagicMock())
    monkeypatch.setattr(proof_worker, "PlainProof", MagicMock())
    monkeypatch.setattr(proof_worker, "generate_proof_for_cert_version", panic)
    monkeypatch.setattr(proof_worker.os, "_exit", fake_exit)

    with pytest.raises(SystemExit):
        proof_worker.prove(4, b"header", "proof")

    assert exits == [1]


def test_worker_reraises_ordinary_errors(monkeypatch):
    def fail(*_args):
        raise ValueError("bad proof")

    monkeypatch.setattr(proof_worker, "IncompleteBlockHeader", MagicMock())
    monkeypatch.setattr(proof_worker, "PlainProof", MagicMock())
    monkeypatch.setattr(proof_worker, "generate_proof_for_cert_version", fail)

    with pytest.raises(ValueError, match="bad proof"):
        proof_worker.prove(4, b"header", "proof")
