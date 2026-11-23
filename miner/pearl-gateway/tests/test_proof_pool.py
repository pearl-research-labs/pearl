"""ProofPool lifecycle against a real spawned worker process."""

import asyncio
import time

import pytest
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
        worker.join(timeout=10)
        assert not worker.is_alive()
    busy.cancel()
