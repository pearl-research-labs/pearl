"""Standalone connection test against alphapool. No miner_base imports.

Connects via stratum to us2.alphapool.tech:5566 with the DECOY wallet, runs
the handshake (configure/subscribe/authorize), and reports the first few jobs
received over a 60-second window. NEVER submits a share.
"""

import asyncio
import logging
import sys
import time

sys.path.insert(0, "src")

from pearl_stratum.stratum_client import StratumClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("connect_test")

DECOY = "prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg"


async def main():
    jobs_seen = []
    diffs_seen = []
    disconnect_count = [0]

    def on_new_job(job):
        jobs_seen.append((time.time(), job.job_id))
        log.info(
            "NEW JOB: id=%s clean=%s incomplete_header=%d bytes nbits=%s",
            job.job_id, job.clean_jobs, len(job.incomplete_header_bytes),
            job.nbits if hasattr(job, "nbits") else "?",
        )

    def on_set_difficulty(diff):
        diffs_seen.append((time.time(), diff))
        log.info("SET DIFFICULTY: %s", diff)

    def on_disconnect(reason):
        disconnect_count[0] += 1
        log.warning("DISCONNECT: %s", reason)

    client = StratumClient(
        host="us2.alphapool.tech",
        port=5566,
        address=DECOY,
        worker="connect-test",
        password="x;d=1048576",
        user_agent="pearl-stratum/0.1",
    )
    client.on_new_job = on_new_job
    client.on_set_difficulty = on_set_difficulty
    client.on_disconnect = on_disconnect

    task = asyncio.create_task(client.run())
    try:
        await asyncio.sleep(60.0)
    finally:
        log.info("Asking client to stop...")
        await client.stop()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("client.run() didn't exit in 5s; task=%r", task)

    print("\n=== SUMMARY ===")
    print(f"Jobs received: {len(jobs_seen)}")
    print(f"Difficulty updates: {len(diffs_seen)}")
    print(f"Disconnects: {disconnect_count[0]}")
    print(f"Stats: {client.stats}")
    if jobs_seen:
        first_t = jobs_seen[0][0]
        print(f"First job latency from start: ~{first_t - start_t:.1f}s")
    return 0 if jobs_seen else 1


if __name__ == "__main__":
    start_t = time.time()
    sys.exit(asyncio.run(main()))
