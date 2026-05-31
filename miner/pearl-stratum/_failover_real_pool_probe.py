"""Real-pool failover-feasibility probe.

WHAT IT TESTS:
  1. The pool ACCEPTS two simultaneous connections from one address with
     distinct worker suffixes (`worker.primary` and `worker.spare`).
     If the pool refuses the second connection or rate-limits the IP, the
     whole failover approach is dead-on-arrival.
  2. Both connections receive the SAME `mining.notify` stream (job_id is
     pool-global as STRATUM_CAPTURE.md §3f claims).
  3. A submit on the SPARE returns the same response shape as on primary.
  4. The probe runs for 5 minutes by default; we just observe connection
     stability + dedup metric over time. No mining workload required — the
     pool's challenge solve is the only meaningful CPU work.

USAGE (on CPU01 or CPU02):
  PYTHONPATH=/home/pearl-deploy/_failover/pearl-stratum-failover/src \\
    /tmp/_failover_venv/bin/python _failover_real_pool_probe.py \\
    --pool us1.alphapool.tech:5566 \\
    --address prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg \\
    --worker cpu01-failover-probe \\
    --duration-s 300

NOTE: Submits would require valid PoW; we don't submit during this probe.
We just confirm the pool accepts both sockets, both finish handshake, and
both receive mining.notify with matching job_ids over the observation window.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from pearl_stratum.failover_client import FailoverStratumClient
from pearl_stratum.job import Job
from pearl_stratum.stratum_client import parse_pool_url


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="us1.alphapool.tech:5566")
    ap.add_argument("--address", required=True)
    ap.add_argument("--worker", default="failover-probe")
    ap.add_argument("--duration-s", type=int, default=300)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    host, port = parse_pool_url(args.pool)

    jobs_per_peer = [[], []]  # type: list[list[Job]]
    # Watch raw job arrivals on each peer (before the wrapper dedups).
    def make_recorder(idx: int):
        def _rec(job: Job) -> None:
            jobs_per_peer[idx].append((time.monotonic(), job.job_id))
        return _rec

    consumer_fires: list = []
    def on_new_job(job: Job) -> None:
        consumer_fires.append((time.monotonic(), job.job_id))

    client = FailoverStratumClient(
        host=host,
        port=port,
        address=args.address,
        worker=args.worker,
        password="x;d=1048576",
        n_peers=2,
        on_new_job=on_new_job,
    )
    # Wire per-peer recorders too (bypassing the dedup) so we can correlate.
    for i, p in enumerate(client.peers):
        prev = p.on_new_job
        rec = make_recorder(i)
        def _chained(job: Job, prev=prev, rec=rec) -> None:
            rec(job)
            if prev is not None:
                prev(job)
        p.on_new_job = _chained

    task = asyncio.create_task(client.run())
    print(f"[probe] starting; will run for {args.duration_s}s")
    t_start = time.monotonic()
    last_report = t_start
    try:
        while time.monotonic() - t_start < args.duration_s:
            await asyncio.sleep(5)
            now = time.monotonic()
            if now - last_report >= 30:
                # Report
                p0_jobs = len(jobs_per_peer[0])
                p1_jobs = len(jobs_per_peer[1])
                p0_conn = client.peers[0].connected
                p1_conn = client.peers[1].connected
                consumer = len(consumer_fires)
                print(
                    f"[probe] t={now - t_start:.0f}s "
                    f"primary={'OK' if p0_conn else 'DOWN'} "
                    f"spare={'OK' if p1_conn else 'DOWN'} "
                    f"jobs_p0={p0_jobs} jobs_p1={p1_jobs} consumer_fires={consumer}"
                )
                last_report = now
    finally:
        await client.stop()
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            pass

    # Final report
    p0_jids = [jid for _, jid in jobs_per_peer[0]]
    p1_jids = [jid for _, jid in jobs_per_peer[1]]
    common = set(p0_jids) & set(p1_jids)
    only_p0 = set(p0_jids) - set(p1_jids)
    only_p1 = set(p1_jids) - set(p0_jids)
    consumer_jids = set(jid for _, jid in consumer_fires)

    print("\n[probe] === SUMMARY ===")
    print(f"duration: {time.monotonic() - t_start:.0f}s")
    print(f"primary connection_count: {1 if client.peers[0].connected else 0}")
    print(f"spare connection_count:   {1 if client.peers[1].connected else 0}")
    print(f"primary jobs received: {len(p0_jids)} (unique: {len(set(p0_jids))})")
    print(f"spare jobs received:   {len(p1_jids)} (unique: {len(set(p1_jids))})")
    print(f"common job_ids:       {len(common)}")
    print(f"only_primary:         {len(only_p0)}")
    print(f"only_spare:           {len(only_p1)}")
    print(f"consumer fire count:  {len(consumer_fires)} (unique: {len(consumer_jids)})")
    print(f"dedup ok (consumer fires == unique union): "
          f"{len(consumer_fires) == len(set(p0_jids) | set(p1_jids))}")

    # The headline question: did pool accept both sockets?
    accepted_both = (
        len(p0_jids) > 0 and len(p1_jids) > 0
        and client.peers[0].stats.last_diff > 0
        and client.peers[1].stats.last_diff > 0
    )
    print(f"\n[probe] BOTH SOCKETS ACCEPTED BY POOL: {accepted_both}")

    return 0 if accepted_both else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
