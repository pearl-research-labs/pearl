#!/usr/bin/env python3
"""Pearl A/B pool-credit comparison harness.

Compares pool-credited share rate between alpha-miner (Arm A) and our
wave-15+ kernel-noising miner (Arm B) against the same alphapool endpoint.
Measures the reconnect-drop gap directly from miner logs and overlays
with pool API share counts to derive the "effective TOPS" ratio.

What we measure
---------------
Per arm, over the measurement window:
  - submits         : alpha-miner `component=share submitted` lines (sent to pool)
  - candidates      : alpha-miner `component=share found_candidate` lines (PoW hits)
  - drops_reconnect : `action=reconnect_drop_ambiguous_share` lines (the 42% bug)
  - drops_stratum   : `share dropped reason=stratum_reconnect` lines (collateral)
  - pool_shares     : count of `recent_shares` entries for the worker on
                      the wallet's pool API
  - pool_eff_th     : pool's `hashrate_1h` for the worker (post-vardiff)
  - drop_pct        : drops_reconnect / candidates  (alpha's known ~42%)

For our pearl-stratum arm we additionally parse:
  - submits         : `mining.submit` lines in stratum log
  - accepted        : `result:true` ack lines
  - stale_21        : `error 21` lines (chain advanced, NOT dropped)
  - reconnects      : socket-close lines

Effective TOPS = pool_eff_th  (this IS the pool-credited number; raw
kernel TOPS is the upper bound; the gap is the bug + drop loss.)

Rig assignment
--------------
This harness orchestrates via SSH. Two arms run on two rigs concurrently,
or one arm runs solo for a baseline. The pool isolates each arm by worker
name. Don't run both arms on the same rig unless you split GPUs explicitly.

Run modes
---------
1) --mode observe : passive measurement. Tails an already-running miner's
   logs and polls the pool API. ZERO start/stop on the rig. Safe to use on
   a production rig because nothing is touched. This is the recommended
   first run -- no risk of tripping pool rate-limits.

2) --mode swap-cpu01 : stop production alpha-miner on CPU01, swap to
   `wave15_pool_runv2.sh`, measure for the window, restore alpha-miner
   on the production wallet. Costs ~window minutes of production mining
   on CPU01 (one rig out of ~31). Restores cleanly on SIGINT/SIGTERM.

3) --mode dual : orchestrate both arms on two SSH-reachable rigs. Most
   defensible apples-to-apples but requires two rigs that are off CatStack
   flight sheets AND have the wave-15 .so deployed.

Run examples
------------
    # 30-min passive observation of CPU01's current alpha-miner
    python3 wave16_ab_harness.py --mode observe \\
        --rig-a-host 192.168.71.252 --rig-a-arm-name alpha \\
        --rig-a-worker CPU01 \\
        --wallet prl1pja266dfa7kcg0xdagaacy0y7x60h7qrw3tcau4enx4gwnmmyxxvs7ep7ad \\
        --window 1800 \\
        --out /tmp/ab_observe

    # CPU01 swap: alpha (10 min baseline) -> wave-15 (20 min)
    python3 wave16_ab_harness.py --mode swap-cpu01 \\
        --rig-a-host 192.168.71.252 \\
        --window 1800 --baseline-window 600 \\
        --decoy-wallet prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg \\
        --out /tmp/ab_swap

    # Dual-rig: alpha on rig X, wave-15 on rig Y
    python3 wave16_ab_harness.py --mode dual \\
        --rig-a-host 192.168.71.252 --rig-a-arm-name alpha \\
        --rig-b-host 192.168.70.162 --rig-b-arm-name pearlw15 \\
        --decoy-wallet prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg \\
        --window 1800 \\
        --out /tmp/ab_dual
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# --- Constants ---------------------------------------------------------------

POOL_API_BASE = "https://pearl.alphapool.tech/api/miner"
POOL_HOST = "us2.alphapool.tech"
POOL_PORT = 5566

ALPHA_LOG_PATH = "/var/log/mfarm/miner.log"
ALPHA_SERVICE = "alpha-miner.service"
WAVE_DEPLOY_DIR = "/home/pearl-deploy"
WAVE_LOG_PATH = "/var/log/pearl-wave15.log"

# Sampling cadences
POOL_POLL_S = 30
LOG_TAIL_PERIOD_S = 5      # how often we drain remote log buffer over ssh

# SSH identity & options
SSH_KEY = os.environ.get("SSH_KEY", "/c/Users/benef/.ssh/id_ed25519")
SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-i", SSH_KEY,
]


# --- Log grammar -------------------------------------------------------------

# alpha-miner v1.4/1.5 line examples (taken from /var/log/mfarm/miner.log):
#   ... component=share found_candidate ... job=<id> kernel_hash=...
#   ... component=share submitted job=<id>
#   ... component=pool connection_lost phase=submit job=<id> ... action=reconnect_drop_ambiguous_share
#   ... component=share dropped reason=stratum_reconnect
#   ... component=miner status attempts=N hits=M hashrate_th_s=X share_equiv_th_s=Y
ALPHA_RE_CANDIDATE = re.compile(r"component=share found_candidate.*?job=(\S+)")
ALPHA_RE_SUBMIT = re.compile(r"component=share submitted job=(\S+)")
ALPHA_RE_DROP_RECONNECT = re.compile(r"action=reconnect_drop_ambiguous_share.*?job=(\S+)|reconnect_drop_ambiguous_share")
ALPHA_RE_DROP_STRATUM = re.compile(r"component=share dropped reason=stratum_reconnect")
ALPHA_RE_STATUS = re.compile(
    r"component=miner status attempts=(\d+) hits=(\d+) hashrate_th_s=([\d.]+).*?share_equiv_th_s=([\d.]+)"
)

# pearl-stratum (our miner) log examples (from stratum_client.py + driver):
#   ... mining.submit id=N job=<id> proof_bytes=N
#   ... share accepted job=<id> latency_ms=N
#   ... StaleShareError ... error 21
#   ... rate=NN.N/s main_TOPS=NN.NN
PEARL_RE_SUBMIT = re.compile(r"mining\.submit.*?id=(\d+)")
PEARL_RE_ACCEPT = re.compile(r'result.*?true|raw_result=True|share accepted|"result":\s*true')
PEARL_RE_STALE = re.compile(r"StaleShareError|error.*?21|\"error\":\s*\[21")
PEARL_RE_TOPS = re.compile(r"main_TOPS=([\d.]+)")
PEARL_RE_RATE = re.compile(r"rate=([\d.]+)")


# --- Data classes ------------------------------------------------------------

@dataclass
class ArmMetrics:
    """Per-arm tallies. `kind` is 'alpha' or 'pearl'."""

    kind: str
    worker: str
    rig: str

    # alpha-side counters
    candidates: int = 0
    submits: int = 0
    drops_reconnect: int = 0
    drops_stratum: int = 0

    # pearl-side counters
    accepts: int = 0
    stales: int = 0

    # status line readings (latest)
    latest_status: dict[str, float] = field(default_factory=dict)

    # pool API snapshots
    pool_samples: list[dict] = field(default_factory=list)

    # per-job dedupe so we don't double-count when a line appears in two reads
    seen_lines: set[int] = field(default_factory=set)

    def feed_alpha_line(self, line: str) -> None:
        # Deduplicate on hash so we don't double-count log re-reads.
        h = hash(line)
        if h in self.seen_lines:
            return
        self.seen_lines.add(h)

        if ALPHA_RE_DROP_RECONNECT.search(line):
            self.drops_reconnect += 1
            return
        if ALPHA_RE_DROP_STRATUM.search(line):
            self.drops_stratum += 1
            return
        if ALPHA_RE_SUBMIT.search(line):
            self.submits += 1
            return
        if ALPHA_RE_CANDIDATE.search(line):
            self.candidates += 1
            return
        m = ALPHA_RE_STATUS.search(line)
        if m:
            self.latest_status = {
                "attempts": int(m.group(1)),
                "hits": int(m.group(2)),
                "hashrate_th_s": float(m.group(3)),
                "share_equiv_th_s": float(m.group(4)),
            }

    def feed_pearl_line(self, line: str) -> None:
        h = hash(line)
        if h in self.seen_lines:
            return
        self.seen_lines.add(h)

        if PEARL_RE_SUBMIT.search(line):
            self.submits += 1
            return
        if PEARL_RE_ACCEPT.search(line):
            self.accepts += 1
            return
        if PEARL_RE_STALE.search(line):
            self.stales += 1
            return
        m = PEARL_RE_TOPS.search(line)
        if m:
            self.latest_status["main_tops"] = float(m.group(1))
        m = PEARL_RE_RATE.search(line)
        if m:
            self.latest_status["rate_s"] = float(m.group(1))

    @property
    def drop_pct(self) -> float:
        """Reconnect-drop rate. Only meaningful for the alpha arm."""
        denom = max(self.candidates, 1)
        return 100.0 * self.drops_reconnect / denom

    @property
    def accept_pct(self) -> float:
        """Accept rate. Pearl arm: accepts/submits. Alpha arm: submits/candidates."""
        if self.kind == "pearl":
            denom = max(self.submits, 1)
            return 100.0 * self.accepts / denom
        denom = max(self.candidates, 1)
        return 100.0 * self.submits / denom

    def latest_pool_eff_th(self) -> float:
        """Pool-reported hashrate_1h (TH/s). 0 if no samples or worker missing."""
        for snap in reversed(self.pool_samples):
            w = snap.get("worker_match")
            if w is None:
                continue
            hr = w.get("hashrate_1h") or ""
            return _parse_th_string(hr)
        return 0.0

    def pool_share_count(self) -> int:
        """Distinct (ts, worker) recent_shares entries seen for this worker."""
        seen: set[tuple[int, str]] = set()
        for snap in self.pool_samples:
            for s in snap.get("recent_shares_for_worker", []):
                seen.add((s.get("ts", 0), str(s.get("worker", ""))))
        return len(seen)


# --- Pool API ----------------------------------------------------------------

def fetch_pool_for_wallet(wallet: str) -> dict | None:
    url = f"{POOL_API_BASE}/{wallet}"
    req = urllib.request.Request(url, headers={"User-Agent": "wave16_ab/1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        print(f"[pool] fetch error: {e}", file=sys.stderr)
        return None


def extract_pool_snapshot(pool_doc: dict, worker_name: str) -> dict:
    """Pull the worker's row out of the wallet doc plus recent_shares for it."""
    workers = pool_doc.get("workers") or []
    # workers may be reported as `worker` or `rig.worker` -- match substring.
    worker_match = None
    for w in workers:
        name = w.get("name") or ""
        if name == worker_name or name.endswith(f".{worker_name}") or worker_name in name:
            worker_match = w
            break

    recent = pool_doc.get("recent_shares") or []
    if not isinstance(recent, list):
        recent = []
    matches = [
        s for s in recent
        if (s.get("worker") or "") == worker_name
           or (s.get("worker") or "").endswith(f".{worker_name}")
    ]
    return {
        "t_wall": time.time(),
        "worker_match": worker_match,
        "recent_shares_for_worker": matches,
        "wallet_estHash1h": pool_doc.get("estHash1h"),
        "wallet_shares24h": pool_doc.get("shares24h"),
    }


# --- SSH helpers -------------------------------------------------------------

def ssh_run(host: str, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a remote command, return (rc, stdout, stderr)."""
    full = ["ssh", *SSH_OPTS, f"root@{host}", cmd]
    try:
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "ssh timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


def ssh_tail_since(host: str, log_path: str, since_ts: float, timeout: int = 15) -> list[str]:
    """Tail remote log, returning new lines since `since_ts` (epoch seconds).

    We use `awk` with the alpha-miner ISO-8601 prefix `YYYY-MM-DDTHH:MM:SS.sssZ`.
    For non-prefixed logs we fall back to `tail -n 5000` so we still drain
    something. Caller dedupes on line hash, so over-read is harmless.
    """
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_ts))
    # `2>/dev/null` because the log path may not exist for the pearl arm yet.
    cmd = (
        f"awk -v cutoff={iso!r} '$0 >= cutoff {{print}}' {log_path} 2>/dev/null "
        f"|| tail -n 5000 {log_path} 2>/dev/null"
    )
    rc, out, err = ssh_run(host, cmd, timeout=timeout)
    if rc != 0 and not out:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


# --- Harness core ------------------------------------------------------------

class Harness:
    def __init__(
        self,
        mode: str,
        rig_a_host: str,
        rig_a_arm: str,
        rig_a_worker: str,
        rig_a_log: str,
        wallet_a: str,
        rig_b_host: str | None,
        rig_b_arm: str,
        rig_b_worker: str,
        rig_b_log: str,
        wallet_b: str,
        window: int,
        baseline_window: int,
        out_prefix: str,
    ):
        self.mode = mode
        self.window = window
        self.baseline_window = baseline_window
        self.out_prefix = out_prefix
        self.wallet_a = wallet_a
        self.wallet_b = wallet_b

        self.arm_a = ArmMetrics(
            kind="alpha", worker=rig_a_worker, rig=f"{rig_a_host}",
        )
        self.arm_a_host = rig_a_host
        self.arm_a_log = rig_a_log
        self.arm_a_arm_label = rig_a_arm

        if rig_b_host:
            self.arm_b: ArmMetrics | None = ArmMetrics(
                kind="pearl", worker=rig_b_worker, rig=f"{rig_b_host}",
            )
            self.arm_b_host = rig_b_host
            self.arm_b_log = rig_b_log
            self.arm_b_arm_label = rig_b_arm
        else:
            self.arm_b = None
            self.arm_b_host = None
            self.arm_b_log = None
            self.arm_b_arm_label = rig_b_arm

        self._stop = threading.Event()
        self._started_at = 0.0

    # ---- worker threads ---------------------------------------------------

    def _tail_thread(self, host: str, log_path: str, arm: ArmMetrics, kind: str) -> None:
        since = time.time() - 60  # backfill a small lookback so we don't miss spillover
        while not self._stop.is_set():
            lines = ssh_tail_since(host, log_path, since)
            for ln in lines:
                if kind == "alpha":
                    arm.feed_alpha_line(ln)
                else:
                    arm.feed_pearl_line(ln)
            since = time.time()
            self._stop.wait(LOG_TAIL_PERIOD_S)

    def _pool_thread(self) -> None:
        next_sample = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= next_sample:
                doc_a = fetch_pool_for_wallet(self.wallet_a)
                if doc_a is not None:
                    self.arm_a.pool_samples.append(extract_pool_snapshot(doc_a, self.arm_a.worker))
                if self.arm_b is not None:
                    # Same wallet or different wallet: support both.
                    doc_b = doc_a if self.wallet_b == self.wallet_a else fetch_pool_for_wallet(self.wallet_b)
                    if doc_b is not None:
                        self.arm_b.pool_samples.append(extract_pool_snapshot(doc_b, self.arm_b.worker))
                next_sample = now + POOL_POLL_S
            self._stop.wait(min(POOL_POLL_S, 5))

    # ---- modes ------------------------------------------------------------

    def run_observe(self) -> None:
        """Passive measurement: read logs + poll pool, never start/stop anything."""
        print(f"[observe] window={self.window}s arm_a={self.arm_a.worker}@{self.arm_a_host}")
        if self.arm_b is not None:
            print(f"[observe] arm_b={self.arm_b.worker}@{self.arm_b_host}")
        self._start_threads()
        self._wait_window()
        self._finalize()

    def run_swap_cpu01(self) -> None:
        """Stop alpha on CPU01, run baseline, then run wave-15, then restore alpha.

        Phase 1: passive measure of currently-running alpha-miner for baseline_window.
        Phase 2: stop alpha, start wave-15 pool runner with decoy wallet, measure for
                 (window - baseline_window) seconds.
        Phase 3: restore alpha-miner on production wallet.
        """
        if self.arm_b is None:
            raise SystemExit("swap-cpu01 mode requires --rig-b-host (the same host as --rig-a-host)")
        if self.arm_a_host != self.arm_b_host:
            print(
                "[swap-cpu01] WARNING: rig-a-host != rig-b-host; both arms expected on same rig",
                file=sys.stderr,
            )

        print(f"[swap-cpu01] phase 1 (alpha baseline): {self.baseline_window}s")
        self._start_threads_for(self.arm_a, self.arm_a_host, self.arm_a_log, kind="alpha")
        self._wait(self.baseline_window)

        print("[swap-cpu01] phase 2: stopping alpha-miner, starting wave-15 pool runner")
        self._stop_alpha(self.arm_a_host)
        self._launch_wave15(self.arm_b_host)
        self._start_threads_for(self.arm_b, self.arm_b_host, self.arm_b_log, kind="pearl")

        phase2 = max(self.window - self.baseline_window, 60)
        print(f"[swap-cpu01] phase 2 (wave-15): {phase2}s")
        self._wait(phase2)

        print("[swap-cpu01] phase 3: stopping wave-15, restoring alpha")
        self._stop_wave15(self.arm_b_host)
        self._restore_alpha(self.arm_a_host)
        self._finalize()

    def run_dual(self) -> None:
        """Both arms on separate hosts, concurrent for the full window."""
        if self.arm_b is None:
            raise SystemExit("dual mode requires --rig-b-host")
        print(f"[dual] window={self.window}s; "
              f"arm_a={self.arm_a.worker}@{self.arm_a_host}, "
              f"arm_b={self.arm_b.worker}@{self.arm_b_host}")
        self._start_threads()
        self._wait_window()
        self._finalize()

    # ---- thread orchestration --------------------------------------------

    def _start_threads(self) -> None:
        self._started_at = time.time()
        threads = []
        threads.append(threading.Thread(
            target=self._tail_thread,
            args=(self.arm_a_host, self.arm_a_log, self.arm_a, "alpha"),
            daemon=True, name="tail-a",
        ))
        if self.arm_b is not None:
            kind = "alpha" if self.arm_b_arm_label.startswith("alpha") else "pearl"
            threads.append(threading.Thread(
                target=self._tail_thread,
                args=(self.arm_b_host, self.arm_b_log, self.arm_b, kind),
                daemon=True, name="tail-b",
            ))
        threads.append(threading.Thread(target=self._pool_thread, daemon=True, name="pool"))
        for t in threads:
            t.start()
        self._threads = threads

    def _start_threads_for(self, arm: ArmMetrics, host: str, log_path: str, kind: str) -> None:
        if self._started_at == 0.0:
            self._started_at = time.time()
            # also start pool thread on first call
            pt = threading.Thread(target=self._pool_thread, daemon=True, name="pool")
            pt.start()
            self._threads = [pt]
        t = threading.Thread(
            target=self._tail_thread, args=(host, log_path, arm, kind),
            daemon=True, name=f"tail-{kind}",
        )
        t.start()
        self._threads.append(t)

    def _wait_window(self) -> None:
        self._wait(self.window)

    def _wait(self, secs: int) -> None:
        end = time.time() + secs
        while time.time() < end and not self._stop.is_set():
            time.sleep(1)
            elapsed = int(time.time() - self._started_at)
            if elapsed > 0 and elapsed % 60 == 0:
                self._progress()

    def _progress(self) -> None:
        print(self._summary_line(self.arm_a))
        if self.arm_b is not None:
            print(self._summary_line(self.arm_b))

    def _summary_line(self, arm: ArmMetrics) -> str:
        eff_th = arm.latest_pool_eff_th()
        ps = arm.pool_share_count()
        if arm.kind == "alpha":
            return (
                f"  [{arm.kind:5}] {arm.worker:30s} "
                f"cand={arm.candidates:4d} submit={arm.submits:4d} "
                f"drop_rc={arm.drops_reconnect:3d} ({arm.drop_pct:5.1f}%) "
                f"drop_st={arm.drops_stratum:3d} pool_sh={ps:3d} eff_th={eff_th:6.2f}"
            )
        return (
            f"  [{arm.kind:5}] {arm.worker:30s} "
            f"submit={arm.submits:4d} accept={arm.accepts:4d} ({arm.accept_pct:5.1f}%) "
            f"stale21={arm.stales:3d} pool_sh={ps:3d} eff_th={eff_th:6.2f}"
        )

    # ---- swap-cpu01 service control --------------------------------------

    def _stop_alpha(self, host: str) -> None:
        rc, out, err = ssh_run(host, f"systemctl stop {ALPHA_SERVICE}", timeout=30)
        if rc != 0:
            print(f"[swap-cpu01] alpha stop rc={rc}: {err}", file=sys.stderr)
        time.sleep(3)

    def _restore_alpha(self, host: str) -> None:
        rc, out, err = ssh_run(host, f"systemctl start {ALPHA_SERVICE}", timeout=30)
        if rc != 0:
            print(f"[swap-cpu01] alpha start rc={rc}: {err}", file=sys.stderr)

    def _launch_wave15(self, host: str) -> None:
        # Run inside the pearl-build docker container with the wave-15 runner.
        # Logs to WAVE_LOG_PATH on the rig so our tail thread can pick them up.
        cmd = (
            f"nohup bash {WAVE_DEPLOY_DIR}/wave15_pool_runv2.sh "
            f"> {WAVE_LOG_PATH} 2>&1 &"
        )
        rc, out, err = ssh_run(host, cmd, timeout=15)
        if rc != 0:
            print(f"[swap-cpu01] wave-15 launch rc={rc}: {err}", file=sys.stderr)
        time.sleep(5)  # give container time to start

    def _stop_wave15(self, host: str) -> None:
        # Inside the container the runner runs python3; kill cleanly.
        ssh_run(host, "pkill -f wave15_pool_run; pkill -f wave15_bench.py", timeout=15)
        time.sleep(2)

    # ---- finalize --------------------------------------------------------

    def _finalize(self) -> None:
        self._stop.set()
        for t in getattr(self, "_threads", []):
            t.join(timeout=10)

        ended_at = time.time()
        elapsed_s = ended_at - self._started_at

        # Compute alpha-arm effective TOPS, pool-credit ratio.
        out_dir = os.path.dirname(self.out_prefix) or "."
        os.makedirs(out_dir, exist_ok=True)

        # CSV: per-arm summary
        csv_path = f"{self.out_prefix}.csv"
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([
                "arm", "kind", "rig", "worker",
                "candidates", "submits", "drops_reconnect", "drops_stratum",
                "accepts", "stales21",
                "drop_pct", "accept_pct",
                "pool_share_count", "pool_eff_th",
                "latest_kernel_th_s", "latest_share_equiv_th_s",
                "elapsed_s",
            ])
            for label, arm in self._all_arms():
                wr.writerow([
                    label, arm.kind, arm.rig, arm.worker,
                    arm.candidates, arm.submits, arm.drops_reconnect, arm.drops_stratum,
                    arm.accepts, arm.stales,
                    f"{arm.drop_pct:.2f}", f"{arm.accept_pct:.2f}",
                    arm.pool_share_count(), f"{arm.latest_pool_eff_th():.3f}",
                    arm.latest_status.get("hashrate_th_s", 0.0),
                    arm.latest_status.get("share_equiv_th_s", 0.0),
                    f"{elapsed_s:.1f}",
                ])
        print(f"[output] csv -> {csv_path}")

        # Per-sample CSV: pool snapshots over time
        snaps_path = f"{self.out_prefix}.pool_samples.csv"
        with open(snaps_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["arm", "t_wall", "worker", "hashrate_live", "hashrate_1h",
                         "difficulty", "online", "recent_shares_count"])
            for label, arm in self._all_arms():
                for snap in arm.pool_samples:
                    w = snap.get("worker_match") or {}
                    wr.writerow([
                        label, f"{snap['t_wall']:.1f}", arm.worker,
                        w.get("hashrate_live", ""), w.get("hashrate_1h", ""),
                        w.get("difficulty", ""), w.get("online", ""),
                        len(snap.get("recent_shares_for_worker", [])),
                    ])
        print(f"[output] pool samples -> {snaps_path}")

        # JSON: raw state for analysis
        json_path = f"{self.out_prefix}.json"
        payload = {
            "mode": self.mode,
            "window_s": self.window,
            "elapsed_s": elapsed_s,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "arms": {
                label: {
                    **dataclasses.asdict(arm),
                    "drop_pct": arm.drop_pct,
                    "accept_pct": arm.accept_pct,
                    "pool_share_count": arm.pool_share_count(),
                    "pool_eff_th": arm.latest_pool_eff_th(),
                    # seen_lines is a set of hashes; not JSON-friendly. drop it.
                }
                for label, arm in self._all_arms()
            },
        }
        # Strip un-serializable bits
        for label in payload["arms"]:
            payload["arms"][label].pop("seen_lines", None)
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[output] json -> {json_path}")

        # Summary report
        self._print_report()

    def _all_arms(self) -> list[tuple[str, ArmMetrics]]:
        out = [(self.arm_a_arm_label, self.arm_a)]
        if self.arm_b is not None:
            out.append((self.arm_b_arm_label, self.arm_b))
        return out

    def _print_report(self) -> None:
        print("\n" + "=" * 72)
        print("A/B HARNESS SUMMARY")
        print("=" * 72)
        a = self.arm_a
        b = self.arm_b
        print(f"Arm A: {self.arm_a_arm_label}  worker={a.worker}  rig={a.rig}")
        if a.kind == "alpha":
            print(f"  candidates    : {a.candidates}")
            print(f"  submitted     : {a.submits}")
            print(f"  drop_reconnect: {a.drops_reconnect}  ({a.drop_pct:.1f}% of candidates)")
            print(f"  drop_stratum  : {a.drops_stratum}")
            print(f"  accept_pct    : {a.accept_pct:.1f}% (submitted/candidates)")
        else:
            print(f"  submitted     : {a.submits}")
            print(f"  accepted      : {a.accepts}  ({a.accept_pct:.1f}%)")
            print(f"  stale 21      : {a.stales}")
        print(f"  pool eff TH/s : {a.latest_pool_eff_th():.2f}")
        print(f"  pool shares   : {a.pool_share_count()}")
        if a.latest_status:
            print(f"  last status   : {a.latest_status}")

        if b is not None:
            print()
            print(f"Arm B: {self.arm_b_arm_label}  worker={b.worker}  rig={b.rig}")
            if b.kind == "alpha":
                print(f"  candidates    : {b.candidates}")
                print(f"  submitted     : {b.submits}")
                print(f"  drop_reconnect: {b.drops_reconnect}  ({b.drop_pct:.1f}% of candidates)")
                print(f"  drop_stratum  : {b.drops_stratum}")
            else:
                print(f"  submitted     : {b.submits}")
                print(f"  accepted      : {b.accepts}  ({b.accept_pct:.1f}%)")
                print(f"  stale 21      : {b.stales}")
            print(f"  pool eff TH/s : {b.latest_pool_eff_th():.2f}")
            print(f"  pool shares   : {b.pool_share_count()}")
            if b.latest_status:
                print(f"  last status   : {b.latest_status}")

            # The headline ratio
            print()
            print("=" * 72)
            eff_a = a.latest_pool_eff_th()
            eff_b = b.latest_pool_eff_th()
            if eff_a > 0:
                ratio = eff_b / eff_a
                print(f"POOL-CREDIT RATIO  (Arm B / Arm A) : {ratio:.3f}x")
            ps_a = a.pool_share_count()
            ps_b = b.pool_share_count()
            if ps_a > 0:
                share_ratio = ps_b / ps_a
                print(f"POOL-SHARE RATIO   (Arm B / Arm A) : {share_ratio:.3f}x")
            print("=" * 72)


# --- helpers -----------------------------------------------------------------

def _parse_th_string(s: str) -> float:
    """Parse pool hashrate strings like '45.04 TH/s', '7.51 TH/s', '0 H/s'."""
    if not s:
        return 0.0
    parts = s.strip().split()
    if len(parts) < 2:
        return 0.0
    try:
        val = float(parts[0])
    except ValueError:
        return 0.0
    unit = parts[1].upper()
    if unit.startswith("PH/"):
        return val * 1_000_000.0
    if unit.startswith("TH/"):
        return val
    if unit.startswith("GH/"):
        return val / 1000.0
    if unit.startswith("MH/"):
        return val / 1_000_000.0
    return 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["observe", "swap-cpu01", "dual"], required=True)
    p.add_argument("--window", type=int, default=1800,
                   help="total measurement window in seconds (default 1800 = 30 min)")
    p.add_argument("--baseline-window", type=int, default=600,
                   help="for swap-cpu01: alpha-baseline phase in seconds (default 600)")
    p.add_argument("--out", default="/tmp/wave16_ab",
                   help="output file prefix (writes .csv, .pool_samples.csv, .json)")

    # Arm A (alpha)
    p.add_argument("--rig-a-host", required=True, help="IP or hostname of rig running Arm A")
    p.add_argument("--rig-a-arm-name", default="alpha",
                   help="label for Arm A in CSV/JSON")
    p.add_argument("--rig-a-worker", default="CPU01",
                   help="pool-side worker name for Arm A (must match the miner's --worker)")
    p.add_argument("--rig-a-log", default=ALPHA_LOG_PATH,
                   help=f"path to Arm A miner log on rig (default {ALPHA_LOG_PATH})")
    p.add_argument("--wallet", required=True,
                   help="wallet for Arm A pool API queries")

    # Arm B (pearl / wave-15+)
    p.add_argument("--rig-b-host", default=None, help="IP or hostname of rig running Arm B")
    p.add_argument("--rig-b-arm-name", default="pearl-w15",
                   help="label for Arm B in CSV/JSON")
    p.add_argument("--rig-b-worker", default="cpu01-armB-wave15",
                   help="pool-side worker name for Arm B")
    p.add_argument("--rig-b-log", default=WAVE_LOG_PATH,
                   help=f"path to Arm B miner log on rig (default {WAVE_LOG_PATH})")
    p.add_argument("--decoy-wallet", default=None,
                   help="wallet for Arm B pool API queries (defaults to --wallet)")

    args = p.parse_args(argv)
    wallet_b = args.decoy_wallet or args.wallet

    harness = Harness(
        mode=args.mode,
        rig_a_host=args.rig_a_host,
        rig_a_arm=args.rig_a_arm_name,
        rig_a_worker=args.rig_a_worker,
        rig_a_log=args.rig_a_log,
        wallet_a=args.wallet,
        rig_b_host=args.rig_b_host,
        rig_b_arm=args.rig_b_arm_name,
        rig_b_worker=args.rig_b_worker,
        rig_b_log=args.rig_b_log,
        wallet_b=wallet_b,
        window=args.window,
        baseline_window=args.baseline_window,
        out_prefix=args.out,
    )

    # Clean shutdown on SIGINT
    def _on_int(signum, _frame):
        print(f"\n[signal {signum}] requesting shutdown...", file=sys.stderr)
        harness._stop.set()
    signal.signal(signal.SIGINT, _on_int)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_int)

    try:
        if args.mode == "observe":
            harness.run_observe()
        elif args.mode == "swap-cpu01":
            harness.run_swap_cpu01()
        elif args.mode == "dual":
            harness.run_dual()
        else:
            raise SystemExit(f"unknown mode {args.mode}")
    except KeyboardInterrupt:
        harness._stop.set()
        harness._finalize()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
