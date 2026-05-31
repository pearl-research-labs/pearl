"""Live-capture script: connect to a real Pearl pool, record what it pushes,
write it as a JSON fixture our parity test can consume.

Run this from a Linux deploy box where pearl-stratum is installed (you need
the AVX-512 SIMD pearl.challenge solver compiled in; without it the v1.5
DDoS-handshake takes ~40 min in pure Python).

Usage:
    cd C:/Source/pearl/miner/pearl-stratum-srv
    python tools/capture_alphapool.py \\
        --pool us1.alphapool.tech:5566 \\
        --wallet prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg \\
        --worker capture-bot \\
        --out tests/fixtures/alphapool_capture_$(date +%Y_%m_%d).json

CAUTION
  - Hammering us2 from the same IP that runs production rigs can trigger
    pool-side rate-limit/IP-ban (see memory entry
    "Don't bench on production rigs"). Use us1 for capture and a non-production
    rig/IP if possible.
  - The decoy wallet from session memory is the canonical capture wallet —
    don't use a production wallet here.
  - This script makes ZERO mining.submit calls; it only subscribes, captures
    pushed frames for `capture_secs`, then disconnects.

After capture, update the test fixture path constant in
`tests/test_alphapool_parity.py` to the new file (or symlink the new file as
the canonical name) and re-run pytest to confirm parity still holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    # pearl-stratum's existing client has the framing + pearl.challenge solver
    # already wired (and uses the C SIMD solver when available).
    from pearl_stratum.stratum_client import StratumClient
except ImportError as e:
    print(
        f"pearl-stratum not importable: {e}\n"
        "Run this from a box where pearl-stratum is installed (uv sync in "
        "C:/Source/pearl/miner/pearl-stratum).",
        file=sys.stderr,
    )
    sys.exit(2)


_LOGGER = logging.getLogger("capture")


class CapturingClient:
    """Thin wrapper that uses StratumClient to do the handshake + challenge,
    then records all pushed frames (notify, set_difficulty, set_mining_params)
    for `capture_secs`."""

    def __init__(self, pool: str, wallet: str, worker: str, capture_secs: float):
        self.pool = pool
        self.wallet = wallet
        self.worker = worker
        self.capture_secs = capture_secs
        self.captured: dict[str, Any] = {
            "capture_metadata": {
                "source": pool,
                "captured_via": "pearl-stratum-srv/tools/capture_alphapool.py",
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "decoy_wallet": wallet,
                "worker": worker,
                "capture_secs": capture_secs,
                "notes": [
                    "Captured by pearl-stratum-srv/tools/capture_alphapool.py.",
                    "Drives a real subscribe/authorize handshake and records",
                    "all pool-pushed frames seen during the capture window.",
                ],
            },
            "subscribe_response_template": None,
            "set_mining_params_push": None,
            "notify_sample": None,
            "set_difficulty_push": None,
            "all_pushed_frames": [],
        }

    async def run(self) -> None:
        client = StratumClient(
            pool_url=self.pool,
            wallet=self.wallet,
            worker=self.worker,
            on_notify=self._on_notify,
            on_set_difficulty=self._on_set_difficulty,
            on_set_mining_params=self._on_set_mining_params,
        )
        # Hook the raw frame stream so we can capture EVERY push, including the
        # subscribe response and any methods we didn't anticipate.
        original_dispatch = client._dispatch_frame  # type: ignore[attr-defined]

        def wrapped_dispatch(frame: dict) -> None:
            self._record_raw(frame)
            return original_dispatch(frame)

        client._dispatch_frame = wrapped_dispatch  # type: ignore[attr-defined]

        task = asyncio.create_task(client.run(), name="stratum-client")
        try:
            await asyncio.sleep(self.capture_secs)
        finally:
            await client.stop()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()

    def _record_raw(self, frame: dict) -> None:
        # Strip any session-identifying tag from subscription IDs so the
        # captured fixture is comparable across sessions.
        scrubbed = self._scrub(frame)
        self.captured["all_pushed_frames"].append(scrubbed)
        method = scrubbed.get("method")
        if method == "pearl.set_mining_params" and self.captured["set_mining_params_push"] is None:
            self.captured["set_mining_params_push"] = scrubbed
        elif method == "mining.notify" and self.captured["notify_sample"] is None:
            self.captured["notify_sample"] = scrubbed
        elif method == "mining.set_difficulty" and self.captured["set_difficulty_push"] is None:
            self.captured["set_difficulty_push"] = scrubbed
        elif scrubbed.get("result") is not None and self.captured["subscribe_response_template"] is None:
            # First reply we see is the subscribe response.
            self.captured["subscribe_response_template"] = scrubbed

    @staticmethod
    def _scrub(frame: dict) -> dict:
        """Replace session-tagged subscription IDs with the placeholder used
        in the canonical fixture, so diffs across sessions are clean."""
        out = json.loads(json.dumps(frame))  # deep copy
        result = out.get("result")
        if isinstance(result, list) and len(result) >= 1 and isinstance(result[0], list):
            for row in result[0]:
                if isinstance(row, list) and len(row) >= 2:
                    row[1] = "<session-tag>"
        return out

    async def _on_notify(self, *a, **k): pass
    async def _on_set_difficulty(self, *a, **k): pass
    async def _on_set_mining_params(self, *a, **k): pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Capture alphapool wire frames into a JSON fixture")
    parser.add_argument("--pool", default="us1.alphapool.tech:5566", help="pool host:port")
    parser.add_argument(
        "--wallet",
        default="prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg",
        help="capture wallet (DEFAULT = canonical decoy; do not use production wallet)",
    )
    parser.add_argument("--worker", default="capture-bot", help="worker name suffix")
    parser.add_argument("--capture-secs", type=float, default=60.0, help="capture window in seconds")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output JSON path (e.g. tests/fixtures/alphapool_capture_2026_05_20.json)",
    )
    args = parser.parse_args()

    if "prl1pja266" in args.wallet:
        print("REFUSING: that looks like a production wallet. Use the decoy.", file=sys.stderr)
        sys.exit(2)

    cap = CapturingClient(args.pool, args.wallet, args.worker, args.capture_secs)
    asyncio.run(cap.run())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cap.captured, indent=2))
    _LOGGER.info(
        "captured %d frames (notify=%s, params=%s, diff=%s) → %s",
        len(cap.captured["all_pushed_frames"]),
        bool(cap.captured["notify_sample"]),
        bool(cap.captured["set_mining_params_push"]),
        bool(cap.captured["set_difficulty_push"]),
        args.out,
    )


if __name__ == "__main__":
    main()
