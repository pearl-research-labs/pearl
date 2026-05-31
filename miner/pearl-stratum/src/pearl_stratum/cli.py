"""`pearl-stratum` entrypoint.

Connects to a pool, bootstraps `SharedState`, and (optionally) launches the
miner subprocess. Safety: refuses to connect if `--address` isn't in the
`--allow-wallet` whitelist file. No production wallet is ever defaulted.

Examples:

    pearl-stratum \\
        --pool stratum+tcp://us2.alphapool.tech:5566 \\
        --address prl1.....                              # DECOY \\
        --worker rig04-stratum \\
        --password 'x;d=1048576' \\
        --device 0 \\
        --allow-wallet C:/Source/pearl-investigation/akoya_decoy_wallet.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .gateway_shim import init_shared_state, reset_shared_state
from .stratum_client import StratumClient, default_worker_name, parse_pool_url

logger = logging.getLogger("pearl_stratum")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pearl-stratum",
        description="Stratum-pool shim for the Pearl miner.",
    )
    p.add_argument(
        "--pool", required=True,
        help="Pool URL: stratum+tcp://host:port or host:port.",
    )
    p.add_argument(
        "--address", required=True,
        help="Pearl wallet address to authorize as. Must be in --allow-wallet.",
    )
    p.add_argument(
        "--worker", default=None,
        help="Worker subname (default: $HOSTNAME).",
    )
    p.add_argument(
        "--password", default="x",
        help="Stratum auth password. Use 'x;d=NN' for static diff hint. Default: 'x'.",
    )
    p.add_argument(
        "--user-agent", default="pearl-stratum/0.1",
        help="UA string sent on mining.subscribe.",
    )
    p.add_argument(
        "--allow-wallet", required=True,
        help=(
            "Path to a whitelist file (one address per line, # for comments). "
            "--address must match one of these. NO DEFAULT — must be passed explicitly."
        ),
    )
    p.add_argument(
        "--device", type=int, default=0,
        help="GPU index passed to the miner subprocess (when --spawn-miner is set).",
    )
    p.add_argument(
        "--spawn-miner", default=None,
        help=(
            "Optional path to a miner binary; if set, pearl-stratum execs it as "
            "a child process with env vars set so it'll attach to this stratum "
            "client. If unset, pearl-stratum just runs the stratum loop and the "
            "miner subprocess is the caller's problem."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def load_whitelist(path: str) -> set[str]:
    """Parse a wallet whitelist file. Strips comments and blank lines."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--allow-wallet file does not exist: {path}")
    entries: set[str] = set()
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    if not entries:
        raise SystemExit(f"--allow-wallet file is empty (no allowed addresses): {path}")
    return entries


def assert_wallet_allowed(address: str, allowed: set[str]) -> None:
    if address not in allowed:
        # Display only a prefix in the error to avoid logging full addresses.
        prefix = address[:8] + "..." if len(address) > 12 else address
        raise SystemExit(
            f"address {prefix!r} is NOT in --allow-wallet whitelist. Refusing to connect. "
            f"Add it to the whitelist file if intentional."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    allowed = load_whitelist(args.allow_wallet)
    assert_wallet_allowed(args.address, allowed)

    host, port = parse_pool_url(args.pool)
    worker = args.worker if args.worker is not None else default_worker_name()

    client = StratumClient(
        host=host,
        port=port,
        address=args.address,
        worker=worker,
        password=args.password,
        user_agent=args.user_agent,
    )
    state = init_shared_state(client)

    logger.info(
        "pearl-stratum starting: pool=%s:%d worker=%s.%s device=%d",
        host, port, args.address[:8] + "...", worker, args.device,
    )

    # Wait for the first job before doing anything else so callers downstream
    # see a populated MiningJob immediately.
    if not state.wait_for_first_job(timeout=60.0):
        logger.error("No mining.notify received in 60s; aborting")
        reset_shared_state()
        return 3

    miner_proc = None
    if args.spawn_miner:
        # Pass the device index in env; the miner is expected to use the shim
        # via the in-process SharedState (so this only makes sense if the
        # miner is a Python subprocess that imports pearl_stratum). For an
        # external native miner you'd need a different bridging strategy.
        import subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        env["PEARL_STRATUM_ACTIVE"] = "1"
        miner_proc = subprocess.Popen([args.spawn_miner], env=env)
        logger.info("Spawned miner subprocess pid=%d", miner_proc.pid)

    stop_event = asyncio.Event()

    def _request_stop(signum: int, _frame: object) -> None:
        logger.info("Caught signal %d; shutting down", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
    except (ValueError, AttributeError):
        # signal.signal isn't always available off the main thread or on Windows
        # for SIGTERM. Best-effort.
        pass

    try:
        while not stop_event.is_set():
            if miner_proc is not None and miner_proc.poll() is not None:
                rc = miner_proc.returncode
                logger.warning("Miner subprocess exited rc=%d; tearing down", rc)
                break
            try:
                import time as _t
                _t.sleep(1.0)
            except KeyboardInterrupt:
                break
    finally:
        if miner_proc is not None and miner_proc.poll() is None:
            miner_proc.terminate()
            try:
                miner_proc.wait(timeout=10)
            except Exception:
                miner_proc.kill()
        reset_shared_state()

    return 0


if __name__ == "__main__":
    sys.exit(main())
