"""``pearl-stratum-proxy`` entrypoint.

Bind a loopback listener and forward to the configured upstream pool.

Example::

    pearl-stratum-proxy \\
        --listen 127.0.0.1:5567 \\
        --upstream us2.alphapool.tech:5566

Then reconfigure alpha-miner with
``--pool stratum+tcp://127.0.0.1:5567`` and restart the miner.

See ``DEPLOY.md`` for the per-rig deployment runbook.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys


def _parse_hostport(spec: str, default_host: str = "127.0.0.1") -> tuple[str, int]:
    """Parse a ``host:port`` string.

    Accepts bare ``port`` (uses ``default_host``) and IPv6 ``[::1]:port``.
    """
    if not spec:
        raise argparse.ArgumentTypeError("empty host:port spec")
    if spec.startswith("["):
        # IPv6 with brackets: [::1]:5566
        try:
            host_end = spec.index("]")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"unterminated IPv6 literal: {spec}") from exc
        host = spec[1:host_end]
        rest = spec[host_end + 1 :]
        if not rest.startswith(":"):
            raise argparse.ArgumentTypeError(f"IPv6 spec missing port: {spec}")
        port_str = rest[1:]
    elif ":" in spec:
        host, port_str = spec.rsplit(":", 1)
    else:
        host, port_str = default_host, spec
    try:
        port = int(port_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port in {spec!r}") from exc
    if not (0 < port < 65536):
        raise argparse.ArgumentTypeError(f"port out of range in {spec!r}")
    return host, port


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pearl-stratum-proxy",
        description="Transparent stratum proxy that hides alpha-miner's reconnect bug.",
    )
    p.add_argument(
        "--listen",
        required=True,
        type=str,
        help="host:port to listen on (e.g. 127.0.0.1:5567).",
    )
    p.add_argument(
        "--upstream",
        required=True,
        type=str,
        help="Upstream pool host:port (e.g. us2.alphapool.tech:5566).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    listen_host, listen_port = _parse_hostport(args.listen)
    upstream_host, upstream_port = _parse_hostport(args.upstream, default_host="")
    if not upstream_host:
        print("--upstream requires host:port (got bare port)", file=sys.stderr)
        return 2

    # Import here to keep --help cheap.
    from .proxy import ProxyServer

    server = ProxyServer(
        listen_host=listen_host,
        listen_port=listen_port,
        upstream_host=upstream_host,
        upstream_port=upstream_port,
    )
    await server.start()

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logging.getLogger(__name__).info("shutdown signal — closing")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main thread doesn't support signal handlers.
            pass

    serve_task = asyncio.create_task(server.serve_forever(), name="proxy-serve")
    stop_task = asyncio.create_task(stop_event.wait(), name="proxy-stop")
    done, _pending = await asyncio.wait(
        {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for t in (serve_task, stop_task):
        if not t.done():
            t.cancel()
    await server.stop()
    for t in done:
        exc = t.exception() if not t.cancelled() else None
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            logging.getLogger(__name__).error("server task failed: %r", exc)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
