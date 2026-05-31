"""Entrypoint: `pearl-stratum-srv`.

Usage (env-driven; no flags):
  export PEARL_SRV_RPC_URL=http://127.0.0.1:18334
  export PEARL_SRV_RPC_USER=rpcuser
  export PEARL_SRV_RPC_PASSWORD=rpcpass
  export PEARL_SRV_MINING_ADDRESS=prl1...
  pearl-stratum-srv

Optional:
  PEARL_SRV_LISTEN_HOST   default 0.0.0.0
  PEARL_SRV_LISTEN_PORT   default 5566
  PEARL_SRV_POLL_INTERVAL default 2.0
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.server import PoolServer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    _setup_logging()
    try:
        settings = Settings()  # reads env
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)

    server = PoolServer.from_settings(settings)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
