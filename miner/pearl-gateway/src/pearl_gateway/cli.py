#! /usr/bin/env python3
"""CLI for PearlGateway."""

import argparse
import asyncio
import os

from miner_utils import get_logger

from pearl_gateway.pearl_gateway import PearlGateway

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PearlGateway - Pearl Mining Proxy")
    parser.add_argument("command", choices=["start"], help="Command to execute")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("MINER_DEBUG", "").lower() in ("1", "true", "yes"),
        help="Enable debug mode (default: from MINER_DEBUG env var)",
    )
    return parser.parse_args()


async def start_gateway(debug: bool = False) -> None:
    logger.debug("Starting PearlGateway...")
    gateway = PearlGateway(debug_mode=debug)
    await gateway.start()
    print("PearlGateway is running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await gateway.stop()


def main() -> None:
    args = parse_args()
    asyncio.run(start_gateway(args.debug))


if __name__ == "__main__":
    main()
