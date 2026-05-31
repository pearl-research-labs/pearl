"""Operator CLI: review + execute pending payouts.

Manual-by-design. We do NOT auto-send from the daemon — wrong code drains the
wallet. This tool:

  1. Reads `pending` payouts from the share_db
  2. Prints a human-reviewable table (recipient, amount, share count)
  3. Prompts for confirmation
  4. On confirm, calls `prlctl --wallet sendmany` with the bulk payout
  5. Marks each payout `sent` with the resulting txid

Usage:
    sudo -u pearl /opt/pearl-pool-venv/.venv/bin/python -m pearl_stratum_srv.payouts_send \\
        --db /var/lib/pearl-stratum-srv/shares.sqlite3 \\
        --pearld-rpc-url https://127.0.0.1:44107 \\
        --rpc-user pearlrpc \\
        --rpc-password "$(awk -F= '/RPC_PASS/{print $2}' /etc/pearl-stratum-srv/env)" \\
        --wallet-rpc-url https://127.0.0.1:44207 \\
        --rpc-cert /var/lib/oyster/rpc.cert \\
        --wallet-passphrase "$(jq -r .PrivatePassphrase /etc/pearl-stratum-srv/wallet-seed.json)"

Add --execute to actually send; without it the tool runs in dry-run mode and
just prints what it WOULD send.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp


async def _wallet_call(url: str, user: str, password: str, method: str, params: list) -> dict:
    """Single JSON-RPC call to oyster wallet. ssl=False because LAN + self-signed."""
    auth = aiohttp.BasicAuth(user, password)
    conn = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(auth=auth, connector=conn) as s:
        async with s.post(url, json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1}) as r:
            data = await r.json()
            if data.get("error"):
                raise RuntimeError(f"wallet RPC error: {data['error']}")
            return data["result"]


def _format_pearl(sats: int) -> str:
    """sats → human-readable PRL (8 decimals)."""
    return f"{sats / 100_000_000:.8f}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Review + execute pending pool payouts")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--wallet-rpc-url", required=True)
    parser.add_argument("--rpc-user", required=True)
    parser.add_argument("--rpc-password", required=True)
    parser.add_argument("--wallet-passphrase", required=True)
    parser.add_argument("--unlock-secs", type=int, default=60,
                        help="seconds to unlock wallet for the send (default 60)")
    parser.add_argument("--execute", action="store_true",
                        help="actually send the payouts (default: dry-run)")
    parser.add_argument("--max-batch", type=int, default=200,
                        help="max recipients per sendmany call")
    args = parser.parse_args()

    # Import here so unit tests can import the module without share_db deps.
    from pearl_stratum_srv.share_db import ShareDb

    async with ShareDb(args.db) as db:
        pending = await db.pending_payouts()

    if not pending:
        print("No pending payouts.", file=sys.stderr)
        return

    # Group: payout_id → recipient → amount. Aggregate same-recipient rows.
    by_recipient: dict[str, int] = {}
    ids_by_recipient: dict[str, list[int]] = {}
    total_sats = 0
    for (pid, recipient, amount, _share_count, _created) in pending:
        by_recipient[recipient] = by_recipient.get(recipient, 0) + amount
        ids_by_recipient.setdefault(recipient, []).append(pid)
        total_sats += amount

    print(f"\nPending payouts: {len(by_recipient)} recipients, {len(pending)} entries, total {_format_pearl(total_sats)} PRL")
    print("-" * 90)
    print(f"{'Recipient':<70} {'Amount (PRL)':>18}")
    print("-" * 90)
    for r, a in sorted(by_recipient.items(), key=lambda x: -x[1]):
        print(f"{r:<70} {_format_pearl(a):>18}")
    print("-" * 90)
    print(f"{'TOTAL':<70} {_format_pearl(total_sats):>18}\n")

    if not args.execute:
        print("DRY RUN — pass --execute to actually send.", file=sys.stderr)
        return

    confirm = input("Type 'SEND' to execute these payouts: ").strip()
    if confirm != "SEND":
        print("Aborted.", file=sys.stderr)
        return

    # Batch into sendmany calls of at most max_batch recipients each.
    items = list(by_recipient.items())
    async with ShareDb(args.db) as db:
        # Unlock wallet for the duration of the send.
        await _wallet_call(
            args.wallet_rpc_url, args.rpc_user, args.rpc_password,
            "walletpassphrase", [args.wallet_passphrase, args.unlock_secs],
        )
        try:
            for i in range(0, len(items), args.max_batch):
                batch = items[i:i + args.max_batch]
                # oyster sendmany expects {address: amount_PRL, ...} as string-keyed object.
                amounts = {r: float(_format_pearl(a)) for r, a in batch}
                print(f"Sending batch {i // args.max_batch + 1} ({len(batch)} recipients)...")
                try:
                    txid = await _wallet_call(
                        args.wallet_rpc_url, args.rpc_user, args.rpc_password,
                        "sendmany", ["", amounts, 1, ""],
                    )
                    print(f"  → txid {txid}")
                    # Mark all payout rows in this batch as sent.
                    for r, _ in batch:
                        for pid in ids_by_recipient[r]:
                            await db.mark_payout_sent(pid, txid)
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
                    break
        finally:
            # Re-lock wallet immediately.
            try:
                await _wallet_call(
                    args.wallet_rpc_url, args.rpc_user, args.rpc_password,
                    "walletlock", [],
                )
            except Exception:
                pass


def cli():  # pragma: no cover
    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    cli()
