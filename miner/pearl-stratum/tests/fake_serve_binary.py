#!/usr/bin/env python3
"""A tiny fake serve-mode binary, standing in for the GPU `pearl_miner_sm89`.

It exists so the ServeLoop's REAL Popen + write + flush + pipe delivery can be
exercised locally with no GPU and no ssh. It mirrors the proven binary contract:

  * prints `serve: ready` to STDERR once at startup (then flushes);
  * loops reading STDIN lines (this is exactly the path the asyncio transport
    failed to deliver to);
  * on a `JOB <header_hex> <target_hex>` line, prints `serve: new JOB` to stderr
    and IMMEDIATELY emits one `HIT {json}` to STDOUT — with the SAME echoed
    `header` the real binary returns — then flushes stdout so the parent's HIT
    reader thread sees it at once.

If the parent's stdin write never flushed (the bug), this process would block in
`readline()` forever and emit ZERO HITs — which is precisely what was observed
live. So a HIT arriving here is direct proof the flush delivery works.
"""

import json
import sys


def main() -> int:
    sys.stderr.write("serve: ready\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("JOB "):
            continue
        parts = line.split(" ")
        if len(parts) != 3:
            continue
        header_hex = parts[1]
        sys.stderr.write("serve: new JOB\n")
        sys.stderr.flush()

        # Emit one HIT echoing the job header (the real binary's contract).
        hit = {
            "nonce": 7,
            "seed": 123,
            "tile": [1, 2],
            "a_rows": [0] * 8,
            "b_cols": [0] * 16,
            "transcript": ["00000000"] * 16,
            "gpu_hash": "00" * 32,
            "header": header_hex,
        }
        sys.stdout.write("HIT " + json.dumps(hit) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
