# Pearlhash shim — operator notes

This document covers the partial Pearlhash client added to `pearl-stratum` in
the wave-16 push. Read this before running anything against the live pool
(`84.32.220.219:9000`).

## What's implemented

The `pearl_stratum.pearlhash_*` modules speak just enough of the proprietary
Pearlhash protocol to log in and emit hashrate keepalives. The cipher RE is
documented in [`51_pearlhash_cipher_re.md`](../../../pearl-investigation/wave16-domination/51_pearlhash_cipher_re.md);
implementation is a faithful Python port of §2.4.

| Module | Purpose |
|---|---|
| `pearlhash_keys.py` | 16-entry C->S keystream table (frame indices 0..15) |
| `pearlhash_cipher.py` | XOR encrypt/decrypt, known-plaintext recovery helper |
| `pearlhash_framing.py` | ASCII-hex + LF wire framing |
| `pearlhash_rand.py` | Pure-Python glibc `srand(0)`/`rand()` reimplementation |
| `pearlhash_client.py` | Asyncio client (login + keepalive loop, no submit yet) |

## Running the client

The package isn't yet wired into the `pearl-stratum` CLI (intentionally — see
"missing pieces" below). Manual test path:

```python
import asyncio
from pearl_stratum.pearlhash_client import PearlhashClient

async def main():
    client = PearlhashClient(
        host="84.32.220.219",
        port=9000,
        wallet="prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg",
        # ^ decoy wallet — use this for bake tests, NOT production wallet
    )
    client.set_hashrate(22.0e12)  # report 22 TH/s to the pool
    rc = await client.run()
    print("client exited rc=", rc)

asyncio.run(main())
```

After ~75 seconds the client will exhaust its keystream table (15 keepalive
frames at 5s cadence) and start logging "keystream not available, dropping
frame". The TCP socket stays open; the pool currently treats this as silent.
When pearl-miner v2 reconnects every few hours, the frame index resets — the
same shim behavior would apply (we'd reconnect on circuit-breaker, restart at
index 0, run 75s, drop, repeat).

## What's MISSING for end-to-end deploy

1. **Submit-frame keystream** — `submit_share()` raises
   `SubmitNotYetSupportedError`. Without this, the shim CANNOT credit shares
   to the wallet. This is the critical gap.
2. **Server -> client (S->C) keystreams** — Five distinct S->C frame kinds
   (login resp A/B/C, set_difficulty/new_job, share_result). All unknown. The
   client receives raw ciphertext via `on_message("inbound_raw", inner)` and
   does NOT decode it; consumers can replay captures offline.
3. **Job dispatch** — Once S->C keystreams are recovered, we need to map them
   to `pearl_stratum.job.Job` shapes so the existing kernel pipeline can mine
   against Pearlhash jobs (currently it only consumes alphapool `mining.notify`
   payloads).
4. **CLI integration** — `cli.py` has no `--pool-protocol pearlhash` switch
   yet. Adding it requires the `MiningClient` protocol mentioned in memo 36
   §3.5; out of scope for this initial scaffold.
5. **Worker name handling** — pearl-miner v2 rejects `wallet.worker` syntax
   (memo 36 §1.6). If Pearlhash exposes worker labeling via another field, we
   don't yet know which one.

## How to capture a submit keystream

(Procedure from memo 51 §4.1.)

On a host with pearl-miner v2 installed (e.g. CPU01:/home/pearl-deploy/):

1. Drop the kernel-mode debugger / tampering hooks from prior RE sessions:
   `rm -f /tmp/pearlhash_re_v2/*_preload*.so`.
2. Start a tcpdump filter:
   `tcpdump -i any -w /tmp/pearlhash_submit.pcap 'host 84.32.220.219 and port 9000'`.
3. Start pearl-miner v2 against a **decoy wallet** at default Pearlhash diff:
   `./pearl-miner --host 84.32.220.219:9000 --user <DECOY_WALLET>`.
4. Wait until pearl-miner logs "Share found" or similar. With 22 TH/s on a
   single 4070 Ti SUPER the expected interval is ~1-5 min.
5. As soon as the first share is logged, dump the process heap:
   `cat /proc/$(pgrep pearl-miner)/maps` then `dd if=/proc/<pid>/mem ...` for
   the heap and anon-rwx regions (see `_dump_miner.py` for the exact procedure).
6. Stop tcpdump.
7. In the pcap, find the first C->S frame that is NOT 112 bytes (login) or
   123/124 bytes (keepalive). That's the submit frame.
8. In the heap dump, search for `"method":"submit"` — the surrounding bytes
   are the plaintext JSON. The id field should be 16 or higher (= the frame
   index we need to recover the key for).
9. Run `pearlhash_cipher.recover_keystream(plaintext, ciphertext_body)` — this
   yields the 48-byte KEY_MSG[N] for N = the submit frame index.
10. Add the recovered entry to `pearlhash_keys.KEY_MSG_HEX` and remove the
    `raise SubmitNotYetSupportedError` in `PearlhashClient.submit_share`.

Cost estimate: ~30 min if pearl-miner v2 finds a share quickly; up to 6 hours
if Pearlhash diff is unusually high that day. Use a decoy wallet so a botched
attempt doesn't leak production hashpower.

## How to capture an S->C keystream

Same procedure as submit, but look for the S->C frame matching the size table
in memo 36 §1.2 (37B login resp A, 67B login resp B, 525B initial job/config,
37B/38B keepalive ack). The plaintext templates are guessable from JSON-RPC
norm (`{"id":N,"result":...}` for ack frames). Each unique S->C kind needs
one paired capture to recover its keystream.

## Tests

```
pytest pearl-stratum/tests/test_pearlhash_cipher.py -v
pytest pearl-stratum/tests/test_pearlhash_framing.py -v
pytest pearl-stratum/tests/test_pearlhash_rand.py -v
```

All three modules are pure-Python; no GPU / CUDA / pearl-gateway deps.

## Risk / known-issues

* The cipher RE is empirical, not algorithmic. If pearl-miner v2 ever pushes a
  new binary with a different key schedule (or a different srand seed), the
  shim breaks silently — frame decoding produces garbage JSON and the pool
  drops us. Mitigation: periodic regression captures (weekly).
* glibc rand sequence is verified for `srand(0)` with the modern (2.34+)
  warmup count of 310 iterations. Older glibc used 313 or 344; if a future
  pearl-miner v2 build was linked against ancient glibc, we'd need to support
  both warmup counts.
* The login plaintext is built with `json.dumps(... separators=(",", ":"))`
  to match pearl-miner v2's compact JSON. Any divergence (whitespace, key
  ordering) might or might not be tolerated by the pool — untested.
