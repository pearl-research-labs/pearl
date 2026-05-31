# Pearl Stratum Proxy — Deployment Runbook

**Status: SKELETON ONLY. NOT APPROVED FOR PRODUCTION. Do NOT deploy until
a human has reviewed and explicitly authorised.**

## What this is

A transparent TCP proxy that hides alpha-miner's
`reconnect_drop_share` bug from the pool by:

- terminating each alpha-miner socket locally on `127.0.0.1:5567`,
- maintaining one long-lived TCP connection to the real pool per
  `worker.gpuN`, and
- swallowing the alpha-miner's spurious FIN on `error[21]` while
  re-attaching the next client connect to the same persistent upstream.

Expected gain: recover the **42% share loss** characterised in
`C:/Source/pearl-investigation/STRATUM_CAPTURE.md` §4.

## What this is NOT

- It is **not** a replacement for the alpha-miner binary or the
  pearl-gemm kernel. Mining still runs through the existing closed-source
  miner.
- It is **not** a new mining-pool. The wire dialect is forwarded
  byte-identically.
- It is **not** wired into the mfarm-agent stack. It runs as a separate
  systemd unit. Removing the unit and reverting one `--pool` flag
  restores the prior state with no other side effects.

## Files / footprint

- ~1 MB on-disk (pure Python, stdlib-only, no torch/CUDA).
- ~64 MB RSS for 6 upstream connections (one per GPU on a 6-GPU rig).
- 0 GPU memory used, 0 GPU SMs used.
- Single CPU core (asyncio is single-threaded).

## Pre-deploy checklist

1. **Code review:** human reviewer signs off on the proxy source.
2. **Local smoke test (NOT on a production rig):** install on
   CPU01 or CPU02 (idle hardware per
   `project_pearl_ab_test_2026_05_17.md`), point at decoy wallet on
   alphapool, verify alpha-miner submits shares end-to-end through
   the proxy for ≥ 30 minutes with no protocol errors.
3. **Verify rollback path works** by stopping the proxy unit and
   confirming alpha-miner reverts to direct pool connection in <60 s.
4. **Get explicit human authorisation** to enable on any production rig
   (rig03 / rig04 / rig05 / minis).

## Per-rig deploy steps (manual, one rig at a time)

### 1. Stage the package

```bash
# As root on the target rig
mkdir -p /opt/pearl-stratum-proxy
rsync -av --delete /path/to/pearl-stratum-proxy/ /opt/pearl-stratum-proxy/
python3.12 -m venv /opt/pearl-stratum-proxy/.venv
/opt/pearl-stratum-proxy/.venv/bin/pip install -e /opt/pearl-stratum-proxy
```

### 2. systemd unit

`/etc/systemd/system/pearl-stratum-proxy.service`:

```
[Unit]
Description=Pearl Stratum Proxy (hides alpha-miner reconnect bug)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=/opt/pearl-stratum-proxy/.venv/bin/pearl-stratum-proxy \
    --listen 127.0.0.1:5567 \
    --upstream us2.alphapool.tech:5566 \
    --log-level INFO
Restart=on-failure
RestartSec=2
LimitNOFILE=4096
# It only talks loopback for inbound and one outbound TCP host. No need
# for elevated privileges, broad capabilities, or filesystem write
# beyond /var/log.
DynamicUser=yes
ProtectSystem=strict
ProtectHome=true
NoNewPrivileges=true

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now pearl-stratum-proxy.service
journalctl -u pearl-stratum-proxy.service -f   # watch first 60 s
```

### 3. Cut alpha-miner over

Edit the alpha-miner flight-sheet (CatStack) for the target rig:

```
--pool stratum+tcp://us2.alphapool.tech:5566
```

becomes

```
--pool stratum+tcp://127.0.0.1:5567
```

Apply the flight sheet. The mfarm-agent will restart alpha-miner, which
will dial the local proxy instead of the pool directly.

### 4. Verify

Within ~5 min, on the rig:

```bash
journalctl -u pearl-stratum-proxy.service --since "5 min ago" | grep -E "client OPEN|upstream OPEN|cache WARM"
```

You should see:

- 6 × `client OPEN` (one per GPU)
- 6 × `upstream OPEN` (one per worker.gpuN)
- 6 × `cache WARM` after the first reconnect cycle (≤ 30 s typically)
- subsequent `client OPEN` lines (every reconnect) should NOT be paired
  with `upstream OPEN` — that's the proof the upstream is persisting.

Pool dashboard: hashrate should be at least flat versus the
pre-deploy baseline. If the bug-fix theory holds, +35-45% credited
hashrate per
`project_pearl_ab_test_2026_05_17.md` (58% effective rate baseline).

### 5. Acceptance criteria for ramp

Hold one rig on the proxy for **≥ 6 hours** before deciding to roll out
to a second rig. Compare pool-credited hashrate against
non-proxy rigs running identical configs (rig03 vs rig04 etc).
Acceptance:

- proxy rig pool-credit ≥ non-proxy rig pool-credit, **and**
- no `journalctl` ERROR lines, **and**
- ≤ 1 % share rejection beyond baseline noise (stale rejects should
  *fall*, but other reject codes must not appear).

If any of these fail, ROLL BACK before continuing.

## Rollback

```bash
# 1. Revert the alpha-miner flight sheet to direct pool URL.
#    (CatStack UI; or manually edit the on-rig config + restart mfarm-agent.)

# 2. Stop the proxy unit. (Doing this first while alpha-miner still
#    points at 127.0.0.1:5567 would just black-hole the miner.)
systemctl disable --now pearl-stratum-proxy.service

# 3. (Optional) Uninstall package
rm -rf /opt/pearl-stratum-proxy
rm /etc/systemd/system/pearl-stratum-proxy.service
systemctl daemon-reload
```

Rollback is order-sensitive: revert the flight sheet **first**, then
stop the proxy. Otherwise alpha-miner will fail to connect for the
window between proxy-stop and flight-sheet-apply.

## Known unknowns / risks

- **Cache invalidation on protocol change.** `pearl.set_mining_params`
  is empirically byte-identical across reconnects in the 60-s capture.
  If the pool ever rotates this payload (e.g. seasonal protocol
  upgrade), every replayed handshake would lie to alpha-miner. Mitigation:
  the proxy parses every upstream notification and updates the cache
  on the fly — but a single client cycle may use stale params. Add a
  pre-deploy check: if we see the pool send a *changed* set_mining_params
  vs the cached one, restart the alpha-miner socket on the next FIN
  cycle so it sees the fresh value.
- **Pool-side fingerprinting.** alphapool today binds session state
  to TCP-connection, not to worker name. If they ever start
  rate-limiting per-source-IP based on inferred reconnect cadence,
  our long-lived connections might look anomalous. Low risk per our
  capture (the pool reset zero connections in 60 s) but worth a check
  during the 6-hour single-rig hold.
- **Per-worker locking is exclusive.** If, during a reconnect race, two
  alpha-miner sockets ever co-exist for the same worker.gpuN, the second
  attach will be REJECTED with a synthetic error response. That's
  intentional — multiplexing onto a shared upstream would scramble
  share accounting. The cost is that one of the two sockets will
  re-attempt to connect after its synthetic error, eating one extra
  reconnect cycle.

## Where to look when it breaks

- `journalctl -u pearl-stratum-proxy.service` — proxy logs
- `/var/log/mfarm/miner.log` (alpha-miner) — miner-side view
- `ss -tnp | grep alphapool` — should show one ESTABLISHED per worker
  (i.e. 6 on a 6-GPU rig), not flapping
- `ss -tnp | grep 127.0.0.1:5567` — should show one ESTABLISHED per
  alpha-miner client thread (also 6); these *will* flap on every
  stale share, that's the bug we're masking

## Out-of-scope follow-ups (do not block deploy)

- TLS termination (alphapool doesn't speak TLS today).
- Prometheus metrics export (currently only journal logs).
- Multi-pool failover (one `--upstream` per process; failover via
  systemd unit swap if needed).
- Submit deduplication (alpha-miner doesn't double-submit; would be
  defence-in-depth, not a correctness fix).
