# Operations runbook — pearl-stratum-srv

On-call reference for the Pearl solo mining pool. Skim section 1 the first time;
afterwards jump straight to symptom-based section 4.

## 1. Topology snapshot

```
  rigs (36 × 4070 Ti SUPER)              CPU02 (pool host)
  ──────────────────────────             ───────────────────────────────────
  alpha-miner --pool                     pearld          :44108 P2P
    stratum+tcp://cpu02.lan:5566   ───►  oyster wallet   :44207 RPC
                                         pearl-stratum-srv :5566 stratum
                                                          :9101 /metrics + /health
```

| Service               | Where     | Listens                        | Logs                                  |
|-----------------------|-----------|--------------------------------|---------------------------------------|
| `pearld.service`      | CPU02     | `:44108` (P2P), `:44107` (RPC) | `journalctl -u pearld -f`             |
| `oyster.service`      | CPU02     | `:44207` (wallet RPC)          | `journalctl -u oyster -f`             |
| `pearl-stratum-srv`   | CPU02     | `:5566` (stratum), `:9101`     | `journalctl -u pearl-stratum-srv -f`  |
| `alpha-miner` (rigs)  | rigs      | egress only                    | `/var/log/mfarm/miner.log` per rig    |

Decoy wallet for benches/captures: `prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg`.
Production payout address: whatever `oyster-cli getnewaddress` returned during bootstrap (`/etc/pearl-stratum-srv/env` `PEARL_SRV_MINING_ADDRESS`).

## 2. Quick commands

```bash
# State
systemctl status pearl-stratum-srv
curl -sf http://cpu02:9101/health
curl -s  http://cpu02:9101/metrics | head -40
/opt/pearl/miner/pearl-stratum-srv/deploy/healthcheck.sh

# Logs
journalctl -u pearl-stratum-srv -n 100
journalctl -u pearl-stratum-srv -f
journalctl -u pearld -n 50

# Lifecycle
sudo systemctl restart pearl-stratum-srv      # ~5s outage, miners auto-reconnect
sudo systemctl stop pearl-stratum-srv         # graceful shutdown

# Block-find audit (last 24h)
journalctl -u pearl-stratum-srv --since "24 hours ago" | grep "BLOCK ACCEPTED"
oyster-cli getbalance                          # accrued coinbase
oyster-cli listunspent 100                     # mature (≥100 conf) outputs
```

## 3. Healthy steady-state numbers

After 5 minutes of normal operation against the full 36-rig fleet:

- `pearl_stratum_srv_connected_miners` ≈ **36**
- `pearl_stratum_srv_template_age_seconds` between **0–93** (Pearl observed block time)
- `pearl_stratum_srv_template_height` advances by **~1 every 60–100s**
- Per-worker `rate(pearl_stratum_srv_shares_total{outcome="accepted"}[2m])` ≈ **0.017/s** (~1 share/min at default `d=2^20`)
- `pearl_stratum_srv_shares_total{outcome="malformed"}` rate ≈ **0** (any sustained nonzero = page)
- `pearl_stratum_srv_shares_total{outcome="stale"}` rate ≈ **0.001/s/rig** (occasional, on chain advance)
- `pearl_stratum_srv_blocks_total{outcome="error"}` total = **0**

Expected block-find at 2.6 PH/s vs 10 EH/s network: **~1 block every 4 days mean, P95 13 days** (Poisson). See `pearl-investigation/CHAIN_PROJECTIONS.md` for the underlying math.

## 4. Symptom → cause → fix

### `/health` returns 503 "template age N seconds exceeds 60s"

**Cause**: pearld stopped returning `getblocktemplate` responses, or our long-poll fetcher is stuck.

**Diagnose**:
```bash
journalctl -u pearld -n 50            # pearld crashed? Out of disk? Reorg storm?
journalctl -u pearl-stratum-srv -n 50 | grep -E "RPC error|getblocktemplate"
curl -u $RPCUSER:$RPCPASS \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","id":1,"method":"getblockcount","params":[]}' \
     http://127.0.0.1:18334/         # node responds at all?
```

**Fix**:
1. If pearld is dead: `sudo systemctl restart pearld`. Stratum-srv will reset the longpollid and reconnect on the next poll cycle (≤2s).
2. If pearld is alive but stratum-srv is stuck: `sudo systemctl restart pearl-stratum-srv`. Miners will reconnect within ~5s.
3. If both healthy but symptom persists: **possible long-poll bug** — set `PEARL_SRV_LONG_POLL=false` in `/etc/pearl-stratum-srv/env` and restart. Falls back to 2s fixed polling.

### `pearl_stratum_srv_connected_miners == 0`

**Cause**: rigs can't reach us OR all rigs IDLE OR the listener crashed.

**Diagnose**:
```bash
ss -tlnp | grep 5566                          # are we listening?
# From a rig:
nc -vz cpu02.lan 5566                         # can the rig reach us?
journalctl -u pearl-stratum-srv | grep "conn .* open"   # recent connects?
```

**Fix**:
1. Listener down → check status, restart.
2. Network → confirm firewall, check mfarm flight sheet points to right host:port.
3. All rigs IDLE → see "PearlRigIdle" below.

### `PearlRigIdle` alert — a specific worker stopped submitting

**Cause** (from `idle_rig_recovery_2026_05_20`): GPU driver crash, alpha-miner crash, xmrig wrapper bug killing GPU, mfarm-agent on `<v0.3.0` bouncing the GPU when CPU config applies.

**Diagnose**:
```bash
# Identify the worker from the Grafana alert label, then:
ssh root@<rig-ip>
systemctl status alpha-miner mfarm-agent
nvidia-smi                                    # GPU alive?
tail -50 /var/log/mfarm/miner.log
```

**Fix**:
1. `nvidia-smi` shows GPU fallen off bus → physical reboot (memory: mini16/19/24 hit this).
2. alpha-miner crashed but service active → `systemctl restart alpha-miner`.
3. mfarm-agent <0.3.0 → upgrade (`mfarm deploy agent --version 0.3.0 <rig>`).
4. xmrig wrapper EXTRA_ARGS quoting bug → check `/opt/mfarm/miner-wrapper.sh` line 153 for unquoted `EXTRA_ARGS` in eval (memory: April-30 MeowOS image).

### `pearl_stratum_srv_blocks_total{outcome="error"}` incremented

**Cause**: We submitted a block, pearld errored (not a normal `rejected: low-difficulty`). This is rare and load-bearing — we may have lost a real block.

**Diagnose**:
```bash
journalctl -u pearl-stratum-srv | grep "BLOCK\|submit_block\|error submitting"
journalctl -u pearld | grep -i "submit\|orphan\|invalid"
```

**Fix**: Check `getblockcount` matches the network (blockbook.pearlresearch.ai/api/v2). If we're behind by 1 and the timestamp matches when our submit fired, the block was orphaned in a race — nothing to do, expected at our hashrate. If `pearld` reports `invalid` for our submitted block, that's a **bug in the share→block pipeline** (likely a pearl-gateway or pearl_mining version skew) — file an issue, do NOT restart blindly.

### Stale-share rate climbed across the fleet

**Cause**: chain reorg, OR long-poll is stuck delivering an old template.

**Diagnose**:
```bash
# Compare our template_height with the network tip.
curl -s http://cpu02:9101/metrics | grep template_height
curl -s https://blockbook.pearlresearch.ai/api/v2 | jq .blockbook.bestHeight
# If we're behind: reorg or stuck.
```

**Fix**: `sudo systemctl restart pearl-stratum-srv` (5s outage, fetcher state cleared).

### Malformed-share rate nonzero for a specific worker

**Cause**: alpha-miner version mismatch (rare; v1.4 → v1.5 protocol bump) OR garbled plain_proof OR a non-alpha-miner client connecting.

**Diagnose**:
```bash
# Find the conn id from logs, then trace back to source IP.
journalctl -u pearl-stratum-srv | grep "bad plain_proof"
# Audit miner version on the offending rig.
ssh root@<rig> 'cat /opt/mfarm/miner-version'
```

**Fix**: Upgrade rig to alpha-miner v1.5+ via mfarm. See `project_alpha_miner_v15_upgrade_2026_05_18`.

## 5. Rollback to alphapool

Single mfarm flight-sheet revert. Restores production pool URL on all rigs.

```bash
# In CatStack web UI: Flight Sheets → PEARL_alpha-miner_v2 → Apply to all
# OR via dual_flight_sheet.py pattern:
python C:/Source/dual_flight_sheet.py \
  --pool-url stratum+tcp://us2.alphapool.tech:5566 \
  --rigs all
```

mfarm-managed rigs: ~30 min wall-clock to push to all 31 via 8-way parallel SSH.

Kerrigan rigs (rig03, rig05) bypass mfarm-agent — `ssh + systemctl edit alpha-miner.service` manually, revert `--pool` arg, `systemctl daemon-reload && systemctl restart alpha-miner.service`.

**Rolled-back state is the pre-deploy baseline**: alpha-miner on alphapool with 5% pool fee + 3% dev-fee leakage. No code rollback needed in our repo; pearl-stratum-srv can keep running idle for re-flip.

## 6. Pre-rollout sanity check

Run before each fleet flip (e.g., after kernel updates, after pearld upgrades, after long uptime):

```bash
# 1. Refresh alphapool capture and diff against our baseline.
python tools/capture_alphapool.py \
  --pool us1.alphapool.tech:5566 \
  --out tests/fixtures/alphapool_capture_$(date +%Y_%m_%d).json
pytest tests/test_alphapool_parity.py::test_no_drift_in_latest_capture

# 2. Healthcheck loop for 5 min, expect zero failures.
for i in $(seq 1 30); do
  deploy/healthcheck.sh || echo "FAIL at iter $i"
  sleep 10
done

# 3. Confirm one rig still mines via current setup before flipping more.
```

If parity test fails → see "drift detected" in section 7.

## 7. Drift detected — what to do

The drift test catches the case where alphapool changed something on the wire that our pool doesn't mirror. If we deploy without matching, alpha-miner may reject our jobs or refuse to subscribe.

1. Read the pytest failure — it lists exactly which fields drifted.
2. Common drifts and their fixes:
   - **`rank` changed** (e.g., 128 → 256): mainnet consensus changed. Update `PEARL_SRV_PARAM_RANK` in env, but ALSO confirm the pearl-gemm kernel supports the new rank (registers may overflow on sm_89 — see `wave18_session` memo).
   - **`m`/`n`/`k` changed**: mainnet shape change. Update `PEARL_SRV_PARAM_M/N/K`.
   - **`mma_type` changed**: kernel ISA change. Likely needs a coordinated miner-side upgrade too.
   - **New field appeared (`+xxx:`)**: alphapool added something. Read pearl-stratum/test_stratum_dialogue.py for hints; likely needs adding to `Settings.mining_params_payload()`.
3. After updating: re-run `pytest tests/` until all green. Rotate the baseline:
   ```bash
   mv tests/fixtures/alphapool_capture_2026_05_18.json tests/fixtures/alphapool_capture_2026_05_18.legacy.json
   mv tests/fixtures/alphapool_capture_NEW.json tests/fixtures/alphapool_capture_2026_05_18.json
   # (or update _BASELINE_PATH in test_alphapool_parity.py to point at the new file)
   ```

## 8. Known limitations

- **No vardiff**: every share returns `result:true` regardless of difficulty. Fine for solo (we only care about blocks); would need vardiff for multi-user mode.
- **No `pearl.challenge`**: skipped because LAN-only, no DDoS exposure. If you expose `:5566` externally, implement per `wave16-domination/58_pearl_challenge_protocol.md`.
- **No TLS**: front with stunnel/Caddy for external exposure.
- **Single-host**: no HA, no failover pearld. If CPU02 goes down, the pool goes down and rigs need a flight-sheet flip back to alphapool. Standby plan: deploy a passive instance on CPU01 with the same env, flip flight sheet on disaster.
- **Pearl-gateway dependency**: we reuse `BlockTemplate.from_get_block_template`, `SubmissionService`, `ProofGenerator`. Upstream changes there can break us; pin versions in `pyproject.toml` for production deploys.

## 9. Where things live

```
/opt/pearl/miner/pearl-stratum-srv/      repo checkout (read-only at runtime)
/etc/pearl-stratum-srv/env               RPC creds + mining address (mode 0640)
/var/log/pearl-stratum-srv/              (reserved; we log to journald)
/etc/systemd/system/pearl-stratum-srv.service   the unit
```

Config defaults (override in `/etc/pearl-stratum-srv/env`):
```
PEARL_SRV_LISTEN_HOST=0.0.0.0           PEARL_SRV_LISTEN_PORT=5566
PEARL_SRV_METRICS_HOST=0.0.0.0          PEARL_SRV_METRICS_PORT=9101
PEARL_SRV_POLL_INTERVAL=2.0             PEARL_SRV_LONG_POLL=true
PEARL_SRV_LONG_POLL_TIMEOUT_SECS=30.0   PEARL_SRV_JOB_HISTORY_SIZE=16
PEARL_SRV_METRICS_MAX_TEMPLATE_AGE_SECONDS=60.0
PEARL_SRV_DEBUG_VERIFY=false            (mainnet params: rank=128, mma=Int7xInt7ToInt32)
```
