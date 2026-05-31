# pearl-stratum-srv

Solo Pearl mining pool. Stratum-v1 TCP listener that translates the alpha-miner
wire protocol into calls against a local `pearld` node, so the fleet can mine
direct to one wallet without going through `us2.alphapool.tech` (no pool fee,
no alpha-miner dev-fee leakage).

## Design

```
  pearld (CPU02)  ──RPC──►  PearlNodeClient ◄──── pearl-stratum-srv
                                                       │
                                                       │ stratum :5566
                                                       ▼
                                                36 alpha-miner rigs
```

Reuses three services from `pearl-gateway`:

- `PearlNodeClient` — `getblocktemplate` / `submitblock` over HTTPS
- `WorkCache` — caches the latest template
- `SubmissionService` — wraps `pearl_mining.generate_proof` + `submitblock`

The only new code here is the **stratum protocol layer** (`protocol.py`,
`connection.py`, `server.py`, `job_registry.py`). It speaks the exact wire
format `pearl-stratum` (our client shim) was built against — see
`C:/Source/pearl-investigation/STRATUM_CAPTURE.md` for the capture this
mirrors.

## Run

```bash
export PEARL_SRV_RPC_URL=http://127.0.0.1:18334
export PEARL_SRV_RPC_USER=rpcuser
export PEARL_SRV_RPC_PASSWORD=rpcpass
export PEARL_SRV_MINING_ADDRESS=prl1...  # from `oyster-cli getnewaddress`
pearl-stratum-srv
```

## Implemented stratum methods

| Method                | Direction | Behavior |
|-----------------------|-----------|----------|
| `mining.configure`    | C→S       | ack `{"pearl/v1": true, "pearl/v1.share_format": "base64"}` |
| `mining.subscribe`    | C→S       | empty extranonce + push `pearl.set_mining_params` + push `set_difficulty 1` + push current `notify` |
| `mining.authorize`    | C→S       | always `true` (LAN trusted) |
| `mining.submit`       | C→S       | decode b64 plain_proof; hand to `SubmissionService`; ack `true` always (block-find is implicit in coinbase) |
| `mining.notify`       | S→C       | pushed on every new chain tip with `clean_jobs=true` |
| `mining.set_difficulty` | S→C     | pushed once at subscribe time, value 1 (we accept all shares) |
| `pearl.set_mining_params` | S→C   | pushed once at subscribe with mainnet `{m,n,k,rank,patterns,mma_type}` |

## Template fetching: long-poll by default

`getblocktemplate` is called via pearld's **long-poll handshake** — pearld
holds the request open until the chain tip advances (or `long_poll_timeout_secs`
elapses, default 30s), at which point it returns immediately. This drops the
stale-share window on chain advance from up to `poll_interval` (~2s default in
poll-mode) to <100ms.

Falls back to fixed-interval polling on RPC error (and resets the `longpollid`
so the next call doesn't hang on a stale handle). Set `PEARL_SRV_LONG_POLL=false`
to force the old fixed-interval behavior.

## Observability

`GET http://<host>:9101/metrics` — Prometheus text exposition:

```
pearl_stratum_srv_connected_miners              # gauge
pearl_stratum_srv_template_age_seconds          # gauge (alert if > 60)
pearl_stratum_srv_template_height               # gauge
pearl_stratum_srv_jobs_in_registry              # gauge
pearl_stratum_srv_shares_total{worker,outcome}  # counter — per-worker liveness
pearl_stratum_srv_blocks_total{outcome}         # counter
```

`GET http://<host>:9101/health` — 200 OK if template age < 60s, else 503. Wire
this into `systemd` unit healthcheck or external uptime monitor.

The per-worker `shares_total{worker=...}` series catches IDLE rigs: scrape
every 15s, alert if any worker's accepted-share rate drops to 0 for >120s. This
is the same failure mode `project_idle_rig_recovery_2026_05_20` describes —
when a rig's GPU FS dies the alpha-miner stops submitting but the mfarm-agent
reports green.

## NOT implemented (intentional)

- **`pearl.challenge`** — alphapool's v1.5 DDoS-pacing handshake. Skipped because
  this server is for trusted LAN clients only. Add later via the captured
  algorithm in `C:/Source/pearl-investigation/wave16-domination/58_pearl_challenge_protocol.md`
  if you ever expose the listener publicly.
- **vardiff** — not needed for solo. Every share returns `true`; only blocks
  matter, and the `nbits` in `mining.notify` already carries the network target.
- **TLS** — add `stunnel` or Caddy in front if exposing externally.
- **Per-user accounting / payouts** — solo mode; coinbase pays one address.

## Tests

```bash
cd C:/Source/pearl/miner/pearl-stratum-srv
uv run pytest tests/
```

46 tests covering: wire framing, job registry semantics, end-to-end TCP
handshake/submit/notify, **byte-for-byte alphapool parity** (against captured
`pearl.set_mining_params` and `mining.notify` frames from
`STRATUM_CAPTURE.md`), and the Prometheus/health endpoint. Runs in ~0.15s
with no pearld dep — all external services are stubbed in `conftest.py`.

Refresh the alphapool capture fixture from a live pool with:

```bash
python tools/capture_alphapool.py --pool us1.alphapool.tech:5566 \
    --out tests/fixtures/alphapool_capture_$(date +%Y_%m_%d).json
```

## Deploy on CPU02

```bash
sudo /opt/pearl/miner/pearl-stratum-srv/deploy/bootstrap_cpu02.sh
# edit /etc/pearl-stratum-srv/env to set rpcpass + mining address
sudo systemctl enable --now pearl-stratum-srv
deploy/healthcheck.sh   # → "OK: N miners, height=H, template_age=Ts"
```

`bootstrap_cpu02.sh` is idempotent: creates the `pearl` user, sets up the env
file, `uv sync`s deps, runs the test suite as a smoke check, installs the
systemd unit. Run the healthcheck from cron/systemd-timer for ongoing
monitoring.

### Prometheus + Grafana

- Merge `deploy/prometheus_scrape.yml` into your Prometheus config (or use it as a starting point). Includes commented-out alerting rules for the four failure modes that matter: template stale, no miners, rig IDLE, block-submit error.
- Import `deploy/grafana_dashboard.json` in Grafana (Dashboards → New → Import). The per-worker share-rate panel is the load-bearing "which rig is IDLE?" view; the template-age gauge surfaces pearld going dark; block-find annotations turn block discoveries into pinned vertical lines on every chart.

## Rollout

See `C:/Source/pearl-investigation/PEARL_WIKI.md` for the canonical
flight-sheet swap procedure. TL;DR: pilot on CPU01 (one rig, 30 min),
canary rig04 (6 h vs rig03/05 A/B), then `dual_flight_sheet.py` pattern
to fleet. Kerrigan rigs (rig03/05) need manual `systemctl edit` because
they bypass mfarm-agent. Rollback = one flight-sheet revert API call.
