# prlmon — Pearl Network Monitor

A Prometheus monitoring sidecar for [pearld](../node/) nodes. Runs alongside each
node and exposes metrics, a diagnostic JSON API, and optional log tailing on
port `:9105`.

## Features

- **Node health**: reachability, peer counts, RPC responsiveness
- **Chain state**: tip height, tip age, difficulty
- **Network traffic**: bytes sent/received
- **Reorg detection**: frequency and depth via WebSocket block notifications
- **Diagnostic API**: aggregated `/node` snapshot plus peers, mempool, chaintips, and logs

## Build

From the repository root:

```bash
task build:prlmon
# or
go build -o bin/prlmon ./prlmon
```

## Run as a sidecar (binary)

Start pearld, then point prlmon at its RPC:

```bash
# Terminal 1 — pearld (example: simnet, no TLS)
./bin/pearld --simnet --notls \
  --rpcuser=monitor --rpcpass=monitor \
  --rpclisten=127.0.0.1:18556

# Terminal 2 — prlmon
./bin/prlmon \
  --rpchost=127.0.0.1:18556 \
  --rpcuser=monitor \
  --rpcpass=monitor \
  --notls \
  --listen=:9105 \
  --poll=10s \
  --node-log-file=/path/to/pearld/logs/simnet/pearld.log
```

With TLS (production), drop `--notls` and pass `--rpccert=/path/to/rpc.cert`.

Node identity labels (`node`, `region`, …) are applied by Prometheus at scrape
time, not by prlmon itself.

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--rpchost` | Yes | — | Host:port of the pearld RPC server |
| `--rpcuser` | Yes | — | RPC username |
| `--rpcpass` | Yes | — | RPC password |
| `--rpccert` | No* | — | Path to RPC TLS certificate (`rpc.cert`) |
| `--notls` | No | false | Disable TLS (development only) |
| `--listen` | No | `:9105` | HTTP listener for `/metrics` and the diagnostic API |
| `--poll` | No | `10s` | Polling interval for RPC calls |
| `--debuglevel` | No | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `--node-log-file` | No | — | Path to pearld log file; enables `/node/logs` |
| `--logs-max-lines` | No | `10000` | Max lines a single log request can return |
| `--self-log-buffer-lines` | No | `4096` | In-memory buffer size for `/logs` |

\*Required unless `--notls` is set.

## Run as a sidecar (Docker)

Images are published to `ghcr.io/pearl-research-labs/pearl/prlmon`.

```bash
docker run --rm -p 9105:9105 \
  -v /path/to/rpc.cert:/etc/node/rpc.cert:ro \
  -v /path/to/pearld/logs:/var/lib/pearld/logs:ro \
  ghcr.io/pearl-research-labs/pearl/prlmon:latest \
  --listen=:9105 \
  --rpchost=host.docker.internal:44109 \
  --rpcuser=monitor \
  --rpcpass="$RPC_PASS" \
  --rpccert=/etc/node/rpc.cert \
  --node-log-file=/var/lib/pearld/logs/mainnet/pearld.log
```

Or use the compose template that runs pearld, node-exporter, and prlmon together:

```bash
cd prlmon/deploy
export RPC_PASS=secret
# Place rpc.cert next to node-compose.yml
docker compose -f node-compose.yml up -d
```

See [`deploy/node-compose.yml`](deploy/node-compose.yml).

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | Prometheus metrics |
| `GET /healthz` | Health check (`ok`) |
| `GET /logs` | prlmon's own logs (in-memory ring buffer); `head`/`tail` |
| `GET /node` | Aggregated node snapshot (chain, peers, mempool, net, reorg) |
| `GET /node/peers` | Augmented `getpeerinfo` (`direction`, `subver`, `sort`, `limit`) |
| `GET /node/mempool` | Mempool size + bytes; `?top=N` for top-N by feerate |
| `GET /node/chaintips` | Passthrough of `getchaintips` |
| `GET /node/logs` | Stream pearld's active log (`head`/`tail`/`follow`; needs `--node-log-file`) |
| `GET /node/logs/files` | List active log + rotated archives |
| `GET /node/logs/files/<name>` | Download a single log file |

These endpoints are read-only and unauthenticated. Restrict access with a host
firewall or private network, the same way you would for `/metrics`.

```bash
curl -s http://localhost:9105/node | jq .
curl -s 'http://localhost:9105/node/logs?tail=200'
curl -s http://localhost:9105/metrics | grep prlmon_
```

## Metrics

All metrics use the `prlmon_` namespace. Attach identity labels in your Prometheus
scrape config (for example via `static_configs` / `file_sd_configs`).

| Metric | Type | Description |
|--------|------|-------------|
| `prlmon_node_up` | Gauge | Node reachability (1=up, 0=down) |
| `prlmon_chain_tip_height` | Gauge | Current chain tip height |
| `prlmon_chain_tip_age_seconds` | Gauge | Age of current tip in seconds |
| `prlmon_chain_difficulty` | Gauge | Current proof-of-work difficulty |
| `prlmon_blocks_connected_total` | Counter | Block connected events (WebSocket) |
| `prlmon_blocks_disconnected_total` | Counter | Block disconnected events (WebSocket) |
| `prlmon_reorg_total` | Counter | Reorg events |
| `prlmon_reorg_depth` | Histogram | Reorg depth distribution |
| `prlmon_p2p_peer_count` | Gauge | Connected peers |
| `prlmon_p2p_inbound_peers` | Gauge | Inbound peers |
| `prlmon_p2p_outbound_peers` | Gauge | Outbound peers |
| `prlmon_p2p_pingtime_seconds` | Histogram | Peer ping time |
| `prlmon_p2p_lastrecv_age_seconds` | Histogram | Age since last message from each peer |
| `prlmon_net_totalbytes_sent_total` | Counter | Bytes sent |
| `prlmon_net_totalbytes_recv_total` | Counter | Bytes received |
| `prlmon_mempool_tx_count` | Gauge | Mempool transaction count |
| `prlmon_mempool_bytes` | Gauge | Mempool size in bytes |

Example scrape target:

```yaml
scrape_configs:
  - job_name: prlmon
    static_configs:
      - targets: ['node-1.example:9105']
        labels:
          node: node-1
          region: us-east
```

## Development

```bash
# Unit tests
go test ./prlmon/...

# Integration tests (builds pearld via rpctest)
go test -tags=rpctest,xmss,zkpow ./prlmon/...
```

## License

ISC — see the repository [LICENSE](../LICENSE).
