# Pearl Gateway

A lightweight bridge between a Pearl full node and a local Miner process, providing low-latency work distribution and safe submission of candidate blocks.

## Overview

PearlGateway sits between your Pearl node (e.g., pearld) and a local Miner process, exposing a minimal JSON-RPC surface to the Miner while managing the full Pearl Core RPC dialogue internally.

### Features

- Low-latency work distribution via dedicated memory cache
- Safe block submission with full template management
- JSON-RPC interface with validation using JSON Schema
- Support for Unix Domain Socket and TCP communication
- Configurable refresh intervals and authentication

### Architecture

```
                 Miner Process
             (JSON-RPC via UDS/loopback)

          ▲                        ▼
 getMiningInfo              submitPlainProof(PlainProof)
          │                        │
 ┌────────────────────────────────────────────┐
 │             PearlGateway                   │
 │                                            │
 │ ┌────────────┐   ┌───────────────────────┐ │
 │ │ Miner RPC  │──►│      Work Cache       │ │
 │ │  Server    │   └───────────────────────┘ │
 │ └────────────┘               ▲             │
 │         │                    │             │
 │         ▼                    │             │
 │ ┌───────────────────┐  ┌─────────────────┐ │
 │ │  ProofGenerator   │──►│ Submission Svc │ │
 │ └───────────────────┘  └─────────────────┘ │
 └────────────────────────────────────────────┘
           │ getblocktemplate      │ submitblock
           ▼                       ▲
      PearlNode Client (pearld)
```

### Data Flow

```mermaid
sequenceDiagram
    participant Miner
    participant AsyncLoopManager
    participant Gateway
    participant ProofPool
    participant ProofGenerator
    participant PearlNode

    Miner->>AsyncLoopManager: OpenedBlockInfo (A, B, A_noised, B_noised)
    AsyncLoopManager->>AsyncLoopManager: secondary_testing (uses noised)
    AsyncLoopManager->>AsyncLoopManager: create_proof (uses non-noised A, B)
    AsyncLoopManager->>Gateway: submitPlainProof(base64_string)
    Gateway->>ProofPool: prove(cert_version, header, plain_proof) [worker process]
    ProofPool-->>Gateway: (public_data, proof_data)
    Gateway->>ProofGenerator: build_block(public_data, proof_data, template)
    ProofGenerator-->>Gateway: PearlBlock
    Gateway->>PearlNode: submit block
```

ZK proof generation is CPU-bound and holds the GIL, so it runs in a separate
worker process (the `ProofPool`) rather than on the gateway's event loop; the
loop assembles and submits the block from the returned proof bytes.


### Installation

To install the package from this directory:

```bash
# Standard installation
uv pip install .

# Development installation (includes test dependencies)
uv pip install -e . --group dev
```

The `zk-pow-rust` module will be automatically built and installed from the local `zk-pow` directory.

## Configuration

PearlGateway uses environment variables for configuration with sensible defaults built-in. No configuration file is required.

### Configuration Options

All configuration can be set via environment variables:

**Pearl Node Connection:**
- `PEARLD_RPC_URL` - Pearl node RPC URL
- `PEARLD_RPC_USER` - RPC username (default: `user`)
- `PEARLD_RPC_PASSWORD` - RPC password (default: `pass`)
- `PEARLD_REFRESH_INTERVAL_SECONDS` - Template refresh interval (default: `1`)
- `PEARLD_MINING_ADDRESS` - Taproot address for block rewards (optional, overrides node's mining address)

**Miner RPC Server:**
- `MINER_RPC_TRANSPORT` - Transport type: `uds` or `tcp` (default: `uds`)
- `MINER_RPC_SOCKET_PATH` - Unix socket path (default: `/tmp/pearlgw.sock`)
- `MINER_RPC_PORT` - TCP port for miners (default: `8337`)
- `MINER_RPC_HOST` - TCP host (default: `localhost`)
- `MINER_RPC_MAX_PENDING` - Max proof submissions queued before new ones are
  rejected (positive integer, default: `64`). Bounds gateway memory when miners
  outpace the worker.

**Logging:**
- `LOGGING_LEVEL` - Log level: `debug`, `info`, `warning`, `error` (default: `info`)

**Pool Mode:**

Setting `POOL_STRATUM_URL` switches the gateway from solo mode (templates from
pearld, blocks submitted to pearld) to pool mode: it takes work from a Pearl
Stratum v1 pool and sends each proof to the pool as a share. Proving still
happens locally, and the pool receives the ZK certificate. Miners need no
changes: their job target becomes the pool's share target. pearld is not used,
so the `PEARLD_RPC_*` settings are ignored.

- `POOL_STRATUM_URL` - Pool endpoint, e.g. `stratum+ssl://host:3443` (`ssl`/`tls` = TLS, `tcp` = plaintext; port defaults to `3333`)
- `POOL_ADDRESS` - Payout address sent as the Stratum username (default: `PEARLD_MINING_ADDRESS`; one of the two is required)
- `POOL_WORKER` - Worker name, appended as `address.worker` (default: `gateway`)
- `POOL_PASSWORD` - Stratum password (default: `x`)
- `POOL_DIFFICULTY` - Start share difficulty, in the pool's difficulty unit (the one its dashboard shows). Sent as `d=N` appended to the password, e.g. `x;d=0.5`. Unset: the pool default. The pool then adjusts it.
- `POOL_RECONNECT_DELAY_SECONDS` / `POOL_MAX_RECONNECT_DELAY_SECONDS` - Reconnect backoff (defaults: `1` / `60`)

```bash
export PEARLD_MINING_ADDRESS="prl1p..."
export POOL_STRATUM_URL="stratum+ssl://pool.example.com:3443"
export POOL_WORKER="rig-01"
pearl-gateway start
```

## Usage

### Starting the Gateway

After installing the package, you can use the `pearl-gateway` command:

```bash
# Start with default configuration
pearl-gateway start

# Start with custom configuration via environment variables
export PEARLD_RPC_URL="https://my-pearl-node:44107"
export PEARLD_RPC_USER="myuser"
export PEARLD_RPC_PASSWORD="mypassword"
export LOGGING_LEVEL="debug"
# Start with a custom mining address (Taproot only)
export PEARLD_MINING_ADDRESS="bc1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr"
pearl-gateway start

# Enable verbose logging
pearl-gateway start -v
```

### Running Tests

The project uses pytest for testing:

```bash
# Run all tests
pytest
```


