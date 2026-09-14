# Pearl

[![Blockchain / Build and Test](https://github.com/pearl-research-labs/pearl/actions/workflows/blockchain_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/blockchain_ci.yml)
[![Integration Tests CI](https://github.com/pearl-research-labs/pearl/actions/workflows/integration_tests_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/integration_tests_ci.yml)
[![Miner CI](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_ci.yml)
[![Miner GPU CI](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_gpu_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_gpu_ci.yml)
[![Desktop Wallet CI/CD](https://github.com/pearl-research-labs/pearl/actions/workflows/pearl-desktop-wallet.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/pearl-desktop-wallet.yml)
[![Plonky2 Tests](https://github.com/pearl-research-labs/pearl/actions/workflows/plonky2_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/plonky2_ci.yml)
[![Rust CI](https://github.com/pearl-research-labs/pearl/actions/workflows/rust_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/rust_ci.yml)
[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)

Pearl is an L1 blockchain based on the **Proof-of-Useful-Work** protocol, where mining is done as a by-product of arbitrary matrix multiplication, as proposed [in this paper](https://arxiv.org/abs/2504.09971).

This monorepo contains the full node, wallet, SPV light client, ZK proving
system, vLLM miner, and supporting tools.

## Repository Layout

| Directory | Description |
|-----------|-------------|
| [`node/`](node/) | **pearld** — reference implementation of the Pearl Protocol (full node) |
| [`wallet/`](wallet/) | **Oyster** — HD wallet daemon with JSON-RPC and gRPC interfaces, plus [**oystercli**](wallet/cmd/oystercli/), an interactive terminal client for it |
| [`spv/`](spv/) | **Pearl light client** — privacy-preserving SPV client using compact block filters |
| [`dnsseeder/`](dnsseeder/) | DNS seeder for the Pearl network |
| [`coredns-dnsseed/`](coredns-dnsseed/) | CoreDNS plugin — production DNS seeder |
| [`proxy/`](proxy/) | Caddy reverse-proxy sidecar for RPC TLS termination and rate limiting |
| [`xmss/`](xmss/) | XMSS post-quantum signature scheme (C + Go FFI) |
| [`zk-pow/`](zk-pow/) | ZK proof-of-work circuit and verifier (Rust, Plonky2/STARKy) |
| [`pearl-blake3/`](pearl-blake3/) | Blake3 hashing utilities (Rust) |
| [`plonky2/`](plonky2/) | Plonky2 SNARK proving system (Rust, vendored) |
| [`miner/`](miner/) | vLLM miner — GPU mining infrastructure (Python/CUDA, uv workspace) |
| [`py-pearl-mining/`](py-pearl-mining/) | Python bindings for Pearl mining (Rust/PyO3) |
| [`apps/`](apps/) | Frontend applications (website, desktop wallet — pnpm/Turborepo) |
| [`tools/`](tools/) | Go development tool dependencies |

## Install (prebuilt binaries)

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
```

Windows:

```powershell
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex
```

Installs `pearld`, `prlctl`, `oyster`, and `oystercli` with localhost-only mainnet defaults and
shared RPC credentials (oyster uses SPV by default). Binaries go to
`${XDG_BIN_HOME:-$HOME/.local/bin}` on macOS/Linux, or `%LOCALAPPDATA%\Pearl\bin`
on Windows.

| Tool | Linux | macOS | Windows |
|------|-------|-------|---------|
| pearld | `~/.pearld/pearld.conf` | `~/Library/Application Support/Pearld/pearld.conf` | `%LOCALAPPDATA%\Pearld\pearld.conf` |
| oyster | `~/.oyster/oyster.conf` | `~/Library/Application Support/Oyster/oyster.conf` | `%LOCALAPPDATA%\Oyster\oyster.conf` |
| prlctl | `~/.prlctl/prlctl.conf` | `~/Library/Application Support/Prlctl/prlctl.conf` | `%LOCALAPPDATA%\Prlctl\prlctl.conf` |

Pin a version or install directory with `--version` / `--bin-dir` (or `-Version` /
`-BinDir` on Windows). See [node/docs/installation.md](node/docs/installation.md)
for upgrade, removal, and other details. To build from source, see **Building** below.

## Prerequisites

- [Go](https://golang.org) 1.26 or newer
- [Rust](https://rustup.rs) toolchain (for ZK and hashing crates)
- C compiler (for XMSS library)
- [Python](https://python.org) 3.12 and [uv](https://docs.astral.sh/uv/) (for vLLM miner packages)
- [Task](https://taskfile.dev) runner
- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (for vLLM miner)

## Building

```bash
task build              # build everything (blockchain + vLLM miner)
task build:blockchain   # pearld, prlctl, oyster, oystercli → bin/
task build:miner        # install vLLM miner Python packages
task build:pearld       # pearld only
```

## Running a Node and vLLM Miner

The setup flow: **build** > **create wallet** > **start node** > **start vLLM miner**.

### 1. Create a wallet and get a mining address

The interactive way — [`oystercli`](wallet/cmd/oystercli/) configures oyster
automatically and walks you through wallet creation, starting the daemon, and
generating an address (Receive menu). With the release installer it is already
on your PATH (`oystercli`); from a source build:

```bash
cd bin && ./oystercli
```

Or manually:

```bash
./bin/oyster -u rpcuser -P rpcpass --create
```

Follow the prompts to set a passphrase and record your seed. Then start the
wallet and generate a Taproot mining address:

```bash
./bin/oyster -u rpcuser -P rpcpass &
./bin/prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
```

### 2. Start the node

```bash
./bin/pearld \
  --rpcuser=rpcuser \
  --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --miningaddr=<your-taproot-address> \
  --txindex
```

Key flags: `--testnet` / `--simnet` for non-mainnet, `--notls` to disable TLS,
`--debuglevel=debug` for verbose logs. See `node/sample-pearld.conf` for all
options.

| Network  | RPC   | P2P   | Wallet Server |
|----------|-------|-------|---------------|
| Mainnet  | 44107 | 44108 | 44207         |
| Testnet  | 44109 | 44110 | 44209         |
| Testnet2 | 44111 | 44112 | 44211         |
| Simnet   | 18556 | 18555 | 18554         |
| Regtest  | 18334 | 18444 | 18332         |

### 3. Start the vLLM miner

The vLLM miner has two components: **pearl-gateway** (bridge to the node) and
**vllm-miner** (GPU mining via vLLM).

```bash
export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="rpcuser"
export PEARLD_RPC_PASSWORD="rpcpass"
export PEARLD_MINING_ADDRESS="<your-taproot-address>"
pearl-gateway start
```

The gateway connects to pearld over JSON-RPC and exposes a mining interface
on `/tmp/pearlgw.sock` (UDS) or port 8337 (TCP, set `MINER_RPC_TRANSPORT=tcp`).

To run the full stack with Docker:

```bash
docker buildx build -t vllm_miner . -f miner/vllm-miner/Dockerfile

docker run --rm -it --gpus all --network host \
  -e PEARLD_RPC_URL=http://localhost:44107 \
  -e PEARLD_RPC_USER=rpcuser \
  -e PEARLD_RPC_PASSWORD=rpcpass \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm_miner:latest \
  pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 --port 8000
```

## Testing

```bash
task test               # run all tests (Go + Python)
task test:go            # Go tests with race detector
task test:python        # full Python test suite
task test:python:basic  # Python tests (excludes integration/perf/slow)
```

## Formatting and Linting

```bash
task fmt            # format all (Go + Rust + Python)
task lint:python    # lint Python code with ruff
task tidy           # tidy Go dependencies
```

Scoped variants are available: `task fmt:go`, `task fmt:rust`, `task fmt:python`,
`task lint:go`, `task lint:rust`, `task lint:python`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

Pearl is licensed under the [copyfree](http://copyfree.org) ISC License.
See [LICENSE](LICENSE) for details.

## Acknowledgments

Pearl's blockchain infrastructure was originally forked from the following
open-source projects:

- [btcd](https://github.com/btcsuite/btcd) — full node implementation
- [btcwallet](https://github.com/btcsuite/btcwallet) — wallet daemon
- [neutrino](https://github.com/lightninglabs/neutrino) — SPV light client
