# Installation

## Prebuilt binaries (recommended)

The release installer downloads the platform archive from GitHub Releases,
verifies its SHA-256 against `checksums.txt`, and installs `pearld`, `prlctl`,
and `oyster` into `${XDG_BIN_HOME:-$HOME/.local/bin}`.

It also writes mainnet default configs into the OS default app-data paths when
missing, with shared auto-generated RPC credentials. Oyster defaults to SPV
sync (`usespv=1`), so a local pearld is optional for the wallet. After install,
no `-u` / `-P` / `-C` is required: `prlctl getinfo` targets local pearld, and
`prlctl --wallet getinfo` targets local oyster. Existing configs that already
have credentials are left unchanged. RPC stays localhost-only.

| Tool | Linux | macOS |
|------|-------|-------|
| pearld | `~/.pearld/pearld.conf` | `~/Library/Application Support/Pearld/pearld.conf` |
| oyster | `~/.oyster/oyster.conf` | `~/Library/Application Support/Oyster/oyster.conf` |
| prlctl | `~/.prlctl/prlctl.conf` | `~/Library/Application Support/Prlctl/prlctl.conf` |

Supported platforms: macOS and Linux on amd64 and arm64.

### Download, inspect, then run

```bash
curl -fsSL -o install.sh https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh
less install.sh
sh install.sh
```

### One-line convenience form

```bash
curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
```

### Pin a version or install directory

```bash
sh install.sh --version v0.1.0
sh install.sh --bin-dir "$HOME/bin"
```

### Upgrade

Rerun the installer (with the same `--bin-dir` if you customized it). Existing
binaries are replaced atomically. Existing config files are left unchanged.

### Remove

```bash
rm -f "${XDG_BIN_HOME:-$HOME/.local/bin}/pearld" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/prlctl" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/oyster" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/sample-pearld.conf"
```

Configs are not removed automatically. Delete them from the paths in the table
above if you also want to discard RPC credentials and settings.
### macOS Gatekeeper / quarantine

Installing via `curl`/`sh` normally does not attach the browser quarantine
flag, so Gatekeeper typically does not block the binaries. Archives downloaded
in a browser can still be quarantined when the app is not Developer ID signed
and notarized. The installer never deletes quarantine metadata or otherwise
bypasses Gatekeeper.

## Requirements (build from source)

- [Go](https://golang.org) 1.26 or newer
- [Rust](https://rustup.rs) toolchain (for ZK verification library)
- C compiler (for XMSS library)
- [Task](https://taskfile.dev) runner

## Build from Source

Clone the repository and build the blockchain binaries:

```bash
git clone https://github.com/pearl-research-labs/pearl.git
cd pearl
task build:blockchain
```

Binaries are placed in `bin/`:
- `pearld` — full node
- `prlctl` — CLI control tool
- `oyster` — wallet daemon

To build only the node:

```bash
task build:pearld
```

## Startup

pearld will run and start downloading the block chain with no extra
configuration necessary. See the
[configuration documentation](configuration.md) for advanced options.

```bash
./bin/pearld
```
