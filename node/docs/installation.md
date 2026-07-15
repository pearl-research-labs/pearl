# Installation

## Prebuilt binaries (recommended)

The release installer downloads the platform archive from GitHub Releases,
verifies its SHA-256 against `checksums.txt`, and installs `pearld`, `prlctl`,
`oyster`, and `oystercli` (the interactive wallet CLI, in releases that
include it):

- macOS/Linux: `install.sh` → `${XDG_BIN_HOME:-$HOME/.local/bin}`
- Windows: `install.ps1` → `%LOCALAPPDATA%\Pearl\bin`

It also writes mainnet default configs into the OS default app-data paths when
missing, with shared auto-generated RPC credentials. Oyster defaults to SPV
sync (`usespv=1`), so a local pearld is optional for the wallet. After install,
no `-u` / `-P` / `-C` is required: `prlctl getinfo` targets local pearld, and
`prlctl --wallet getinfo` targets local oyster. Existing configs that already
have credentials are left unchanged. RPC stays localhost-only.

| Tool | Linux | macOS | Windows |
|------|-------|-------|---------|
| pearld | `~/.pearld/pearld.conf` | `~/Library/Application Support/Pearld/pearld.conf` | `%LOCALAPPDATA%\Pearld\pearld.conf` |
| oyster | `~/.oyster/oyster.conf` | `~/Library/Application Support/Oyster/oyster.conf` | `%LOCALAPPDATA%\Oyster\oyster.conf` |
| prlctl | `~/.prlctl/prlctl.conf` | `~/Library/Application Support/Prlctl/prlctl.conf` | `%LOCALAPPDATA%\Prlctl\prlctl.conf` |

Supported platforms: macOS and Linux on amd64 and arm64; Windows on amd64.

### macOS / Linux — download, inspect, then run

```bash
curl -fsSL -o install.sh https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh
less install.sh
sh install.sh
```

### macOS / Linux — one-line convenience form

```bash
curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
```

### Windows — download, inspect, then run

```powershell
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 -OutFile install.ps1
notepad install.ps1
pwsh -File .\install.ps1
```

### Windows — one-line convenience form

```powershell
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex
```

### Pin a version or install directory

```bash
sh install.sh --version v0.1.0
sh install.sh --bin-dir "$HOME/bin"
```

```powershell
.\install.ps1 -Version v0.1.0
.\install.ps1 -BinDir "$env:USERPROFILE\bin"
```

### Upgrade

Rerun the installer (with the same `--bin-dir` / `-BinDir` if you customized it).
Existing binaries are replaced atomically. Existing config files are left
unchanged.

### Remove

```bash
rm -f "${XDG_BIN_HOME:-$HOME/.local/bin}/pearld" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/prlctl" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/oyster" \
      "${XDG_BIN_HOME:-$HOME/.local/bin}/oystercli"
```

```powershell
Remove-Item "$env:LOCALAPPDATA\Pearl\bin\pearld.exe", `
            "$env:LOCALAPPDATA\Pearl\bin\prlctl.exe", `
            "$env:LOCALAPPDATA\Pearl\bin\oyster.exe", `
            "$env:LOCALAPPDATA\Pearl\bin\oystercli.exe" -ErrorAction SilentlyContinue
```

Configs are not removed automatically. Delete them from the paths in the table
above if you also want to discard RPC credentials and settings.

### macOS Gatekeeper / Windows SmartScreen

Installing via `curl`/`sh` or `irm` normally does not attach browser quarantine
/ Mark-of-the-Web the same way a browser download does, so Gatekeeper and
SmartScreen typically do not block the binaries. Archives downloaded in a
browser can still be blocked when the app is not platform-signed. The installer
never deletes quarantine metadata, Zone.Identifier streams, or otherwise
bypasses OS protections.

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
- `oystercli` — interactive wallet CLI

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
