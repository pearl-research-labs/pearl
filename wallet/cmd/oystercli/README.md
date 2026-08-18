# oystercli

An interactive terminal client for the [oyster](../../README.md) wallet
daemon. It puts the core wallet workflows behind a menu-driven UI and, since
it is aimed at technical users, doubles as a troubleshooting tool.

Installed by the release installers (`install.sh` / `install.ps1`) alongside
the daemon, with both on `$PATH`:

```
oystercli
```

For a source checkout:

```
task build:oystercli
cd bin && ./oystercli
```

(When oyster is not on `$PATH`, the create-wallet and start-daemon flows ask
for its exact location — point them at the built `oyster`, or pass
`--oysterbin`.)

## What it does

- **Overview** — per-account balances, pending funds, recent activity.
- **Send** — guided flow with address/amount validation, a review step, and
  automatic unlock prompting; the fee rate defaults to the network's minimum
  relay fee (0.00001 PRL/kB).
- **Receive** — fresh or current addresses, rendered with a scannable QR code.
- **Transactions** — paged history browser with filtering and full detail view.
- **Accounts** — list, create, rename, and inspect addresses.
- **Coins** — UTXO listing plus lock/unlock coin control.
- **Security** — lock/unlock, passphrase change, WIF import/export (guarded),
  message signing and verification.
- **Node & sync** — oyster and pearld state at a glance.
- **Troubleshoot**
  - *RPC console*: run any wallet RPC (or node RPC via oyster's passthrough)
    with method autocompletion and pretty-printed results.
  - *Doctor*: pass/warn/fail checks for config, certificates, wallet.db, TCP
    reachability, TLS, auth, sync, and pearld connectivity, with an
    exportable, credential-free report.
  - *Logs*: tail or follow the local oyster log with level colorization.

## Connecting

oystercli talks to a running oyster daemon over its legacy JSON-RPC port
(mainnet `44207`; see `wallet/netparams`). It resolves connection settings
optimistically and needs no configuration from you:

1. Flags win: `--connect`, `--rpcuser`, `--rpcpass`, `--cafile`, `--notls`,
   `--testnet`/`--testnet2`/`--simnet`/`--signet`, `--appdata`.
2. Otherwise it reads the existing `~/.oyster/oyster.conf`
   (`username=`/`password=`, `noservertls=`, `rpclisten=`) and connects. An
   explicit `rpclisten=` is where the daemon actually listens, so oystercli
   dials it — whatever the active network's default port would be.
3. If there is no config at all, oystercli **writes a secure one itself**
   (see below) — you are never asked to make configuration choices.

The TLS certificate defaults to `~/.oyster/rpc.cert`. Every resolution step is
visible: whenever the CLI has to act it prints a compact table of what it
found and where (network, appdata, config, credentials, TLS, certificate,
connect target, wallet, oyster binary), and always under `--verbose`.

## Zero-config first run

When no daemon is reachable, oystercli provisions and manages one for you:

- **Auto-provisioned config.** If `~/.oyster/oyster.conf` is missing, oystercli
  writes a secure one — generated RPC credentials and `usespv=1` (no pearld
  needed). TLS and listeners are left at oyster's defaults, which are TLS on
  and **loopback-only** listeners on the active network's RPC port — so the
  same conf serves mainnet and testnet. An existing config is never
  overridden: if it merely lacks credentials, only those are appended; any
  other settings are left exactly as they are. If an existing config is
  insecure (`noservertls`, a non-loopback listener), oystercli warns but
  respects it.
- **Create a wallet** (when none exists) — drives `oyster --createfromfile`
  (the desktop wallet's mechanism), with a seed backup ceremony for new
  wallets. Restores recover funds by rescanning from the wallet birthday you
  enter, or from the chain's genesis when you leave it empty.
- **Start oyster now** — spawns the daemon detached (it keeps running after
  oystercli exits; the PID and log path are printed), waits for its RPC, and
  connects.
- **Stop the daemon** — from the **Node & sync** screen, sends oyster's
  authenticated `stop` RPC (graceful shutdown, keys unloaded) and exits the
  CLI. Only a client holding the daemon's credentials can stop it. Restart by
  running oystercli again and choosing "Start oyster now".

The daemon binary is resolved from `--oysterbin` if given, otherwise from
`$PATH` — the release installers put it there, which is the supported setup.
There is deliberately no implicit lookup in the current directory or next to
the executable: oystercli hands wallet passphrases and seeds to the binary it
runs, so it only executes explicitly trusted paths. When oyster is not on
`$PATH` (e.g. a `task build` tree), the CLI asks for its exact location and
remembers it for the session; the chosen binary and its origin are printed
before use.

If something is listening on `127.0.0.1:8335`, triage points out that this is
the desktop wallet's private oyster instance (random per-session credentials)
and steers toward running a dedicated daemon instead.

## Remote daemons

oystercli can drive an oyster running on another machine:

```
oystercli -c wallet-host:44207 -u <user> -P <pass> --cafile <copied rpc.cert>
```

- On the daemon's machine: set a non-loopback `rpclisten=` in its
  `oyster.conf` and open the port in the firewall. The auto-generated
  certificate already covers that machine's hostname and interface IPs; if
  its address changed since generation, delete `rpc.cert`/`rpc.key` and
  restart oyster to regenerate.
- On the client: copy the daemon's `rpc.cert` and point `--cafile` at it —
  or skip both by tunnelling (`ssh -L 44207:127.0.0.1:44207 wallet-host`)
  and connecting to `localhost`, which every generated certificate covers.
- Flag-free alternative: keep a dedicated client appdata dir
  (`oystercli -A ~/.oyster-remote`) whose `oyster.conf` holds the remote
  daemon's credentials and `rpclisten=wallet-host:44207` (oystercli dials
  the listener address), with the copied `rpc.cert` beside it.

With a remote target the local bootstrapping steps — config provisioning,
wallet creation, starting or stopping-and-restarting the daemon from triage —
are not offered; that machine's operator owns its configuration.

## Testnet

`oystercli --testnet` runs the exact same flow against the test network:
credentials come from the shared `oyster.conf`, the wallet lives in the
per-network directory (`~/.oyster/testnet/wallet.db`), "Create a wallet" and
"Start oyster now" pass `--testnet` through to the daemon, and the RPC port
defaults to the testnet port (`44209`). Mainnet and testnet daemons can run
side by side from the same config. The one caveat: a config with an explicit
`rpclisten=` pins the daemon (and oystercli) to that address on every
network, so port-sharing setups should drop the line and rely on the
per-network defaults.

## Flags

| Flag | Purpose |
| --- | --- |
| `-c, --connect` | Host[:port] of the oyster RPC server (default `localhost`) |
| `-u, --rpcuser` / `-P, --rpcpass` | RPC credentials |
| `--cafile` | RPC server certificate |
| `--notls` | Disable TLS |
| `-A, --appdata` | Oyster data directory (config/cert discovery, diagnostics) |
| `--testnet`, `--testnet2`, `--simnet`, `--signet` | Network selection |
| `--oysterbin` | Path to the oyster binary for wallet creation and daemon start |
| `-v, --verbose` | Trace every RPC call (method, duration, outcome) to stderr |
| `-V, --version` | Print the version |

## Secrets

oystercli's security posture is the same as oyster's — both are sensitive
local services, and secrets live where oyster keeps them (`oyster.conf` and
the wallet DB, mode `0600`). oystercli adds no weaker assumptions: passphrase
prompts are masked, `--verbose` traces method names only (never parameters),
the wallet is re-locked on exit if this session unlocked it, and the
auto-provisioned config keeps TLS on with a loopback-only listener. The wallet
passphrase is forwarded to the daemon to unlock it and is not persisted by
oystercli.

## Accessibility

Set `ACCESSIBLE=1` to switch every prompt to plain, screen-reader friendly
input instead of the TUI renderer.

## Notes

- The wallet is locked again on exit when it was this session that unlocked it.
- oystercli is pure Go (no cgo); it builds with `CGO_ENABLED=0` and does not
  need the xmss/zkpow toolchains.
