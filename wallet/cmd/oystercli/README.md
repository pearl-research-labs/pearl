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
  automatic unlock prompting.
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
(mainnet `44207`; see `wallet/netparams`). Connection settings resolve in
this order:

1. Flags: `--connect`, `--rpcuser`, `--rpcpass`, `--cafile`, `--notls`,
   `--testnet`/`--testnet2`/`--simnet`/`--signet`, `--appdata`.
2. `~/.oyster/oyster.conf`: `username=`/`password=` (or `rpcuser=`/`rpcpass=`)
   and `noservertls=`.
3. An interactive prompt for anything still missing.

The TLS certificate defaults to `~/.oyster/rpc.cert`. Use `--notls` only when
the daemon runs with `--noservertls`.

Every resolution step is visible: whenever the CLI has to ask for something
(and always under `--verbose`), it prints a compact table of what it looked
for, where, and what it found — network, appdata, config file, credentials,
TLS, certificate, connect target, and wallet database, each with its source.

## First run and daemon management

When the daemon is unreachable, a triage screen classifies the failure
(daemon not running / TLS mismatch / bad credentials / no wallet) and offers
the matching next step:

- **Guided setup** — writes (or completes) `oyster.conf` with generated or
  chosen RPC credentials, a chain backend (SPV or a local pearld), and the
  TLS choice. The daemon and the CLI then share one config file, which fixes
  the common "oyster was always started with ad-hoc flags and the credentials
  are gone" situation. Existing config content is never rewritten, only
  missing keys are appended.
- **Create a wallet** — drives `oyster --createfromfile` (the desktop
  wallet's mechanism), with a seed backup ceremony for new wallets.
- **Start oyster now** — once the config carries credentials, spawns the
  daemon detached (it keeps running after oystercli exits; the PID and log
  path are printed), waits for its RPC to come up, and connects.
- **Retry / Doctor / Quit.**

The daemon binary is resolved from `--oysterbin` if given, otherwise from
`$PATH` — the release installers put it there, which is the supported setup.
There is deliberately no implicit lookup in the current directory or next to
the executable: oystercli hands wallet passphrases and seeds to the binary it
runs, so it only executes explicitly trusted paths. When oyster is not on
`$PATH` (e.g. a `task build` tree), the CLI asks for its exact location and
remembers it for the session; the chosen binary and its origin are always
printed before use.

If something is listening on `127.0.0.1:8335`, triage points out that this is
the desktop wallet's private oyster instance (random per-session credentials)
and steers toward running a dedicated daemon instead.

## Flags

| Flag | Purpose |
| --- | --- |
| `-c, --connect` | Host[:port] of the oyster RPC server (default `localhost`) |
| `-u, --rpcuser` / `-P, --rpcpass` | RPC credentials |
| `--cafile` | RPC server certificate |
| `--notls` | Disable TLS |
| `-A, --appdata` | Oyster data directory (config/cert discovery, diagnostics) |
| `--testnet`, `--testnet2`, `--simnet`, `--signet` | Network selection |
| `--oysterbin` | Path to the oyster binary for the creation wizard and daemon start |
| `-v, --verbose` | Trace every RPC call (method, duration, outcome) to stderr |
| `-V, --version` | Print the version |

## Handling of secrets

oystercli deals with two secrets: the **RPC password** (authenticates to the
daemon) and the **wallet passphrase** (unlocks the wallet). How they are
handled:

- **The wallet's decryption key never enters oystercli.** The passphrase is
  forwarded to the daemon via the `walletpassphrase` RPC; oyster holds the
  derived key. oystercli keeps the passphrase only as a short-lived local
  while a single unlock/create/sign call runs — it is never cached, and it is
  never written to the config.
- **Prompts are masked** (`EchoModePassword`); typed secrets are not echoed.
- **Nothing secret is logged.** `--verbose` traces method name, duration, and
  error code only — never parameters. The doctor report is credential-free by
  construction.
- **No secrets on the command line.** The daemon is spawned with only
  `--appdata`/network flags; credentials come from `oyster.conf`. Wallet
  creation passes the passphrase/seed to `oyster --createfromfile` through a
  file in a private `0700` temp directory (file mode `0600`), removed
  immediately after.
- **On-disk credentials** live only in `oyster.conf` (mode `0600`), which the
  daemon itself requires; the guided setup writes it and never widens its
  permissions.
- **The wallet is re-locked on exit** if this session unlocked it.
- Use TLS (the default) so the RPC password and passphrase are encrypted in
  transit; `--notls` is only appropriate for a loopback-only daemon.

Two inherent limits worth knowing: Go strings cannot be reliably zeroed in
memory, so secrets may persist in the heap until garbage collected; and the
RPC password is held for the whole session because HTTP POST auth needs it on
every call. Passing `-P`/`--rpcpass` on the command line also exposes it via
`ps` — prefer `oyster.conf` discovery or the interactive prompt.

## Accessibility

Set `ACCESSIBLE=1` to switch every prompt to plain, screen-reader friendly
input instead of the TUI renderer.

## Notes

- The wallet is locked again on exit when it was this session that unlocked it.
- oystercli is pure Go (no cgo); it builds with `CGO_ENABLED=0` and does not
  need the xmss/zkpow toolchains.
