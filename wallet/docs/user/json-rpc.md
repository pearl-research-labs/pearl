# Oyster JSON-RPC

Oyster's wallet RPC is JSON-RPC over HTTP. This is the supported interface.
`oystercli` and `prlctl --wallet` speak it, and so can any HTTP client.

`help` prints each method's parameters and result fields. This page is the
contract around that: how to connect, which methods exist, and what is stable.

## Connecting

| | |
| --- | --- |
| POST | `/` |
| Websocket | `/ws` |

Both require HTTP Basic authentication. Oyster will not start the RPC server
unless `--username` and `--password` are set. A missing or wrong password is
HTTP 401.

POST bodies are JSON-RPC 1.0 (`jsonrpc`, `method`, `params`, `id`) and are
limited to 4 MiB. At most 10 POST clients and 25 websocket clients are served
at once (`--rpcmaxclients`, `--rpcmaxwebsockets`).

The websocket endpoint uses the same methods. Authenticate with HTTP Basic on
the upgrade request. Server pushes are not sent on this socket; clients poll.

## TLS

Oyster generates a self-signed certificate at startup and writes it next to
the key (`rpc.cert` and `rpc.key` under the app data directory). Clients
should trust that certificate.

| Flag | Effect |
| --- | --- |
| `--rpccert`, `--rpckey` | Use this certificate pair instead of the generated one |
| `--onetimetlskey` | Generate a new pair each startup and write only the certificate. Refuses to start if the key file already exists |
| `--noservertls` | Serve plain HTTP |

When TLS is on, the server requires TLS 1.3 and pins ALPN to HTTP/1.1 so
websocket upgrades keep working. Clients that only support TLS 1.2 cannot
connect.

## Listen address

With no `--rpclisten`, Oyster listens on localhost at the network's wallet
port. Setting `--rpclisten` replaces that default. Repeat the flag to listen
on more than one address.

| Network | Wallet RPC port |
| --- | --- |
| Mainnet | 44207 |
| Testnet | 44209 |
| Testnet2 | 44211 |
| Simnet | 18554 |
| Signet | 38332 |
| Regtest | 18332 |

Logs for this server use the `RPCS` subsystem (`--debuglevel=RPCS=trace`).

## Methods that need pearld

These handlers run only when the chain backend is a pearld RPC client. With
`--usespv`, or before that client is connected, they return
`Chain RPC is inactive`:

- `getinfo`
- `help` (once a chain client is connected; a pearld client is required for the chain-aware help text)
- `listsinceblock`
- `sendfrom`
- `signrawtransaction`

Any method Oyster does not implement itself is forwarded to pearld on that
same connection. Forwarding has the same requirement. `sendrawtransaction`
is one of those forwarded methods: Oyster does not implement it, and it is
not available in SPV mode.

## Methods

Parameter and result details come from `help` with the method name.

### Addresses and accounts

- `createnewaccount`
- `dumpprivkey`
- `getaccount`
- `getaccountaddress`
- `getaddressesbyaccount`
- `getnewaddress`
- `getrawchangeaddress`
- `importprivkey`
- `keypoolrefill`
- `renameaccount`
- `validateaddress`

### Balances and history

- `getbalance`
- `getreceivedbyaccount`
- `getreceivedbyaddress`
- `gettransaction`
- `getunconfirmedbalance`
- `listaccounts`
- `listaddresstransactions`
- `listalltransactions`
- `listreceivedbyaccount`
- `listreceivedbyaddress`
- `listtransactions`

`getunconfirmedbalance` takes an account. Bitcoin Core's method of the same
name does not.

### Coins and sending

- `listlockunspent`
- `listunspent`
- `lockunspent`
- `sendmany`
- `sendtoaddress`
- `settxfee`

### Chain

- `chainsynced`
- `getbestblock`
- `getbestblockhash`
- `getblockcount`
- `getsyncprogress`

### Messages and locking

- `signmessage`
- `verifymessage`
- `walletislocked`
- `walletlock`
- `walletpassphrase`
- `walletpassphrasechange`

### Oyster extensions

These are not part of the Bitcoin Core wallet API. They are stable under the
policy below.

- `chainsynced`
- `createnewaccount`
- `getbestblock`
- `getsyncprogress`
- `getunconfirmedbalance`
- `listaddresstransactions`
- `listalltransactions`
- `rebroadcasttransaction`
- `removetransaction`
- `renameaccount`
- `walletislocked`

### Process

- `stop` shuts the oyster process down. It is not a wallet method; an
  authorized client may call it on POST or the websocket.

## Unimplemented

These names are recognized and return "Method unimplemented":

- `backupwallet`
- `dumpwallet`
- `getwalletinfo`
- `importwallet`
- `listaddressgroupings`

## Unsupported

These Bitcoin Core methods do not fit Oyster's wallet and return
"Request unsupported by Oyster":

- `encryptwallet` (wallets are created encrypted)
- `move`
- `setaccount`

## Stability

Method names, parameters, and result fields listed by `help` are stable.

Removing a method, renaming it, or changing the meaning of a parameter or
result field is a breaking change and is called out as one.

Adding a method, or adding a parameter or result field that existing clients
can ignore, can ship in any release.
