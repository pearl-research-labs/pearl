# Configuration

pearld has a number of [configuration](https://pkg.go.dev/github.com/pearl-research-labs/pearl/node)
options, which can be viewed by running: `$ pearld --help`.

## Peer server listen interface

pearld allows you to bind to specific interfaces which enables you to setup
configurations with varying levels of complexity.  The listen parameter can be
specified on the command line as shown below with the -- prefix or in the
configuration file without the -- prefix (as can all long command line options).
The configuration file takes one entry per line.

**NOTE:** The listen flag can be specified multiple times to listen on multiple
interfaces as a couple of the examples below illustrate.

Command Line Examples:

|Flags|Comment|
|----------|------------|
|--listen=|all interfaces on default port which is changed by `--testnet` and `--regtest` (**default**)|
|--listen=0.0.0.0|all IPv4 interfaces on default port which is changed by `--testnet` and `--regtest`|
|--listen=::|all IPv6 interfaces on default port which is changed by `--testnet` and `--regtest`|
|--listen=:44108|all interfaces on port 44108|
|--listen=0.0.0.0:44108|all IPv4 interfaces on port 44108|
|--listen=[::]:44108|all IPv6 interfaces on port 44108|
|--listen=127.0.0.1:44108|only IPv4 localhost on port 44108|
|--listen=[::1]:44108|only IPv6 localhost on port 44108|
|--listen=:8336|all interfaces on non-standard port 8336|
|--listen=0.0.0.0:8336|all IPv4 interfaces on non-standard port 8336|
|--listen=[::]:8336|all IPv6 interfaces on non-standard port 8336|
|--listen=127.0.0.1:8337 --listen=[::1]:44108|IPv4 localhost on port 8337 and IPv6 localhost on port 44108|
|--listen=:44108 --listen=:8337|all interfaces on ports 44108 and 8337|

The following config file would configure pearld to only listen on localhost for both IPv4 and IPv6:

```text
[Application Options]

listen=127.0.0.1:44108
listen=[::1]:44108
```

In addition, if you are starting pearld with TLS and want to make it
available via a hostname, then you will need to generate the TLS
certificates for that host. For example,

```
gencerts --host=myhostname.example.com --directory=/home/me/.pearld/
```

## RPC server listen interface

pearld allows you to bind the RPC server to specific interfaces which enables you
to setup configurations with varying levels of complexity.  The `rpclisten`
parameter can be specified on the command line as shown below with the -- prefix
or in the configuration file without the -- prefix (as can all long command line
options).  The configuration file takes one entry per line.

A few things to note regarding the RPC server:

* The RPC server will **not** be enabled unless the `rpcuser` and `rpcpass`
  options are specified.
* When the `rpcuser` and `rpcpass` and/or `rpclimituser` and `rpclimitpass`
  options are specified, the RPC server will only listen on localhost IPv4 and
  IPv6 interfaces by default.  You will need to override the RPC listen
  interfaces to include external interfaces if you want to connect from a remote
  machine.
* The RPC server has TLS enabled by default, even for localhost.  You may use
  the `--notls` option to disable it, but only when all listeners are on
  localhost interfaces.
* The `--rpclisten` flag can be specified multiple times to listen on multiple
  interfaces as a couple of the examples below illustrate.
* The RPC server is disabled by default when using the `--regtest` and
  `--simnet` networks.  You can override this by specifying listen interfaces.

Command Line Examples:

|Flags|Comment|
|----------|------------|
|--rpclisten=|all interfaces on default port which is changed by `--testnet`|
|--rpclisten=0.0.0.0|all IPv4 interfaces on default port which is changed by `--testnet`|
|--rpclisten=::|all IPv6 interfaces on default port which is changed by `--testnet`|
|--rpclisten=:44107|all interfaces on port 44107|
|--rpclisten=0.0.0.0:44107|all IPv4 interfaces on port 44107|
|--rpclisten=[::]:44107|all IPv6 interfaces on port 44107|
|--rpclisten=127.0.0.1:44107|only IPv4 localhost on port 44107|
|--rpclisten=[::1]:44107|only IPv6 localhost on port 44107|
|--rpclisten=:8336|all interfaces on non-standard port 8336|
|--rpclisten=0.0.0.0:8336|all IPv4 interfaces on non-standard port 8336|
|--rpclisten=[::]:8336|all IPv6 interfaces on non-standard port 8336|
|--rpclisten=127.0.0.1:8337 --listen=[::1]:44107|IPv4 localhost on port 8337 and IPv6 localhost on port 44107|
|--rpclisten=:44107 --listen=:8337|all interfaces on ports 44107 and 8337|

The following config file would configure the pearld RPC server to listen to all interfaces on the default port, including external interfaces, for both IPv4 and IPv6:

```text
[Application Options]

rpclisten=
```

## Default ports

While pearld is highly configurable when it comes to the network configuration,
the following is intended to be a quick reference for the default ports used so
port forwarding can be configured as required.

pearld provides a `--upnp` flag which can be used to automatically map the Pearl
peer-to-peer listening port if your router supports UPnP.  If your router does
not support UPnP, or you don't wish to use it, please note that only the Pearl
peer-to-peer port should be forwarded unless you specifically want to allow RPC
access to your pearld from external sources such as in more advanced network
configurations.

|Name|Port|
|----|----|
|Default Pearl peer-to-peer port|TCP 44108|
|Default RPC port|TCP 44107|

## Using bootstrap.dat

### What is bootstrap.dat?

It is a flat, binary file containing Pearl blockchain data starting from the
genesis block and continuing through a relatively recent block height depending
on the last time it was updated.

See the Pearl block explorer at [explorer.pearlresearch.ai](https://explorer.pearlresearch.ai) for more details.

**NOTE:** Using bootstrap.dat is entirely optional.  Pearld will download the
block chain from other peers through the Pearl protocol with no extra
configuration needed.

### What are the pros and cons of using bootstrap.dat?

Pros:

* Typically accelerates the initial process of bringing up a new node as it
  downloads from public P2P nodes and generally is able to achieve faster
  download speeds
* It is particularly beneficial when bringing up multiple nodes as you only need
  to download the data once

Cons:

* Requires you to setup and configure a torrent client if you don't already have
  one available
* Requires roughly twice as much disk space since you'll need the flat file as
  well as the imported database

### Where do I get bootstrap.dat?

The bootstrap.dat file is made available via a torrent. Check the
[Pearl Discord](https://discord.gg/joinpearl) or [explorer.pearlresearch.ai](https://explorer.pearlresearch.ai)
for the torrent download details.

### How do I know I can trust the bootstrap.dat I downloaded?

You don't need to trust the file as the `addblock` utility verifies every block
using the same rules that are used when downloading the block chain normally
through the Pearl protocol.  Additionally, the chain rules contain hard-coded
checkpoints for the known-good block chain at periodic intervals.  This ensures
that not only is it a valid chain, but it is the same chain that everyone else
is using.

### How do I use bootstrap.dat with pearld?

pearld comes with a separate utility named `addblock` which can be used to import
`bootstrap.dat`.  This approach is used since the import is a one-time operation
and we prefer to keep the daemon itself as lightweight as possible.

1. Stop pearld if it is already running.  This is required since addblock needs to
   access the database used by pearld and it will be locked if pearld is using it.
2. Note the path to the downloaded bootstrap.dat file.
3. Run the addblock utility with the `-i` argument pointing to the location of
   bootstrap.dat:

**Windows:**

```bat
"%PROGRAMFILES%\Pearl Suite\Pearld\addblock" -i C:\Path\To\bootstrap.dat
```

**Linux/Unix/BSD/POSIX:**

```bash
./bin/addblock -i /path/to/bootstrap.dat
```
