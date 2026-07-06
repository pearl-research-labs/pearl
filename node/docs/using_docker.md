# Using Docker

- [Using Docker](#using-docker)
  - [Introduction](#introduction)
  - [Docker volumes](#docker-volumes)
  - [Known error messages when starting the pearld container](#known-error-messages-when-starting-the-pearld-container)
  - [Examples](#examples)
    - [Preamble](#preamble)
    - [Full node without RPC port](#full-node-without-rpc-port)
    - [Full node with RPC port](#full-node-with-rpc-port)
    - [Full node with RPC port running on TESTNET](#full-node-with-rpc-port-running-on-testnet)

## Introduction

With Docker you can easily set up *pearld* to run your Pearl full node. The official *pearld* Docker images are published to the GitHub Container Registry (GHCR) at `ghcr.io/pearl-research-labs/pearl/pearld`. Images are multi-arch and run natively on both `linux/amd64` and `linux/arm64`. The Docker source file of this image is located at [Dockerfile](https://github.com/pearl-research-labs/pearl/blob/master/node/Dockerfile).

Image tags follow a strict naming discipline so you can always tell an official release from a build artifact:

- **Official releases** — `:latest` (most recent release) and `:vX.Y.Z` (a specific release version). Version-shaped tags are only ever produced by the release process.
- **Build artifacts** — `:sha-<commit>` is a per-commit build published on every merge to `master` (and on manual side-branch builds), intended for internal and testnet testing. These are not releases; pin to a `:sha-<commit>` tag when you need a specific pre-release commit.

Any *pearld* flag can be appended to the container `command` (see the examples below), so indexing options such as `--txindex`/`--addrindex` or network options such as `--testnet` are added simply by extending `command`.

This documentation focuses on running Docker container with *docker-compose.yml* files. These files are better to read and you can use them as a template for your own use. For more information about Docker and Docker compose visit the official [Docker documentation](https://docs.docker.com/).

## Docker volumes

**Special diskspace hint**: The following examples are using a Docker managed volume. The volume is named *pearld-data* This will use a lot of disk space, because it contains the full Pearl blockchain. Please make yourself familiar with [Docker volumes](https://docs.docker.com/storage/volumes/).

The *pearld-data* volume will be reused, if you upgrade your *docker-compose.yml* file. Keep in mind, that it is not automatically removed by Docker, if you delete the pearld container. If you don't need the volume anymore, please delete it manually with the command:

```bash
docker volume ls
docker volume rm pearld-data
```

For binding a local folder to your *pearld* container please read the [Docker documentation](https://docs.docker.com/). The preferred way is to use a Docker managed volume.

## Known error messages when starting the pearld container

We pass all needed arguments to *pearld* as command line parameters in our *docker-compose.yml* file. It doesn't make sense to create a *pearld.conf* file. This would make things too complicated. Anyhow *pearld* will complain with following log messages when starting. These messages can be ignored:

```bash
Error creating a default config file: open /sample-pearld.conf: no such file or directory
...
[WRN] PRLD: open /root/.pearld/pearld.conf: no such file or directory
```

## Examples

### Preamble

All following examples uses some defaults:

- container_name: pearld
  Name of the docker container that is be shown by e.g. ```docker ps -a```

- hostname: pearld **(very important to set a fixed name before first start)**
  The internal hostname in the docker container. By default, docker is recreating the hostname every time you change the *docker-compose.yml* file. The default hostnames look like *ef00548d4fa5*. This is a problem when using the *pearld* RPC port. The RPC port is using a certificate to validate the hostname. If the hostname changes you need to recreate the certificate. To avoid this, you should set a fixed hostname before the first start. This ensures, that the docker volume is created with a certificate with this hostname.

- restart: unless-stopped
  Starts the *pearld* container when Docker starts, except that when the container is stopped (manually or otherwise), it is not restarted even after Docker restarts.

To use the following examples create an empty directory. In this directory create a file named *docker-compose.yml*, copy and paste the example into the *docker-compose.yml* file and run it.

```bash
mkdir ~/pearld-docker
cd ~/pearld-docker
touch docker-compose.yaml
nano docker-compose.yaml (use your favourite editor to edit the compose file)
docker-compose up (creates and starts a new pearld container)
```

With the following commands you can control *docker-compose*:

```docker-compose up -d``` (creates and starts the container in background)

```docker-compose down``` (stops and delete the container. **The docker volume pearld-data will not be deleted**)

```docker-compose stop``` (stops the container)

```docker-compose start``` (starts the container)

```docker ps -a``` (list all running and stopped container)

```docker volume ls``` (lists all docker volumes)

```docker logs pearld``` (shows the log )

```docker-compose help``` (brings up some helpful information)

### Full node without RPC port

Let's start with an easy example. If you just want to create a full node without the need of using the RPC port, you can use the following example. This example will launch *pearld* and exposes only the default p2p port 44108 to the outside world:

```yaml
version: "2"

services:
  pearld:
    container_name: pearld
    hostname: pearld
    image: ghcr.io/pearl-research-labs/pearl/pearld:latest
    restart: unless-stopped
    volumes:
      - pearld-data:/root/.pearld
    ports:
      - 44108:44108

volumes:
  pearld-data:
```

### Full node with RPC port

To use the RPC port of *pearld* you need to specify a *username* and a very strong *password*. If you want to connect to the RPC port from the internet, you need to expose port 44107(RPC) as well.

```yaml
version: "2"

services:
  pearld:
    container_name: pearld
    hostname: pearld
    image: ghcr.io/pearl-research-labs/pearl/pearld:latest
    restart: unless-stopped
    volumes:
      - pearld-data:/root/.pearld
    ports:
      - 44108:44108
      - 44107:44107
    command: [
        "--rpcuser=[CHOOSE_A_USERNAME]",
        "--rpcpass=[CREATE_A_VERY_HARD_PASSWORD]"
    ]

volumes:
  pearld-data:
```

### Full node with RPC port running on TESTNET

To run a node on testnet, you need to provide the *--testnet* argument. The ports for testnet are 44110 (p2p) and 44109 (RPC):

```yaml
version: "2"

services:
  pearld:
    container_name: pearld
    hostname: pearld
    image: ghcr.io/pearl-research-labs/pearl/pearld:latest
    restart: unless-stopped
    volumes:
      - pearld-data:/root/.pearld
    ports:
      - 44110:44110
      - 44109:44109
    command: [
        "--testnet",
        "--rpcuser=[CHOOSE_A_USERNAME]",
        "--rpcpass=[CREATE_A_VERY_HARD_PASSWORD]"
    ]

volumes:
  pearld-data:
```
