# Pearl

[English](README.md) | **简体中文**

> 本文档是 [`README.md`](README.md) 的简体中文翻译。如有内容差异，请以英文原文为准。

[![Blockchain / Build and Test](https://github.com/pearl-research-labs/pearl/actions/workflows/blockchain_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/blockchain_ci.yml)
[![Integration Tests CI](https://github.com/pearl-research-labs/pearl/actions/workflows/integration_tests_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/integration_tests_ci.yml)
[![Miner CI](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_ci.yml)
[![Miner GPU CI](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_gpu_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/miner_gpu_ci.yml)
[![Desktop Wallet CI/CD](https://github.com/pearl-research-labs/pearl/actions/workflows/pearl-desktop-wallet.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/pearl-desktop-wallet.yml)
[![Plonky2 Tests](https://github.com/pearl-research-labs/pearl/actions/workflows/plonky2_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/plonky2_ci.yml)
[![Rust CI](https://github.com/pearl-research-labs/pearl/actions/workflows/rust_ci.yml/badge.svg)](https://github.com/pearl-research-labs/pearl/actions/workflows/rust_ci.yml)
[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)

Pearl 是一条基于**有用工作量证明（Proof-of-Useful-Work）**协议的 L1 区块链。按照[这篇论文](https://arxiv.org/abs/2504.09971)提出的方法，其挖矿过程是执行任意矩阵乘法时产生的副产品。

本 monorepo 包含完整节点、钱包、SPV 轻客户端、零知识证明系统、vLLM 矿工以及相关辅助工具。

## 仓库结构

| 目录 | 说明 |
|------|------|
| [`node/`](node/) | **pearld**：Pearl 协议的参考实现（完整节点） |
| [`wallet/`](wallet/) | **Oyster**：提供 JSON-RPC 和 gRPC 接口的 HD 钱包守护进程 |
| [`spv/`](spv/) | **Pearl 轻客户端**：使用紧凑区块过滤器、保护隐私的 SPV 客户端 |
| [`dnsseeder/`](dnsseeder/) | Pearl 网络的 DNS 种子节点程序 |
| [`coredns-dnsseed/`](coredns-dnsseed/) | 用于生产环境 DNS 种子节点的 CoreDNS 插件 |
| [`proxy/`](proxy/) | Caddy 反向代理 sidecar，用于 RPC TLS 终止和速率限制 |
| [`xmss/`](xmss/) | XMSS 后量子签名方案（C + Go FFI） |
| [`zk-pow/`](zk-pow/) | 零知识工作量证明电路和验证器（Rust、Plonky2/STARKy） |
| [`pearl-blake3/`](pearl-blake3/) | Blake3 哈希工具（Rust） |
| [`plonky2/`](plonky2/) | Plonky2 SNARK 证明系统（Rust，vendored） |
| [`miner/`](miner/) | vLLM 矿工：GPU 挖矿基础设施（Python/CUDA、uv workspace） |
| [`py-pearl-mining/`](py-pearl-mining/) | Pearl 挖矿的 Python 绑定（Rust/PyO3） |
| [`apps/`](apps/) | 前端应用（网站、桌面钱包，使用 pnpm/Turborepo） |
| [`tools/`](tools/) | Go 开发工具依赖 |

## 安装（预编译二进制文件）

macOS / Linux：

```bash
curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
```

Windows：

```powershell
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex
```

安装脚本会安装 `pearld`、`prlctl` 和 `oyster`，使用仅限本机访问的主网默认配置和共享 RPC 凭据（`oyster` 默认使用 SPV）。二进制文件会被安装到 macOS/Linux 上的 `${XDG_BIN_HOME:-$HOME/.local/bin}`，或 Windows 上的 `%LOCALAPPDATA%\Pearl\bin`。

| 工具 | Linux | macOS | Windows |
|------|-------|-------|---------|
| pearld | `~/.pearld/pearld.conf` | `~/Library/Application Support/Pearld/pearld.conf` | `%LOCALAPPDATA%\Pearld\pearld.conf` |
| oyster | `~/.oyster/oyster.conf` | `~/Library/Application Support/Oyster/oyster.conf` | `%LOCALAPPDATA%\Oyster\oyster.conf` |
| prlctl | `~/.prlctl/prlctl.conf` | `~/Library/Application Support/Prlctl/prlctl.conf` | `%LOCALAPPDATA%\Prlctl\prlctl.conf` |

可以使用 `--version` / `--bin-dir`（Windows 上为 `-Version` / `-BinDir`）指定版本或安装目录。升级、卸载及其他详细说明请参阅 [`node/docs/installation.md`](node/docs/installation.md)。如需从源代码构建，请继续阅读下方的**构建**部分。

## 前置依赖

- [Go](https://golang.org) 1.26 或更高版本
- [Rust](https://rustup.rs) 工具链（用于零知识证明和哈希相关 crate）
- C 编译器（用于 XMSS 库）
- [Python](https://python.org) 3.12 和 [uv](https://docs.astral.sh/uv/)（用于 vLLM 矿工软件包）
- [Task](https://taskfile.dev) 任务运行器
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)（用于 vLLM 矿工）

## 构建

```bash
task build              # 构建全部组件（区块链 + vLLM 矿工）
task build:blockchain   # 构建 pearld、prlctl、oyster，输出到 bin/
task build:miner        # 安装 vLLM 矿工 Python 软件包
task build:pearld       # 仅构建 pearld
```

## 运行节点和 vLLM 矿工

基本流程：**构建** > **创建钱包** > **启动节点** > **启动 vLLM 矿工**。

### 1. 创建钱包并获取挖矿地址

```bash
./bin/oyster -u rpcuser -P rpcpass --create
```

按照提示设置密码短语并妥善记录助记词。然后启动钱包并生成一个 Taproot 挖矿地址：

```bash
./bin/oyster -u rpcuser -P rpcpass &
./bin/prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
```

### 2. 启动节点

```bash
./bin/pearld \
  --rpcuser=rpcuser \
  --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --miningaddr=<your-taproot-address> \
  --txindex
```

常用参数：`--testnet` / `--simnet` 用于非主网环境，`--notls` 用于禁用 TLS，`--debuglevel=debug` 用于输出详细日志。所有选项请参阅 `node/sample-pearld.conf`。

| 网络 | RPC | P2P | 钱包服务器 |
|------|-----|-----|------------|
| 主网 | 44107 | 44108 | 44207 |
| 测试网 | 44109 | 44110 | 44209 |
| 测试网 2 | 44111 | 44112 | 44211 |
| Simnet | 18556 | 18555 | 18554 |
| Regtest | 18334 | 18444 | 18332 |

### 3. 启动 vLLM 矿工

vLLM 矿工包含两个组件：**pearl-gateway**（连接节点的桥接服务）和 **vllm-miner**（通过 vLLM 使用 GPU 挖矿）。

```bash
export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="rpcuser"
export PEARLD_RPC_PASSWORD="rpcpass"
export PEARLD_MINING_ADDRESS="<your-taproot-address>"
pearl-gateway start
```

网关通过 JSON-RPC 连接 `pearld`，并在 `/tmp/pearlgw.sock`（UDS）或 8337 端口上公开挖矿接口（设置 `MINER_RPC_TRANSPORT=tcp` 可使用 TCP）。

如需使用 Docker 运行完整组件栈：

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

## 测试

```bash
task test               # 运行全部测试（Go + Python）
task test:go            # 使用 race detector 运行 Go 测试
task test:python        # 运行完整 Python 测试套件
task test:python:basic  # 运行 Python 测试（排除集成、性能和慢速测试）
```

## 格式化与代码检查

```bash
task fmt            # 格式化全部代码（Go + Rust + Python）
task lint:python    # 使用 ruff 检查 Python 代码
task tidy           # 整理 Go 依赖
```

还可以使用范围更小的任务：`task fmt:go`、`task fmt:rust`、`task fmt:python`、`task lint:go`、`task lint:rust`、`task lint:python`。

## 参与贡献

请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 安全

请参阅 [`SECURITY.md`](SECURITY.md)。

## 许可证

Pearl 使用 [copyfree](http://copyfree.org) ISC 许可证。详情请参阅 [`LICENSE`](LICENSE)。

## 致谢

Pearl 的区块链基础设施最初由以下开源项目派生而来：

- [btcd](https://github.com/btcsuite/btcd)：完整节点实现
- [btcwallet](https://github.com/btcsuite/btcwallet)：钱包守护进程
- [neutrino](https://github.com/lightninglabs/neutrino)：SPV 轻客户端
