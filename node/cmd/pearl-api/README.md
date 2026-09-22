# pearl-api —— Pearl 移动钱包数据服务

把 `pearld` 的 JSON-RPC（需开启 `--txindex --addrindex`）包装为移动应用友好的
HTTPS JSON API。私钥与交易签名全部在移动设备本地完成，本服务只做**只读查询**与
**交易广播**，不接触任何密钥材料。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/status` | 链高度、网络、服务器时间 |
| GET | `/v1/balance?address=` | 已确认/未确认余额（单位 Grain，字符串） |
| GET | `/v1/utxos?address=` | 当前 UTXO 集（txid、vout、金额、scriptPubKey、确认数） |
| GET | `/v1/history?address=&skip=&count=` | 交易历史（最新在前，游标分页） |
| GET | `/v1/fee-estimate` | 推荐费率（Grain/vbyte） |
| POST | `/v1/broadcast` | `{ "txHex": "..." }` → `{ "txid": "..." }` |

金额一律使用最小单位 **Grain**（1 PRL = 1e8 Grain），以字符串承载避免 JSON 精度丢失。
响应模型与 `apps/packages/pearl-mobile-core/src/api-types.ts` 一一对应。

## 运行

```bash
# 1) 全节点（必须开启交易索引与地址索引）
pearld --txindex --addrindex --rpcuser=u --rpcpass=p

# 2) API 服务（开发环境可用 --notls；生产建议放在 proxy/ 的 Caddy 反代之后）
go run ./node/cmd/pearl-api \
  --rpcuser=u --rpcpass=p --notls \
  --listen=0.0.0.0:8333 --apitoken=<bearer-token>
```

参数说明：

- `--network`：`mainnet`（默认）| `testnet2` | `regtest`，需与 pearld 一致
- `--rpchost`：pearld RPC 地址（主网默认 `127.0.0.1:44107`，测试网 `44109`）
- `--apitoken`：设置后所有请求需携带 `Authorization: Bearer <token>`
- 与 pearld 之间默认要求 TLS（`--rpccert` 指向节点 `rpc.cert`）；同机部署可用 `--notls`

## 部署建议

- 生产环境将本服务置于 `proxy/` 目录的 Caddy 之后获得 TLS 与缓存；只暴露上述
  `/v1/*` 路径。
- 不持有密钥，因此无热钱包风险；仍建议对广播接口做限流。
- 历史/UTXO 通过 `searchrawtransactions` 派生，依赖 addrindex；单地址交易数
  超过 10 万会被防御性截断（返回错误），属于 MVP 限制。
