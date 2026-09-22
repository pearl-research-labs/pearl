# Pearl 移动钱包（pearl-mobile）

面向 Pearl 网络的 iOS/Android 自托管移动钱包。密钥与交易签名全部在设备本地完成，
数据服务只做只读查询与广播。

## 架构

```
┌─ apps/apps/pearl-mobile        React Native + Expo 应用（本目录）
│   创建/导入助记词、余额、历史、收款二维码、扫码转账、网络切换、备份/删除
│
├─ apps/packages/pearl-mobile-core
│   纯 TypeScript 密码学核心：BIP39/32/86 派生、BIP341 Taproot sighash、
│   Schnorr 签名、交易构建、币选择、地址发现（gap-limit）
│   测试向量由仓库 node 权威实现生成：
│     go run ./node/cmd/genmobilevectors          # 生成 test/vectors.json
│     go run ./node/cmd/genmobilevectors --verify # Go 引擎自校验
│
└─ node/cmd/pearl-api
    托管数据服务：包装 pearld JSON-RPC（需 --txindex --addrindex）
    /v1/balance /v1/utxos /v1/history /v1/fee-estimate /v1/broadcast
```

关键参数与链对齐：Bech32m HRP `prl`/`tprl`，BIP86 路径 `m/86'/808276'/0'/*`
（测试网 coin type 1），金额单位 Grain（1 PRL = 1e8），仅 Taproot（P2TR）地址。

## 开发

```bash
cd apps
pnpm install                        # 安装 workspace 依赖
pnpm --filter @pearl/pearl-mobile-core test   # 核心库黄金向量测试
pnpm --filter @pearl/pearl-mobile typecheck   # 应用类型检查

cd apps/pearl-mobile
pnpm start                          # 启动 Metro，扫码用 Expo Go 预览
pnpm ios                            # 本机编译 iOS（需 macOS + Xcode）
pnpm android                        # 本机编译 Android
```

首次使用前生成图标：`pnpm assets`（需 Python + Pillow）。

## 构建与发布

### GitHub Actions 自签名 IPA（默认，推荐）

工作流 `.github/workflows/pearl-mobile.yml`：macOS runner 上执行
`expo prebuild → pod install → xcodebuild archive`（关闭签名），
把 `Payload/*.app` 打成**未签名 IPA** 并作为 artifact 上传（保留 30 天）。

- 手动触发：Actions → "Pearl Mobile Wallet (iOS)" → Run workflow；
- 自动触发：push 到 `master` 且涉及移动端路径；
- 每次构建前先跑 `@pearl/pearl-mobile-core` 黄金向量测试与 iOS bundle 冒烟验证。

下载 IPA 后自行重签名安装到真机（任选其一）：

| 工具 | 说明 |
| --- | --- |
| Sideloadly | Windows/macOS，免费 Apple ID 即可签名（7 天有效期需续签） |
| AltStore / SideStore | 通过电脑端服务器为手机侧载应用 |
| 爱思助手 | Windows，一键 IPA 签名安装 |

免费 Apple ID 签名的应用 7 天后过期（数据保留，重新签名安装即可）；
付费开发者账号签名则一年有效。此方式**不需要** Expo/EAS 账号与仓库 secrets。

### EAS Build（可选）

如需 TestFlight/商店分发，可改用 EAS 托管构建：

```bash
npm i -g eas-cli
eas login && eas init          # 链接项目（写入 eas.json 的 projectId）
eas build --platform ios --profile production
eas submit --platform ios      # 上传 TestFlight（需 Apple 付费账号）
```

## 数据服务部署

应用默认请求 `https://api.pearlnetwork.net`（可在设置页改自定义节点）。
自建参考 `node/cmd/pearl-api/README.md`：pearld 开启 `--txindex --addrindex` 后，
把 pearl-api 置于 `proxy/` 目录的 Caddy 反代之后即可。

## 安全模型

- 助记词仅存 iOS Keychain / Android Keystore（`WHEN_UNLOCKED_THIS_DEVICE_ONLY`），
  App 退后台即清除内存副本
- 签名在设备本地完成（BIP340 Schnorr，SIGHASH_DEFAULT），服务端永不接触密钥
- 备份助记词需生物识别（Face ID/指纹）确认
- 转账需二次确认页展示完整费用明细
