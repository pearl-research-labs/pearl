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

## 构建与发布（EAS）

```bash
npm i -g eas-cli
eas login
eas build --platform ios --profile production     # TestFlight/App Store
eas build --platform android --profile production
```

iOS 发布需要 Apple Developer 账号；`app.json` 中的 `bundleIdentifier`、
相机/FaceID 权限说明均已配置。

### GitHub Actions（推荐）

仓库内置工作流 `.github/workflows/pearl-mobile.yml`：

- **手动触发**：Actions → "Pearl Mobile Wallet (iOS)" → Run workflow，
  可选 `preview`（内测 IPA）/ `production`（正式 IPA）/ `development`（模拟器包），
  `submit=true` 时 production 构建完成后自动上传 TestFlight；
- **自动触发**：push 到 `master` 且涉及移动端路径时，自动构建 `preview` 内测包；
- 每次构建前先跑 `@pearl/pearl-mobile-core` 黄金向量测试与 iOS JS bundle 冒烟验证，
  失败即终止，不会产出包。

需要在仓库 Settings → Secrets and variables → Actions 配置：

| Secret | 用途 |
| --- | --- |
| `EXPO_TOKEN` | Expo 账户 token（`npx expo login` 后在 https://expo.dev/settings/access-tokens 创建），EAS 构建与签名托管 |

签名证书由 EAS 托管（首次 `eas build` 按提示创建），仓库内不落任何证书私钥。

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
