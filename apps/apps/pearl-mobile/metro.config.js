// Metro 配置：支持 pnpm monorepo（workspace 包 + 提升的依赖根）
const {getDefaultConfig} = require('expo/metro-config');
const path = require('path');

const projectRoot = __dirname;
const workspaceRoot = path.resolve(projectRoot, '../..');

const config = getDefaultConfig(projectRoot);

// 监视整个 workspace（@pearl/pearl-mobile-core 源码直接参与打包）
config.watchFolders = [workspaceRoot];

config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(workspaceRoot, 'node_modules'),
];

// pnpm 使用符号链接，需保留真实路径解析
config.resolver.unstable_enableSymlinks = true;
config.resolver.unstable_enablePackageExports = true;

module.exports = config;
