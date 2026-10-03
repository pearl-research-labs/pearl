# Pearl Apps Monorepo

This monorepo contains the user-facing applications for the Pearl blockchain network.

## 📦 Applications

### 💼 Pearl Desktop Wallet (`@pearl/pearl-desktop-wallet`)

A modern, secure desktop wallet application for managing Pearl blockchain assets. Built with
Electron, React, and TypeScript.

**Location:** [apps/pearl-desktop-wallet](./apps/pearl-desktop-wallet)

## 🚀 Prerequisites

Before getting started, ensure you have the following installed:

- **Node.js** >= 22.0.0
- **pnpm** >= 8.15.6

You can check your versions with:

```bash
node --version
pnpm --version
```

If you need to install pnpm:

```bash
npm install -g pnpm@8.15.6
```

## 📥 Installation

1. Clone the repository and navigate to the apps directory:

```bash
cd pearl/apps
```

2. Install dependencies for all applications:

```bash
pnpm install
```

This will install dependencies for all apps and packages in the monorepo workspace.

## 🛠️ Development

### Running the Desktop Wallet

To start the Pearl Desktop Wallet in development mode:

```bash
pnpm --filter @pearl/pearl-desktop-wallet dev
```

## 🏗️ Building

### Building the Desktop Wallet

To build the desktop wallet for distribution:

```bash
# From the apps root directory
pnpm --filter @pearl/pearl-desktop-wallet build

# Or from the pearl-desktop-wallet directory
cd apps/pearl-desktop-wallet
pnpm build
```

To build platform-specific distributables:

```bash
# Build for macOS
pnpm build:mac

# Build for Windows
pnpm build:win

# Build for Linux
pnpm build:linux
```

### Building All Applications

To build all applications:

```bash
pnpm build
```

## 📦 Packages

This monorepo also includes shared packages:

- **`@pearl/ui`**: Shared UI components and design system
- **`@pearl/address-validation`**: Address validation utilities for Pearl blockchain
- **`@pearl/eslint-config`**: Shared ESLint configuration

## 🧹 Code Quality

### Formatting

Format all code:

```bash
pnpm format
```

Check formatting:

```bash
pnpm format-check
```

### Linting

Run linting across all apps:

```bash
pnpm lint
```

## 🔧 Useful Commands

| Command             | Description                      |
| ------------------- | -------------------------------- |
| `pnpm dev`          | Run all apps in development mode |
| `pnpm build`        | Build all apps for production    |
| `pnpm lint`         | Lint all apps                    |
| `pnpm format`       | Format all code with Prettier    |
| `pnpm deps:check`   | Check for dependency updates     |
| `pnpm deps:upgrade` | Upgrade all dependencies         |

## 📁 Project Structure

```
apps/
├── apps/
│   └── pearl-desktop-wallet/ # Desktop wallet application
├── packages/
│   ├── ui/                    # Shared UI components
│   ├── address-validation/    # Address validation utilities
│   └── eslint-config/         # Shared ESLint config
├── package.json               # Root package.json with scripts
├── pnpm-workspace.yaml        # pnpm workspace configuration
└── turbo.json                 # Turbo configuration
```

## 🤝 Contributing

When contributing to this repository:

1. Follow the existing code style
2. Run `pnpm format` before committing
3. Ensure all lint checks pass with `pnpm lint`
4. Test your changes in development mode
5. Update documentation as needed

## 📄 License

See the main Pearl repository for license information.

## 🔗 Related

- [Pearl Monorepo](../README.md) - Main project documentation

## 💬 Support

For support and questions, please visit:

- Website: [pearlresearch.ai](https://pearlresearch.ai)
- Email: support@pearlresearch.ai
