import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';
import { Socket } from 'net';
import { promisify } from 'util';
import dns from 'dns';
import { ManagerApi } from '../../types/app-bridge.ts';
import { WalletService } from './wallet-service/wallet-service.ts';
import { WalletProcess } from './wallet-process.ts';
import { displayToFs, fsToDisplay } from '../../utils/filename-utils.ts';
import { getCurrentNetworkConfig, getCurrentNetwork, setCurrentNetwork, getAllNetworks, type Network } from '../config/network-config';
import { getCustomPeer, getPeerSettings as getConfigPeerSettings, setCustomPeer, resetToDefaultPeer } from '../config/peer-settings';
import { randomBytes } from 'crypto';


let sessionRpcCreds: { rpcUser: string; rpcPassword: string } | null = null;
function getSessionRpcCreds() {
  if (!sessionRpcCreds) {
    sessionRpcCreds = {
      rpcUser: randomBytes(16).toString('hex'),
      rpcPassword: randomBytes(32).toString('hex'),
    };
  }
  return sessionRpcCreds;
}


const dnsLookup = promisify(dns.lookup);
interface WalletData {
  name: string;
}

const baseWalletDir = path.join(os.homedir(), '.pearl-wallet', 'wallet-data');

function getBaseConfig() {
  const networkConfig = getCurrentNetworkConfig();
  const customPeer = getCustomPeer();
  return {
    rpcHost: 'http://127.0.0.1',
    ...getSessionRpcCreds(),
    rpcPort: networkConfig.rpcPort,
    network: networkConfig.name,
    peerAddress: customPeer?.address,
    peerPort: customPeer?.port,
  };
}

class ManagerService implements ManagerApi {
  private walletService: WalletService | null = null;
  private currentWallet: WalletData | null = null;
  private walletProcess: WalletProcess | null = null;

  private async loadWallet(walletName: string, mode: 'create' | 'open') {
    const walletDataDir = path.join(baseWalletDir, displayToFs(walletName));
    const existingWallets = await this.getExistingWallets();
    const isValidWalletExist = existingWallets.walletNames.includes(walletName);
    const isInvalidSeedWalletCreated = fs.existsSync(walletDataDir) && !isValidWalletExist;

    if (isValidWalletExist && mode === 'create') {
      throw new Error('Wallet data directory already exist, please use a different wallet name');
    } else if (!isValidWalletExist && mode === 'open') {
      throw new Error('Wallet data directory does not exist, please use a different wallet name');
    }

    try {
      fs.mkdirSync(walletDataDir, { recursive: true });
    } catch (error) {
      if (isInvalidSeedWalletCreated) {
        console.log(
          'A wallet with invalid seed was created previously - hence only the wallet name dir was created - we should not throw'
        );
      } else {
        throw new Error(
          `Failed to create wallet data directory: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    }

    const config = {
      ...getBaseConfig(),
      dataDir: walletDataDir,
    };

    this.walletService = new WalletService(config);
    this.walletProcess = new WalletProcess(config, this.walletService);
  }

  async stopWalletProcess(options: { force?: boolean } = {}) {
    if (this.walletProcess && this.walletProcess.getStatus().isRunning) {
      await this.walletProcess.stop(options);
    }
  }

  async startWalletProcess() {
    if (!this.walletProcess) {
      throw new Error('Wallet process not initialized');
    }

    if (this.walletProcess.getStatus().isRunning === false) {
      const startResult = await this.walletProcess.start();
      if (!startResult.success) {
        throw new Error(`Failed to start wallet service: ${startResult.message}`);
      }
    }
  }

  async lockWallet() {
    if (this.walletService) {
      await this.walletService.lockWallet();
    }
    await this.stopWalletProcess();
    this.currentWallet = null;
  }

  // Fast path for the Lock button while the wallet is mid-block-recovery. The
  // `walletlock` RPC would sit on Wallet.walletLocker -> `<-endRecovery()` for
  // up to a minute, so we skip it entirely and SIGKILL the child. Safe because
  // bbolt write txns are atomic at commit; a mid-batch kill just replays that
  // batch on the next open.
  async forceLockWallet() {
    await this.stopWalletProcess({ force: true });
    this.currentWallet = null;
  }

  ensureWalletService() {
    const ws = this.walletService;
    if (!ws) {
      throw new Error('Wallet service not initialized');
    }
    return ws;
  }

  getWalletsStats() {
    return {
      name: this.currentWallet?.name,
    };
  }

  async selectWallet(walletName: string) {
    // Validate custom peer before starting wallet (if one is configured).
    // When no custom peer is set the daemon falls back to DNS seeding, so
    // there is nothing to pre-validate here. If a custom peer is set but
    // unreachable, we warn and continue — the daemon still discovers peers
    // via DNS seeding.
    const customPeer = getCustomPeer();

    if (customPeer) {
      console.log(`[ManagerService] Validating custom peer before starting wallet: ${customPeer.address}:${customPeer.port}`);
      const validation = await this.validatePeerAddress(customPeer.address, customPeer.port);

      if (!validation.valid) {
        console.warn(`[ManagerService] ⚠️ Custom peer unreachable: ${validation.error}. Falling back to DNS seeders.`);
      } else {
        console.log(`[ManagerService] ✅ Peer validation passed, starting wallet...`);
      }
    } else {
      console.log(`[ManagerService] No custom peer configured — relying on DNS seeders for peer discovery`);
    }

    try {
      await this.stopWalletProcess();
    } catch (error) {
      console.log('Failed to stop wallet process:', error);
    }

    try {
      await this.loadWallet(walletName, 'open');
    } catch (error) {
      console.log('Failed to load wallet:', error);
    }

    try {
      await this.startWalletProcess();
    } catch (error) {
      console.log('Failed to start wallet process:', error);
    }

    this.currentWallet = { name: walletName };
  }

  async create(options: { name: string; password: string }) {
    const { name, password } = options;

    await this.stopWalletProcess();

    await this.loadWallet(name, 'create');

    const createResult = await this.walletProcess?.createWalletAndGetSeed(password || 'walletpass');
    if (!createResult || !createResult.success || !createResult.seed) {
      throw new Error(
        `Failed to create wallet${createResult && 'error' in createResult ? `: ${createResult.error}` : ''}`
      );
    }

    const generatedSeed = createResult.seed;

    await this.startWalletProcess();

    this.currentWallet = { name };

    return { seed: generatedSeed };
  }

  async import(options: { name: string; seed: string; password?: string }) {
    const { name, seed, password = 'walletpass' } = options;
    const finalPassword = password && password.trim() ? password.trim() : 'walletpass';

    if (!name) {
      throw new Error('Wallet name is required');
    }
    if (!seed) {
      throw new Error('Seed phrase is required');
    }
    await this.stopWalletProcess();

    await this.loadWallet(name, 'create');
    const importResult = await this.walletProcess?.importWalletFromSeed(seed, finalPassword);
    if (!importResult || !importResult.success) {
      throw new Error(
        `Failed to import wallet: ${importResult && 'error' in importResult ? importResult.error : 'Unknown error'}`
      );
    }

    try {
      await this.startWalletProcess();
    } catch (error) {
      console.log('Failed to start wallet process:', error);
    }

    this.currentWallet = { name };
    return { name, seed };
  }

  async getExistingWallets() {
    let walletNames: string[] = [];
    const networkConfig = getCurrentNetworkConfig();

    if (fs.existsSync(baseWalletDir)) {
      const entries = await fs.promises.readdir(baseWalletDir, { withFileTypes: true });
      walletNames = entries
        .filter(entry => entry.isDirectory())
        .map(entry => fsToDisplay(entry.name))
        .filter(name => {
          const walletDbPath = path.join(baseWalletDir, name, networkConfig.dataSubdir, 'wallet.db');
          return fs.existsSync(walletDbPath);
        });
    }

    return { walletNames, defaultWallet: walletNames.length > 0 ? walletNames[0] : undefined };
  }

  // Network management methods
  async getNetworkInfo() {
    const current = getCurrentNetwork();
    const networks = getAllNetworks();
    const config = getCurrentNetworkConfig();

    return {
      currentNetwork: current,
      availableNetworks: networks,
      networkConfig: config,
    };
  }

  async setNetwork(network: string) {
    // Validate network
    if (network !== 'mainnet' && network !== 'testnet') {
      throw new Error(`Invalid network: ${network}`);
    }

    // Stop wallet if running
    await this.stopWalletProcess();

    // Update network
    setCurrentNetwork(network as Network);

    return { success: true, network };
  }

  // Peer settings management
  async getPeerSettings() {
    return getConfigPeerSettings();
  }

  async validatePeerAddress(address: string, port: number): Promise<{ valid: boolean; error?: string }> {
    try {
      // Step 1: Check if host resolves via DNS
      console.log(`[ManagerService] Validating peer: ${address}:${port}`);

      try {
        await dnsLookup(address);
        console.log(`[ManagerService] ✅ DNS lookup successful for ${address}`);
      } catch (dnsError) {
        console.error(`[ManagerService] ❌ DNS lookup failed for ${address}:`, dnsError);
        return {
          valid: false,
          error: `Please make sure your internet connection is stable and that ${address} is responsive`
        };
      }

      // Step 2: Try to connect to the port
      return new Promise((resolve) => {
        const socket = new Socket();
        const timeout = 5000; // 5 second timeout

        const cleanup = () => {
          socket.destroy();
        };

        socket.setTimeout(timeout);

        socket.on('connect', () => {
          console.log(`[ManagerService] ✅ Successfully connected to ${address}:${port}`);
          cleanup();
          resolve({ valid: true });
        });

        socket.on('timeout', () => {
          console.error(`[ManagerService] ❌ Connection timeout for ${address}:${port}`);
          cleanup();
          resolve({
            valid: false,
            error: `Connection timeout: ${address}:${port} is not responding. Please verify the address and port.`
          });
        });

        socket.on('error', (err) => {
          console.error(`[ManagerService] ❌ Connection error for ${address}:${port}:`, err.message);
          cleanup();
          resolve({
            valid: false,
            error: `Cannot connect to ${address}:${port}. Error: ${err.message}`
          });
        });

        socket.connect(port, address);
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[ManagerService] ❌ Peer validation failed:`, errorMessage);
      return { valid: false, error: `Validation failed: ${errorMessage}` };
    }
  }

  async setCustomPeerAddress(address: string, port: number) {
    // Stop wallet if running
    await this.stopWalletProcess();

    // Update peer settings (don't validate here - validate when unlocking)
    setCustomPeer(address, port);

    return { success: true };
  }

  async resetPeerToDefault() {
    // Stop wallet if running
    await this.stopWalletProcess();

    // Reset to default
    resetToDefaultPeer();

    return { success: true };
  }
}

export { ManagerService };
