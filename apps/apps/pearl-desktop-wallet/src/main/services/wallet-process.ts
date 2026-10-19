import { spawn, ChildProcess } from 'child_process';
import { join } from 'path';
import { app } from 'electron';
import log from 'electron-log';
import fs from 'fs';
import path from 'path';
import { WalletService } from './wallet-service/wallet-service';
import { getCurrentNetworkConfig } from '../config/network-config';
import { removeSetupFile, setupFileGeneration, untrackSetupFile, writeSetupFile } from './setup-file';

const binaryNameMap: Record<string, Record<string, string>> = {
  win32: {
    x64: 'oyster-windows-x64.exe',
    ia32: 'oyster-windows-ia32.exe',
  },
  linux: {
    x64: 'oyster-linux-x64',
    arm64: 'oyster-linux-arm64',
  },
  darwin: {
    x64: 'oyster-darwin-x64',
    arm64: 'oyster-darwin-arm64',
  },
};

interface WalletProcessConfig {
  dataDir: string;
  rpcUser: string;
  rpcPassword: string;
  peerAddress?: string;
  peerPort?: number;
}

class WalletProcess {
  private process: ChildProcess | null = null;
  private setupChild: ChildProcess | null = null;
  private setupFile: string | null = null;
  private isProcessRunning = false;
  private walletPassphrase: string = 'walletpass'; // Default passphrase - Don't delete this line. It's used to create the wallet.

  constructor(
    private readonly config: WalletProcessConfig,
    private readonly walletService: WalletService
  ) { }

  isRunning() {
    return this.isProcessRunning;
  }

  getStatus() {
    return {
      isRunning: this.isProcessRunning,
      pid: this.process?.pid,
    };
  }

  setPassphrase(passphrase: string) {
    this.walletPassphrase = passphrase;
  }

  getBinaryPath(): string {
    const { platform, arch } = process;
    let binaryName = binaryNameMap[platform][arch];
    if (!binaryName) {
      throw new Error(`Unsupported platform: ${platform} architecture: ${arch}`);
    }

    const isDev = !app.isPackaged;
    const basePath = isDev ? join(__dirname, '../../bin') : join(process.resourcesPath, 'bin');

    const binaryPath = join(basePath, binaryName);

    try {
      if (!fs.existsSync(binaryPath)) {
        throw new Error(`Binary not found at path: ${binaryPath}`);
      }

      fs.accessSync(binaryPath, fs.constants.F_OK | fs.constants.X_OK);
    } catch (error) {
      throw new Error(
        `Binary not accessible: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }

    return binaryPath;
  }

  getWalletArgs(): string[] {
    const networkConfig = getCurrentNetworkConfig();
    const args = [
      '--usespv',
      `--appdata=${this.config.dataDir}`,
      `--username=${this.config.rpcUser}`,
      `--password=${this.config.rpcPassword}`,
      `--rpclisten=127.0.0.1:${networkConfig.rpcPort}`,
      '--noservertls',
    ];

    // Only pass --addpeer when the user has configured a custom peer.
    // Otherwise the daemon falls back to its built-in DNS seeding.
    if (this.config.peerAddress && this.config.peerPort) {
      args.push(`--addpeer=${this.config.peerAddress}:${this.config.peerPort}`);
    }

    // Add network flag if not mainnet
    if (networkConfig.walletFlag) {
      args.splice(1, 0, networkConfig.walletFlag);
    }

    return args;
  }

  createWalletAndGetSeed(passphrase?: string) {
    return this.startWalletProcess(passphrase);
  }

  importWalletFromSeed(seed: string, passphrase?: string) {
    return this.startWalletProcess(passphrase, seed);
  }

  private async startWalletProcess(
    passphrase: string = 'walletpass',
    seed?: string
  ): Promise<{ success: true; message: string; seed?: string } | { success: false; error: string }> {
    const binaryPath = this.getBinaryPath();

    try {
      const networkConfig = getCurrentNetworkConfig();
      const networkDir = path.join(this.config.dataDir, networkConfig.dataSubdir);
      if (fs.existsSync(networkDir)) {
        fs.rmSync(networkDir, { recursive: true, force: true });
      }

      const walletDbPath = path.join(this.config.dataDir, 'wallet.db');
      if (fs.existsSync(walletDbPath)) {
        fs.unlinkSync(walletDbPath);
      }

      const isImport = !!seed;
      const walletConfig = {
        seed,
        privatepassphrase: passphrase,
        bday: isImport ? '1724644369' : undefined,
      };

      const walletConfigFile = writeSetupFile(this.config.dataDir, JSON.stringify(walletConfig, null, 2));
      const setupGeneration = setupFileGeneration(walletConfigFile);

      return new Promise<
        { success: true; message: string; seed?: string } | { success: false; error: string }
      >(resolve => {
        try {
          const args = this.getWalletArgs();

          args.push(`--createfromfile=${walletConfigFile}`);

          const childProcess = spawn(binaryPath, args, {
            stdio: isImport ? ['ignore', 'pipe', 'pipe'] : ['pipe', 'pipe', 'pipe'],
          });
          this.setupChild = childProcess;
          this.setupFile = walletConfigFile;

          let output = '';
          let errorOutput = '';
          let hasResolved = false;
          let extractedSeed = '';

          childProcess.stdout?.on('data', data => {
            const text = data.toString();
            output += text;

            if (!isImport) {
              // Match a 12-word BIP39 mnemonic on a single line.
              let seedMatch = text.match(/^([a-z]+(?: [a-z]+){11})$/m);

              if (seedMatch) {
                extractedSeed = seedMatch[1] ? seedMatch[1].trim() : seedMatch[0].trim();
              }
            }
          });

          childProcess.stderr?.on('data', data => {
            const text = data.toString();
            errorOutput += text;
          });

          const cleanup = () => {
            // Ignore a close that arrives after a retry has rewritten this path.
            removeSetupFile(walletConfigFile, setupGeneration);
          };

          childProcess.on('close', code => {
            if (this.setupChild === childProcess) {
              this.setupChild = null;
              this.setupFile = null;
            }
            cleanup();

            if (!hasResolved) {
              hasResolved = true;
              if (code === 0) {
                if (isImport) {
                  resolve({ success: true, message: 'Wallet imported successfully' });
                } else if (extractedSeed) {
                  resolve({
                    success: true,
                    seed: extractedSeed,
                    message: 'Wallet created successfully',
                  });
                } else {
                  resolve({
                    success: false,
                    error: 'Wallet creation succeeded but no seed found in output',
                  });
                }
              } else {
                resolve({
                  success: false,
                  error: `Wallet ${isImport ? 'import' : 'creation'} failed with code: ${code}. Output: ${errorOutput || output}`,
                });
              }
            }
          });

          childProcess.on('error', error => {
            // A failed kill emits 'error' while the child is still alive. Wiping
            // here would publish zeros to a process that has not read the file.
            // Spawn failures already have an exit code; their close handler also
            // removes the file.
            if (childProcess.exitCode !== null || childProcess.signalCode !== null) {
              cleanup();
            }

            if (!hasResolved) {
              hasResolved = true;
              resolve({ success: false, error: (error as Error).message });
            }
          });

          setTimeout(() => {
            if (!hasResolved) {
              hasResolved = true;
              try {
                childProcess.kill('SIGTERM');
              } catch { }

              setTimeout(() => {
                if (childProcess.exitCode === null && childProcess.signalCode === null) {
                  try {
                    childProcess.kill('SIGKILL');
                  } catch { }
                }
              }, 2000);

              // Wipe only from the close handler, after the child has exited.
              resolve({
                success: false,
                error: `Wallet ${isImport ? 'import' : 'creation'} timed out after 30 seconds`,
              });
            }
          }, 30000);
        } catch (error) {
          removeSetupFile(walletConfigFile);
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          resolve({ success: false, error: errorMessage });
        }
      });
    } catch (error) {
      return {
        success: false,
        error: `Failed to setup wallet process: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  async start() {
    if (this.isProcessRunning) {
      return { success: false as const, message: 'Wallet is already running' };
    }

    const binaryPath = this.getBinaryPath();

    if (!fs.existsSync(binaryPath)) {
      throw new Error(`Wallet binary not found at: ${binaryPath}`);
    }

    try {
      await this.killExistingWalletProcesses();
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch { }

    return new Promise<{ success: true; message: string }>((resolve, reject) => {
      try {
        const args = [...this.getWalletArgs()];

        this.process = spawn(binaryPath, args, {
          stdio: ['ignore', 'pipe', 'pipe'],
        });

        let isResolved = false;
        let pollInterval: NodeJS.Timeout | null = null;
        let timeoutTimer: NodeJS.Timeout | null = null;

        this.process.stdout?.on('data', data => {
          const output = data.toString();
          console.log(`### ${output}`);
        });

        this.process.stderr?.on('data', data => {
          console.log(`error with data: ${data}`);
        });

        function cleanup() {
          if (pollInterval) clearInterval(pollInterval);
          if (timeoutTimer) clearTimeout(timeoutTimer);
        }

        const startPolling = () => {
          const maxWaitMs = 60000;
          const intervalMs = 1000;
          const startTime = Date.now();

          pollInterval = setInterval(async () => {
            if (isResolved) return;

            const elapsed = Date.now() - startTime;

            try {
              await this.walletService.getBalance('*', 0);
              if (!isResolved) {
                isResolved = true;
                this.isProcessRunning = true;
                cleanup();
                resolve({ success: true, message: 'Wallet started successfully' });
              }
            } catch (e: any) {
              // Wallet not ready yet, keep trying
              if (elapsed >= maxWaitMs && !isResolved) {
                isResolved = true;
                cleanup();
                reject(
                  new Error(
                    `Timed out waiting for wallet to be ready: ${e?.message || 'Unknown error'}`
                  )
                );
              }
            }
          }, intervalMs);

          timeoutTimer = setTimeout(() => {
            if (!isResolved) {
              isResolved = true;
              cleanup();
              reject(new Error('Timed out starting wallet'));
            }
          }, maxWaitMs + 5000);
        };

        startPolling();

        this.process.on('error', error => {
          if (!isResolved) {
            isResolved = true;
            if (pollInterval) clearInterval(pollInterval);
            if (timeoutTimer) clearTimeout(timeoutTimer);
            reject(new Error(`Failed to start wallet: ${error.message}`));
          }
        });

        this.process.on('exit', () => {
          this.isProcessRunning = false;
          this.process = null;
          if (!isResolved) {
            isResolved = true;
            cleanup();
            reject(new Error('Wallet process exited before it was ready'));
          }
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  async stopSetupChild(): Promise<void> {
    const child = this.setupChild;
    if (!child || child.exitCode !== null || child.signalCode !== null) {
      this.setupChild = null;
      this.setupFile = null;
      return;
    }

    const file = this.setupFile;
    await new Promise<void>(resolve => {
      let settled = false;
      let killTimer: NodeJS.Timeout | undefined;
      let giveUpTimer: NodeJS.Timeout | undefined;
      const finish = (giveUp: boolean) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(killTimer);
        clearTimeout(giveUpTimer);
        if (giveUp && file && child.exitCode === null && child.signalCode === null) {
          // Quit is proceeding and the child is still alive. Drop the path so
          // will-quit does not zero a file oyster may still read.
          untrackSetupFile(file);
        }
        if (this.setupChild === child) {
          this.setupChild = null;
          this.setupFile = null;
        }
        resolve();
      };

      child.once('close', () => finish(false));
      killTimer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) {
          try {
            child.kill('SIGKILL');
          } catch { }
        }
      }, 2000);
      giveUpTimer = setTimeout(() => finish(true), 5000);
      try {
        child.kill('SIGTERM');
      } catch { }
    });
  }

  async stop(options: { force?: boolean } = {}) {
    if (!this.isProcessRunning || !this.process) {
      return { success: false as const, message: 'Wallet is not running' };
    }

    return new Promise<{ success: true; message: string }>(resolve => {
      if (!this.process) {
        resolve({ success: true, message: 'Wallet is not running' });
        return;
      }

      const proc = this.process;

      proc.once('exit', () => {
        this.isProcessRunning = false;
        this.process = null;
        resolve({
          success: true,
          message: options.force ? 'Wallet force stopped' : 'Wallet stopped successfully',
        });
      });

      // When `force` is set, skip the polite SIGTERM. The Go wallet's Stop()
      // waits for `<-endRecovery()`, which only returns after the current
      // 2000-block recovery batch completes; during block scanning that can
      // take a minute or more. bbolt commits are atomic at txn boundaries, so
      // a mid-batch SIGKILL is safe to replay on the next launch.
      if (options.force) {
        proc.kill('SIGKILL');
        return;
      }

      proc.kill('SIGTERM');

      // `ChildProcess.killed` flips to true as soon as *any* signal is
      // dispatched, regardless of whether the process actually died. Use our
      // own `isProcessRunning` flag (cleared in the 'exit' handler above) to
      // detect a process that ignored SIGTERM.
      setTimeout(() => {
        if (this.isProcessRunning && this.process === proc) {
          proc.kill('SIGKILL');
        }
      }, 5000);
    });
  }

  async killExistingWalletProcesses() {
    return new Promise<void>(resolve => {
      const lsofProcess = spawn('lsof', ['-i', ':8335'], { stdio: 'pipe' });
      let output = '';

      lsofProcess.stdout?.on('data', data => {
        output += data.toString();
      });

      lsofProcess.on('close', code => {
        // The shipped daemon binary is named `oyster-<platform>-<arch>`, which
        // lsof reports (truncated) as `oyster-...` in its COMMAND column. The
        // previous match string ('pearlwall') never appears, so a wedged prior
        // daemon holding the RPC port was never reaped. Match on 'oyster'.
        if (code === 0 && output.includes('oyster')) {
          const lines = output.split('\n');
          const pids: string[] = [];

          for (const line of lines) {
            if (line.includes('oyster')) {
              const parts = line.split(/\s+/);
              if (parts.length > 1) {
                pids.push(parts[1]);
              }
            }
          }

          if (pids.length > 0) {
            for (const pid of pids) {
              try {
                spawn('kill', ['-9', pid]);
              } catch { }
            }
          }
        }
        resolve();
      });

      lsofProcess.on('error', () => {
        resolve();
      });
    });
  }
}

export { WalletProcess };
export type { WalletProcessConfig };
