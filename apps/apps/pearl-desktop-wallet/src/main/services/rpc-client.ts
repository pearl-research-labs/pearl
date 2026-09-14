import axios from 'axios';

interface RpcConfig {
  rpcHost: string;
  rpcPort: number;
  rpcUser: string;
  rpcPassword: string;
}

class RpcClient {
  constructor(private config: RpcConfig) { }

  async call<Result>(
    method: string,
    params: unknown[] = [],
    timeoutMs: number = 60000,
  ): Promise<Result> {
    const rpcData = {
      jsonrpc: '2.0',
      method,
      params,
      id: Date.now(),
    };

    const rpcUrl = `${this.config.rpcHost}:${this.config.rpcPort}`;
    try {
      const response = await axios.post(rpcUrl, rpcData, {
        auth: {
          username: this.config.rpcUser,
          password: this.config.rpcPassword,
        },
        headers: { 'Content-Type': 'application/json' },
        // Heavy methods such as `listalltransactions` walk the full wallet
        // history and can exceed the previous 10s cap on wallets with a large
        // history, causing the dashboard to time out and never render even
        // when the node is fully synced. Default to 60s and allow per-call
        // overrides for hot paths.
        timeout: timeoutMs,
      });

      if (response.data.error) {
        throw new Error(response.data.error.message || 'RPC call failed');
      }

      return response.data.result as Result;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error(`Cannot connect to RPC server: ${rpcUrl}`);
        }
        throw new Error(`RPC call failed: ${rpcUrl} ${error.message}`);
      }
      throw error as Error;
    }
  }
}

export { RpcClient };
export type { RpcConfig };
