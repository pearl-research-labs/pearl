import axios from 'axios';

interface RpcConfig {
  rpcHost: string;
  rpcPort: number;
  rpcUser: string;
  rpcPassword: string;
}

class RpcError extends Error {
  constructor(
    readonly code: number,
    message: string
  ) {
    super(message);
    this.name = 'RpcError';
  }
}

class RpcClient {
  constructor(private config: RpcConfig) {}

  async call<Result>(method: string, params: unknown[] = [], timeoutMs: number = 60000): Promise<Result> {
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
        headers: {'Content-Type': 'application/json'},
        // Methods such as `listalltransactions` walk the full wallet history and can exceed 10s on large wallets.
        // Default to 60s so the dashboard can render, and allow per-call overrides for hot paths.
        timeout: timeoutMs,
      });

      if (response.data.error) {
        const {code, message} = response.data.error;
        const detail = typeof message === 'string' && message ? message : 'RPC call failed';
        if (Number.isInteger(code)) throw new RpcError(code, detail);
        throw new Error(detail);
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

export {RpcClient, RpcError};
export type {RpcConfig};
