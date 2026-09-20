import { PEARL_MAINNET } from "../chains/pearl/network";

interface RpcResult<T> {
  jsonrpc?: string;
  result: T | null;
  error: { code: number; message: string } | null;
  id: number | string | null;
}

interface RawTxVout {
  value: number;
  n: number;
  scriptPubKey: {
    address?: string;
    addresses?: string[];
    hex?: string;
  };
}

interface RawTxVin {
  txid?: string;
  vout?: number;
}

interface RawTx {
  txid: string;
  vin: RawTxVin[];
  vout: RawTxVout[];
  confirmations?: number;
}

const PER_ENDPOINT_ATTEMPTS = 2;
const INTRA_ENDPOINT_BACKOFF_MS = 250;
const PEARL_RPC_URL = PEARL_MAINNET.rpcUrl;

function describeError(err: unknown): string {
  if (err instanceof Error) {
    const parts = [err.name, err.message].filter(Boolean);
    const cause = (err as Error & { cause?: unknown }).cause;
    if (cause instanceof Error) {
      parts.push(`cause=${cause.name}: ${cause.message}`);
    } else if (cause && typeof cause === "object") {
      const c = cause as Record<string, unknown>;
      const extra = [c.code, c.errno, c.syscall, c.address, c.port, c.path]
        .filter((value) => value !== undefined && value !== null && value !== "")
        .map(String);
      if (extra.length > 0) {
        parts.push(`cause=${extra.join(" ")}`);
      } else {
        parts.push(`cause=${JSON.stringify(cause)}`);
      }
    }
    return parts.join(": ");
  }
  return String(err);
}

async function fetchOnce<T>(url: string, method: string, params: unknown[]): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", method, params, id: 1 }),
    });
  } catch (err) {
    throw new Error(`rpc fetch ${url} failed for ${method}: ${describeError(err)}`);
  }
  if (!res.ok) {
    throw new Error(`rpc http ${res.status}`);
  }
  const body = (await res.json()) as RpcResult<T>;
  if (body.error) {
    throw new Error(`rpc ${body.error.code}: ${body.error.message}`);
  }
  if (body.result === null) throw new Error("rpc null result");
  return body.result;
}

async function call<T>(method: string, params: unknown[]): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < PER_ENDPOINT_ATTEMPTS; attempt++) {
    if (attempt > 0) {
      await new Promise((r) => setTimeout(r, INTRA_ENDPOINT_BACKOFF_MS));
    }
    try {
      return await fetchOnce<T>(PEARL_RPC_URL, method, params);
    } catch (err) {
      lastErr = err;
      if (err instanceof Error && /^rpc -?\d+:/.test(err.message)) throw err;
      if (err instanceof Error && /^rpc fetch .* failed for /.test(err.message)) continue;
      if (err instanceof TypeError) continue;
      throw err;
    }
  }
  throw lastErr ?? new Error("rpc endpoint exhausted");
}

function prlToGrains(value: number): bigint {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error("E_INVALID_RPC_VALUE");
  }
  const [whole, frac = ""] = value.toFixed(8).split(".");
  const fracPadded = (frac + "00000000").slice(0, 8);
  return BigInt(whole) * 100_000_000n + BigInt(fracPadded);
}

function voutPaysAddress(vout: RawTxVout, address: string): boolean {
  if (vout.scriptPubKey.address === address) return true;
  return Array.isArray(vout.scriptPubKey.addresses) && vout.scriptPubKey.addresses.includes(address);
}

export const MAX_UTXO_WALK_PAGES = 60;
export const MAX_UTXO_WALK_PAGES_HARD = 200;
export const MAX_RPC_PAGE_LENGTH = 500;

export interface PrlBalanceResult {
  grains: bigint;
  degraded: boolean;
}

export interface PrlUtxo {
  txid: string;
  vout: number;
  valueGrains: bigint;
  scriptHex: string;
}

export interface PrlUtxoSet {
  utxos: PrlUtxo[];
  degraded: boolean;
  droppedNoScript: number;
}

export interface FetchUtxosOptions {
  maxPages?: number;
}

export async function fetchPrlBalanceGrains(
  address: string,
  opts: FetchUtxosOptions = {},
): Promise<PrlBalanceResult> {
  const { utxos, degraded, droppedNoScript } = await fetchPrlUtxos(address, opts);
  let total = 0n;
  for (const u of utxos) total += u.valueGrains;
  return { grains: total, degraded: degraded || droppedNoScript > 0 };
}

export async function fetchPrlUtxos(
  address: string,
  opts: FetchUtxosOptions = {},
): Promise<PrlUtxoSet> {
  const PAGE = 100;
  const maxPages = Math.min(Math.max(1, opts.maxPages ?? MAX_UTXO_WALK_PAGES), MAX_UTXO_WALK_PAGES_HARD);
  let skip = 0;
  const utxo = new Map<string, { valueGrains: bigint; scriptHex: string }>();
  const seenOutputs = new Set<string>();
  let pageCount = 0;
  let degraded = false;
  let droppedNoScript = 0;

  while (true) {
    if (pageCount >= maxPages) {
      degraded = true;
      break;
    }
    let page: RawTx[];
    try {
      page = await call<RawTx[]>("searchrawtransactions", [address, 1, skip, PAGE]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("No information available about address")) {
        return { utxos: [], degraded: false, droppedNoScript: 0 };
      }
      throw err;
    }
    if (!page || page.length === 0) break;
    if (page.length > MAX_RPC_PAGE_LENGTH) {
      degraded = true;
      page = page.slice(0, MAX_RPC_PAGE_LENGTH);
    }

    for (const tx of page) {
      for (const vout of tx.vout) {
        if (!voutPaysAddress(vout, address)) continue;
        const key = `${tx.txid}:${vout.n}`;
        if (seenOutputs.has(key)) continue;
        seenOutputs.add(key);
        const scriptHex = vout.scriptPubKey.hex;
        if (!scriptHex || !/^[0-9a-fA-F]+$/.test(scriptHex)) {
          droppedNoScript++;
          continue;
        }
        utxo.set(key, { valueGrains: prlToGrains(vout.value), scriptHex });
      }
    }
    for (const tx of page) {
      for (const vin of tx.vin) {
        if (!vin.txid || vin.vout === undefined) continue;
        utxo.delete(`${vin.txid}:${vin.vout}`);
      }
    }

    pageCount++;
    if (degraded) break;
    if (page.length < PAGE) break;
    skip += page.length;
  }

  const out: PrlUtxo[] = [];
  for (const [key, held] of utxo) {
    const [txid, voutStr] = key.split(":");
    out.push({
      txid: txid!,
      vout: Number(voutStr),
      valueGrains: held.valueGrains,
      scriptHex: held.scriptHex,
    });
  }
  return { utxos: out, degraded, droppedNoScript };
}

export async function broadcastPearlTx(rawHex: string): Promise<string> {
  if (typeof window !== "undefined" && window.appBridge?.wallet?.broadcastPearlTx) {
    return await window.appBridge.wallet.broadcastPearlTx(rawHex);
  }
  return await call<string>("sendrawtransaction", [rawHex]);
}
