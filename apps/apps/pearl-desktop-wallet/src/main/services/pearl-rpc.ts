const PEARL_RPC_POOL: readonly string[] = [
  "https://rpc.pearlbridge.xyz/",
  "https://pearl-sentry-fsn1-1.pearlbridge.xyz/rpc",
  "https://pearl-sentry-nbg1-1.pearlbridge.xyz/rpc",
  "https://pearl-sentry-hel1-1.pearlbridge.xyz/rpc",
];

interface RpcResult<T> {
  result: T | null;
  error: { code: number; message: string } | null;
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
}

const ENDPOINT_COOLDOWN_MS = 60_000;
const endpointUnhealthyUntil = new Map<string, number>();
const PER_ENDPOINT_ATTEMPTS = 2;
const INTRA_ENDPOINT_BACKOFF_MS = 250;
const MAX_UTXO_WALK_PAGES = 60;
const MAX_UTXO_WALK_PAGES_HARD = 200;
const MAX_RPC_PAGE_LENGTH = 500;

function isEndpointHealthy(url: string, now: number): boolean {
  return now >= (endpointUnhealthyUntil.get(url) ?? 0);
}

function markEndpointUnhealthy(url: string, now: number): void {
  endpointUnhealthyUntil.set(url, now + ENDPOINT_COOLDOWN_MS);
}

function orderedAttempts(now: number): string[] {
  const healthy: string[] = [];
  const cooled: string[] = [];
  for (const url of PEARL_RPC_POOL) {
    if (isEndpointHealthy(url, now)) healthy.push(url);
    else cooled.push(url);
  }
  return healthy.length > 0 ? [...healthy, ...cooled] : cooled;
}

async function fetchOnce<T>(url: string, method: string, params: unknown[]): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", method, params, id: 1 }),
  });
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

function isRetryableSameEndpoint(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  if (err instanceof Error) {
    const m = /^rpc http (\d+)$/.exec(err.message);
    if (m) {
      const status = Number(m[1]);
      return status === 408 || status === 429 || (status >= 500 && status < 600);
    }
  }
  return false;
}

function isChainLevelError(err: unknown): boolean {
  return err instanceof Error && /^rpc -?\d+:/.test(err.message);
}

async function call<T>(method: string, params: unknown[]): Promise<T> {
  const attempts = orderedAttempts(Date.now());
  let lastErr: unknown;
  for (const url of attempts) {
    let endpointFailedAllAttempts = true;
    for (let attempt = 0; attempt < PER_ENDPOINT_ATTEMPTS; attempt++) {
      if (attempt > 0) {
        await new Promise((resolve) => setTimeout(resolve, INTRA_ENDPOINT_BACKOFF_MS));
      }
      try {
        return await fetchOnce<T>(url, method, params);
      } catch (err) {
        lastErr = err;
        if (isChainLevelError(err)) {
          throw err;
        }
        if (isRetryableSameEndpoint(err)) {
          continue;
        }
        endpointFailedAllAttempts = false;
        throw err;
      }
    }
    if (endpointFailedAllAttempts) {
      markEndpointUnhealthy(url, Date.now());
    }
  }
  throw lastErr ?? new Error("rpc pool exhausted");
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

export async function fetchPrlUtxos(address: string): Promise<PrlUtxoSet> {
  const PAGE = 100;
  const maxPages = Math.min(MAX_UTXO_WALK_PAGES, MAX_UTXO_WALK_PAGES_HARD);
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

export async function fetchPrlBalanceGrains(address: string): Promise<PrlBalanceResult> {
  const { utxos, degraded, droppedNoScript } = await fetchPrlUtxos(address);
  let total = 0n;
  for (const u of utxos) total += u.valueGrains;
  return { grains: total, degraded: degraded || droppedNoScript > 0 };
}

export async function broadcastPearlTx(rawHex: string): Promise<string> {
  return await call<string>("sendrawtransaction", [rawHex]);
}
