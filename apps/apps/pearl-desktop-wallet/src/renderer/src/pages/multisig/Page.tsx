import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Layers2, Plus, ShieldCheck, ClipboardList } from "lucide-react";
import { Header } from "@/pages/multisig/components/header";
import { copyToClipboard, formatGrains, shortHex } from "./helpers";
import { listVaults, listPendingTxs, type VaultRecord } from "@/services/multisig";
import { fetchVaultBalance } from "@/services/multisig";

type VaultSummary = VaultRecord & {
  balance: bigint | null;
  balanceError: string | null;
  pendingCount: number;
};

type BalanceFetchResult =
  | {
      grains: bigint | null;
      degraded: boolean;
      error?: undefined;
    }
  | {
      grains: null;
      degraded: boolean;
      error: string;
    };

export default function MultisigPage() {
  const navigate = useNavigate();
  const [vaults, setVaults] = useState<VaultSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async (options: { silent?: boolean } = {}) => {
    const silent = options.silent ?? false;
    if (silent) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    try {
      const records = await listVaults();
      const summaries = await Promise.all(
        records.map(async (vault) => {
          const [balanceResult, pendingTxs] = await Promise.all([
            fetchVaultBalance(vault).catch((err: unknown): BalanceFetchResult => ({
              grains: null,
              degraded: true,
              error: err instanceof Error ? err.message : "Unable to load balance",
            })),
            listPendingTxs(vault.id).catch(() => []),
          ]);
          return {
            ...vault,
            balance: balanceResult.grains,
            balanceError: ("error" in balanceResult ? balanceResult.error : null) ?? null,
            pendingCount: pendingTxs.length,
          };
        }),
      );
      setVaults(summaries);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load multisig vaults");
    } finally {
      if (silent) {
        setRefreshing(false);
      } else {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void refresh({ silent: true });
    }, 15_000);
    const onFocus = () => {
      void refresh({ silent: true });
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
      <Header onBack={() => navigate("/wallet")} name="Multisig" />

      <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
          <section className="rounded-3xl border border-slate-200 bg-slate-950 px-6 py-7 text-white shadow-xl shadow-slate-950/10 sm:px-8">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div className="max-w-2xl space-y-3">
                <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs uppercase tracking-[0.18em] text-white/70">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  Pearl multisig
                </div>
                <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                  Vaults, proposals, and signatures in one place.
                </h1>
                <p className="max-w-2xl text-sm leading-6 text-white/70 sm:text-base">
                  Create a Pearl vault, share your cosigner descriptor, sign proposal PSBTs,
                  and broadcast finalized transactions from the desktop wallet.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => navigate("/multisig/create")}
                  className="inline-flex items-center gap-2 rounded-2xl bg-amber-300 px-4 py-3 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5"
                >
                  <Plus className="h-4 w-4" />
                  Create vault
                </button>
                <button
                  type="button"
                  onClick={() => void refresh({ silent: true })}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/15 bg-white/5 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-white/10"
                >
                  <ClipboardList className="h-4 w-4" />
                  {refreshing ? "Refreshing..." : "Refresh"}
                </button>
              </div>
            </div>
          </section>

          {error ? (
            <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              {error}
            </div>
          ) : null}

          {loading && vaults.length === 0 ? (
            <div className="rounded-3xl border border-slate-200 bg-white p-8 text-sm text-slate-500 shadow-sm">
              Loading vaults...
            </div>
          ) : vaults.length === 0 ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-3xl border border-dashed border-slate-300 bg-white p-8 shadow-sm">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-100 text-amber-700">
                  <Layers2 className="h-6 w-6" />
                </div>
                <h2 className="mt-5 text-xl font-semibold text-slate-900">No vaults yet</h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
                  Create a vault to start exporting descriptors, composing PSBTs, and managing
                  multisig spends locally.
                </p>
                <button
                  type="button"
                  onClick={() => navigate("/multisig/create")}
                  className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
                >
                  <Plus className="h-4 w-4" />
                  Create vault
                </button>
              </div>
            </div>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              {vaults.map((vault) => (
                <article
                  key={vault.id}
                  className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-lg"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                        <Layers2 className="h-3.5 w-3.5" />
                        Vault
                      </div>
                      <h2 className="mt-2 truncate text-2xl font-semibold text-slate-950">
                        {vault.label}
                      </h2>
                      <p className="mt-1 text-sm text-slate-500">
                        {vault.threshold}-of-{vault.total} on Pearl mainnet
                      </p>
                    </div>
                    <div className="rounded-2xl bg-slate-950 px-3 py-2 text-right text-xs text-white/75">
                      <div>Balance</div>
                      <div className="mt-1 text-base font-semibold text-white">
                        {vault.balance === null
                          ? "Unavailable"
                          : `${formatGrains(vault.balance)} PRL`}
                      </div>
                      {vault.balanceError ? (
                        <div className="mt-1 text-[10px] text-white/50">{vault.balanceError}</div>
                      ) : null}
                    </div>
                  </div>

                  <div className="mt-5 space-y-3 rounded-2xl bg-slate-50 p-4 text-sm text-slate-700">
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-slate-500">Address</span>
                      <button
                        type="button"
                        onClick={async () => {
                          await copyToClipboard(vault.pearlAddress);
                          setCopied(vault.id);
                          window.setTimeout(() => setCopied(null), 1200);
                        }}
                        className="font-mono text-xs text-slate-900 transition-colors hover:text-amber-700"
                      >
                        {copied === vault.id ? "Copied" : shortHex(vault.pearlAddress, 10)}
                      </button>
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-slate-500">My signer</span>
                      <span className="font-mono text-xs text-slate-900">
                        {shortHex(vault.myPubkeyHex)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-slate-500">Pending</span>
                      <span className="font-semibold text-slate-900">{vault.pendingCount}</span>
                    </div>
                  </div>

                  <div className="mt-5 flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => navigate(`/multisig/${vault.id}`)}
                      className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
                    >
                      Open vault
                      <ArrowRight className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => navigate(`/multisig/${vault.id}/send`)}
                      className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 px-4 py-3 text-sm font-semibold text-slate-900 transition-colors hover:bg-slate-100"
                    >
                      Compose spend
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
