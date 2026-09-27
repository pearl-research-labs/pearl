import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowRight, ArrowUpRight, Copy, Download, Plus, Trash2 } from "lucide-react";
import { Header } from "@/pages/multisig/components/header";
import {
  copyToClipboard,
  formatGrains,
  shortHex,
} from "./helpers";
import {
  deletePendingTx,
  deleteVault,
  importProposalArtifactOrPsbt,
  exportMyCosignerDescriptor,
  fetchVaultBalance,
  getVault,
  listPendingTxs,
  listSentTxs,
  type VaultSentTxRecord,
  type VaultPendingTxRecord,
  type VaultRecord,
} from "@/services/multisig";

const formatTimeAgo = (timestamp: number): string => {
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
};

const formatFullDate = (timestamp: number): string => {
  return new Date(timestamp).toLocaleString();
};

export default function MultisigDashboard() {
  const navigate = useNavigate();
  const { multisigId } = useParams<{ multisigId: string }>();
  const [vault, setVault] = useState<VaultRecord | null>(null);
  const [pendingTxs, setPendingTxs] = useState<VaultPendingTxRecord[]>([]);
  const [sentTxs, setSentTxs] = useState<VaultSentTxRecord[]>([]);
  const [balance, setBalance] = useState<bigint | null>(null);
  const [descriptorJson, setDescriptorJson] = useState("");
  const [proposalArtifact, setProposalArtifact] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const refresh = async () => {
    if (!multisigId) return;
    setError(null);
    const record = await getVault(multisigId);
    if (!record) {
      if (!vault) {
        setVault(null);
        setPendingTxs([]);
        setSentTxs([]);
        setBalance(null);
      }
      return;
    }
    setVault(record);
    const balanceResult = await fetchVaultBalance(record);
    const pending = await listPendingTxs(record.id);
    const sent = await listSentTxs(record.id);
    setBalance(balanceResult.grains);
    setPendingTxs(pending);
    setSentTxs(sent);
    if (balanceResult.degraded) {
      setError("Vault balance is partial because the RPC walk was degraded");
    }
    const exported = await exportMyCosignerDescriptor({
      vaultAccount: record.myVaultAccount,
      keyIndex: record.myKeyIndex,
      label: record.label,
    }).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : "Unable to derive multisig key");
      return null;
    });
    setDescriptorJson(exported?.json ?? "");
  };

  useEffect(() => {
    void refresh().catch((err) => setError(err instanceof Error ? err.message : "Unable to load vault"));
  }, [multisigId]);

  useEffect(() => {
    if (!multisigId) return;
    const interval = window.setInterval(() => {
      void refresh().catch(() => {
        // ignore background refresh errors; the next tick will retry
      });
    }, 15_000);
    const onFocus = () => {
      void refresh().catch(() => {
        // ignore background refresh errors; the next focus will retry
      });
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
  }, [multisigId]);

  if (!multisigId) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 text-sm text-slate-700 shadow-sm">
          Missing vault id.
        </div>
      </div>
    );
  }

  if (!vault) {
    return (
      <div className="flex h-full w-full flex-col bg-gradient-to-br from-slate-50 via-white to-amber-50">
        <Header onBack={() => navigate("/multisig")} name="Vault" />
        <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
            <h2 className="text-2xl font-semibold text-slate-950">Vault not found</h2>
            <p className="mt-2 text-sm text-slate-600">
              This vault id is not present in local storage on this machine.
            </p>
            <button
              type="button"
              onClick={() => navigate("/multisig")}
              className="mt-6 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white"
            >
              Back to multisig
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
      <Header onBack={() => navigate("/multisig")} name={vault.label} />

      <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto grid w-full max-w-6xl gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <section className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                    Vault overview
                  </div>
                  <h2 className="mt-2 text-3xl font-semibold text-slate-950">{vault.label}</h2>
                  <p className="mt-1 text-sm text-slate-500">
                    {vault.threshold}-of-{vault.total} multisig, mainnet only
                  </p>
                </div>
                <div className="rounded-3xl bg-slate-950 px-4 py-3 text-right text-white">
                  <div className="text-xs text-white/70">Balance</div>
                  <div className="mt-1 text-2xl font-semibold">
                    {balance === null ? "..." : `${formatGrains(balance)} PRL`}
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <ActionButton
                  label="Compose spend"
                  onClick={() => navigate(`/multisig/${vault.id}/send`)}
                  icon={<ArrowRight className="h-4 w-4" />}
                />
                <ActionButton
                  label="Import proposal"
                  onClick={() => navigate(`/multisig/${vault.id}/sign`)}
                  icon={<Download className="h-4 w-4" />}
                />
                <ActionButton
                  label="Copy address"
                  onClick={async () => {
                    await copyToClipboard(vault.pearlAddress);
                    setCopied("address");
                    window.setTimeout(() => setCopied(null), 1200);
                  }}
                  icon={<Copy className="h-4 w-4" />}
                  secondary
                  labelSuffix={copied === "address" ? "Copied" : shortHex(vault.pearlAddress, 12)}
                />
                <ActionButton
                  label="Copy descriptor"
                  onClick={async () => {
                    if (!descriptorJson) return;
                    await copyToClipboard(descriptorJson);
                    setCopied("descriptor");
                    window.setTimeout(() => setCopied(null), 1200);
                  }}
                  icon={<Copy className="h-4 w-4" />}
                  secondary
                  labelSuffix={copied === "descriptor" ? "Copied" : shortHex(vault.myPubkeyHex)}
                />
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-xl font-semibold text-slate-950">Pending transactions</h3>
                <button
                  type="button"
                  onClick={() => void refresh()}
                  className="text-sm font-medium text-amber-700 transition-colors hover:text-amber-800"
                >
                  Refresh
                </button>
              </div>

              <div className="mt-4 space-y-3">
                {pendingTxs.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-600">
                    No pending transactions yet.
                  </div>
                ) : (
                  pendingTxs.map((pending) => (
                    <div
                      key={pending.id}
                      className="rounded-3xl border border-slate-200 bg-slate-50 p-4"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <div className="text-sm font-semibold text-slate-950">
                            {pending.preview.destination.slice(0, 12)}...
                          </div>
                          <div className="mt-1 text-xs text-slate-500">
                            {pending.status.toUpperCase()} • {formatGrains(BigInt(pending.preview.amountGrains))} PRL
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => navigate(`/multisig/${vault.id}/tx/${pending.id}`)}
                            className="rounded-2xl bg-slate-950 px-3 py-2 text-xs font-semibold text-white"
                          >
                            Open
                          </button>
                          <button
                            type="button"
                            onClick={async () => {
                              await deletePendingTx(pending.id);
                              await refresh();
                            }}
                            className="rounded-2xl border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-xl font-semibold text-slate-950">Sent activity</h3>
                <div className="text-sm text-slate-500">{sentTxs.length} transactions</div>
              </div>

              <div className="mt-4 space-y-3">
                {sentTxs.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-600">
                    No sent transactions yet.
                  </div>
                ) : (
                  sentTxs.map((sent) => (
                    <div key={sent.id} className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-100">
                            <ArrowUpRight className="h-5 w-5 text-amber-700" />
                          </div>
                          <div>
                            <div className="text-sm font-semibold text-slate-950">
                              Sent to {sent.destination.slice(0, 12)}...
                            </div>
                            <div className="mt-1 text-xs text-slate-500">
                              {formatTimeAgo(sent.time)} • {formatFullDate(sent.time)}
                            </div>
                            <div className="mt-1 font-mono text-[11px] text-slate-500">
                              Tx ID: {shortHex(sent.txid, 10)}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-red-500">
                            -{formatGrains(BigInt(sent.amountGrains))} PRL
                          </div>
                          <div className="text-sm text-slate-600">
                            {sent.confirmations} confirmations
                          </div>
                          <div className="text-xs text-slate-500">
                            {sent.status === "confirmed" ? "Confirmed" : "Broadcast"}
                          </div>
                          {BigInt(sent.feeGrains) > 0n ? (
                            <div className="text-xs text-slate-500">
                              Fee: {formatGrains(BigInt(sent.feeGrains))} PRL
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-white/60">
                <Plus className="h-3.5 w-3.5" />
                Signer details
              </div>
              <div className="mt-4 space-y-4 text-sm text-white/80">
                <div>
                  <div className="text-white/60">Address</div>
                  <div className="mt-1 break-all font-mono text-xs text-white">
                    {vault.pearlAddress}
                  </div>
                </div>
                <div>
                  <div className="text-white/60">My origin path</div>
                  <div className="mt-1 font-mono text-xs text-white">{vault.myOriginPath}</div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-2xl bg-white/5 p-3">
                    <div className="text-white/60">My pubkey</div>
                    <div className="mt-1 font-mono text-xs text-white">
                      {shortHex(vault.myPubkeyHex)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-white/5 p-3">
                    <div className="text-white/60">Pending</div>
                    <div className="mt-1 text-lg font-semibold text-white">{pendingTxs.length}</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-950">Import proposal</h3>
              <p className="mt-2 text-sm text-slate-600">
                Paste a proposal artifact from Telegram, or a raw PSBT from a partner. The
                wallet will import either format into this local vault.
              </p>
              <textarea
                value={proposalArtifact}
                onChange={(e) => setProposalArtifact(e.target.value)}
                placeholder='{"version":1,"kind":"pearl-multisig-proposal",...} or cHNidP8B...'
                className="mt-4 min-h-32 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 font-mono text-xs text-slate-900 outline-none focus:border-slate-400"
              />
              <button
                type="button"
                disabled={proposalArtifact.trim().length === 0}
                onClick={async () => {
                  try {
                    const pending = await importProposalArtifactOrPsbt({
                      vault,
                      artifactText: proposalArtifact,
                    });
                    navigate(`/multisig/${vault.id}/tx/${pending.id}`);
                  } catch (err) {
                    setError(err instanceof Error ? err.message : "Unable to import proposal");
                  }
                }}
                className="mt-3 w-full rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Import proposal
              </button>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-950">Danger zone</h3>
              <p className="mt-2 text-sm text-slate-600">
                This removes the vault and all locally stored pending transactions.
              </p>
              <button
                type="button"
                onClick={async () => {
                  if (!window.confirm("Delete this vault from local storage?")) return;
                  await deleteVault(vault.id);
                  navigate("/multisig");
                }}
                className="mt-4 inline-flex items-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700"
              >
                <Trash2 className="h-4 w-4" />
                Delete vault
              </button>
            </div>
          </aside>
        </div>

        {error ? (
          <div className="mx-auto mt-6 max-w-6xl rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ActionButton(props: {
  label: string;
  onClick: () => void;
  icon: ReactNode;
  secondary?: boolean;
  labelSuffix?: string;
}) {
  return (
    <button
      type="button"
      onClick={props.onClick}
      className={`flex items-center justify-between rounded-2xl px-4 py-3 text-sm font-semibold transition-colors ${
        props.secondary
          ? "border border-slate-200 bg-white text-slate-900 hover:bg-slate-50"
          : "bg-slate-950 text-white hover:bg-slate-800"
      }`}
    >
      <span className="flex items-center gap-2">
        {props.icon}
        {props.label}
      </span>
      {props.labelSuffix ? (
        <span className="ml-3 max-w-28 truncate font-mono text-[11px] font-normal opacity-70">
          {props.labelSuffix}
        </span>
      ) : null}
    </button>
  );
}
