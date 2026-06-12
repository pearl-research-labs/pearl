import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AlertTriangle, CheckCircle2, Copy, Loader2, Send } from "lucide-react";
import { Header } from "@/pages/multisig/components/header";
import { copyToClipboard, formatGrains, shortHex } from "./helpers";
import {
  exportProposalArtifact,
  importProposalArtifactOrPsbt,
} from "@/services/multisig";
import {
  getVault,
  inspectPsbt,
  broadcastPendingTx,
  signVaultPsbt,
  savePendingTx,
  type VaultPendingTxRecord,
  type VaultRecord,
  type PsbtSignerInfo,
} from "@/services/multisig";

export default function MultisigRelaySign() {
  const navigate = useNavigate();
  const { multisigId } = useParams<{ multisigId: string }>();
  const [vault, setVault] = useState<VaultRecord | null>(null);
  const [artifactText, setArtifactText] = useState("");
  const [proposal, setProposal] = useState<VaultPendingTxRecord | null>(null);
  const [info, setInfo] = useState<PsbtSignerInfo | null>(null);
  const [exportedArtifact, setExportedArtifact] = useState("");
  const [busy, setBusy] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!multisigId) return;
    void getVault(multisigId).then((value) => setVault(value ?? null));
  }, [multisigId]);

  const canLoad = useMemo(() => artifactText.trim().length > 0, [artifactText]);

  const loadProposal = async () => {
    if (!vault) throw new Error("Vault not loaded");
    const imported = await importProposalArtifactOrPsbt({ vault, artifactText });
    const signerInfo = inspectPsbt(imported.psbtBase64, vault.threshold, vault.sortedPubkeysHex);
    setProposal(imported);
    setInfo(signerInfo);
    return { imported, signerInfo };
  };

  const handleLoadAndSign = async () => {
    setBusy(true);
    setError(null);
    setCopyState("idle");
    setExportedArtifact("");
    try {
      const { imported } = await loadProposal();
      const psbt = await signVaultPsbt({ vault: vault!, psbtBase64: imported.psbtBase64 });
      const signed = {
        ...imported,
        psbtBase64: psbt.psbtBase64,
        updatedAt: Date.now(),
      };
      const updatedInfo = inspectPsbt(signed.psbtBase64, vault!.threshold, vault!.sortedPubkeysHex);
      const signedRecord = {
        ...signed,
        signersHex: updatedInfo.signersHex,
      };
      await savePendingTx(signedRecord);
      setProposal(signedRecord);
      setInfo(updatedInfo);
      setExportedArtifact(exportProposalArtifact({ vault: vault!, pending: signedRecord }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to sign proposal");
    } finally {
      setBusy(false);
    }
  };

  const handleBroadcast = async () => {
    if (!vault || !proposal || !info || !info.thresholdMet) return;
    setBusy(true);
    setError(null);
    try {
      const broadcast = await broadcastPendingTx({ vault, pending: proposal });
      setProposal(broadcast);
      setExportedArtifact(exportProposalArtifact({ vault, pending: broadcast }));
      navigate(`/multisig/${vault.id}/tx/${broadcast.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to broadcast transaction");
    } finally {
      setBusy(false);
    }
  };

  if (!multisigId) {
    return <div className="flex h-full items-center justify-center">Missing vault id.</div>;
  }

  if (!vault) {
    return (
      <div className="flex h-full w-full flex-col bg-gradient-to-br from-slate-50 via-white to-amber-50">
        <Header onBack={() => navigate(`/multisig/${multisigId}`)} name="Proposal sign" />
        <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
            Loading vault...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
      <Header onBack={() => navigate(`/multisig/${vault.id}`)} name="Proposal sign" />

      <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto grid w-full max-w-5xl gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-2xl font-semibold text-slate-950">Sign proposal artifact</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Paste the proposal blob from Telegram, or a raw PSBT if your partner sent one.
              The wallet will import it locally, sign it, and let you copy the updated payload
              back to the next signer.
            </p>

            <div className="mt-6 space-y-4">
              <label className="space-y-2">
                <div className="text-sm font-medium text-slate-700">Proposal artifact</div>
                <textarea
                  value={artifactText}
                  onChange={(e) => setArtifactText(e.target.value)}
                  className="min-h-56 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 font-mono text-sm outline-none focus:border-slate-400"
                  placeholder='{"version":1,"kind":"pearl-multisig-proposal",...}'
                />
              </label>

              <button
                type="button"
                disabled={busy || !canLoad}
                onClick={() => void handleLoadAndSign()}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                Import and sign
              </button>
            </div>

            {error ? (
              <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                {error}
              </div>
            ) : null}

            {proposal ? (
              <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
                Proposal imported locally. You can now sign, copy the updated artifact, or
                broadcast once the threshold is met.
              </div>
            ) : null}
          </section>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
              <h3 className="text-xl font-semibold">Vault context</h3>
              <div className="mt-4 space-y-3 text-sm text-white/80">
                <div>
                  <div className="text-white/60">Vault</div>
                  <div className="mt-1 text-lg font-semibold text-white">{vault.label}</div>
                </div>
                <div>
                  <div className="text-white/60">Address</div>
                  <div className="mt-1 break-all font-mono text-xs text-white">{vault.pearlAddress}</div>
                </div>
                <div>
                  <div className="text-white/60">Signer</div>
                  <div className="mt-1 font-mono text-xs text-white">{shortHex(vault.myPubkeyHex)}</div>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-950">Proposal preview</h3>
              {proposal && info ? (
                <div className="mt-4 space-y-3 text-sm text-slate-700">
                  <div className="flex items-center justify-between gap-4">
                    <span>Kind</span>
                    <span className="font-semibold text-slate-950">proposal artifact</span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>Threshold met</span>
                    <span className="font-semibold text-slate-950">
                      {info.thresholdMet ? "Yes" : "No"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>Fee</span>
                    <span className="font-semibold text-slate-950">
                      {info.feeUnknown ? "Unknown" : `${formatGrains(info.feeGrains)} PRL`}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="mt-4 text-sm text-slate-500">
                  Paste a proposal artifact to inspect it locally.
                </div>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-950">Updated artifact</h3>
              {exportedArtifact ? (
                <>
                  <textarea
                    readOnly
                    value={exportedArtifact}
                    className="mt-3 min-h-48 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 font-mono text-xs text-slate-800 outline-none"
                  />
                  <button
                    type="button"
                    onClick={async () => {
                      await copyToClipboard(exportedArtifact);
                      setCopyState("copied");
                      window.setTimeout(() => setCopyState("idle"), 1500);
                    }}
                    className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
                  >
                    {copyState === "copied" ? <CheckCircle2 className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copyState === "copied" ? "Copied" : "Copy updated artifact"}
                  </button>
                </>
              ) : (
                <div className="mt-3 flex items-center gap-2 text-sm text-slate-600">
                  <AlertTriangle className="h-4 w-4" />
                  The next signer needs the same proposal blob after your signature is added.
                </div>
              )}
              {proposal && info?.thresholdMet ? (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void handleBroadcast()}
                  className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-900 transition-colors hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  Broadcast final transaction
                </button>
              ) : null}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
