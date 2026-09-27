import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowRight, Loader2 } from "lucide-react";
import { Header } from "@/pages/multisig/components/header";
import { formatGrains } from "./helpers";
import {
  composeVaultSend,
  getVault,
  persistComposedAsPending,
  type ComposedVaultSend,
  type VaultRecord,
} from "@/services/multisig";

function parsePearlAmountToGrains(value: string): bigint {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error("Please enter an amount");
  }
  if (!/^\d+(\.\d+)?$/.test(trimmed)) {
    throw new Error("Please enter a valid amount");
  }

  const [whole, frac = ""] = trimmed.split(".");
  if (frac.length > 8) {
    throw new Error("Pearl amounts can have at most 8 decimal places");
  }

  const wholeGrains = BigInt(whole) * 100_000_000n;
  const fracGrains = BigInt((frac + "00000000").slice(0, 8));
  return wholeGrains + fracGrains;
}

function parseFeeRateSatPerVbyte(value: string): bigint {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error("Please enter a fee rate");
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error("Please enter a valid fee rate");
  }
  return BigInt(Math.ceil(parsed));
}

export default function MultisigSend() {
  const navigate = useNavigate();
  const { multisigId } = useParams<{ multisigId: string }>();
  const [vault, setVault] = useState<VaultRecord | null>(null);
  const [destination, setDestination] = useState("");
  const [amount, setAmount] = useState("");
  const [feerate, setFeerate] = useState("2");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ComposedVaultSend | null>(null);

  useEffect(() => {
    if (!multisigId) return;
    void getVault(multisigId).then((value) => setVault(value ?? null));
  }, [multisigId]);

  if (!multisigId) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-600">
        Missing vault id.
      </div>
    );
  }

  if (!vault) {
    return (
      <div className="flex h-full w-full flex-col bg-gradient-to-br from-slate-50 via-white to-amber-50">
        <Header onBack={() => navigate(`/multisig/${multisigId}`)} name="Compose spend" />
        <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
            Vault not found.
          </div>
        </div>
      </div>
    );
  }

  const onCompose = async () => {
    setBusy(true);
    setError(null);
    try {
      const amountGrains = parsePearlAmountToGrains(amount);
      const feerateSatPerVbyte = parseFeeRateSatPerVbyte(feerate);
      const composed = await composeVaultSend({
        vault,
        destination: destination.trim(),
        amountGrains,
        feerateSatPerVbyte,
      });
      setPreview(composed);
      const pending = await persistComposedAsPending({
        vault,
        psbtBase64: composed.psbtBase64,
        preview: {
          destination: composed.destination,
          amountGrains: composed.amountGrains.toString(),
          feeGrains: composed.feeGrains.toString(),
          changeGrains: composed.changeGrains.toString(),
          inputCount: composed.utxos.length,
        },
      });
      navigate(`/multisig/${vault.id}/tx/${pending.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to compose spend");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
      <Header onBack={() => navigate(`/multisig/${vault.id}`)} name="Compose spend" />

      <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto grid w-full max-w-4xl gap-6 xl:grid-cols-[1.15fr_0.85fr]">
          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-2xl font-semibold text-slate-950">Create a vault PSBT</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              The wallet will select UTXOs, build a PSBT, and store it locally as a pending
              multisig transaction.
            </p>

            {error ? (
              <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                {error}
              </div>
            ) : null}

            <div className="mt-6 space-y-4">
              <label className="space-y-2">
                <div className="text-sm font-medium text-slate-700">Destination address</div>
                <input
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 font-mono text-sm outline-none focus:border-slate-400"
                  placeholder="prl1..."
                />
              </label>
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="space-y-2">
                  <div className="text-sm font-medium text-slate-700">Amount (PRL)</div>
                  <input
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 font-mono text-sm outline-none focus:border-slate-400"
                    placeholder="0.5"
                    inputMode="decimal"
                  />
                </label>
                <label className="space-y-2">
                  <div className="text-sm font-medium text-slate-700">Fee rate (sat/vB)</div>
                  <input
                    value={feerate}
                    onChange={(e) => setFeerate(e.target.value)}
                    className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 font-mono text-sm outline-none focus:border-slate-400"
                    placeholder="2"
                    inputMode="decimal"
                  />
                </label>
              </div>

              <button
                type="button"
                disabled={busy}
                onClick={() => void onCompose()}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
                Compose and save draft
              </button>
            </div>
          </section>

          <aside className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
            <h3 className="text-xl font-semibold">Vault context</h3>
            <div className="mt-4 space-y-4 text-sm text-white/80">
              <div>
                <div className="text-white/60">Vault</div>
                <div className="mt-1 text-lg font-semibold text-white">{vault.label}</div>
              </div>
              <div>
                <div className="text-white/60">Address</div>
                <div className="mt-1 break-all font-mono text-xs text-white">{vault.pearlAddress}</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-white/5 p-3">
                  <div className="text-white/60">Threshold</div>
                  <div className="mt-1 text-lg font-semibold text-white">
                    {vault.threshold}-of-{vault.total}
                  </div>
                </div>
                <div className="rounded-2xl bg-white/5 p-3">
                  <div className="text-white/60">My signer</div>
                  <div className="mt-1 font-mono text-xs text-white">{vault.myOriginPath}</div>
                </div>
              </div>
            </div>

            {preview ? (
              <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4 text-sm text-white/80">
                <div className="flex items-center justify-between">
                  <span>Fee</span>
                  <span className="font-semibold text-white">
                    {formatGrains(preview.feeGrains)} PRL
                  </span>
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span>Change</span>
                  <span className="font-semibold text-white">
                    {formatGrains(preview.changeGrains)} PRL
                  </span>
                </div>
              </div>
            ) : null}
          </aside>
        </div>
      </div>
    </div>
  );
}
