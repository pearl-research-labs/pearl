import { ArrowLeft, CheckCircle2, Copy, Loader2, Send } from "lucide-react";
import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { pearlTxExplorerUrl } from "@/chains/pearl/network";
import { Header } from "@/pages/multisig/components/header";
import {
	broadcastPendingTx,
	exportProposalArtifact,
	getPendingTx,
	getSentTx,
	getVault,
	inspectPsbt,
	type PsbtSignerInfo,
	type VaultSentTxRecord,
	signPendingTx,
	type VaultPendingTxRecord,
	type VaultRecord,
} from "@/services/multisig";
import { copyToClipboard, formatGrains, shortHex } from "./helpers";

export default function MultisigTx() {
	const navigate = useNavigate();
	const { multisigId, txId } = useParams<{ multisigId: string; txId: string }>();
	const [vault, setVault] = useState<VaultRecord | null>(null);
	const [pending, setPending] = useState<VaultPendingTxRecord | null>(null);
	const [sent, setSent] = useState<VaultSentTxRecord | null>(null);
	const [info, setInfo] = useState<PsbtSignerInfo | null>(null);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [copied, setCopied] = useState(false);

	const refresh = useCallback(async () => {
		if (!multisigId || !txId) return;
		const record = await getVault(multisigId);
		const draft = await getPendingTx(txId);
		const sentRecord = await getSentTx(txId);
		setVault(record ?? null);
		setPending(draft ?? null);
		setSent(sentRecord ?? null);
		if (record && draft) {
			setInfo(inspectPsbt(draft.psbtBase64, record.threshold, record.sortedPubkeysHex));
		} else if (!draft && sentRecord) {
			setInfo(null);
		} else {
			setInfo(null);
		}
	}, [multisigId, txId]);

	useEffect(() => {
		void refresh().catch((err) =>
			setError(err instanceof Error ? err.message : "Unable to load transaction"),
		);
	}, [refresh]);

	useEffect(() => {
		if (!multisigId || !txId) return;
		const interval = window.setInterval(() => {
			void refresh().catch(() => {
				// ignore background refresh errors; the next tick will retry
			});
		}, 10_000);
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
	}, [multisigId, refresh, txId]);

	if (!multisigId || !txId) {
		return <div className="flex h-full items-center justify-center">Missing route parameters.</div>;
	}

	if (!vault) {
		return (
			<div className="flex h-full w-full flex-col bg-gradient-to-br from-slate-50 via-white to-amber-50">
				<Header onBack={() => navigate(`/multisig/${multisigId}`)} name="Pending tx" />
				<div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
					<div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
						<h2 className="text-2xl font-semibold text-slate-950">Transaction not found</h2>
						<p className="mt-2 text-sm text-slate-600">
							The local PSBT for this vault id is missing.
						</p>
					</div>
				</div>
			</div>
		);
	}

	if (!pending && sent) {
		const sentLabel = sent.status === "confirmed" ? "confirmed" : "broadcast";
		return (
			<div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
				<Header onBack={() => navigate(`/multisig/${multisigId}`)} name="Pending tx" />

				<div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
					<div className="mx-auto grid w-full max-w-4xl gap-6">
						<section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
							<div className="text-xs uppercase tracking-[0.16em] text-slate-500">Multisig activity</div>
							<h2 className="mt-2 text-3xl font-semibold text-slate-950">
								Transaction {sentLabel}
							</h2>
							<p className="mt-2 text-sm text-slate-600">
								This transaction left the pending queue and was moved into sent activity after
								{sent.status === "confirmed" ? " confirmation." : " broadcast."}
							</p>
						</section>

						<section className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
							<div className="flex items-start justify-between gap-4">
								<div>
									<div className="text-xs uppercase tracking-[0.16em] text-white/60">Tx ID</div>
									<div className="mt-2 font-mono text-sm text-white">{sent.txid}</div>
								</div>
								<div className="rounded-2xl bg-emerald-500/15 px-3 py-2 text-right">
									<div className="text-xs text-emerald-200">
										{sent.status === "confirmed" ? "Confirmed" : "Broadcast"}
									</div>
									<div className="mt-1 text-sm font-semibold text-white">
										{sent.confirmations} confirmations
									</div>
								</div>
							</div>
							<div className="mt-6 grid gap-3 text-sm text-white/80 sm:grid-cols-2">
								<div className="rounded-2xl border border-white/10 bg-white/5 p-4">
									<div className="text-xs text-white/50">Destination</div>
									<div className="mt-1 break-all font-mono text-xs text-white">{sent.destination}</div>
								</div>
								<div className="rounded-2xl border border-white/10 bg-white/5 p-4">
									<div className="text-xs text-white/50">Amount</div>
									<div className="mt-1 text-lg font-semibold text-white">
										{formatGrains(BigInt(sent.amountGrains))} PRL
									</div>
								</div>
								<div className="rounded-2xl border border-white/10 bg-white/5 p-4">
									<div className="text-xs text-white/50">Broadcast time</div>
									<div className="mt-1 text-sm text-white">{new Date(sent.time).toLocaleString()}</div>
								</div>
								<div className="rounded-2xl border border-white/10 bg-white/5 p-4">
									<div className="text-xs text-white/50">Fee</div>
									<div className="mt-1 text-lg font-semibold text-white">
										{formatGrains(BigInt(sent.feeGrains))} PRL
									</div>
								</div>
							</div>

							<div className="mt-6 flex flex-wrap gap-3">
								<ActionButton
									label="Back to vault"
									onClick={() => navigate(`/multisig/${multisigId}`)}
									icon={<ArrowLeft className="h-4 w-4" />}
								/>
							</div>
						</section>
					</div>
				</div>
			</div>
		);
	}

	if (!pending || !info) {
		return (
			<div className="flex h-full w-full flex-col bg-gradient-to-br from-slate-50 via-white to-amber-50">
				<Header onBack={() => navigate(`/multisig/${multisigId}`)} name="Pending tx" />
				<div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
					<div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
						<h2 className="text-2xl font-semibold text-slate-950">Transaction not found</h2>
						<p className="mt-2 text-sm text-slate-600">
							The local PSBT for this vault id is missing.
						</p>
					</div>
				</div>
			</div>
		);
	}

	const canBroadcast = info.thresholdMet && !info.foreignSignersHex.length;

	const handleSign = async () => {
		setBusy(true);
		setError(null);
		try {
			await new Promise((resolve) => window.setTimeout(resolve, 0));
			const updated = await signPendingTx({ vault, pending });
			setPending(updated);
			await refresh();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Unable to sign PSBT");
		} finally {
			setBusy(false);
		}
	};

	const handleBroadcast = async () => {
		setBusy(true);
		setError(null);
		try {
			await new Promise((resolve) => window.setTimeout(resolve, 0));
			const updated = await broadcastPendingTx({ vault, pending });
			setPending(updated);
			await refresh();
		} catch (err) {
			setError(err instanceof Error ? err.message : "Unable to broadcast transaction");
		} finally {
			setBusy(false);
		}
	};

	return (
		<div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-white to-amber-50">
			<Header onBack={() => navigate(`/multisig/${vault.id}`)} name="Pending tx" />

			<div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
				<div className="mx-auto grid w-full max-w-5xl gap-6 xl:grid-cols-[1.05fr_0.95fr]">
					<section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
						<div className="flex flex-wrap items-start justify-between gap-4">
							<div>
								<div className="text-xs uppercase tracking-[0.16em] text-slate-500">PSBT</div>
								<h2 className="mt-2 text-3xl font-semibold text-slate-950">
									{pending.preview.destination.slice(0, 12)}...
								</h2>
								<p className="mt-1 text-sm text-slate-500">
									{pending.status.toUpperCase()} • {info.signerCount} signatures collected
								</p>
							</div>
							<div className="rounded-3xl bg-slate-950 px-4 py-3 text-right text-white">
								<div className="text-xs text-white/60">Amount</div>
								<div className="mt-1 text-2xl font-semibold">
									{formatGrains(BigInt(pending.preview.amountGrains))} PRL
								</div>
							</div>
						</div>

						<div className="mt-6 grid gap-3 sm:grid-cols-2">
							<ActionButton
								label="Add my signature"
								onClick={() => void handleSign()}
								disabled={busy}
								icon={
									busy ? (
										<Loader2 className="h-4 w-4 animate-spin" />
									) : (
										<CheckCircle2 className="h-4 w-4" />
									)
								}
							/>
							<ActionButton
								label={busy ? "Sending..." : "Send transaction"}
								onClick={() => void handleBroadcast()}
								disabled={busy || !canBroadcast}
								icon={
									busy ? (
										<Loader2 className="h-4 w-4 animate-spin" />
									) : (
										<Send className="h-4 w-4" />
									)
								}
							/>
							<ActionButton
								label="Copy proposal artifact"
								onClick={async () => {
									await copyToClipboard(exportProposalArtifact({ vault, pending }));
									setCopied(true);
									window.setTimeout(() => setCopied(false), 1200);
								}}
								icon={<Copy className="h-4 w-4" />}
								secondary
								disabled={busy}
								labelSuffix={copied ? "Copied" : shortHex(pending.id)}
							/>
							<ActionButton
								label="Copy PSBT"
								onClick={async () => {
									await copyToClipboard(pending.psbtBase64);
									setCopied(true);
									window.setTimeout(() => setCopied(false), 1200);
								}}
								icon={<Copy className="h-4 w-4" />}
								secondary
								disabled={busy}
							/>
							<ActionButton
								label="Copy txid"
								onClick={async () => {
									const txid = pending.expectedTxid ?? pending.txid;
									if (!txid) return;
									await copyToClipboard(txid);
									setCopied(true);
									window.setTimeout(() => setCopied(false), 1200);
								}}
								icon={<Copy className="h-4 w-4" />}
								secondary
								disabled={busy || !pending.expectedTxid}
								labelSuffix={pending.expectedTxid ? shortHex(pending.expectedTxid, 10) : "Pending"}
							/>
							{pending.txid ? (
								<ActionButton
									label="Open explorer"
									onClick={() =>
										window.open(
											pearlTxExplorerUrl("mainnet", pending.txid!),
											"_blank",
											"noopener,noreferrer",
										)
									}
									icon={<ArrowLeft className="h-4 w-4 rotate-180" />}
									secondary
									disabled={busy}
								/>
							) : null}
						</div>

						{busy ? (
							<div className="mt-4 inline-flex items-center gap-2 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
								<Loader2 className="h-4 w-4 animate-spin" />
								{canBroadcast ? "Sending transaction..." : "Signing PSBT..."}
							</div>
						) : null}

						{error ? (
							<div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
								{error}
							</div>
						) : null}

						<div className="mt-6 rounded-3xl border border-slate-200 bg-slate-50 p-4">
							<div className="text-sm font-semibold text-slate-950">Output summary</div>
							<div className="mt-3 space-y-3 text-sm text-slate-700">
								{info.outputs.map((output, index) => (
									<div
										key={`${output.scriptHex}_${index}`}
										className="flex items-center justify-between gap-4"
									>
										<span className="font-mono text-xs text-slate-500">
											{output.address ?? shortHex(output.scriptHex)}
										</span>
										<span className="font-semibold text-slate-950">
											{formatGrains(output.amountGrains)} PRL
										</span>
									</div>
								))}
							</div>
						</div>
					</section>

          <aside className="min-w-0 space-y-6">
						<div className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
							<h3 className="text-xl font-semibold">Signature state</h3>
							<div className="mt-4 space-y-3 text-sm text-white/80">
								<div className="flex items-center justify-between gap-4">
									<span>Threshold met</span>
									<span className="font-semibold text-white">
										{info.thresholdMet ? "Yes" : "No"}
									</span>
								</div>
								<div className="flex items-center justify-between gap-4">
									<span>Predicted txid</span>
									<span className="max-w-[12rem] truncate font-mono text-xs text-white/90">
										{pending.expectedTxid ?? "Computing..."}
									</span>
								</div>
								<div className="flex items-center justify-between gap-4">
									<span>Foreign signers</span>
									<span className="font-semibold text-white">{info.foreignSignersHex.length}</span>
								</div>
								<div className="flex items-center justify-between gap-4">
									<span>Fee</span>
									<span className="font-semibold text-white">
										{info.feeUnknown ? "Unknown" : `${formatGrains(info.feeGrains)} PRL`}
									</span>
								</div>
								<div className="flex items-center justify-between gap-4">
									<span>Change</span>
									<span className="font-semibold text-white">
										{formatGrains(BigInt(pending.preview.changeGrains))} PRL
									</span>
								</div>
							</div>
						</div>

						<div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
							<h3 className="text-lg font-semibold text-slate-950">Collected signers</h3>
							<div className="mt-4 space-y-2">
								{info.signersHex.length === 0 ? (
									<div className="text-sm text-slate-500">No signatures yet.</div>
								) : (
									info.signersHex.map((signer) => (
										<div
											key={signer}
											className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-700"
										>
											{signer}
										</div>
									))
								)}
							</div>
						</div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-950">Raw PSBT</h3>
              <pre className="mt-4 max-h-60 max-w-full overflow-auto whitespace-pre-wrap break-words rounded-2xl bg-slate-950 p-4 font-mono text-[11px] leading-5 text-slate-100">
                {pending.psbtBase64}
              </pre>
            </div>
					</aside>
				</div>
			</div>
		</div>
	);
}

function ActionButton(props: {
	label: string;
	onClick: () => void;
	icon: ReactNode;
	disabled?: boolean;
	secondary?: boolean;
	labelSuffix?: string;
}) {
	return (
		<button
			type="button"
			onClick={props.onClick}
			disabled={props.disabled}
			className={`flex items-center justify-between rounded-2xl px-4 py-3 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
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
