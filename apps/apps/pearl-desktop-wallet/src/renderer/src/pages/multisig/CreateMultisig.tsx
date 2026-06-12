import { Check, Copy } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Header } from "@/pages/multisig/components/header";
import { createVault, exportMyCosignerDescriptor, listVaults } from "@/services/multisig";
import { copyToClipboard, descriptorTextToPubkeyHex, shortHex } from "./helpers";

type CosignerInput = {
	id: string;
	value: string;
	error: string | null;
};

function newCosignerInput(): CosignerInput {
	return { id: crypto.randomUUID(), value: "", error: null };
}

function createCosignerSlots(count: number): CosignerInput[] {
	return Array.from({ length: count }, () => newCosignerInput());
}

export default function CreateMultisig() {
	const navigate = useNavigate();
	const [label, setLabel] = useState("Pearl Vault");
	const [threshold, setThreshold] = useState(2);
	const [totalSigners, setTotalSigners] = useState(3);
	const [myVaultAccount, setMyVaultAccount] = useState(0);
	const [myKeyIndex, setMyKeyIndex] = useState(0);
	const [myDescriptorJson, setMyDescriptorJson] = useState("");
	const [myPubkeyHex, setMyPubkeyHex] = useState("");
	const [myOriginPath, setMyOriginPath] = useState("");
	const [copiedDescriptor, setCopiedDescriptor] = useState(false);
	const [cosigners, setCosigners] = useState<CosignerInput[]>(() => createCosignerSlots(2));
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		const run = async () => {
			try {
				setError(null);
				const exported = await exportMyCosignerDescriptor({
					vaultAccount: myVaultAccount,
					keyIndex: myKeyIndex,
					label: label.trim() || "Pearl Vault",
				});
				if (cancelled) return;
				setMyDescriptorJson(exported.json);
				setMyPubkeyHex(exported.pubkeyHex);
				setMyOriginPath(exported.originPath);
			} catch (err) {
				if (cancelled) return;
				setError(err instanceof Error ? err.message : "Unable to derive local multisig key");
			}
		};
		void run();
		return () => {
			cancelled = true;
		};
	}, [label, myKeyIndex, myVaultAccount]);

	useEffect(() => {
		let cancelled = false;
		const run = async () => {
			try {
				const existing = await listVaults();
				if (cancelled) return;
				const max = existing.reduce((currentMax, vault) => {
					return vault.myVaultAccount > currentMax ? vault.myVaultAccount : currentMax;
				}, -1);
				setMyVaultAccount(max + 1);
			} catch {
				if (cancelled) return;
				setMyVaultAccount(0);
			}
		};
		void run();
		return () => {
			cancelled = true;
		};
	}, []);

	useEffect(() => {
		const targetCount = Math.max(0, Number.isFinite(totalSigners) ? totalSigners - 1 : 0);
		setCosigners((items) => {
			if (items.length === targetCount) return items;
			if (items.length < targetCount) {
				return [...items, ...createCosignerSlots(targetCount - items.length)];
			}
			return items.slice(0, targetCount);
		});
	}, [totalSigners]);

	const totalCosigners = useMemo(() => {
		const imported = cosigners
			.map((item) => {
				try {
					return descriptorTextToPubkeyHex(item.value).pubkeyHex.toLowerCase();
				} catch {
					return null;
				}
			})
			.filter((v): v is string => Boolean(v));
		const local = myPubkeyHex.trim().length > 0 ? [myPubkeyHex.toLowerCase()] : [];
		const unique = new Set<string>([...local, ...imported]);
		return unique.size;
	}, [cosigners, myPubkeyHex]);

	const updateCosigner = (id: string, value: string) => {
		setCosigners((items) =>
			items.map((item) => (item.id === id ? { ...item, value, error: null } : item)),
		);
	};

	const validateCosigners = () => {
		let hasError = false;
		setCosigners((items) =>
			items.map((item) => {
				if (item.value.trim().length === 0) {
					return item;
				}
				try {
					descriptorTextToPubkeyHex(item.value);
					return { ...item, error: null };
				} catch (err) {
					hasError = true;
					return { ...item, error: err instanceof Error ? err.message : "Invalid cosigner" };
				}
			}),
		);
		return !hasError;
	};

	const handleCreate = async () => {
		setError(null);
		if (!validateCosigners()) return;
		if (!myPubkeyHex) {
			setError("Local signer is not ready yet");
			return;
		}
		if (!Number.isInteger(totalSigners) || totalSigners < 1) {
			setError("Total signers must be at least 1");
			return;
		}
		if (threshold > totalSigners) {
			setError("Threshold cannot exceed total signers");
			return;
		}
		if (cosigners.length !== Math.max(0, totalSigners - 1)) {
			setError("Cosigner slots do not match total signers");
			return;
		}
		const parsed = cosigners
			.map((item) => item.value.trim())
			.filter(Boolean)
			.map((value) => descriptorTextToPubkeyHex(value).pubkeyHex.toLowerCase());
		const cosignerPubkeysHex = Array.from(new Set([myPubkeyHex.toLowerCase(), ...parsed]));
		if (parsed.length !== cosigners.length) {
			setError("Fill in every cosigner slot");
			return;
		}
		if (cosignerPubkeysHex.length !== totalSigners) {
			setError("Each signer must be unique");
			return;
		}
		if (threshold > cosignerPubkeysHex.length) {
			setError("Threshold cannot exceed the number of cosigners");
			return;
		}
		setBusy(true);
		try {
			const vault = await createVault({
				label,
				threshold,
				cosignerPubkeysHex,
				myPubkeyHex,
				myVaultAccount,
				myKeyIndex,
				network: "mainnet",
			});
			navigate(`/multisig/${vault.id}`);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Unable to create vault");
		} finally {
			setBusy(false);
		}
	};

	return (
		<div className="flex h-full w-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-amber-50/40 to-white">
			<Header onBack={() => navigate("/multisig")} name="Create vault" />

			<div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
				<div className="mx-auto grid w-full max-w-5xl gap-6 xl:grid-cols-[1.3fr_0.9fr]">
					<section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
						<h2 className="text-2xl font-semibold text-slate-950">Vault details</h2>
						<p className="mt-2 text-sm leading-6 text-slate-600">
							Pick a label, threshold, and derivation slot for your local signer. The generated
							descriptor is the piece you share with the other cosigners.
						</p>

						{error ? (
							<div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
								{error}
							</div>
						) : null}

						<div className="mt-6 grid gap-4 sm:grid-cols-2">
							<label className="space-y-2">
								<div className="text-sm font-medium text-slate-700">Threshold</div>
								<input
									type="number"
									min={1}
									max={15}
									value={threshold}
									onChange={(e) => setThreshold(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none ring-0 transition-colors focus:border-slate-400"
								/>
							</label>
							<label className="space-y-2">
								<div className="text-sm font-medium text-slate-700">Total signers</div>
								<input
									type="number"
									min={1}
									max={15}
									value={totalSigners}
									onChange={(e) => setTotalSigners(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none ring-0 transition-colors focus:border-slate-400"
								/>
							</label>
							<label className="space-y-2 sm:col-span-2">
								<div className="text-sm font-medium text-slate-700">Label</div>
								<input
									value={label}
									onChange={(e) => setLabel(e.target.value)}
									className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none ring-0 transition-colors placeholder:text-slate-400 focus:border-slate-400"
									placeholder="Treasury vault"
								/>
							</label>
							<label className="space-y-2">
								<div className="text-sm font-medium text-slate-700">Vault account</div>
								<input
									type="number"
									min={0}
									value={myVaultAccount}
									onChange={(e) => setMyVaultAccount(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none ring-0 transition-colors focus:border-slate-400"
								/>
							</label>
							<label className="space-y-2">
								<div className="text-sm font-medium text-slate-700">Key index</div>
								<input
									type="number"
									min={0}
									value={myKeyIndex}
									onChange={(e) => setMyKeyIndex(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none ring-0 transition-colors focus:border-slate-400"
								/>
							</label>
						</div>

						<div className="mt-6 rounded-3xl border border-slate-200 bg-slate-50 p-5">
							<div className="flex items-center justify-between gap-4">
								<div>
									<div className="text-sm font-medium text-slate-700">Your cosigner</div>
									<div className="mt-1 text-xs text-slate-500">{myOriginPath || "Deriving..."}</div>
								</div>
								<button
									type="button"
									onClick={async () => {
										await copyToClipboard(myDescriptorJson);
										setCopiedDescriptor(true);
										window.setTimeout(() => setCopiedDescriptor(false), 1500);
									}}
									className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
								>
									{copiedDescriptor ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
									{copiedDescriptor ? "Copied" : "Copy descriptor"}
								</button>
							</div>
							<div className="mt-4 break-all rounded-2xl bg-white p-4 font-mono text-xs text-slate-700">
								{myDescriptorJson || "Descriptor will appear here once the signer key is derived."}
							</div>
							<div className="mt-3 text-xs text-slate-500">
								The wallet always includes your own pubkey in the vault set. Share this JSON with
								the other cosigners, and paste their descriptors below.
							</div>
							<div className="mt-3 flex items-center justify-between gap-4 text-sm text-slate-600">
								<span>m-of-n</span>
								<span className="font-semibold text-slate-900">
									{threshold}-of-{totalSigners}
								</span>
							</div>
						</div>
					</section>

					<aside className="rounded-3xl border border-slate-200 bg-slate-950 p-6 text-white shadow-xl shadow-slate-950/10">
						<div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-white/60">
							Cosigners
						</div>
						<h2 className="mt-2 text-2xl font-semibold">Add the other pubkeys</h2>
						<p className="mt-2 text-sm leading-6 text-white/70">
							Paste the exported descriptor JSON from each other signer, or a raw x-only pubkey.
						</p>

						<div className="mt-5 space-y-3">
							{cosigners.map((item, index) => (
								<div key={item.id} className="rounded-3xl border border-white/10 bg-white/5 p-4">
									<div className="text-sm font-medium text-white">Cosigner {index + 1}</div>
									<textarea
										value={item.value}
										onChange={(e) => updateCosigner(item.id, e.target.value)}
										placeholder='{"version":1,"type":"pearl-multisig-pubkey"...}'
										className="mt-3 min-h-28 w-full rounded-2xl border border-white/10 bg-slate-950/70 p-3 font-mono text-xs text-white outline-none placeholder:text-white/30 focus:border-amber-300"
									/>
									{item.error ? (
										<div className="mt-2 text-xs text-amber-300">{item.error}</div>
									) : null}
								</div>
							))}
						</div>

						<div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4 text-sm text-white/80">
							<div className="mt-2 flex items-center justify-between gap-4">
								<span>Threshold</span>
								<span className="font-semibold text-white">{threshold}</span>
							</div>
							<div className="mt-2 flex items-center justify-between gap-4">
								<span>Total signers</span>
								<span className="font-semibold text-white">{totalSigners}</span>
							</div>
							<div className="mt-2 flex items-center justify-between gap-4">
								<span>Configured pubkeys</span>
								<span className="font-semibold text-white">{totalCosigners}</span>
							</div>
							<div className="mt-2 flex items-center justify-between gap-4">
								<span>Local pubkey</span>
								<span className="font-mono text-xs text-white">{shortHex(myPubkeyHex || "")}</span>
							</div>
						</div>

						<button
							type="button"
							disabled={busy || !myPubkeyHex}
							onClick={() => void handleCreate()}
							className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-amber-300 px-4 py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-60"
						>
							{busy ? "Creating..." : "Create vault"}
						</button>
					</aside>
				</div>
			</div>
		</div>
	);
}
