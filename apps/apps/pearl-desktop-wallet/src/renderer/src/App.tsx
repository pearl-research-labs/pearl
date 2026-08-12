import { Route, HashRouter as Router, Routes, useLocation, useNavigate } from "react-router-dom";
import { MajorUpgradeBanner } from "./components/MajorUpgradeBanner";
import ActivityPage from "./pages/ActivityPage";
import ChangePassword from "./pages/ChangePassword";
import CreateWallet from "./pages/create-wallet/CreateWallet";
import ImportAccount from "./pages/ImportAccount";
import CreateMultisigPage from "./pages/multisig/CreateMultisig";
import MultisigDashboardPage from "./pages/multisig/MultisigDashboard";
import MultisigRelaySignPage from "./pages/multisig/MultisigRelaySign";
import MultisigSendPage from "./pages/multisig/MultisigSend";
import MultisigTxPage from "./pages/multisig/MultisigTx";
import MultisigPage from "./pages/multisig/Page";
import ReceiveTransaction from "./pages/ReceiveTransaction";
import SendTransaction from "./pages/send-transaction/SendTransaction";
import WalletDashboard from "./pages/WalletDashboard";
import WalletUnlock from "./pages/WalletUnlock";
import WelcomePage from "./pages/WelcomePage";
import { SyncWallet } from "./SyncWallet";
import "./App.css";

function AppContent() {
	const location = useLocation();
	const navigate = useNavigate();

	return (
		<div className="relative flex h-screen w-full flex-col overflow-hidden bg-gradient-to-br from-gray-50 to-gray-100 font-sans text-gray-900 antialiased">
			<MajorUpgradeBanner />
			<div className={`min-h-0 flex-1`}>
				<Routes>
					<Route path="/" element={<WelcomePage />} />
					<Route path="/wallet" element={<WalletDashboard />} />
					<Route path="/send" element={<SendTransaction />} />
					<Route path="/receive" element={<ReceiveTransaction />} />
					<Route path="/unlock" element={<WalletUnlock />} />
					{/*  */}
					<Route path="/multisig" element={<MultisigPage />} />
					<Route path="/multisig/create" element={<CreateMultisigPage />} />
						<Route path="/multisig/:multisigId" element={<MultisigDashboardPage />} />
						<Route path="/multisig/:multisigId/send" element={<MultisigSendPage />} />
						<Route path="/multisig/:multisigId/tx/:txId" element={<MultisigTxPage />} />
						<Route path="/multisig/:multisigId/sign" element={<MultisigRelaySignPage />} />
						{/*  */}
					<Route path="/change-password" element={<ChangePassword />} />
					<Route path="/import-account" element={<ImportAccount />} />
					<Route path="/onboarding/create" element={<CreateWallet />} />
					<Route path="/activity" element={<ActivityPage onBack={() => navigate("/wallet")} />} />
					<Route
						path="*"
						element={
							<div style={{ color: "red", padding: "20px" }}>
								Route not found: {location.pathname}
							</div>
						}
					/>
				</Routes>
			</div>
		</div>
	);
}

function App() {
	return (
		<Router>
			<SyncWallet />
			<AppContent />
		</Router>
	);
}

export default App;
